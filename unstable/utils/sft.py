"""
Supervised Fine-Tuning (SFT) script for PEFT models.

Data format: CSV files with columns pid, prompt, completion, reward, env_id, step_info
(same format as training_data/train_data_step_*.csv, or curated output from memory.py).

Checkpoints are saved under outputs/ in the same format used by the RL framework,
so they can be loaded directly via checkpoint_cfg in grpo.yaml / build_peft_model().

Workflow:
    # 1. Curate data
    python memory.py \\
        --data "outputs/.../training_data/*.csv" \\
        --wins_only --from_step 100 \\
        --out curated/simpletak_wins.csv

    # 2. Train
    python -m unstable.utils.sft \\
        --model Qwen/Qwen3-1.7B-Base \\
        --data curated/simpletak_wins.csv \\
        --run SimpleTak-SFT \\
        --epochs 3 --lr 5e-5 --batch_size 4
"""

import argparse
import csv
import datetime
import glob
import pathlib

import torch
import wandb
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler

from unstable.learner.models import build_peft_model
from unstable.utils.logger import setup_logger
from unstable.utils.templates import DEFAULT_LORA_CFG


class SFTDataset(Dataset):
    def __init__(self, csv_paths: list[str], tokenizer, max_len: int = 4096):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples: list[dict] = []
        for path in csv_paths:
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.samples.append({
                        "prompt": row["prompt"],
                        "completion": row["completion"],
                    })

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt_ids = self.tokenizer(sample["prompt"], add_special_tokens=False).input_ids
        completion_ids = self.tokenizer(sample["completion"], add_special_tokens=False).input_ids
        eos = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []
        full_ids = (prompt_ids + completion_ids + eos)[: self.max_len]
        prompt_len = min(len(prompt_ids), self.max_len)
        return {"input_ids": full_ids, "prompt_len": prompt_len}


def collate_fn(batch):
    max_len = max(len(s["input_ids"]) for s in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, s in enumerate(batch):
        ids = torch.tensor(s["input_ids"], dtype=torch.long)
        seq_len = len(ids)
        input_ids[i, :seq_len] = ids
        attention_mask[i, :seq_len] = 1
        comp_start = s["prompt_len"]
        labels[i, comp_start:seq_len] = ids[comp_start:]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _build_output_dirs(run_name: str) -> dict[str, pathlib.Path]:
    now = datetime.datetime.now()
    base = pathlib.Path("outputs") / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S") / run_name
    dirs = {}
    for name in ("checkpoints", "logs"):
        d = base / name
        d.mkdir(parents=True, exist_ok=True)
        dirs[name] = d
    return dirs


def train_sft(args):
    dirs = _build_output_dirs(args.run)
    logger = setup_logger("sft", str(dirs["logs"]))
    logger.info(f"Output dir: {dirs['checkpoints'].parent}")
    use_wandb = getattr(args, "wandb_project", None) is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run,
            config=vars(args),
        )
        wandb.define_metric("*", step_metric="step")
    csv_paths = sorted({p for pattern in args.data for p in glob.glob(pattern, recursive=True)})
    if not csv_paths: raise FileNotFoundError(f"No CSV files matched: {args.data}")
    logger.info(f"Found {len(csv_paths)} CSV file(s)")
    lora_cfg = {
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": args.target_modules or DEFAULT_LORA_CFG["target_modules"],
    }
    checkpoint_cfg = {"path": args.checkpoint, "revision": None} if args.checkpoint else {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = build_peft_model(
        args.model, device, lora_cfg, checkpoint_cfg=checkpoint_cfg, freeze_base=True
    )
    model.config.use_cache = False
    if args.gradient_checkpointing: model.gradient_checkpointing_enable()
    dataset = SFTDataset(csv_paths, tokenizer, max_len=args.max_len)
    logger.info(f"Dataset size: {len(dataset)} samples")
    if len(dataset) == 0: raise RuntimeError("Dataset is empty — check data paths")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False)
    policy_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(policy_params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * (len(loader) + args.grad_accum_steps - 1) // args.grad_accum_steps
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer,
        num_warmup_steps=int(args.lr_warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    global_step = 0
    accum_loss = 0.0
    last_logits = None
    last_labels = None
    optimizer.zero_grad()
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for micro_step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss / args.grad_accum_steps
            loss.backward()
            accum_loss += loss.item()
            last_logits = out.logits
            last_labels = labels
            is_last_micro = (micro_step + 1) % args.grad_accum_steps == 0 or (micro_step + 1) == len(loader)
            if not is_last_micro:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy_params, args.grad_clip if args.grad_clip > 0 else float("inf")
            )
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            epoch_loss += accum_loss
            global_step += 1
            if global_step % args.log_every == 0:
                lr = optimizer.param_groups[0]['lr']
                logger.info(f"epoch {epoch+1}/{args.epochs}  step {global_step}/{total_steps}  loss={accum_loss:.4f}  lr={lr:.2e}")
                if use_wandb:
                    metrics = {
                        "train/loss": accum_loss,
                        "train/perplexity": torch.exp(torch.tensor(accum_loss)).item(),
                        "train/lr": lr,
                        "train/grad_norm": grad_norm.item(),
                        "train/epoch": epoch + 1,
                        "step": global_step,
                    }
                    with torch.no_grad():
                        comp_mask = last_labels != -100
                        if comp_mask.any() and last_logits is not None:
                            comp_logits = last_logits[comp_mask]  # [N, V]
                            entropy = torch.distributions.Categorical(logits=comp_logits).entropy().mean()
                            metrics["train/entropy"] = entropy.item()
                    wandb.log(metrics)
            accum_loss = 0.0
            if args.save_every and global_step % args.save_every == 0:
                ckpt_dir = dirs["checkpoints"] / f"step-{global_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(ckpt_dir), save_adapter=True)
                tokenizer.save_pretrained(str(ckpt_dir))
                logger.info(f"Saved checkpoint: {ckpt_dir}")
        avg_loss = epoch_loss / len(loader)
        logger.info(f"Epoch {epoch+1}/{args.epochs} complete — avg loss: {avg_loss:.4f}")
        if use_wandb:
            wandb.log({"train/epoch_loss": avg_loss, "train/epoch": epoch + 1, "step": global_step})
        ckpt_dir = dirs["checkpoints"] / f"epoch-{epoch+1}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_dir), save_adapter=True)
        tokenizer.save_pretrained(str(ckpt_dir))
        logger.info(f"Saved epoch checkpoint: {ckpt_dir}")

    final_dir = dirs["checkpoints"] / "iteration-0"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir), save_adapter=True)
    tokenizer.save_pretrained(str(final_dir))
    logger.info(f"Final checkpoint: {final_dir}")
    if use_wandb: wandb.finish()
    return str(final_dir)


def main():
    parser = argparse.ArgumentParser(description="SFT trainer for PEFT models (LoRA)")
    # Model
    parser.add_argument("--model", required=True, help="Base model name or path")
    parser.add_argument("--checkpoint", default=None, help="Existing LoRA checkpoint to resume from")
    # Data (use memory.py to curate before passing here)
    parser.add_argument("--data", nargs="+", required=True, help="Glob pattern(s) for curated CSV files")
    parser.add_argument("--max_len", type=int, default=4096, help="Maximum token length for prompt+completion")
    # Training
    parser.add_argument("--run", default="SFTRun", help="Run name for output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lr_scheduler", default="cosine", choices=["cosine", "linear", "constant", "constant_with_warmup"])
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.05)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    # LoRA
    parser.add_argument("--lora_rank", type=int, default=DEFAULT_LORA_CFG["lora_rank"])
    parser.add_argument("--lora_alpha", type=int, default=DEFAULT_LORA_CFG["lora_alpha"])
    parser.add_argument("--lora_dropout", type=float, default=DEFAULT_LORA_CFG["lora_dropout"])
    parser.add_argument("--target_modules", nargs="*", default=None)
    # Logging / checkpointing
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=None, help="Extra mid-epoch checkpoint every N steps")
    parser.add_argument("--wandb_project", default=None, help="W&B project name (omit to disable)")

    args = parser.parse_args()
    final_path = train_sft(args)
    print(f"\nTraining complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
