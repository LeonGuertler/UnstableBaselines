#!/usr/bin/env python
# sft.py
# Minimal supervised-fine-tuning script for LoRA adapters on (observation → action) data
#
# Usage example:
#   python sft.py \
#       --model_name "Qwen/Qwen3-0.6B" \
#       --train_file "data/sft_train.jsonl" \
#       --val_file "data/sft_val.jsonl" \
#       --output_dir "outputs/sft_lora" \
#       --batch_size 32 \
#       --epochs 3 \
#       --lora_rank 32 \
#       --lora_alpha 32 \
#       --wandb_project "UnstableBaselines" \
#       --wandb_name "tictactoe-sft"
#
# The script saves only the LoRA weights in <output_dir>/checkpoint-<epoch>

import os, json, math, argparse, time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
import wandb

from learners.lora_utils import build_lora_model   # same helper you already use

# ---------- Dataset ---------- #
class ObservationActionDataset(Dataset):
    """
    Expects a .jsonl file with lines
    {"observation": "...", "action": "..."}
    """

    def __init__(self, path: str, tokenizer, max_len: int = 1024):
        self.samples = [json.loads(l) for l in open(path, "r", encoding="utf-8")]
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ex = self.samples[idx]
        # Put <obs> before the action to mimic RL prompts.
        prompt = ex["observation"].rstrip()
        target = ex["action"].lstrip()

        # “prompt + target” is fed once; only the target tokens have labels.
        prompt_ids = self.tok(prompt, add_special_tokens=False).input_ids
        target_ids = self.tok(target, add_special_tokens=False, max_length=self.max_len - len(prompt_ids) - 1).input_ids + [self.tok.eos_token_id]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids  # ignore obs tokens in the loss

        if len(input_ids) > self.max_len:  # truncate from the left (safer for dialogue)
            excess = len(input_ids) - self.max_len
            input_ids = input_ids[excess:]
            labels = labels[excess:]

        return dict(input_ids=torch.tensor(input_ids, dtype=torch.long), labels=torch.tensor(labels, dtype=torch.long))


# ---------- Train / eval helpers ---------- #
def collate(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = torch.stack(
        [torch.nn.functional.pad(x["input_ids"], (0, max_len - len(x["input_ids"])), value=tokenizer.pad_token_id)
         for x in batch]
    )
    labels = torch.stack(
        [torch.nn.functional.pad(x["labels"], (0, max_len - len(x["labels"])), value=-100)
         for x in batch]
    )
    return {"input_ids": input_ids, "labels": labels}


def run_epoch(model, loader, optimizer, scheduler, device, train=True):
    model.train(train)
    total_loss, n_tokens = 0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item() * input_ids.numel()
        n_tokens += input_ids.numel()
    return total_loss / n_tokens


# ---------- Main ---------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=False)
    parser.add_argument("--output_dir", type=str, default="outputs/sft_lora")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    # LoRA
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    # WandB
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---------- Load model & tokenizer ---------- #
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    model = build_lora_model(
        model=base, r=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ---------- Data ---------- #
    train_ds = ObservationActionDataset(args.train_file, tokenizer)
    val_ds = ObservationActionDataset(args.val_file, tokenizer) if args.val_file else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate
    ) if val_ds else None

    # ---------- Optim / sched ---------- #
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    total_steps = args.epochs * math.ceil(len(train_loader))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ---------- WandB ---------- #
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
        )

    # ---------- Training loop ---------- #
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, scheduler, device, train=True)
        val_loss = (
            run_epoch(model, val_loader, optimizer, scheduler, device, train=False)
            if val_ds
            else None
        )
        elapsed = time.time() - t0

        log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ppl": math.exp(train_loss),
            "time": elapsed,
        }
        if val_loss is not None:
            log["val_loss"] = val_loss
            log["val_ppl"] = math.exp(val_loss)
        if args.wandb_project:
            wandb.log(log)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in log.items()})

        # ---------- Save LoRA checkpoint ---------- #
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)
        # Save *only* the LoRA adapter weights
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

    if args.wandb_project:
        wandb.finish()

    print(f"✓ Finished. LoRA checkpoints stored at: {args.output_dir}")
