import os, json, math, argparse, time
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
import wandb
from learners.lora_utils import build_lora_model
from peft.tuners.lora import LoraModel


def format_trace(observation, reasoning, action):
    instruction = "You are playing a two-player zero-sum game. Make valid moves to win. You should first reason about your next move, and then submit the move enclosed by \\boxed{}."
    full_observation = instruction + f"\nObservation: {observation}\n"
    full_trace = f"{reasoning}\n\\boxed{{{action}}}"
    return full_observation, full_trace

class ObservationActionDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=4096):
        raw_samples = [json.loads(l) for l in open(path, "r", encoding="utf-8")]

        print(f"Len raw data{len(raw_samples)}")
        self.samples = []
        for ex in raw_samples:
            prompt, target = format_trace(observation=ex["observation"], reasoning=ex["reasoning"], action=ex["answer"])
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            target_ids = tokenizer(target, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
            if len(prompt_ids) + len(target_ids) <= max_len and ex["answer"]!="":
                self.samples.append(ex)

        print(f"Len data{len(self.samples)}")

        self.tok = tokenizer; self.max_len = max_len

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        ex = self.samples[idx]
        prompt, target = format_trace(observation=ex["observation"], reasoning=ex["reasoning"], action=ex["answer"])
        prompt_ids = self.tok(prompt, add_special_tokens=False).input_ids
        target_ids = self.tok(target, add_special_tokens=False, max_length=self.max_len).input_ids + [self.tok.eos_token_id]
        ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        return dict(input_ids=torch.tensor(ids), labels=torch.tensor(labels))

def collate(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    pad_id = tokenizer.pad_token_id
    inp = torch.stack([torch.nn.functional.pad(x["input_ids"], (0, max_len - len(x["input_ids"])), value=pad_id) for x in batch])
    lbl = torch.stack([torch.nn.functional.pad(x["labels"], (0, max_len - len(x["labels"])), value=-100) for x in batch])
    return {"input_ids": inp, "labels": lbl}

def run_epoch(model, loader, optimizer, scheduler, device, train=True):
    model.train(train)
    total_loss, n_tok = 0.0, 0
    for i, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        loss = model(input_ids=input_ids, labels=labels).loss / args.gradient_accumulation_steps
        if train:
            loss.backward()
            if (i + 1) % args.gradient_accumulation_steps == 0 or (i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item() * args.gradient_accumulation_steps * input_ids.numel()
        n_tok += input_ids.numel()
    return total_loss / n_tok


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="Qwen/Qwen3-0.6B")
parser.add_argument("--train_file", required=True)
parser.add_argument("--val_file")
parser.add_argument("--output_dir", default="outputs/sft_lora")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=6)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--grad_clip", type=float, default=1.0)
parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
parser.add_argument("--lora_rank", type=int, default=32)
parser.add_argument("--lora_alpha", type=int, default=32)
parser.add_argument("--lora_dropout", type=float, default=0.0)
parser.add_argument("--wandb_project"); parser.add_argument("--wandb_name")
args = parser.parse_args(); os.makedirs(args.output_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
base = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
model = build_lora_model(base, r=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)

train_ds = ObservationActionDataset(args.train_file, tokenizer)
val_ds = ObservationActionDataset(args.val_file, tokenizer) if args.val_file else None
micro_batch_size = args.batch_size // args.gradient_accumulation_steps
if micro_batch_size == 0:
    raise ValueError("batch_size must be ≥ gradient_accumulation_steps")
train_loader = DataLoader(train_ds, batch_size=micro_batch_size, shuffle=True, collate_fn=collate, drop_last=True, )
val_loader = DataLoader(val_ds, batch_size=micro_batch_size, shuffle=False, collate_fn=collate, ) if val_ds else None

updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
total_steps = args.epochs * updates_per_epoch
scheduler = get_linear_schedule_with_warmup(torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr), int(total_steps * args.warmup_ratio), total_steps)

if args.wandb_project:
    wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    train_loss = run_epoch(model, train_loader, scheduler.optimizer, scheduler, device, True)
    val_loss = run_epoch(model, val_loader, scheduler.optimizer, scheduler, device, False) if val_ds else None
    print(f"Epoch {epoch}: train_ppl={math.exp(train_loss):.2f}" + (f", val_ppl={math.exp(val_loss):.2f}" if val_loss else ""))
    if args.wandb_project:
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
    ckpt = os.path.join(args.output_dir, f"checkpoint-{epoch}"); os.makedirs(ckpt, exist_ok=True)
    model.save_pretrained(ckpt); tokenizer.save_pretrained(ckpt)
print("✓ LoRA checkpoints at", args.output_dir)
