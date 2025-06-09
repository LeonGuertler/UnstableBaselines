import pathlib
from pathlib import Path
from typing import Any, Dict, Optional, List
import time
import os
import ray, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file as safe_load
from ray.data import from_items
from ray.train.torch import TorchTrainer, prepare_model
from ray.train.torch import TorchConfig
from ray.train import ScalingConfig, get_context
from ray.train import get_dataset_shard

# local imports
from unstable.algorithms import BaseAlgo

def load_lora_state(peft_model, ckpt_dir: str | pathlib.Path):
    ckpt_dir = pathlib.Path(ckpt_dir)
    candidates = [ckpt_dir / "adapter_model.safetensors", ckpt_dir / "adapter_model.bin", ckpt_dir / "pytorch_model.bin"]
    for path in candidates:
        if path.exists():
            print(f"[loader] found LoRA adapter → {path.name}")
            lora_sd = safe_load(str(path)) if path.suffix == ".safetensors" else torch.load(path, map_location="cpu")
            set_peft_model_state_dict(peft_model, lora_sd, adapter_name="default")
            return
    raise FileNotFoundError(f"No adapter_model.* found in {ckpt_dir}")

def _build_peft_model(base_name: str, lora_cfg: Dict[str, Any] | None, initial_lora_path: Optional[str], freeze_base: bool = True):
    print(f"[Learner] Loading base model: {base_name} …")
    base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
    if freeze_base:
        for p in base.parameters():
            p.requires_grad_(False)

    lcfg = LoraConfig(
        r=lora_cfg.get("lora_rank", 32),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    )
    model = get_peft_model(base, lcfg)

    if initial_lora_path:
        print(f"[Learner] Loading initial LoRA weights from {initial_lora_path}")
        load_lora_state(model, initial_lora_path)

    return model

def _train_loop(config):
    tracker = config["tracker"]
    model_pool = config["model_pool"]
    rank = get_context().get_world_rank()
    print(f"[Learner rank {rank}/{get_context().get_world_size()}]")
    model = _build_peft_model(config["model_name"], config["lora_cfg"], config["initial_lora_path"])
    model = prepare_model(model)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config["learning_rate"])

    _step = 0 
    _samples_seen = 0
    print('[Trainer] Starting training loop')
    while True:
        if ray.get(config["buffer"].size.remote()) >= config["batch_size"] * config["batch_delay_buffer"]:
            mini_batch = ray.get(config["buffer"].get_batch.remote(config["mini_batch_size"]))
            _samples_seen += len(mini_batch)
            print(f"[Learner] Starting training step with minibatch size {len(mini_batch)}")
            optimizer.zero_grad(set_to_none=True)
            tokenizer.pad_token = tokenizer.eos_token
            model.train()
            micro_bs = config["mini_batch_size"] // config["grad_accum_per_learner"]
            metrics_acc: Dict[str, float] = {}
            for i in range(config["grad_accum_per_learner"]):
                micro_batch = mini_batch[i * micro_bs : (i + 1) * micro_bs]

                # REINFORCE
                obs, acts, advs = zip(*[(s.obs, s.act, s.reward) for s in micro_batch])
                advs = torch.tensor(advs, dtype=torch.float32, device=model.device)
                enc = tokenizer([o + a for o, a in zip(obs, acts)], return_tensors="pt", padding=True)
                enc = enc.to(model.device)
                print(f"[Learner] Tokenized and predicting now")
                out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                logp = torch.nn.functional.log_softmax(out.logits, dim=-1)
                tgt_ids = enc["input_ids"][:, 1:]
                tok_logp = logp[:, :-1, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
                mask = torch.ones_like(enc["input_ids"], dtype=torch.bool, device=model.device)
                for k, z in enumerate(obs):
                    L = len(tokenizer(z, add_special_tokens=False)["input_ids"])
                    mask[k, :L] = False
                mask = mask[:, 1:]
                seq_logp = (tok_logp * mask).sum(1) / mask.sum(1).clamp(min=1)
                loss = -(advs * seq_logp).mean() / config["gradient_accum_steps"]
                print(f"[Learner] Loss: {loss.item()}")
                loss.backward()

                metrics_acc["loss"] = metrics_acc.get("loss", 0.0) + loss.item()
                metrics_acc["logp_mean"] = metrics_acc.get("logp_mean", 0.0) + seq_logp.mean().item()
                metrics_acc["logp_std"] = metrics_acc.get("logp_std", 0.0) + seq_logp.std().item()
                metrics_acc["num_steps"] = metrics_acc.get("num_steps", 0) + len(micro_batch)

            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            print(f"[Learner] Before optimizer step")
            optimizer.step()
            print(f"[Learner] After optimizer step")
            _step += 1

            # --- LOGGING ---
            if tracker is not None and rank == 0:
                log = {f"learner/{k}": v / config["gradient_accum_steps"] for k, v in metrics_acc.items()}
                log.update({
                    "learner/step": _step, 
                    "learner/samples_seen": _samples_seen * config["mini_batch_size"], 
                    "learner/lr": optimizer.param_groups[0]["lr"], 
                    "learner/grad_norm": sum(p.grad.data.norm(2).item()**2 for p in model.parameters() if p.grad is not None) ** 0.5
                })
                tracker.log_learner.remote(log)

            # === SAVE CHECKPOINT ===
            if rank == 0 and _step % config["save_every"] == 0:
                ckpt_dir = config["ckpt_root"] / f"iteration-{_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                model = model.module if hasattr(model,'module') else model
                model.save_pretrained(ckpt_dir, save_adapter=True)
                _last_ckpt = ckpt_dir
                print(f"[Learner] saved → {ckpt_dir}")

                if model_pool and _last_ckpt:
                    model_pool.add_checkpoint.remote(str(_last_ckpt), _step)
                    print(f"[Learner] ↪registered → {_last_ckpt}")
                    model_pool.snapshot.remote(_step)
        else:
            print(f"[Learner] waiting for buffer to fill. Current size: {ray.get(config['buffer'].size.remote())} ({config['batch_size'] * config['batch_delay_buffer']})")
            time.sleep(5.0)

        
    
    return {"steps": _step, "samples_seen": _samples_seen}

@ray.remote
class Learner:
    def __init__(self, model_name, step_buffer, model_pool, algorithm_cls,
                 num_learners=1, batch_size=384, gradient_accum_steps=32,
                 learning_rate=5e-6, grad_clip=1.0, batch_delay_buffer=1.5,
                 lora_cfg={}, initial_lora_path=None, ckpt_root="checkpoints",
                 save_every=1, tracker=None):
        assert batch_size % num_learners == 0, "batch_size must be divisible by num_learners"
        assert gradient_accum_steps % num_learners == 0, "gradient_accum_steps must be divisible by num_learners"
        assert batch_size % gradient_accum_steps == 0, "batch_size must be divisible by gradient_accum_steps"
        assert batch_size >= gradient_accum_steps, "batch_size must be at least gradient_accum_steps"

        self.model_name = model_name
        self.algorithm = algorithm_cls()
        self.num_learners = num_learners
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.grad_accum = gradient_accum_steps
        self.grad_clip = grad_clip
        self.batch_delay_buffer = batch_delay_buffer
        self.step_buffer = step_buffer
        self.model_pool = model_pool
        self.tracker = tracker
        self.save_every = save_every
        self.lora_cfg = lora_cfg
        self.initial_lora_path = initial_lora_path
        self.ckpt_root = Path(ray.get(self.tracker.get_checkpoints_dir.remote()) if self.tracker else Path(ckpt_root))
        print(f"[Learner] ckpt_root: {self.ckpt_root}")
        self.ckpt_root.mkdir(parents=True, exist_ok=True)

        # Distributed attributes
        self.mini_batch_size = batch_size // num_learners
        self.grad_accum_per_learner = gradient_accum_steps // num_learners

        print(f"[Learner] initialized with {self.num_learners} workers")

    def train(self, iterations: int):
        print("[Learner] starting training loop …")
        TorchTrainer(
            _train_loop,
            train_loop_config={
                "model_name": self.model_name,
                "buffer": self.step_buffer,
                "lora_cfg": self.lora_cfg,
                "initial_lora_path": self.initial_lora_path,
                "learning_rate": self.learning_rate,
                "grad_clip": self.grad_clip,
                "batch_size": self.batch_size,
                "mini_batch_size": self.mini_batch_size,
                "batch_delay_buffer": self.batch_delay_buffer,
                "gradient_accum_steps": self.grad_accum,
                "grad_accum_per_learner": self.grad_accum_per_learner,
                "ckpt_root": self.ckpt_root,
                "tracker": self.tracker,
                "model_pool": self.model_pool,
                "save_every": self.save_every

            },
            scaling_config=ScalingConfig(
                num_workers=self.num_learners,
                use_gpu=True
            ),
        ).fit()
