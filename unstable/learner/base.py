import ray, torch, time, pathlib
from typing import List, Dict, Any, Optional

from unstable.collection.buffers import BaseBuffer
from unstable.collection.trackers import BaseTracker
from unstable.learner.models import build_peft_model, enable_full_activation_ckpt
from unstable.utils.logging import setup_logger

from transformers import get_scheduler


class BaseLearner:
    def __init__(
        self, 
        model_name: str, 
        total_training_steps: int, 
        lora_cfg: Dict[str,Any], 
        batch_size: int, 
        mini_batch_size: int, 
        learning_rate: float, 
        grad_clip: float, 
        lr_scheduler_type: str,
        lr_warmup_ratio: float,
        buffer: BaseBuffer, 
        tracker: BaseTracker, 
        model_registry, 
        epochs: int = 1,
        max_generation_len: Optional[int] = None,
        max_train_len: Optional[int] = None,
        activation_checkpointing: bool=True, 
        gradient_checkpointing: bool=True, 
        use_trainer_cache: bool=False, 
        value_head: bool=False, 
        initial_lora_path: Optional[str]=None
    ): 
        self.total_training_steps = total_training_steps
        self.model_name, self.lora_cfg = model_name, lora_cfg
        self.initial_lora_path = initial_lora_path
        self.buffer, self.tracker, self.model_registry = buffer, tracker, model_registry
        self.logger = setup_logger("learner", ray.get(tracker.get_log_dir.remote()))
        self.use_trainer_cache, self.gradient_checkpointing, self.activation_checkpointing = use_trainer_cache, gradient_checkpointing, activation_checkpointing
        self.max_generation_len, self.max_train_len = max_generation_len, max_train_len
        self.grad_accumulation_steps = batch_size // mini_batch_size
        self.batch_size, self.mini_batch_size, self.lr, self.grad_clip, self.epochs = batch_size, mini_batch_size, learning_rate, grad_clip, epochs
        self.ckpt_dir = pathlib.Path(ray.get(self.tracker.get_checkpoints_dir.remote())); self.ckpt_dir.mkdir(parents=True, exist_ok=True) # create ckpt dir
        torch.set_float32_matmul_precision('high')
        torch.set_default_dtype(torch.bfloat16)
        gpu_ids = ray.get_gpu_ids()
        self.device = (torch.device(f"cuda:{gpu_ids[0]}") if gpu_ids else torch.device("cpu"))
        self.model, self.tokenizer = build_peft_model(model_name, self.device, lora_cfg, initial_lora_path, value_head=value_head)
        self.model.to(torch.bfloat16)
        if not self.use_trainer_cache:      self.model.config.use_cache = False
        if self.gradient_checkpointing:     self.model.gradient_checkpointing_enable()
        if self.activation_checkpointing:   enable_full_activation_ckpt(self.model)
        self.actor_params = [p for n, p in self.model.named_parameters() if f".{self.model.actor_adapter_name}." in n]
        self.actor_optimizer = torch.optim.AdamW(self.actor_params, lr=self.lr)
        total_actor_optimizer_steps = int(self.total_training_steps * self.epochs)
        self.actor_lr_scheduler = get_scheduler(lr_scheduler_type, self.actor_optimizer, num_warmup_steps=int(lr_warmup_ratio * total_actor_optimizer_steps), num_training_steps=total_actor_optimizer_steps)
        self._step = 1; self._samples_seen = 0 # training counters

    def _update(self, batch):               raise NotImplementedError
    def train(self, iterations: int):
        self.logger.info("Starting training loop")
        while self._step < iterations:
            try: # Wait and collect data
                while (ray.get(self.buffer.size.remote()) < self.batch_size * 1.5): time.sleep(0.2)
                self.logger.info("Enough data, starting learning step")
                batch: List = ray.get(self.buffer.get_batch.remote(self.batch_size)); self._samples_seen += self.batch_size
                accumulated_metrics = self._update(batch=batch)
                # Metrics
                log = {f"{k}": v for k, v in accumulated_metrics.items()}
                log.update({"step": self._step,  "samples_seen": self._samples_seen,  "lr": self.actor_optimizer.param_groups[0]["lr"]})
                if "actor_grad_norm" not in log: log.update({"actor_grad_norm": sum(p.grad.data.norm(2).item()**2 for p in self.model.parameters() if p.grad is not None) ** 0.5})
                self.tracker.log_learner.remote(log)
                # Save & register the updated checkpoint
                ckpt_path = self._save_checkpoint()
                try:
                    self.model_registry.add_checkpoint.remote(uid=f"ckpt-{self._step}", path=ckpt_path, iteration=self._step)
                    self.logger.info(f"Registered new ckpt: {ckpt_path}, ckpt-{self._step}")
                except Exception as exc: self.logger.info(f"Exception when adding checkpoint: {exc}")
                self.logger.info(f"registered new ckpt -> {ckpt_path} for iteration{self._step}")
                self._step += 1
            except Exception as exc: self.logger.info(f"Exception in learner loop: {exc}")
        self.logger.info("[Learner] training finished.")
        self.buffer.stop.remote()

    def _save_checkpoint(self):
        ckpt_dir = self.ckpt_dir / f"iteration-{self._step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(ckpt_dir, save_adapter=True)
        return ckpt_dir

    def _masked_mean(self, x, mask, axis=None): return (x * mask).sum(dim=axis) / mask.sum(dim=axis) if axis is not None else (x * mask).sum() / mask.sum()
    def _masked_std(self, x, mask, axis=None, eps: float = 1e-8):
        if axis is None: return x[mask.bool()].std(unbiased=False).clamp_min(eps)
        mean = self._masked_mean(x, mask, axis=axis).unsqueeze(axis)
        var = ((x - mean) ** 2 * mask).sum(dim=axis) / mask.sum(dim=axis)
        return var.clamp_min(eps).sqrt()
