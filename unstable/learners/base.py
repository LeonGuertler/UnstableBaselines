
import ray, torch, time, pathlib
from typing import List, Dict, Any, Optional

from unstable.buffers import BaseBuffer
from unstable.trackers import BaseTracker
from unstable.learners.utils import build_peft_model, enable_full_activation_ckpt, remap_device
from unstable.utils import setup_logger


class BaseLearner:
    def __init__(self, model_name: str, lora_cfg: Dict[str,Any], batch_size: int, mini_batch_size: int, learning_rate: float, grad_clip: float, buffer: BaseBuffer, tracker: BaseTracker, model_registry, activation_checkpointing: bool=True, gradient_checkpointing: bool=True, use_trainer_cache: bool=False, initial_lora_path: Optional[str]=None): 
        # basically build the policy model and optimizer for policy model
        self.model_name, self.lora_cfg = model_name, lora_cfg
        self.buffer, self.tracker, self.model_registry = buffer, tracker, model_registry
        self.logger = setup_logger("learner", ray.get(tracker.get_log_dir.remote()))
        self.use_trainer_cache, self.gradient_checkpointing, self.activation_checkpointing = use_trainer_cache, gradient_checkpointing, activation_checkpointing
        self.batch_size, self.mini_batch_size, self.lr, self.grad_clip = batch_size, mini_batch_size, learning_rate, grad_clip
        self.gradient_acc_steps = self.batch_size // self.mini_batch_size # TODO maybe assert that divisible 
        self.ckpt_dir = pathlib.Path(ray.get(self.tracker.get_checkpoints_dir.remote())); self.ckpt_dir.mkdir(parents=True, exist_ok=True) # create ckpt dir

        torch.set_float32_matmul_precision('high')
        torch.set_default_dtype(torch.bfloat16)

        gpu_ids = ray.get_gpu_ids()
        self.device = (remap_device(torch.device(f"cuda:{gpu_ids[0]}")) if gpu_ids else torch.device("cpu"))
        self.policy_model, self.tokenizer = build_peft_model(model_name, self.device, lora_cfg, initial_lora_path)
        self.policy_model.to(torch.bfloat16)

        if not self.use_trainer_cache:      self.policy_model.config.use_cache = False
        if self.gradient_checkpointing:     self.policy_model.gradient_checkpointing_enable() # gradient checkpointing
        if self.activation_checkpointing:   enable_full_activation_ckpt(self.policy_model)       # activation checkpointing. Affords most of the vRAM savings
        
        self.policy_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.policy_model.parameters()), lr=learning_rate)
        self._step = 1; self._samples_seen = 0 # training counters

    def initialize_algorithm(self, cfg):    raise NotImplementedError
    def _update(self, batch):               raise NotImplementedError
    def train(self, iterations: int):
        self.logger.info("Starting training loop")

        while self._step < iterations:
            try:
                while (ray.get(self.buffer.size.remote()) < self.batch_size * 1.5): time.sleep(0.2) # wait until enough data is available
                self.logger.info("Enough data, starting learning step")
                batch: List = ray.get(self.buffer.get_batch.remote(self.batch_size)); self._samples_seen += self.batch_size
                accumulated_metrics = self._update(batch=batch) # handled by specific algo implementations

                log = {f"{k}": v for k, v in accumulated_metrics.items()}
                log.update({"step": self._step,  "samples_seen": self._samples_seen,  "lr": self.policy_optimizer.param_groups[0]["lr"], "policy_grad_norm": sum(p.grad.data.norm(2).item()**2 for p in self.policy_model.parameters() if p.grad is not None) ** 0.5})
                self.tracker.log_learner.remote(log)

                # save & register the updated checkpoint
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
        self.policy_model.save_pretrained(ckpt_dir, save_adapter=True)
        return ckpt_dir

