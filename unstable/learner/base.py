import ray, pathlib, torch, time, random, os, json
from ray.experimental.internal_kv import _internal_kv_put, _internal_kv_get, _internal_kv_exists
from ray.util import get_node_ip_address as _ray_ip
from typing import Dict, Any, Optional, List
from transformers import get_scheduler

from unstable.collection.buffers import BaseBuffer
from unstable.collection.trackers import BaseTracker
from unstable.learner.models import build_peft_model, enable_full_activation_ckpt
from unstable.utils.logging import setup_logger

class BaseLearner:
    def __init__(
        self, 
        model_name: str, 
        total_training_steps: int, 
        lora_cfg: Dict[str,Any], 
        local_batch_size: int, 
        micro_batch_size: int, 
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
        use_trainer_cache: bool=False, 
        value_head: bool=False, 
        initial_lora_path: Optional[str]=None,
        zero_optimization: Optional[Dict[str,Any]]=None,
        rank: int = 0,
        world_size: int = 1,
        **kwargs
    ):
        self.model_name = model_name
        self.total_training_steps = total_training_steps; self.epochs = int(epochs)
        self.lora_cfg = lora_cfg; self.initial_lora_path = initial_lora_path
        self.buffer, self.tracker, self.model_registry = buffer, tracker, model_registry
        self.logger = setup_logger(f"learner-{rank}", ray.get(tracker.get_log_dir.remote()))
        self.use_trainer_cache = use_trainer_cache
        self.max_generation_len = max_generation_len
        self.max_train_len = max_train_len
        self.local_batch_size = local_batch_size
        self.micro_batch_size = micro_batch_size
        if self.local_batch_size % self.micro_batch_size != 0: raise ValueError("local_batch_size must be divisible by micro_batch_size for gradient accumulation.")
        self.grad_accumulation_steps = self.local_batch_size // self.micro_batch_size
        self.lr = learning_rate
        self.grad_clip = grad_clip
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_warmup_ratio = lr_warmup_ratio
        self.ckpt_dir = pathlib.Path(ray.get(self.tracker.get_checkpoints_dir.remote()))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.set_float32_matmul_precision("high")
        self.device = torch.device(f"cuda:0") if ray.get_gpu_ids() else torch.device("cpu")
        model, self.tokenizer = build_peft_model(model_name, self.device, lora_cfg, initial_lora_path, value_head=value_head)
        params = [{'params': [p for n, p in model.named_parameters() if f".{model.actor_adapter_name}." in n], 'lr': self.lr}]
        self.optimizer = torch.optim.AdamW(params, lr=self.lr, fused=True)
        total_optimizer_steps = int(self.total_training_steps * self.epochs)
        self.lr_scheduler = get_scheduler(lr_scheduler_type, self.optimizer, num_warmup_steps=int(lr_warmup_ratio * total_optimizer_steps), num_training_steps=total_optimizer_steps)
        # DeepSpeed
        import deepspeed; self.logger.info(f"Initializing DeepSpeed with rank: {rank}, world_size: {world_size}")
        if rank == 0: 
            master_node = {'address': _ray_ip(), 'port': random.randint(20000, 40000)}
            _internal_kv_put("learner/master_node", json.dumps(master_node).encode("utf-8"), overwrite=True, namespace="")
        else: 
            while not bool(_internal_kv_exists("learner/master_node", namespace="")): time.sleep(1)
            master_node = json.loads(_internal_kv_get("learner/master_node", namespace="").decode("utf-8"))
        os.environ["MASTER_ADDR"] = master_node['address']; os.environ["MASTER_PORT"] = str(master_node['port'])
        deepspeed.init_distributed(dist_backend="nccl", rank=rank, world_size=world_size)
        self.engine, self.actor_optimizer, _, self.actor_lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            config={
                "train_batch_size": world_size*self.local_batch_size,
                "train_micro_batch_size_per_gpu": self.micro_batch_size,
                "gradient_accumulation_steps": self.grad_accumulation_steps,
                "gradient_clipping": self.grad_clip,
                "zero_optimization": zero_optimization if zero_optimization is not None else {"stage": 0},
                "bf16": {"enabled": True},
                "fp16": {"enabled": False},
                "steps_per_print": 100
            }
        )
        self.logger.info("DeepSpeed initialized successfully")
        self.model = self.engine.module; self.rank = rank; self.world_size = world_size; self._step = 1; self._samples_seen = 0

    def _update(self, batch):               raise NotImplementedError
    def train(self, iterations: int):
        from deepspeed import comm as dist; from torch.distributed import ReduceOp
        self.logger.info("Starting training loop")
        while self._step < iterations:
            try: # Wait and collect data
                while (ray.get(self.buffer.size.remote()) < self.local_batch_size * self.world_size): time.sleep(0.2)
                self.logger.info("Enough data, starting learning step")
                batch: List = ray.get(self.buffer.get_batch.remote(self.local_batch_size)); self._samples_seen += self.local_batch_size
                accumulated_metrics = self._update(batch=batch)

                # Metrics
                t = torch.tensor([float(v) for v in accumulated_metrics.values()], device=self.device, dtype=torch.float32)
                total_samples = torch.tensor(self._samples_seen, device=self.device, dtype=torch.long)
                dist.reduce(t, dst=0, op=ReduceOp.SUM); dist.reduce(total_samples, dst=0, op=ReduceOp.SUM)
                if self.rank == 0:
                    t /= self.world_size
                    log = dict(zip(accumulated_metrics.keys(), t.tolist()))
                    log.update({"step": self._step,  "samples_seen": int(total_samples.item()), "lr": self.optimizer.param_groups[0]["lr"]})
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
