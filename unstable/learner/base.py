import ray, pathlib, torch, time, random, os, json
from ray.experimental.internal_kv import _internal_kv_put, _internal_kv_get, _internal_kv_exists
from ray.util import get_node_ip_address as _ray_ip
from typing import Dict, Any, Optional, List
from transformers import get_scheduler
import deepspeed

from unstable.collection.buffers import BaseBuffer
from unstable.collection.trackers import BaseTracker
from unstable.learner.models import build_peft_model, enable_full_activation_ckpt
from unstable.utils.logger import setup_logger

class BaseLearner:
    def __init__(
        self, 
        model_name: str, 
        total_training_steps: int, 
        lora_cfg: Dict[str,Any], 
        checkpoint_cfg: Dict[str,Any],
        local_batch_size: int, 
        micro_batch_size: int, 
        learning_rate: float, 
        grad_clip: float, 
        lr_scheduler_type: str,
        lr_warmup_ratio: float,
        buffer: BaseBuffer, 
        tracker: BaseTracker, 
        model_sampler, 
        epochs: int = 1,
        eval_steps: int = 100,
        max_generation_len: Optional[int] = None,
        max_train_len: Optional[int] = None,
        temperature: float = 1.0,
        use_trainer_cache: bool=False,
        initial_lora_path: Optional[str]=None,
        zero_optimization: Optional[Dict[str,Any]]=None,
        gradient_checkpointing: bool = False,
        activation_checkpointing:  bool = False,
        rank: int = 0,
        world_size: int = 1,
        env_vars = {},
        **kwargs
    ):
        self.model_name = model_name
        self.total_training_steps = total_training_steps; self.epochs = int(epochs)
        self.lora_cfg = lora_cfg; self.initial_lora_path = initial_lora_path; self.eval_steps = eval_steps
        self.buffer, self.tracker, self.model_sampler = buffer, tracker, model_sampler
        self.logger = setup_logger(f"learner-{rank}", ray.get(tracker.get_log_dir.remote()))
        self.checkpoint_cfg = checkpoint_cfg
        self.lora_cfg = lora_cfg
        self.temperature = temperature
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
        for k, v in env_vars.items(): os.environ[k] = v
        self.ckpt_dir = pathlib.Path(ray.get(self.tracker.get_checkpoints_dir.remote()))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(f"cuda:0") if ray.get_gpu_ids() else torch.device("cpu")
        model, self.tokenizer = build_peft_model(model_name, self.device, lora_cfg, checkpoint_cfg=self.checkpoint_cfg.get('policy', {}), value_head=False)
        if not self.use_trainer_cache or gradient_checkpointing or activation_checkpointing:      model.config.use_cache = False
        if gradient_checkpointing:     model.gradient_checkpointing_enable()
        if activation_checkpointing:   enable_full_activation_ckpt(model)
        self._policy_params = [p for n, p in model.named_parameters() if f".{model.adapter_name}." in n and p.requires_grad]
        self.optimizer = torch.optim.AdamW(self._policy_params, lr=self.lr, fused=True)
        self.total_optimizer_steps = int(self.total_training_steps * self.epochs)
        self.lr_scheduler = get_scheduler(lr_scheduler_type, self.optimizer, num_warmup_steps=int(lr_warmup_ratio * self.total_optimizer_steps), num_training_steps=self.total_optimizer_steps)
        # DeepSpeed
        self.rank = rank; self.world_size = world_size
        self.zero_optimization = zero_optimization
        if self.rank == 0: 
            master_node = {'address': _ray_ip(), 'port': random.randint(20000, 40000)}
            _internal_kv_put("learner/master_node", json.dumps(master_node).encode("utf-8"), overwrite=True, namespace="")
        else: 
            while not bool(_internal_kv_exists("learner/master_node", namespace="")): time.sleep(1)
            master_node = json.loads(_internal_kv_get("learner/master_node", namespace="").decode("utf-8"))
        os.environ["MASTER_ADDR"] = master_node['address']; os.environ["MASTER_PORT"] = str(master_node['port']); 
        os.environ['RANK'] = str(self.rank); os.environ['LOCAL_RANK'] = '0'; os.environ['WORLD_SIZE'] = str(self.world_size)
        deepspeed.init_distributed(dist_backend="nccl", rank=self.rank, world_size=self.world_size, auto_mpi_discovery=False)
        self.engine, self.actor_optimizer, _, self.actor_lr_scheduler = self.init_distributed_training(model, self.optimizer)
        self.logger.info("DeepSpeed initialized successfully")
        self.model = self.engine.module; self.rank = self.rank; self.world_size = self.world_size; self._step = checkpoint_cfg['iteration']+1; self._samples_seen = self._step * self.local_batch_size

    def init_distributed_training(self, model, optimizer):
        self.logger.info(f"Initializing DeepSpeed Engine with rank: {self.rank}, world_size: {self.world_size}")
        return deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config={
                "train_batch_size": self.world_size*self.local_batch_size,
                "train_micro_batch_size_per_gpu": self.micro_batch_size,
                "gradient_accumulation_steps": self.grad_accumulation_steps,
                "gradient_clipping": self.grad_clip,
                "zero_optimization": self.zero_optimization if self.zero_optimization is not None else {"stage": 0},
                "bf16": {"enabled": True},
                "steps_per_print": 100
            }
        )


    def _prepare_batch(self, steps):
        raw_prompt_ids, completion_ids, rewards, vllm_logprobs, returns = zip(*[(s.prompt_ids, s.completion_ids, s.reward, s.completion_logprobs, s.step_info.get("return", 0.0)) for s in steps])
        prompt_ids = self._zero_pad_right([torch.tensor(pid, dtype=torch.long) for pid in raw_prompt_ids]).to(self.device)
        prompt_attention_mask = self._zero_pad_right([torch.ones(len(pid), dtype=torch.long) for pid in raw_prompt_ids]).to(self.device)
        input_ids = self._zero_pad_right([torch.tensor(pid+cid, dtype=torch.long) for pid, cid in zip(raw_prompt_ids, completion_ids)]).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        vllm_logprobs = self._zero_pad_right([torch.tensor(vllm_logprobs, dtype=torch.float32, device=self.device) for vllm_logprobs in vllm_logprobs]).to(self.device)
        lengths = [len(step.prompt_ids) + len(step.completion_ids) for step in steps]
        prompt_lengths = [len(step.prompt_ids) for step in steps]
        if self.max_train_len is not None: input_ids = input_ids[:, :self.max_train_len]; lengths = [min(l, self.max_train_len) for l in lengths]; prompt_lengths = [min(pl, self.max_train_len) for pl in prompt_lengths]
        attention_mask = self._zero_pad_right([torch.ones(lengths[i], dtype=torch.long) for i in range(len(steps))]).to(self.device)
        completion_mask = attention_mask.clone().bool()
        for i in range(len(steps)): completion_mask[i, :prompt_lengths[i]] = False
        response_mask = completion_mask[:, 1:]
        vllm_logprobs = self._zero_pad_right(vllm_logprobs)
        pct_truncated = sum(l > self.max_train_len for l in lengths) / len(lengths) if self.max_train_len else 0
        return prompt_ids, prompt_attention_mask, input_ids, rewards, vllm_logprobs, lengths, attention_mask, response_mask, pct_truncated, returns

    def _update(self, batch):               raise NotImplementedError
    def train(self, iterations: int):
        from deepspeed import comm as dist; from torch.distributed import ReduceOp
        self.logger.info("Starting training loop")
        while self._step < iterations:
            try: 
                # Wait and collect data
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
                    log.update({"step": self._step, "grad_norm": accumulated_metrics.get("grad_norm", 0.0), "samples_seen": int(total_samples.item()), "lr": self.optimizer.param_groups[0]["lr"]})
                    self.tracker.log_learner.remote(log)
                    # Save & register the updated checkpoint
                    ckpt_path = self._save_checkpoint()
                    try:
                        self.model_sampler.add_checkpoint.remote(uid=f"ckpt-{self._step}", path=ckpt_path, iteration=self._step, eval=(self._step > 0 and self._step % self.eval_steps == 0))
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
        vh = getattr(self.model.base_model, "value_head", None)
        if vh is not None: torch.save(vh.state_dict(), ckpt_dir / "value_head.pt")
        return ckpt_dir

    def _masked_mean(self, x, mask, axis=None): return (x * mask).sum(dim=axis) / mask.sum(dim=axis) if axis is not None else (x * mask).sum() / mask.sum()
    def _masked_std(self, x, mask, axis=None, eps: float = 1e-8):
        if axis is None: return x[mask.bool()].std(unbiased=False).clamp_min(eps)
        mean = self._masked_mean(x, mask, axis=axis).unsqueeze(axis)
        var = ((x - mean) ** 2 * mask).sum(dim=axis) / mask.sum(dim=axis)
        return var.clamp_min(eps).sqrt()
    def _zero_pad_right(self, seq):
        max_len = max(s.size(-1) for s in seq); padded_seq = []
        for s in seq:
            pad_len = max_len - s.size(-1)
            padded_seq.append(torch.nn.functional.pad(s, (0, pad_len), value=0))
        return torch.stack(padded_seq, dim=0)
