import ray
import torch
import torch.nn.functional as F
import tree
from typing import List, Dict, Optional

from unstable.algorithms.base import BaseLearner
from unstable.common.utils.models import build_peft_model, masked_mean, masked_std

from transformers import get_scheduler

@ray.remote
class GRPOLearner(BaseLearner):
    def __init__(
        self,
        infer_mini_batch_size: int,
        clip_ratio: float = 0.2,
        epochs: int = 1,
        entropy_coeff: float = 0.0,
        beta: float = 0.0,
        max_generation_len: Optional[int] = None,
        max_train_len: Optional[int] = None,
        actor_grad_accumulation_steps: int = 4,
        actor_lr_scheduler_type: str = "linear",
        actor_lr_warmup_ratio: float = 0.025,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert self.batch_size % (self.mini_batch_size * actor_grad_accumulation_steps) == 0, "batch_size must be divisible by mini_batch_size * actor_grad_accumulation_steps"
        self.infer_mini_batch_size = infer_mini_batch_size
        self.clip_ratio = clip_ratio
        self.update_epochs = epochs
        self.entropy_coeff = entropy_coeff
        self.beta = beta
        self.max_generation_len = max_generation_len
        self.max_train_len = max_train_len
        self.actor_grad_accumulation_steps = actor_grad_accumulation_steps
        num_actor_optimizer_steps = int(self.num_training_steps * epochs * ((self.batch_size // self.mini_batch_size) // self.actor_grad_accumulation_steps))
        self.actor_lr_scheduler = get_scheduler(actor_lr_scheduler_type, self.policy_optimizer, num_warmup_steps=int(actor_lr_warmup_ratio * num_actor_optimizer_steps), num_training_steps=num_actor_optimizer_steps)
        gpu_ids = ray.get_gpu_ids()
        self.policy_device = self.device
        if self.beta > 0.0:
            self.ref_policy_device = torch.device(f"cuda:{gpu_ids[1]}") if len(gpu_ids) > 1 else self.policy_device
            self.ref_policy_model, _ = build_peft_model(self.model_name, self.ref_policy_device, self.lora_cfg, self.initial_lora_path)
            self.ref_policy_model.to(torch.bfloat16); self.ref_policy_model.eval()
            for param in self.ref_policy_model.parameters(): param.requires_grad = False

    def _prepare_batch(self, steps: List) -> tuple:
        obs, acts = zip(*[(s.obs, s.act)for s in steps])
        combined  = [o + a for o, a in zip(obs, acts)]
        lengths   = [len(self.tokenizer(text, add_special_tokens=False)["input_ids"]) for text in combined]
        avg_len   = sum(lengths) / len(lengths)
        pct_truncated = (sum(l > self.max_train_len for l in lengths) / len(lengths) if self.max_train_len else 0.0)
        enc = self.tokenizer(combined, return_tensors="pt", padding=True, truncation=True, max_length=self.max_train_len)
        response_mask = enc.attention_mask.bool().clone()
        for i, text in enumerate(obs):
            prompt_len = len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
            response_mask[i, :prompt_len] = False
        response_mask = response_mask[:, 1:]
        return enc.input_ids, enc.attention_mask, response_mask, avg_len, pct_truncated

    def _forward_minibatch(self, input_ids, attention_mask, response_mask, policy_model):
        out = policy_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits
        logp = F.log_softmax(logits, dim=-1)
        tgt_ids = input_ids[:, 1:]
        tok_logp = logp[:, :-1, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
        tok_entropy = -(torch.exp(logp[:, :-1, :]) * logp[:, :-1, :]).sum(dim=-1)  # [B, T-1]
        entropy = masked_mean(tok_entropy, response_mask)
        return tok_logp, entropy, response_mask

    def _mini_batch_update_step(self, input_ids, attention_mask, response_mask, logps, advantages, logps_ref) -> Dict[str, float]:
        new_logps, entropy, _ = self._forward_minibatch(input_ids.to(self.policy_device), attention_mask.to(self.policy_device), response_mask.to(self.policy_device), self.policy_model)
        ratio = torch.exp(new_logps - logps)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2)
        policy_loss = masked_mean(policy_loss, response_mask.to(self.policy_device), axis=1).mean()
        entropy_loss = self.entropy_coeff * entropy
        total_loss = policy_loss - entropy_loss
        if self.beta > 0.0:
            kl = masked_mean(torch.exp(logps_ref - new_logps) - (logps_ref - new_logps) - 1, response_mask.to(self.policy_device))
            total_loss += self.beta * kl
        total_loss = total_loss / self.actor_grad_accumulation_steps
        total_loss.backward()
        return {
            "policy_loss": policy_loss.item(),
            "kl": kl.item() if self.beta > 0.0 else 0.0,
            "entropy": entropy.item(),
            "logp_mean": masked_mean(new_logps, response_mask.to(self.policy_device)).item(),
            "logp_std": masked_std(new_logps, response_mask.to(self.policy_device)).item()
        }

    def _update(self, batch):
        all_steps = tree.flatten(batch)
        input_ids, attention_mask, response_mask, avg_len, pct_truncated = self._prepare_batch(all_steps)
        # Values and logps
        logps = torch.zeros(input_ids.shape[0], input_ids.shape[1]-1)
        if self.beta > 0.0: logps_ref = torch.zeros(input_ids.shape[0], input_ids.shape[1]-1)
        for i in range(0, len(all_steps), self.infer_mini_batch_size):
            mb_input_ids, mb_attention_mask = input_ids[i : i + self.infer_mini_batch_size], attention_mask[i : i + self.infer_mini_batch_size]
            with torch.no_grad():
                mb_logps = self._forward_minibatch(mb_input_ids.to(self.policy_device), mb_attention_mask.to(self.policy_device), response_mask[i : i + self.infer_mini_batch_size].to(self.policy_device), self.policy_model)[0].float().cpu()
                logps[i : i + self.infer_mini_batch_size, :mb_logps.shape[1]] = mb_logps
                if self.beta > 0.0:
                    mb_logps_ref = self._forward_minibatch(mb_input_ids.to(self.ref_policy_device), mb_attention_mask.to(self.ref_policy_device), response_mask[i : i + self.infer_mini_batch_size].to(self.ref_policy_device), self.ref_policy_model)[0].float().cpu()
                    logps_ref[i : i + self.infer_mini_batch_size, :mb_logps_ref.shape[1]] = mb_logps_ref
        # Advantages are already computed - set A_t,i = R_i
        advantages = torch.zeros(logps.shape[0], logps.shape[1])
        for i in range(len(all_steps)): 
            advantages[i, torch.where(response_mask[i])[0]] = all_steps[i].reward
        # Training loop
        metrics_acc: Dict[str, float] = {}
        for _ in range(self.update_epochs):
            N = input_ids.size(0); num_mini_batches = N // self.mini_batch_size
            idx = torch.randperm(N)
            for i in range(num_mini_batches):
                mb_idx = idx[i * self.mini_batch_size : (i + 1) * self.mini_batch_size]
                mb_input_ids = input_ids[mb_idx]
                mb_attention_mask = attention_mask[mb_idx]
                mb_response_mask = response_mask[mb_idx]
                mb_logps = logps[mb_idx].to(self.policy_device)
                mb_advantages = advantages[mb_idx]
                mb_advantages = mb_advantages.to(self.policy_device)
                if self.beta > 0.0: mb_logps_ref = logps_ref[mb_idx].to(self.policy_device) 
                else: mb_logps_ref = None
                update_metrics = self._mini_batch_update_step(mb_input_ids, mb_attention_mask, mb_response_mask, mb_logps, mb_advantages, mb_logps_ref)
                if (i+1) % self.actor_grad_accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.grad_clip)
                    update_metrics["policy_grad_norm"] = update_metrics.get("policy_grad_norm", 0.0) + float(grad_norm)
                    self.policy_optimizer.step()
                    self.policy_optimizer.zero_grad(set_to_none=True)
                    self.actor_lr_scheduler.step()
                # Metrics
                for k, v in update_metrics.items(): metrics_acc[k] = metrics_acc.get(k, 0.0) + v
                self.logger.info(f"Mini-step metrics: {update_metrics}")
        log = {k: v / (self.update_epochs * num_mini_batches) for k, v in metrics_acc.items()}
        return {**log, "avg_train_len": avg_len, "pct_truncated": pct_truncated, "step": self._step, "samples_seen": self._samples_seen}
