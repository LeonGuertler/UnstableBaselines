import ray
import torch
import torch.nn.functional as F
import tree
from typing import List, Dict

from unstable.learner.base import BaseLearner

from transformers import get_scheduler

@ray.remote
class GRPOLearner(BaseLearner):
    def __init__(
        self,
        clip_ratio: float,
        entropy_coeff: float,
        beta: float,
        **kwargs
    ):
        super().__init__(value_head=True, **kwargs)
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.beta = beta

    def _prepare_batch(self, steps: List) -> tuple:
        obs, acts = zip(*[(s.obs, s.act)for s in steps])
        combined  = [o + a for o, a in zip(obs, acts)]
        lengths   = [len(self.tokenizer(text, add_special_tokens=False)["input_ids"]) for text in combined]
        avg_len   = sum(lengths) / len(lengths)
        pct_truncated = (sum(l > self.max_train_len for l in lengths) / len(lengths) if self.max_train_len else 0.0)
        enc = self.tokenizer(combined, return_tensors="pt", padding=True, truncation=True, max_length=self.max_train_len).to(self.device)
        response_mask = enc.attention_mask.bool().clone()
        for i, text in enumerate(obs):
            prompt_len = len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
            response_mask[i, :prompt_len] = False
        response_mask = response_mask[:, 1:]
        return enc.input_ids, enc.attention_mask, response_mask, avg_len, pct_truncated

    def _get_logps(self, input_ids, attention_mask, response_mask, compute_entropy: bool=False):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits
        logp = F.log_softmax(logits, dim=-1)
        tgt_ids = input_ids[:, 1:]
        tok_logp = logp[:, :-1, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
        if compute_entropy:
            tok_entropy = -(torch.exp(logp[:, :-1, :]) * logp[:, :-1, :]).sum(dim=-1)  # [B, T-1]
            entropy = self._masked_mean(tok_entropy, response_mask)
        return tok_logp, entropy if compute_entropy else None

    def _mini_batch_update_step(self, input_ids, attention_mask, response_mask, logps, advantages, logps_ref) -> Dict[str, float]:
        new_logps, entropy = self._get_logps(input_ids, attention_mask, response_mask, compute_entropy=True)
        ratio = torch.exp(new_logps - logps)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2)
        policy_loss = self._masked_mean(policy_loss, response_mask, axis=1).mean()
        entropy_loss = self.entropy_coeff * entropy
        total_loss = policy_loss - entropy_loss
        if self.beta > 0.0:
            kl = self._masked_mean(torch.exp(logps_ref - new_logps) - (logps_ref - new_logps) - 1, response_mask)
            total_loss += self.beta * kl
        total_loss = total_loss / self.grad_accumulation_steps
        total_loss.backward()
        return {
            "policy_loss": policy_loss.item(),
            "kl": kl.item() if self.beta > 0.0 else 0.0,
            "entropy": entropy.item(),
            "logp_mean": self._masked_mean(new_logps, response_mask).item(),
            "logp_std": self._masked_std(new_logps, response_mask).item()
        }

    def _update(self, batch):
        all_steps = tree.flatten(batch)
        self.model.set_adapter(self.model.actor_adapter_name)
        input_ids, attention_mask, response_mask, avg_len, pct_truncated = self._prepare_batch(all_steps)
        logps = torch.zeros(input_ids.shape[0], input_ids.shape[1]-1, device=self.device)
        if self.beta > 0.0: logps_ref = torch.zeros(input_ids.shape[0], input_ids.shape[1]-1, device=self.device)
        for i in range(0, input_ids.shape[0], self.mini_batch_size):
            mb_input_ids, mb_attention_mask = input_ids[i : i + self.mini_batch_size], attention_mask[i : i + self.mini_batch_size]
            with torch.no_grad():
                mb_logps = self._get_logps(mb_input_ids, mb_attention_mask, response_mask[i : i + self.mini_batch_size])[0]
                logps[i : i + self.mini_batch_size] = mb_logps
                if self.beta > 0.0:
                    with self.model.disable_adapter():
                        mb_logps_ref = self._get_logps(mb_input_ids, mb_attention_mask, response_mask[i : i + self.mini_batch_size])[0]
                    logps_ref[i : i + self.mini_batch_size] = mb_logps_ref
        # Advantages are already computed in sampling reward transformation - Therefore, we set A_{t,i} = R_i. 
        advantages = torch.zeros(logps.shape[0], logps.shape[1], device=self.device)
        for i in range(len(all_steps)): advantages[i, torch.where(response_mask[i])[0]] = all_steps[i].reward
        # Training loop
        metrics_acc: Dict[str, float] = {}
        for _ in range(self.epochs):
            N = input_ids.size(0)
            idx = torch.randperm(N, device=self.device)
            for i in range(self.grad_accumulation_steps):
                mb_idx = idx[i * self.mini_batch_size : (i + 1) * self.mini_batch_size]
                mb_input_ids, mb_attention_mask, mb_response_mask, mb_logps, mb_advantages = input_ids[mb_idx], attention_mask[mb_idx], response_mask[mb_idx], logps[mb_idx], advantages[mb_idx]
                if self.beta > 0.0: mb_logps_ref = logps_ref[mb_idx]
                else: mb_logps_ref = None
                update_metrics = self._mini_batch_update_step(mb_input_ids, mb_attention_mask, mb_response_mask, mb_logps, mb_advantages, mb_logps_ref)
                for k, v in update_metrics.items(): metrics_acc[k] = metrics_acc.get(k, 0.0) + v
                self.logger.info(f"Mini-step metrics: {update_metrics}")
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_params, self.grad_clip)
            metrics_acc["actor_grad_norm"] = metrics_acc.get("actor_grad_norm", 0.0) + float(grad_norm)
            self.actor_optimizer.step(); self.actor_optimizer.zero_grad(set_to_none=True); self.actor_lr_scheduler.step()
        log = {k: v / (self.epochs * self.grad_accumulation_steps) for k, v in metrics_acc.items()if k != "actor_grad_norm"}
        log.update({"actor_grad_norm": metrics_acc["actor_grad_norm"] / self.epochs})
        return {**log, "avg_train_len": avg_len, "pct_truncated": pct_truncated, "step": self._step, "samples_seen": self._samples_seen}
