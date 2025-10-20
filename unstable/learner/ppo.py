import ray
import torch
import torch.nn.functional as F
import tree
from typing import List, Dict

from unstable.learner.base import BaseLearner

from transformers import get_scheduler

@ray.remote
class PPOLearner(BaseLearner):
    def __init__(
        self,
        critic_learning_rate: float,
        critic_lr_scheduler_type: str,
        critic_lr_warmup_ratio: float,
        normalize_adv: bool,
        clip_value: float,
        gamma: float,
        gae_lambda: float,
        value_loss_coeff: float,
        clip_ratio: float,
        entropy_coeff: float,
        beta: float,
        **kwargs
    ):
        super().__init__(value_head=True, **kwargs)
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.beta = beta
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_adv = normalize_adv
        self.clip_value = clip_value
        self.critic_params = [p for n, p in self.model.named_parameters() if f".{self.model.critic_adapter_name}." in n or self.model.value_head_prefix in n]
        self.critic_optimizer = torch.optim.AdamW(self.critic_params, lr=critic_learning_rate)
        num_critic_optimizer_steps = int(self.total_training_steps * self.epochs)
        self.critic_lr_scheduler = get_scheduler(critic_lr_scheduler_type, self.critic_optimizer, num_warmup_steps=int(critic_lr_warmup_ratio * num_critic_optimizer_steps), num_training_steps=num_critic_optimizer_steps)

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
        out = self.engine(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, :-1, :]
        tgt_ids = input_ids[:, 1:]
        lse = torch.logsumexp(logits, dim=-1)
        tgt_logits = logits.gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
        tok_logp = tgt_logits - lse
        entropy = None
        if compute_entropy:
            probs       = torch.exp(logits - lse.unsqueeze(-1))
            tok_entropy = lse - (probs * logits).sum(dim=-1)
            entropy     = self._masked_mean(tok_entropy, response_mask)
        return tok_logp, entropy

    def _mini_batch_update_step(self, input_ids, attention_mask, response_mask, logps, returns, values, advantages, logps_ref) -> Dict[str, float]:
        # Policy Update
        self.model.set_adapter(self.model.actor_adapter_name)
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
        self.engine.backward(total_loss)
        # Critic Update
        self.model.set_adapter(self.model.critic_adapter_name)
        value_pred = self.model.values(input_ids, attention_mask)[:, :-1]
        value_loss = torch.max((value_pred - returns).pow(2), (torch.clamp(value_pred, values-self.clip_value, values+self.clip_value) - returns).pow(2))
        value_loss = (0.5 * self._masked_mean(value_loss, response_mask, axis=1)).mean()
        value_loss = value_loss * self.value_loss_coeff
        self.engine.backward(value_loss)
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "ratio": self._masked_mean(ratio, response_mask).item(),
            "kl": kl.item() if self.beta > 0.0 else 0.0,
            "entropy": entropy.item(),
            "logp_mean": self._masked_mean(new_logps, response_mask).item(),
            "logp_std": self._masked_std(new_logps, response_mask).item(),
            "value_mae": self._masked_mean(torch.abs(value_pred - returns), response_mask).item()
        }

    def _update(self, batch):
        all_steps = tree.flatten(batch)
        input_ids, attention_mask, response_mask, avg_len, pct_truncated = self._prepare_batch(all_steps)
        logps = torch.zeros(input_ids.shape[0], input_ids.shape[1]-1, device=self.device)
        values = torch.zeros(input_ids.shape[0], input_ids.shape[1]-1, device=self.device)
        if self.beta > 0.0: logps_ref = torch.zeros(input_ids.shape[0], input_ids.shape[1]-1, device=self.device)
        for i in range(0, len(all_steps), self.mini_batch_size):
            mb_input_ids, mb_attention_mask = input_ids[i : i + self.mini_batch_size], attention_mask[i : i + self.mini_batch_size]
            with torch.no_grad():
                self.model.set_adapter(self.model.actor_adapter_name)
                mb_logps = self._get_logps(mb_input_ids, mb_attention_mask, response_mask[i : i + self.mini_batch_size])[0]
                self.model.set_adapter(self.model.critic_adapter_name)
                mb_values = self.model.values(input_ids=mb_input_ids, attention_mask=mb_attention_mask)[:, :-1]
                logps[i : i + self.mini_batch_size] = mb_logps
                values[i : i + self.mini_batch_size] = mb_values
                if self.beta > 0.0:
                    with self.model.disable_adapter():
                        mb_logps_ref = self._get_logps(mb_input_ids, mb_attention_mask, response_mask[i : i + self.mini_batch_size])[0]
                    logps_ref[i : i + self.mini_batch_size] = mb_logps_ref
        # GAE
        rewards = torch.zeros(values.shape[0], values.shape[1], device=self.device)
        for i in range(len(all_steps)): rewards[i, torch.where(response_mask[i])[0][-1]] = all_steps[i].reward
        advantages = torch.zeros(values.shape[0], values.shape[1], device=self.device)
        for i in range(len(advantages)):
            action_inds = torch.where(response_mask[i])[0]
            lastgaelam = 0
            for t in reversed(action_inds):
                nextvalues = values[i, t + 1] if t < action_inds[-1] else 0.0
                delta = rewards[i, t] + self.gamma* nextvalues - values[i, t]
                lastgaelam = delta + self.gamma * self.gae_lambda * lastgaelam
                advantages[i, t] = lastgaelam
        returns = advantages + values 
        # Training loop
        metrics_acc: Dict[str, float] = {}
        for _ in range(self.epochs):
            N = input_ids.size(0)
            idx = torch.randperm(N, device=self.device)
            for i in range(self.grad_accumulation_steps):
                mb_idx = idx[i * self.mini_batch_size : (i + 1) * self.mini_batch_size]
                mb_input_ids, mb_attention_mask, mb_response_mask, mb_logps, mb_advantages = input_ids[mb_idx], attention_mask[mb_idx], response_mask[mb_idx], logps[mb_idx], advantages[mb_idx]
                if self.normalize_adv: mb_advantages = (mb_advantages - self._masked_mean(mb_advantages, mb_response_mask)) / self._masked_std(mb_advantages, mb_response_mask)
                mb_advantages, mb_values, mb_returns = mb_advantages, values[mb_idx], returns[mb_idx]
                if self.beta > 0.0: mb_logps_ref = logps_ref[mb_idx]
                else: mb_logps_ref = None
                update_metrics = self._mini_batch_update_step(mb_input_ids, mb_attention_mask, mb_response_mask, mb_logps, mb_returns, mb_values, mb_advantages, mb_logps_ref)
                for k, v in update_metrics.items(): metrics_acc[k] = metrics_acc.get(k, 0.0) + v
                self.logger.info(f"Mini-step metrics: {update_metrics}")
                self.engine.step()
        log = {k: v / (self.epochs * self.grad_accumulation_steps) for k, v in metrics_acc.items()}
        return {**log, "avg_train_len": avg_len, "pct_truncated": pct_truncated, "step": self._step, "samples_seen": self._samples_seen}

    def _save_checkpoint(self):
        ckpt_dir = super()._save_checkpoint()
        torch.save(getattr(self.model, self.model.value_head_prefix).state_dict(), ckpt_dir / "critic" / "value_head.pth")
        return ckpt_dir
