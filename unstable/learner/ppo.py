import os
import ray
import random
import torch
import torch.nn.functional as F
import tree
from dataclasses import replace
from transformers import get_scheduler

from unstable.collection.reward_transformations import NormalizeRewards
from unstable.learner.models import build_peft_model, enable_full_activation_ckpt
from unstable.learner.base import BaseLearner
from unstable.utils.misc import write_training_data_to_file


@ray.remote
class PPOLearner(BaseLearner):
    def __init__(
        self,
        infer_micro_batch_size: int,
        normalize_adv: bool,
        clip_value: float,
        value_coeff: float,
        gamma: float,
        gae_lambda: float,
        upper_clip_ratio: float,
        lower_clip_ratio: float,
        entropy_coeff: float,
        beta: float,
        critic_learning_rate: float = 1e-6,
        critic_lr_warmup_ratio: float = 0.1,
        critic_lr_scheduler_type: str = "constant",
        **kwargs
    ):
        super().__init__(value_head=True, **kwargs)
        self.upper_clip_ratio = upper_clip_ratio
        self.lower_clip_ratio = lower_clip_ratio
        self.infer_micro_batch_size = infer_micro_batch_size
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.beta = beta
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_adv = normalize_adv
        self.clip_value = clip_value
        self.critic_model, _ = build_peft_model(self.model_name, self.device, self.lora_cfg, checkpoint_cfg=self.checkpoint_cfg.get('critic', {}), value_head=True)
        self._value_params = [p for n, p in self.critic_model.named_parameters() if (f".{self.critic_model.adapter_name}." in n or f".{self.critic_model.value_head_prefix}." in n) and p.requires_grad]
        self.critic_optimizer = torch.optim.AdamW(self._value_params, lr=critic_learning_rate, fused=True)
        self.critic_lr_scheduler = get_scheduler(critic_lr_scheduler_type, self.critic_optimizer, num_warmup_steps=int(critic_lr_warmup_ratio * self.total_optimizer_steps), num_training_steps=self.total_optimizer_steps)
        self.critic_engine, self.critic_optimizer, _, self.critic_lr_scheduler = self.init_distributed_training(self.critic_model, self.critic_optimizer)
        self.critic_model = self.critic_engine.module

    def _micro_batch_update_step(self, steps):
        prompt_ids, prompt_attention_mask, input_ids, advs, vllm_logprobs, lengths, attention_mask, response_mask, pct_truncated, returns = self._prepare_batch(steps=steps)
        # Compute reference log probs with base model
        with torch.no_grad():
            self.model.disable_adapter_layers()
            ref_out = self.engine(input_ids=input_ids, attention_mask=attention_mask)
            ref_logp = F.log_softmax(ref_out.logits / self.temperature, dim=-1)
            ref_tok_logp = ref_logp[:, :-1, :].gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            self.model.enable_adapter_layers()
        # Compute policy logps and values
        out = self.engine(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, :-1, :] / self.temperature
        logp = F.log_softmax(logits, dim=-1)
        tok_logp = logp.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        seq_logp = (tok_logp * response_mask).sum(1) / self.max_generation_len
        # Entropy (computed from raw logits, matching TRL's entropy_from_logits)
        probs = torch.softmax(logits, dim=-1)
        entropy_per_token = torch.logsumexp(logits, dim=-1) - (probs * logits).sum(dim=-1)
        entropy_seq = self._masked_mean(entropy_per_token, response_mask, axis=1)
        # KL divergence against base model
        kl_seq = self._masked_mean(torch.exp(tok_logp - ref_tok_logp) - (tok_logp - ref_tok_logp) - 1, response_mask, axis=1)
        # Importance ratio
        vllm_seq_logp = (vllm_logprobs * response_mask).sum(1) / self.max_generation_len
        ratio = torch.exp(seq_logp - vllm_seq_logp)
        clipped_ratio = torch.clamp(ratio, 1 - self.lower_clip_ratio, 1 + self.upper_clip_ratio)
        # Policy loss
        policy_loss = -torch.min(advs * ratio, advs * clipped_ratio).mean()
        kl_loss = self.beta * kl_seq.mean()
        entropy_loss = -self.entropy_coeff * entropy_seq.mean()
        loss = policy_loss + kl_loss + entropy_loss
        self.engine.backward(loss)
        # Value MSE with clipping
        values = self.critic_model.values(prompt_ids, attention_mask=prompt_attention_mask)
        value = values[torch.arange(values.size(0), device=values.device), prompt_attention_mask.sum(dim=1) - 1]
        old_values = torch.tensor([s.step_info["old_value"] for s in steps], dtype=torch.float32, device=self.device)
        value_clipped = old_values + torch.clamp(value - old_values, -self.clip_value, self.clip_value)
        # Value loss
        value_loss = 0.5 * torch.max((value - returns) ** 2, (value_clipped - returns) ** 2).mean()
        self.critic_engine.backward(self.value_coeff * value_loss)

        return {"loss": loss.item(), "policy_loss": policy_loss.item(), "kl_loss": kl_loss.item(), "seq_logp_mean": seq_logp.mean().item(), "ref_logp_mean": (ref_tok_logp * response_mask).sum(1).mean().item(), "ratio": ratio.mean().item(), 
                "vllm_seq_logp_mean": vllm_seq_logp.mean().item(), "avg_train_len": sum(lengths) / len(lengths), "pct_truncated": pct_truncated, "offpolicy_ratio": ratio.mean().item(), "kl_seq": kl_seq.mean().item(), 
                "entropy": entropy_seq.mean().item(), "clip_frac": ((ratio < 1 - self.lower_clip_ratio) | (ratio > 1 + self.upper_clip_ratio)).float().mean().item(), "value_loss": value_loss.item()}

    @staticmethod
    def _compute_gae(rewards, values, gamma, gae_lambda):
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + (gamma * values[t + 1] if t + 1 < len(values) else 0) - values[t]
            advantages[t] = last_advantage = delta + (gamma * gae_lambda * last_advantage)
        returns = advantages + values
        return advantages, returns

    def _update(self, batch):
        from deepspeed.utils import safe_get_full_grad
        total_steps = self.epochs * self.grad_accumulation_steps
        all_samples, all_values, all_rewards, ep_lens = tree.flatten(batch), [], [], [len(ep) for ep in batch]
        for i in range(0, len(all_samples), self.infer_micro_batch_size):
            with torch.no_grad():
                micro_batch = all_samples[i : i + self.infer_micro_batch_size]
                prompt_ids, prompt_attention_mask, _, rewards, _, _, _, _, _, _ = self._prepare_batch(steps=micro_batch)
                values = self.critic_model.values(prompt_ids, attention_mask=prompt_attention_mask)
                all_values.append(values[torch.arange(values.size(0), device=values.device), prompt_attention_mask.sum(dim=1) - 1])
                all_rewards.append(rewards)
        all_values, all_rewards = torch.cat(all_values).float().cpu(), torch.cat(all_rewards).float().cpu()
        ep_values, ep_rewards = torch.split(all_values, ep_lens), torch.split(all_rewards, ep_lens)
        ep_advs, ep_returns = [], []
        for rewards, values in zip(ep_rewards, ep_values):
            advs, returns = self._compute_gae(rewards, values, self.gamma, self.gae_lambda)
            ep_advs.append(advs); ep_returns.append(returns)
        train_batch = []
        for i, ep in enumerate(batch):
            for j, step in enumerate(ep):
                train_batch.append(replace(step, reward=ep_advs[i][j].item(), step_info={**step.step_info, "return": ep_returns[i][j].item(), "old_value": ep_values[i][j].item(), "advantage": ep_advs[i][j].item()}))
        all_advs = torch.cat(ep_advs).float()
        if self.normalize_adv: train_batch = NormalizeRewards(z_score=True)(train_batch)
        train_batch = [replace(s, step_info={**s.step_info, "normalized_adv": s.reward}) for s in train_batch]
        train_dir = ray.get(self.tracker.get_train_dir.remote())
        write_training_data_to_file(train_batch, os.path.join(train_dir, f"train_data_step_{self._step}.csv"), overwrite=True)
        metrics_acc = {'rewards_mean': all_rewards.mean().item(), 'rewards_std': all_rewards.std().item(), 'values_mean': all_values.mean().item(), 'values_std': all_values.std().item(), 'advantages_mean': all_advs.mean().item(), 'advantages_std': all_advs.std().item()}
        for epoch in range(self.epochs):
            random.shuffle(train_batch)
            for i in range(self.grad_accumulation_steps):
                sub = train_batch[i * self.micro_batch_size : (i + 1) * self.micro_batch_size]
                update_metrics = self._micro_batch_update_step(sub)
                for k, v in update_metrics.items(): metrics_acc[k] = metrics_acc.get(k, 0.0) + v
                self.logger.info(f"Epoch {epoch+1}/{self.epochs} actor mini-step metrics: {update_metrics}")
                if self.engine.is_gradient_accumulation_boundary(): metrics_acc['grad_norm_actor'] = (sum(safe_get_full_grad(p).norm(2).cpu()**2 for p in self._policy_params if safe_get_full_grad(p) is not None) ** 0.5).item()
                if self.critic_engine.is_gradient_accumulation_boundary(): 
                    print('GRAD NORM', (sum(safe_get_full_grad(p).norm(2).cpu()**2 for p in self._value_params if safe_get_full_grad(p) is not None) ** 0.5).item())
                    metrics_acc['grad_norm_critic'] = (sum(safe_get_full_grad(p).norm(2).cpu()**2 for p in self._value_params if safe_get_full_grad(p) is not None) ** 0.5).item()
                self.engine.step()
                self.critic_engine.step()
        for k in metrics_acc: 
            if k not in ['rewards_mean', 'rewards_std', 'values_mean', 'values_std', 'advantages_mean', 'advantages_std', 'grad_norm_actor', 'grad_norm_critic']: metrics_acc[k] /= total_steps
        self.logger.info(f"Step metrics: {metrics_acc}")
        metrics_acc['grad_norm'] = (metrics_acc.get('grad_norm_actor', 0.0) + metrics_acc.get('grad_norm_critic', 0.0)) / 2 if 'grad_norm_actor' in metrics_acc and 'grad_norm_critic' in metrics_acc else 0.0
        return metrics_acc
    
    def _save_checkpoint(self):
        ckpt_dir = super()._save_checkpoint()
        critic_dir = ckpt_dir / "critic"
        critic_dir.mkdir(parents=True, exist_ok=True)
        self.critic_model.save_pretrained(critic_dir, save_adapter=True)
        vh = getattr(self.critic_model.base_model, "value_head", None)
        if vh is not None: torch.save(vh.state_dict(), critic_dir / "value_head.pt")
        return ckpt_dir
    