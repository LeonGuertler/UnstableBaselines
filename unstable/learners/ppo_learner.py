import ray
import torch
import torch.nn.functional as F
import tree
import random
from typing import List, Dict, Optional
from dataclasses import replace

from unstable.learners.base import BaseLearner
from unstable.learners.a2c_learner import compute_gae
from unstable.learners.utils import build_peft_model, enable_full_activation_ckpt
from unstable.reward_transformations.transformation_sampling import NormalizeRewardsByEnv


@ray.remote
class PPOLearner(BaseLearner):
    def initialize_algorithm(
        self,
        infer_mini_batch_size: int,
        clip_ratio: float = 0.2,
        update_epochs: int = 4,
        entropy_coeff: float = 0.0,
        value_loss_coeff: float = 0.5,
        critic_learning_rate: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_adv: bool = False,
        max_generation_len: Optional[int] = None,
        max_train_len: Optional[int] = None,
        initial_lora_path: Optional[str] = None,
    ):
        self.infer_mini_batch_size = infer_mini_batch_size
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_adv = normalize_adv
        self.max_generation_len = max_generation_len
        self.max_train_len = max_train_len

        self.critic, _ = build_peft_model(self.model_name, self.device, self.lora_cfg, initial_lora_path, critic_model=True)

        if not self.use_trainer_cache: self.policy_model.config.use_cache = False
        if self.gradient_checkpointing: self.policy_model.gradient_checkpointing_enable()
        if self.activation_checkpointing: enable_full_activation_ckpt(self.policy_model)

        self.critic_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.critic.parameters()), lr=critic_learning_rate)

    def _prepare_batch(self, steps: List) -> tuple:
        obs, acts, old_logps, advs, rets = zip(*[(s.obs, s.act, s.step_info.get("old_logp", torch.nan), s.step_info.get("advantage", torch.nan), s.step_info.get("return", torch.nan))for s in steps])
        old_logps = torch.tensor(old_logps, dtype=torch.float32, device=self.device)
        advs      = torch.tensor(advs,     dtype=torch.float32, device=self.device)
        rets      = torch.tensor(rets,     dtype=torch.float32, device=self.device)
        combined  = [o + a for o, a in zip(obs, acts)]
        lengths   = [len(self.tokenizer(text, add_special_tokens=False)["input_ids"]) for text in combined]
        avg_len   = sum(lengths) / len(lengths)
        pct_truncated = (sum(l > self.max_train_len for l in lengths) / len(lengths) if self.max_train_len else 0.0)
        enc = self.tokenizer(combined, return_tensors="pt", padding=True, truncation=True, max_length=self.max_train_len).to(self.device)
        state_enc = self.tokenizer(obs, return_tensors="pt", padding=True, truncation=True, max_length=self.max_train_len).to(self.device)
        return enc, state_enc, old_logps, advs, rets, obs, avg_len, pct_truncated

    def _forward_minibatch(self, enc, obs):
        out = self.policy_model(**enc)
        logits = out.logits
        logp = F.log_softmax(logits, dim=-1)
        entropy = -(torch.exp(logp) * logp).sum(dim=-1).mean()
        tgt_ids = enc.input_ids[:, 1:]
        tok_logp = logp[:, :-1, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
        mask = torch.ones_like(enc.input_ids, dtype=torch.bool, device=self.device)
        for i, text in enumerate(obs): mask[i, :len(self.tokenizer(text, add_special_tokens=False)["input_ids"])] = False
        mask = mask[:, 1:]
        seq_logp = (tok_logp * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        return seq_logp, entropy, mask

    def _mini_batch_update_step(self, steps: List) -> Dict[str, float]:
        enc, state_enc, old_logps, advs, rets, obs, avg_len, pct_truncated = self._prepare_batch(steps)
        seq_logp, entropy, _ = self._forward_minibatch(enc, obs)

        ratio = torch.exp(seq_logp - old_logps)
        surr1 = ratio * advs
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advs
        policy_loss = -torch.min(surr1, surr2).mean()

        value_pred = self.critic(**state_enc)[:, 0]
        value_loss = 0.5 * ((value_pred - rets) ** 2).mean() * self.value_loss_coeff
        entropy_loss = entropy * self.entropy_coeff

        total_loss = policy_loss + value_loss - entropy_loss
        total_loss.backward()
        torch.cuda.empty_cache()

        return {
            "policy_loss": policy_loss.item(), "value_loss": value_loss.item(), "entropy": entropy.item(), "logp_mean": seq_logp.mean().item(), "logp_std": seq_logp.std().item(),
            "value_mae": (value_pred - rets).abs().mean().item(), "avg_train_len": avg_len, "pct_truncated": pct_truncated,
        }

    def _update(self, batch):
        all_steps = tree.flatten(batch)
        old_values, old_logps = [], []
        for i in range(0, len(all_steps), self.infer_mini_batch_size):
            sub = all_steps[i : i + self.infer_mini_batch_size]
            enc, state_enc, _, _, _, obs, _, _ = self._prepare_batch(sub)
            with torch.no_grad():
                old_values.append(self.critic(**state_enc)[:, 0].cpu())
                seq_logp, _, _ = self._forward_minibatch(enc, obs)
                old_logps.append(seq_logp.cpu())
        old_values = torch.cat(old_values)
        old_logps = torch.cat(old_logps)
        ep_values = torch.split(old_values, [len(ep) for ep in batch])
        ep_rewards = [torch.tensor([s.reward for s in ep]) for ep in batch]
        ep_adv, ep_ret = [], []
        for rewards, values in zip(ep_rewards, ep_values):
            adv = compute_gae(rewards, values)
            ep_adv.append(adv)
            ep_ret.append(adv + values)

        train_data = []
        for i, ep in enumerate(batch):
            for j, step in enumerate(ep):
                step = replace(step, reward=ep_adv[i][j].item())
                step = replace(step, step_info={**step.step_info, "return": ep_ret[i][j].item(), "old_logp": old_logps[sum(len(e) for e in batch[:i]) + j].item(), "advantage": ep_adv[i][j].item()})
                train_data.append(step)

        if self.normalize_adv: train_data = NormalizeRewardsByEnv(True)(train_data)

        metrics_acc: Dict[str, float] = {}
        num_batches = self.batch_size // self.mini_batch_size
        for _ in range(self.update_epochs):
            self.policy_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            random.shuffle(train_data)
            for i in range(num_batches):
                sub = train_data[i * self.mini_batch_size : (i + 1) * self.mini_batch_size]
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    update_metrics = self._mini_batch_update_step(sub)
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.grad_clip)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
                self.policy_optimizer.step()
                self.critic_optimizer.step()
                for k, v in update_metrics.items(): metrics_acc[k] = metrics_acc.get(k, 0.0) + v
                self.logger.info(f"Mini-step metrics: {update_metrics}")

        log = {k: v / (self.update_epochs * num_batches) for k, v in metrics_acc.items()}
        self._step += 1
        return {**log, "step": self._step, "samples_seen": self._samples_seen}
