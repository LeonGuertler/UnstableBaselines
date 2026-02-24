import random
import ray, torch

from unstable.learner.base import BaseLearner

@ray.remote
class REINFORCELearner(BaseLearner):
    def __init__(self, max_train_len: int, max_generation_len: int, kl_coef: float = 0.0, entropy_coeff: float = 0.0, epochs: int = 2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_train_len = max_train_len
        self.max_generation_len = max_generation_len
        self.kl_coef = kl_coef
        self.entropy_coeff = entropy_coeff
        self.epochs = epochs
 
    def _micro_batch_update_step(self, steps):
        _, _, input_ids, advs, vllm_logprobs, lengths, attention_mask, response_mask, pct_truncated, _ = self._prepare_batch(steps=steps)
        # Compute reference log probs with base model
        with torch.no_grad():
            self.model.disable_adapter_layers()
            ref_out = self.engine(input_ids=input_ids, attention_mask=attention_mask)
            ref_logp = torch.nn.functional.log_softmax(ref_out.logits / self.temperature, dim=-1)
            ref_tok_logp = ref_logp[:, :-1, :].gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            self.model.enable_adapter_layers()
        # Compute policy log probs with LoRA adapter
        out = self.engine(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, :-1, :] / self.temperature
        logp = torch.nn.functional.log_softmax(logits, dim=-1)
        tok_logp = logp.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        seq_logp = (tok_logp * response_mask).sum(1) / self.max_generation_len
        # Entropy (computed from raw logits, matching TRL's entropy_from_logits)
        probs = torch.softmax(logits, dim=-1)
        entropy_per_token = torch.logsumexp(logits, dim=-1) - (probs * logits).sum(dim=-1)
        entropy_seq = self._masked_mean(entropy_per_token, response_mask, axis=1)
        # KL divergence against base model
        kl_seq = self._masked_mean(torch.exp(tok_logp - ref_tok_logp) - (tok_logp - ref_tok_logp) - 1, response_mask, axis=1)
        # Importance ratio for off-policy correction
        vllm_seq_logp = (vllm_logprobs * response_mask).sum(1) / self.max_generation_len
        ratio = torch.exp(seq_logp - vllm_seq_logp)
        # Policy loss
        policy_loss = -(advs * ratio).mean()
        kl_loss = self.kl_coef * kl_seq.mean()
        entropy_loss = -self.entropy_coeff * entropy_seq.mean()
        loss = policy_loss + kl_loss + entropy_loss
        self.engine.backward(loss)
        return {"loss": loss.item(), "policy_loss": policy_loss.item(), "kl_loss": kl_loss.item(), "seq_logp_mean": seq_logp.mean().item(), "ref_logp_mean": (ref_tok_logp * response_mask).sum(1).mean().item() / self.max_generation_len, "vllm_seq_logp_mean": vllm_seq_logp.mean().item(), "avg_train_len": sum(lengths) / len(lengths), "pct_truncated": pct_truncated, "offpolicy_ratio": ratio.mean().item(), "kl_seq": kl_seq.mean().item(), "entropy": entropy_seq.mean().item()}
    
    def _update(self, batch):
        from deepspeed.utils import safe_get_full_grad
        metrics_acc = {}
        total_steps = self.epochs * self.grad_accumulation_steps
        for epoch in range(self.epochs):
            random.shuffle(batch)
            for i in range(self.grad_accumulation_steps):
                sub = batch[i * self.micro_batch_size : (i + 1) * self.micro_batch_size]
                update_metrics = self._micro_batch_update_step(sub)
                for k, v in update_metrics.items(): metrics_acc[k] = metrics_acc.get(k, 0.0) + v
                self.logger.info(f"Epoch {epoch+1}/{self.epochs} mini-step metrics: {update_metrics}")
                if self.engine.is_gradient_accumulation_boundary(): metrics_acc['grad_norm'] = metrics_acc.get('grad_norm', 0.0) + (sum(safe_get_full_grad(p).norm(2).cpu()**2 for p in self.model.parameters() if safe_get_full_grad(p) is not None) ** 0.5).item()
                self.engine.step()
        for k in update_metrics: metrics_acc[k] /= total_steps
        self.logger.info(f"Step metrics: {metrics_acc}")
        return metrics_acc
