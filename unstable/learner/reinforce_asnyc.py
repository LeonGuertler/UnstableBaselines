import ray, torch

from unstable.learner.base import BaseLearner

@ray.remote
class REINFORCEAsyncLearner(BaseLearner):
    def __init__(self, max_train_len: int, max_generation_len: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_train_len = max_train_len
        self.max_generation_len = max_generation_len
 
    def _micro_batch_update_step(self, steps):
        input_ids, advs, vllm_logprobs, lengths, attention_mask, response_mask, pct_truncated = self._prepare_batch(steps=steps)
        out = self.engine(input_ids=input_ids, attention_mask=attention_mask)
        logp = torch.nn.functional.log_softmax(out.logits, dim=-1)
        tok_logp = logp[:, :-1, :].gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        seq_logp = (tok_logp * response_mask).sum(1) / self.max_generation_len
        vllm_kl_seq = self._masked_mean(torch.exp(vllm_logprobs - tok_logp) - (vllm_logprobs - tok_logp) - 1, response_mask, axis=1)
        vllm_seq_logp = (vllm_logprobs * response_mask).sum(1) / self.max_generation_len
        ratio = torch.exp(seq_logp - vllm_seq_logp)
        loss = -(advs * ratio).mean()
        self.engine.backward(loss)  
        return {"loss": loss.item() / self.grad_accumulation_steps, "seq_logp_mean": seq_logp.mean().item(), "vllm_seq_logp_mean": vllm_seq_logp.mean().item(), "avg_train_len": sum(lengths) / len(lengths), "pct_truncated": pct_truncated, "offpolicy_ratio": ratio.mean().item(), "vllm_kl_seq": vllm_kl_seq.mean().item()}
    
    def _update(self, batch):
        from deepspeed.utils import safe_get_full_grad
        metrics_acc = {}
        for i in range(self.grad_accumulation_steps):
            sub = batch[i * self.micro_batch_size : (i + 1) * self.micro_batch_size]
            update_metrics = self._micro_batch_update_step(sub)
            for k, v in update_metrics.items(): metrics_acc[k] = metrics_acc.get(k, 0.0) + v /  self.grad_accumulation_steps
            self.logger.info(f"Mini-step metrics: {update_metrics}")
            if self.engine.is_gradient_accumulation_boundary(): metrics_acc['grad_norm'] = (sum(safe_get_full_grad(p).norm(2).cpu()**2 for p in self.model.parameters() if safe_get_full_grad(p) is not None) ** 0.5).item()
            self.engine.step()
        self.logger.info(f"Step metrics: {metrics_acc}")
        return metrics_acc
