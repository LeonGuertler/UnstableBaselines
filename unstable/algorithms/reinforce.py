import ray, torch
from unstable.algorithms.base import BaseLearner

@ray.remote
class REINFORCELearner(BaseLearner):
    def __init__(self, max_train_len: int, max_generation_len: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_train_len = max_train_len
        self.max_generation_len = max_generation_len # for Dr. GRPO trick
 
    def _prepare_batch(self, steps):
        obs, acts, advs = zip(*[(s.obs, s.act, s.reward) for s in steps])
        advs = torch.tensor(advs, dtype=torch.float32, device=self.device)
        combined = [o + a for o, a in zip(obs, acts)]
        lengths = [len(self.tokenizer(text, add_special_tokens=False)["input_ids"]) for text in combined]
        avg_len = sum(lengths) / len(lengths)
        pct_truncated = sum(l > self.max_train_len for l in lengths) / len(lengths) if self.max_train_len else 0
        enc = self.tokenizer(combined, return_tensors="pt", padding=True, truncation=True, max_length=self.max_train_len).to(self.device) # Tokenize with truncation
        return enc, advs, obs, avg_len, pct_truncated

    def _mini_batch_update_step(self, steps, scaling: float = 1.0):
        enc, advs, obs, avg_len, pct_truncated = self._prepare_batch(steps=steps)
        out = self.policy_model(**enc)
        logp = torch.nn.functional.log_softmax(out.logits, dim=-1)
        tgt_ids = enc.input_ids[:, 1:]
        tok_logp = logp[:, :-1, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
        mask = torch.ones_like(enc.input_ids, dtype=torch.bool, device=self.device) # build prompt mask
        for i, o in enumerate(obs): mask[i, :len(self.tokenizer(o, add_special_tokens=False)["input_ids"])] = False
        mask = mask[:, 1:]
        seq_logp = (tok_logp * mask).sum(1) / self.max_generation_len
        loss = -(advs * seq_logp).mean() / scaling
        loss.backward()
        torch.cuda.empty_cache()
        return {"loss": loss.item(), "logp_mean": seq_logp.mean().item(), "avg_train_len": avg_len, "pct_truncated": pct_truncated}
    
    def _update(self, batch):
        metrics_acc = {}
        self.policy_optimizer.zero_grad(set_to_none=True)
        for i in range(self.gradient_acc_steps):
            sub = batch[i * self.mini_batch_size : (i + 1) * self.mini_batch_size]
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16): 
                update_metrics = self._mini_batch_update_step(sub, scaling=self.gradient_acc_steps)
            for k, v in update_metrics.items(): metrics_acc[k] = metrics_acc.get(k, 0.0) + v /  self.gradient_acc_steps
            self.logger.info(f"Mini-step metrics: {update_metrics}")
        self.logger.info(f"Step metrics: {metrics_acc}")
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.grad_clip)
        self.policy_optimizer.step()
        return metrics_acc
