import torch 
from algorithms import BaseAlgo

class Reinforce(BaseAlgo):
    def prepare_batch(self, steps):
        obs, acts, advs = zip(*[(s.obs, s.act, s.reward) for s in steps])
        advs = torch.tensor(advs, dtype=torch.float32, device=self.device)
        enc = self.tokenizer([o + a for o, a in zip(obs, acts)], return_tensors="pt", padding=True).to(self.device)
        return enc, advs, obs

    def update(self, steps):
        enc, advs, obs = self.prepare_batch(steps=steps) # unpack
        out = self.model(**enc) # forward
        logp = torch.nn.functional.log_softmax(out.logits, dim=-1)
        tgt_ids = enc.input_ids[:, 1:]
        tok_logp = logp[:, :-1, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
        mask = torch.ones_like(enc.input_ids, dtype=torch.bool, device=self.device) # build prompt mask
        for i, o in enumerate(obs):
            L = len(self.tokenizer(o, add_special_tokens=False)["input_ids"])
            mask[i, :L] = False
        mask = mask[:, 1:]
        seq_logp = (tok_logp * mask).sum(1) / mask.sum(1).clamp(min=1)
        loss = -(advs * seq_logp).mean()
        loss.backward()
        # update_info_dict = {
        #     "loss": loss.item(),
        #     "adv_mean": advs.mean().item(),
        #     "adv_std": advs.std().item(),
        #     "logp_mean": seq_logp.mean().item(),
        #     "logp_std": seq_logp.std().item(),
        #     "grad_norm": self.get_grad_norm(),  # Optional: requires implementing `get_grad_norm`
        #     "lr": self.optimizer.param_groups[0]["lr"] if hasattr(self, "optimizer") else None,
        # }

        # return loss.item(), update_info_dict

        return {
            "loss": loss.item(),
            "logp_mean": seq_logp.mean().item(),
            "logp_std": seq_logp.std().item(),
            "num_steps": len(steps),
        }

