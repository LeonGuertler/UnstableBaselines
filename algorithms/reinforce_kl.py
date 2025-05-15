import torch 
from algorithms import BaseAlgo

class Reinforce_KL(BaseAlgo):
    def __init__(self, args, model, tokenizer, device, ref_model):
        self.ref_model = ref_model 
        self.kl_coef = 0.2 
        self.ref_model.eval()
        self.ref_model = self.ref_model.to(device)
        super().__init__(args, model, tokenizer, device)


    def prepare_batch(self, steps):
        obs, acts, advs = zip(*[(s.obs, s.act, s.reward) for s in steps])
        advs = torch.tensor(advs, dtype=torch.float32, device=self.device)
        enc = self.tokenizer([o + a for o, a in zip(obs, acts)], return_tensors="pt", padding=True).to(self.device)
        return enc, advs, obs

    def update(self, steps):
        enc, advs, obs = self.prepare_batch(steps=steps)
        
        # Get logits from policy (π)
        out = self.model(**enc)
        logp = torch.nn.functional.log_softmax(out.logits, dim=-1)

        # Get logits from reference policy (π_ref) - no grad
        with torch.no_grad():
            ref_out = self.ref_model(**enc)
            ref_logp = torch.nn.functional.log_softmax(ref_out.logits, dim=-1)

        # Prepare masks for token selection
        tgt_ids = enc.input_ids[:, 1:]  # shifted targets
        logp = logp[:, :-1, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
        ref_logp = ref_logp[:, :-1, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)

        # Mask to ignore prompt tokens
        mask = torch.ones_like(enc.input_ids, dtype=torch.bool, device=self.device)
        for i, o in enumerate(obs):
            L = len(self.tokenizer(o, add_special_tokens=False)["input_ids"])
            mask[i, :L] = False
        mask = mask[:, 1:]  # shift to match token targets

        # REINFORCE loss (policy gradient)
        seq_logp = (logp * mask).sum(1) / mask.sum(1).clamp(min=1)
        pg_loss = -(advs * seq_logp).mean()

        # KL penalty
        kl = (logp - ref_logp) * mask
        kl = kl.sum(1) / mask.sum(1).clamp(min=1)
        kl_loss = kl.mean()

        # Total loss
        loss = pg_loss + self.kl_coef * kl_loss
        loss.backward()

        return {
            "loss": loss.item(), "pg_loss": pg_loss.item(), "kl_loss": kl_loss.item(), "kl_coef": self.kl_coef, 
            "logp_mean": seq_logp.mean().item(), "logp_std": seq_logp.std().item(), "num_steps": len(steps),
        }