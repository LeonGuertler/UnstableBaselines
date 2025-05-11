import torch 
from algorithms import BaseAlgo

import torch.nn as nn

class ActorCriticWrapper(nn.Module):
    def __init__(self, base, hidden):
        super().__init__()
        self.base = base                                     # LoRA-patched LM

        # pick dtype & device from base model
        ref_p   = next(base.parameters())
        dtype   = ref_p.dtype
        device  = ref_p.device

        self.v_head = nn.Linear(hidden, 1, bias=True).to(device=device, dtype=dtype)

    def forward(self, **enc):
        out = self.base(**enc, output_hidden_states=True)
        values = self.v_head(out.hidden_states[-1]).squeeze(-1)   # [B, T]
        return out.logits, values


class PPO(BaseAlgo):
    def __init__(self, args, model, tokenizer, device):
        super().__init__(args, model, tokenizer, device)
        self.clip_eps      = args.ppo_clip_eps     # 0.2
        self.vf_coef       = args.vf_coef          # 0.5
        self.entropy_coef  = args.entropy_coef     # 0.01

    # ---------- helpers ----------------------------------------------------
    # def _encode(self, obs, acts):
    #     enc = self.tokenizer([o + a for o, a in zip(obs, acts)],
    #                          return_tensors="pt", padding=True).to(self.device)
    #     # mask for generated tokens only
    #     mask = torch.ones_like(enc.input_ids, dtype=torch.bool)
    #     for i, o in enumerate(obs):
    #         L = len(self.tokenizer(o, add_special_tokens=False)["input_ids"])
    #         mask[i, :L] = False
    #     return enc, mask[:, 1:]                   # shift for labels

    def _encode(self, obs, acts):
        enc = self.tokenizer([o + a for o, a in zip(obs, acts)], return_tensors="pt", padding=True).to(self.device)
        # full-length mask: False = prompt, True = generated
        full_mask = torch.ones_like(enc.input_ids, dtype=torch.bool)
        for i, o in enumerate(obs):
            L = len(self.tokenizer(o, add_special_tokens=False)["input_ids"])
            full_mask[i, :L] = False

        return enc, full_mask                    # no slicing here




    def _seq_logp(self, logits, tgt_ids, mask):
        logp = torch.nn.functional.log_softmax(logits, dim=-1)
        tok_lp = logp[:, :-1, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
        return (tok_lp * mask).sum(1) / mask.sum(1).clamp(min=1)

    # ---------- public API --------------------------------------------------
    def update(self, steps):
        obs, acts, rewards = zip(*[(s.obs, s.act, s.reward) for s in steps])
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        enc, full_mask = self._encode(obs, acts)
        logits, values = self.model(**enc)                # one forward pass
        tgt = enc.input_ids[:, 1:]
        mask_lp = full_mask[:, 1:]

        # seq_logp = self._seq_logp(logits, tgt, mask)      # new log-probs
        seq_logp = self._seq_logp(logits, tgt, mask_lp)
        old_logp = seq_logp.detach()
        # values   = (values * mask).sum(1) / mask.sum(1).clamp(min=1)
        values = (values * full_mask).sum(1) / full_mask.sum(1).clamp(min=1)
        old_val  = values.detach()

        # advantage = normalized reward (already zero-mean, unit-var)
        adv = rewards
        ratio = torch.exp(seq_logp - old_logp)

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss  = 0.5 * ((values - rewards) ** 2).mean()

        entropy = -(seq_logp.exp() * seq_logp).mean()

        loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy
        loss.backward()

        return {
            "loss": loss.item(),
            "policy": policy_loss.item(),
            "value": value_loss.item(),
            "entropy": entropy.item(),
            "clip_frac": ((ratio > 1+self.clip_eps) |
                          (ratio < 1-self.clip_eps)).float().mean().item(),
            "num_steps": len(steps)
        }



# class PPO(BaseAlgo):
#     def prepare_batch(self, steps):
#         """
#         Turn a list[Step] into tensors on self.dev.
#         Return whatever update() needs.
#         """
#         raise NotImplementedError

#     def update(self, batch):
#         """
#         One gradient update on *this worker only*.
#         Must call .backward() but NOT .step().
#         Return latest loss as float (for logging).
#         """
#         raise NotImplementedError