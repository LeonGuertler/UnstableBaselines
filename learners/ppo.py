import math, gc
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field

import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def masked_mean(x, mask, dim=None, eps=1e-8):
    if dim is None:
        return (x * mask).sum() / (mask.sum() + eps)
    return (x * mask).sum(dim) / (mask.sum(dim) + eps)

def align_for_causal_loss(logits: torch.Tensor, input_ids: torch.Tensor, mask: torch.Tensor):
    logits_s = logits[:, :-1, :]
    labels_s = input_ids[:, 1:]
    mask_s   = mask[:, 1:]
    return logits_s, labels_s, mask_s


def build_adv_no_critic(final_rewards: torch.Tensor,
                        seq_len: int,
                        gamma: float = 0.99,
                        lam: float   = 0.95):
    """
    final_rewards  shape (B,)      – the terminal reward of each trajectory
    returns        shape (B, L-1)  – one advantage for every token *before* <eos>
    """
    B = final_rewards.size(0)
    adv = torch.zeros(B, seq_len, device=final_rewards.device)
    adv[:, -1] = final_rewards                      # put R at the last token
    γλ = gamma * lam
    for t in reversed(range(seq_len - 1)):
        adv[:, t] = γλ * adv[:, t + 1]              # exponential back-prop
    # optional but recommended normalisation
    # adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv


class PPOLearner:
    def __init__(self, args):
        self.args = args 

        # set the device 
        torch.cuda.set_device(0)

        # load the models
        self.model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
        self.ref_model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
        for p in self.ref_model.parameters(): p.requires_grad = False # ensure no grads
        # self.critic = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)


        self.actor_optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        # self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=self.args.lr)

        # gradient checkpointing
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            # self.critic.gradient_checkpointing_enable()

        # bfloat16 training
        if self.args.bf16_training:
            self.model.to(torch.bfloat16)
            self.ref_model.to(torch.bfloat16)
            # self.critic.to(torch.bfloat16)


    def compute_gae(self, rewards, values, masks):
        adv = torch.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(rewards.size(1))):
            next_val = values[:, t+1] if t+1 < rewards.size(1) else 0
            delta = rewards[:, t] + self.args.gamma * next_val - values[:, t]
            lastgaelam = delta + self.args.gamma * self.args.lam * lastgaelam
            adv[:, t] = lastgaelam
        return adv, adv + values

    def get_logps(self, logits, input_ids, mask):
        logits_s, labels_s, mask_s = align_for_causal_loss(logits, input_ids, mask)
        logps = torch.gather(F.log_softmax(logits_s, dim=-1), 2, labels_s.unsqueeze(-1)).squeeze(-1)
        return logps, mask_s


    # def update(self, steps: List):
    #     # sort steps by size for more efficient mini-batching
    #     # steps.sort(key=lambda s: len(self.tokenizer(s.obs + s.act, add_special_tokens=False)["input_ids"]))

    #     # extract from steps
    #     observations, actions, rewards = [], [], []
    #     for step in steps:
    #         observations.append(step.obs)
    #         actions.append(step.act)
    #         rewards.append(step.reward)

    #     # convert rewards to torch
    #     rewards = torch.tensor(rewards, dtype=torch.float32, device="cuda") * self.args.reward_scale

    #     full_texts = [p + c for p, c in zip(observations, actions)]

    #     # iterate over mini-batches
    #     assert len(steps)%self.args.gradient_accumulation_steps==0, f"The batch-size ({len(steps)}) has to be divisible by the number of gradient accumulation steps ({self.args.mini_batch_size})."
    #     batch_size = len(steps)

    #     for _ in range(self.args.ppo_epochs):
    #         self.actor_optimizer.zero_grad(set_to_none=True)
    #         self.critic_optimizer.zero_grad(set_to_none=True)

    #         mini_batch_size = len(steps) // self.args.gradient_accumulation_steps

    #         # for i in range(0, batch_size, mini_batch_size): # TODO for sure randomize order (maybe get list of idx first (still blocks) and randomly pick a block)
    #         for mini_g_step in range(0, self.args.gradient_accumulation_steps):
    #             mb_full_texts = full_texts[mini_g_step*mini_batch_size: (mini_g_step+1)*mini_batch_size]
    #             mb_observations = observations[mini_g_step*mini_batch_size: (mini_g_step+1)*mini_batch_size]
    #             mb_rewards = rewards[mini_g_step*mini_batch_size: (mini_g_step+1)*mini_batch_size]


    #             mb_encodings = self.tokenizer(mb_full_texts, return_tensors="pt", padding=True, add_special_tokens=True).to("cuda")
    #             mb_prompt_lens = [len(self.tokenizer(p, add_special_tokens=False)["input_ids"]) for p in mb_observations]

    #             mb_completion_mask = torch.ones_like(mb_encodings.input_ids, dtype=torch.bool)
    #             for i, l in enumerate(mb_prompt_lens):
    #                 mb_completion_mask[i, :l] = False

    #             print(f"ACTIVE BATCH SIZE: {len(mb_full_texts)}")


    #             with torch.no_grad():
    #                 # mb_logp_old = self.get_logps()
    #                 # mb_logp_ref = 
    #                 # mb_values = 
    #                 # Reference policy
    #                 ref_logits = self.ref_model(**mb_encodings).logits.float()
    #                 mb_logp_ref, shifted_mb_completion_mask = self.get_logps(ref_logits, mb_encodings.input_ids, mb_completion_mask)

    #                 # Value model
    #                 value_logits = self.critic(**mb_encodings).logits.float()
    #                 mb_values = value_logits[..., 0][:, :-1]


    #             # build dense reward tensor
    #             mb_dense_rewards = torch.zeros_like(mb_values)
    #             mb_dense_rewards[:, -1] = mb_rewards # final token reward
    #             mb_advantage, mb_rets = self.compute_gae(mb_dense_rewards, mb_values, shifted_mb_completion_mask) # calcualte the advantage
    #             mb_advantage_norm = (mb_advantage - mb_advantage.mean()) / (mb_advantage.std()+1e-8) # normalize advantage
    #             infos = {} # log dict

    #             # get new log_p
    #             model_outputs = self.model(**mb_encodings)
    #             logits = model_outputs.logits.float()
    #             mb_logp_new, _ = self.get_logps(logits, mb_encodings.input_ids, mb_completion_mask)
    #             mb_ratio = torch.exp(mb_logp_new - mb_logp_ref)  # or logp_old
    #             mb_clipped = torch.clamp(mb_ratio, 1 - self.args.clip, 1 + self.args.clip)
    #             mb_pg_loss = torch.min(-mb_advantage * mb_ratio, -mb_advantage * mb_clipped)
    #             mb_pg_loss = masked_mean(mb_pg_loss, shifted_mb_completion_mask)

    #             # value loss
    #             mb_value_pred = self.critic(mb_encodings.input_ids, attention_mask=mb_encodings.attention_mask).logits.float()[..., 0][:, :-1]
    #             mb_v_clip = mb_values + torch.clamp(mb_value_pred - mb_values, -self.args.ppo_value_clip, self.args.ppo_value_clip)
    #             mb_value_loss = 0.5 * torch.max((mb_value_pred - mb_rets)**2, (mb_v_clip - mb_rets)**2)
    #             mb_value_loss = masked_mean(mb_value_loss, shifted_mb_completion_mask)

    #             # KL loss
    #             mb_kl_loss = torch.tensor(0., device="cuda")
    #             if self.args.kl_penalty_coef != 0:
    #                 mb_kl = (mb_logp_new - mb_logp_ref) * shifted_mb_completion_mask
    #                 mb_kl_loss = mb_kl.mean()

    #             mb_loss = mb_pg_loss + self.args.vf_coef * mb_value_loss + self.args.kl_penalty_coef * mb_kl_loss
    #             mb_loss = mb_loss / self.args.gradient_accumulation_steps
    #             mb_loss.backward()

    #             if mini_g_step == self.args.gradient_accumulation_steps - 1:
    #                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)
    #                 torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.gradient_clip)
    #                 self.actor_optimizer.step()
    #                 self.critic_optimizer.step()


    #         # TODO update info dict

    #     torch.cuda.empty_cache()
    #     gc.collect()
    #     return {k: v.detach().cpu().to(torch.float32).numpy() for k, v in self.model.state_dict().items()}
    def update(self, steps: List):
        # ------------- assemble batch -------------
        obs, acts, advs = [], [], []
        for st in steps:
            obs.append(st.obs)
            acts.append(st.act)
            advs.append(st.reward)          # ← already an advantage!
        advs = torch.tensor(advs, dtype=torch.float32, device="cuda")

        full_texts = [o + a for o, a in zip(obs, acts)]

        assert len(steps) % self.args.gradient_accumulation_steps == 0
        mb_sz   = len(steps) // self.args.gradient_accumulation_steps
        n_epochs = self.args.ppo_epochs

        for _ in range(n_epochs):
            self.actor_optimizer.zero_grad(set_to_none=True)

            for g_step in range(self.args.gradient_accumulation_steps):
                s, e = g_step * mb_sz, (g_step + 1) * mb_sz
                mb_text  = full_texts[s:e]
                mb_obs   = obs[s:e]
                # mb_adv   = advs[s:e]

                enc = self.tokenizer(
                    mb_text, return_tensors="pt", padding=True,
                    add_special_tokens=True).to("cuda")

                # ----- build completion-mask (unchanged) -----
                prompt_lens = [len(self.tokenizer(o, add_special_tokens=False)["input_ids"])
                            for o in mb_obs]
                comp_mask = torch.ones_like(enc.input_ids, dtype=torch.bool)
                for i, L in enumerate(prompt_lens):
                    comp_mask[i, :L] = False
                logits_ref = self.ref_model(**enc).logits.float().detach()
                logp_ref, shifted_mask = self.get_logps(logits_ref,
                                                        enc.input_ids,
                                                        comp_mask)

                # ----- actor forward -----
                logits_new = self.model(**enc).logits.float()
                logp_new, _ = self.get_logps(logits_new,
                                            enc.input_ids,
                                            comp_mask)

                seq_len   = logits_new.size(1) - 1            # T after shift
                mb_advmat = build_adv_no_critic(
                                final_rewards = advs[s:e],    # your old scalar “advantage”
                                seq_len       = seq_len,
                                gamma         = self.args.gamma,
                                lam           = self.args.lam)

                ratio     = torch.exp(logp_new - logp_ref)   # (T,B)
                clipped   = torch.clamp(ratio,
                                        1 - self.args.clip,
                                        1 + self.args.clip)
                # pg_loss   = torch.min(-mb_adv * ratio, -mb_adv * clipped)
                # pg_loss   = masked_mean(pg_loss, shifted_mask)
                pg_loss = torch.min(-mb_advmat * ratio, -mb_advmat * clipped)
                pg_loss = masked_mean(pg_loss, shifted_mask)


                # KL penalty (optional)
                kl_loss = torch.tensor(0., device="cuda")
                if self.args.kl_penalty_coef != 0:
                    kl = (logp_new - logp_ref) * shifted_mask
                    kl_loss = kl.mean()


                print(f"pg_loss: {pg_loss}, kl_loss: {kl_loss}")

                loss = pg_loss + self.args.kl_penalty_coef * kl_loss
                loss = loss / self.args.gradient_accumulation_steps
                loss.backward()

                if g_step == self.args.gradient_accumulation_steps - 1:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                self.args.gradient_clip)
                    self.actor_optimizer.step()

        torch.cuda.empty_cache();  gc.collect()
        return {k: v.detach().cpu().float().numpy()
                for k, v in self.model.state_dict().items()}



    def update_weights(self, weights: dict):
        with torch.no_grad():
            device = self.model.device
            state_dict = self.model.state_dict()
            for k in weights:
                if k in state_dict and state_dict[k].shape == weights[k].shape:
                    tensor = torch.from_numpy(weights[k].copy()).to(device)
                    state_dict[k].copy_(tensor)