import math, gc, os
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field

import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer



class REINFORCELearner:
    # def __init__(self, args):
    #     self.args = args 
    #     torch.cuda.set_device(0)
    #     self.model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    #     self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    #     self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
    #     self.update_step = 0

    #     # # gradient checkpointing
    #     # if self.args.gradient_checkpointing:
    #     #     self.model.gradient_checkpointing_enable()

    #     # # bfloat16 training
    #     # if self.args.bf16_training:
    #     #     self.model.to(torch.bfloat16)
    #     if args.gradient_checkpointing:
    #         self.model.module.gradient_checkpointing_enable() if isinstance(self.model, torch.nn.DataParallel) else self.model.gradient_checkpointing_enable()

    #     if args.bf16_training:
    #         self.model.module.to(torch.bfloat16) if isinstance(self.model, torch.nn.DataParallel) else self.model.to(torch.bfloat16)
    def __init__(self, args):
        self.args = args

        # At this point, CUDA_VISIBLE_DEVICES is already set by RayLearner
        visible_gpus = torch.cuda.device_count()
        print(f"VISIBLE GPUS: {visible_gpus}")

        # 1. Load model onto root device
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to("cuda:0")
        print("MODEL LOADED")

        # 2. Wrap if we have more than 1 visible GPU
        # if visible_gpus > 1:
        self.model = torch.nn.DataParallel(self.model)
        print("MODEL PUSHED TO DATA PARALLEL")
        # 3. Tokenizer and optimizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        self.update_step = 0

        # 4. Enable checkpointing and bfloat16 (respecting DataParallel)
        model_ref = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        if args.gradient_checkpointing:
            model_ref.gradient_checkpointing_enable()
        if args.bf16_training:
            model_ref.to(torch.bfloat16)


     def update(self, batch):
        enc, advs, obs = batch # unpack
        out  = self.model(**enc) # forward
        logp = F.log_softmax(out.logits, dim=-1)

        tgt_ids  = enc.input_ids[:, 1:]
        tok_logp = logp[:, :-1, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)

        # build prompt mask
        mask = torch.ones_like(enc.input_ids, dtype=torch.bool, device=self.dev)
        for i, o in enumerate(obs):
            L = len(self.tok(o, add_special_tokens=False)["input_ids"])
            mask[i, :L] = False
        mask = mask[:, 1:]

        seq_logp = (tok_logp * mask).sum(1) / mask.sum(1).clamp(min=1)
        loss = -(advs * seq_logp).mean()
        loss.backward()
        return loss.item()

    def update(self, steps: List):
        obs, acts, advs = [], [], []
        for st in steps:
            obs.append(st.obs)
            acts.append(st.act)
            advs.append(st.reward)
        advs = torch.tensor(advs, dtype=torch.float32, device="cuda")
        full_texts = [o + a for o, a in zip(obs, acts)]

        # TODO sort sequences by str length for more efficient padding
        assert len(steps) % self.args.gradient_accumulation_steps == 0
        mb_sz = len(steps) // self.args.gradient_accumulation_steps

        self.optimizer.zero_grad(set_to_none=True)
        for g_step in range(self.args.gradient_accumulation_steps):
            s, e = g_step * mb_sz, (g_step + 1) * mb_sz
            mb_text = full_texts[s:e]
            mb_obs = obs[s:e]
            mb_advs = advs[s:e]

            enc = self.tokenizer(mb_text, return_tensors="pt", padding=True, add_special_tokens=True) #.to("cuda")
            prompt_lens = [len(self.tokenizer(o, add_special_tokens=False)["input_ids"]) for o in mb_obs]
            comp_mask = torch.ones_like(enc.input_ids, dtype=torch.bool)
            for i, L in enumerate(prompt_lens):
                comp_mask[i, :L] = False

            # calcualte path probability
            out = self.model(**enc)
            logp = F.log_softmax(out.logits, dim=-1)

            dev = logp.device

            tgt_ids = enc.input_ids[:, 1:].to(dev)
            logp = logp[:, :-1, :]
            tok_logp = logp.gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)

            # mask out prompt 
            comp_mask = comp_mask.to(dev)
            comp_mask_shifted = comp_mask[:, 1:]
            comp_lengths = comp_mask_shifted.sum(dim=1).clamp(min=1)
            seq_logp = (tok_logp * comp_mask_shifted).sum(dim=1) / comp_lengths

            mb_advs = mb_advs.to(dev)

            loss = -(mb_advs * seq_logp).mean() # get reinforce loss
            loss = loss / self.args.gradient_accumulation_steps # scale for gradient accumulation
            loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)
        self.optimizer.step()

        torch.cuda.empty_cache(); gc.collect()
        self.update_step += 1
        if self.update_step % self.args.save_every_n_update_steps and self.args.save_strategy=="steps":
            self._store_model()

        # unwrap before saving weights for the collector
        state_dict = self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict()
        return {k: v.detach().cpu().float().numpy() for k, v in state_dict.items()}
        # return {k: v.detach().cpu().float().numpy() for k, v in self.model.state_dict().items()}

    def store_model(self, checkpoint_folder: Optional[str], checkpoint_filename: Optional[str]):
        # if not provided, use defaults
        if checkpoint_folder is None:
            checkpoint_folder = self.args.output_dir_checkpoints
        if checkpoint_filename is None:
            checkpoint_filename = f"Update_Step_{self.update_step}"

        save_path = os.path.join(checkpoint_folder, checkpoint_filename)
        checkpoint = {"model_state_dict": self.model.state_dict(), "optimizer_state_dict": self.optimizer.state_dict(), "args": vars(self.args)}
        torch.save(checkpoint, save_path)
        print(f"[REINFORCE] Checkpoint saved to {save_path}")

    def load_model(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"[REINFORCE] Checkpoint loaded from {checkpoint_path}")


    def update_weights(self, weights: dict):
        with torch.no_grad():
            device = self.model.device
            state_dict = self.model.state_dict()
            for k in weights:
                if k in state_dict and state_dict[k].shape == weights[k].shape:
                    tensor = torch.from_numpy(weights[k].copy()).to(device)
                    state_dict[k].copy_(tensor)
