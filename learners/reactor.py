import gc, math, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict

# ---------- helpers ---------------------------------------------------------
def masked_mean(x, mask, eps=1e-8):
    return (x * mask).sum() / (mask.sum() + eps)

def align_for_causal(logits, ids, mask):
    return logits[:, :-1], ids[:, 1:], mask[:, 1:]

def token_logp(logits, ids, mask):
    logits, ids, mask = align_for_causal(logits, ids, mask)
    lp = F.log_softmax(logits, -1)
    return torch.gather(lp, 2, ids.unsqueeze(-1)).squeeze(-1), mask

def huber(x, delta):
    if delta == float("inf"):   # disabled
        return x
    return torch.where(x.abs() <= delta, x, delta * x.sign())

# ---------------------------------------------------------------------------
class REACTORLearner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda")
        torch.cuda.set_device(0)

        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name, trust_remote_code=True,
            torch_dtype=torch.bfloat16).to(self.device)

        self.ref_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, trust_remote_code=True,
            torch_dtype=torch.bfloat16).to(self.device).eval()
        for p in self.ref_model.parameters(): p.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, trust_remote_code=True)

        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if args.bf16_training:
            self.model.to(torch.bfloat16)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

        # per-role running baseline
        self.role_baseline: Dict[int, float] = {0: 0.0, 1: 0.0}

    # -----------------------------------------------------------------------
    def update(self, steps: List):
        B = len(steps)
        assert B % self.args.gradient_accumulation_steps == 0

        # -------- pack batch ------------------------------------------------
        obs   = [s.obs for s in steps]
        acts  = [s.act for s in steps]
        roles = torch.tensor([s.pid        for s in steps], device=self.device)
        rs    = torch.tensor([s.reward     for s in steps], device=self.device)
        sds   = torch.tensor([s.reward_sd  for s in steps], device=self.device)

        # -------- baseline & advantage -------------------------------------
        base = torch.tensor([self.role_baseline[int(p)] for p in roles.cpu()],
                            device=self.device)
        adv  = huber(rs - base, self.args.huber_delta)           # [B]

        # update EMA baselines
        for r, p in zip(rs, roles):
            rb = self.role_baseline[int(p)]
            self.role_baseline[int(p)] = (1 - self.args.baseline_tau) * rb \
                                         + self.args.baseline_tau * float(r)

        # -------- SD weights -----------------------------------------------
        w = ((sds - sds.min()) /
             (sds.max() - sds.min() + 1e-8)).pow(self.args.sd_power).sqrt()
        w = w[:, None]                                           # [B,1]

        full_txt = [o + a for o, a in zip(obs, acts)]
        enc = self.tokenizer(full_txt, return_tensors="pt",
                             padding=True).to(self.device)

        # completion mask
        plens = [len(self.tokenizer(o, add_special_tokens=False)["input_ids"])
                 for o in obs]
        mask  = enc.attention_mask.bool()
        comp_mask = mask.clone()
        for i, L in enumerate(plens): comp_mask[i, :L] = False

        # ------------- forward passes --------------------------------------
        logits_new = self.model(**enc).logits
        with torch.no_grad():
            logits_ref = self.ref_model(**enc).logits

        # token log-probs
        lp_new, comp_mask = token_logp(logits_new, enc.input_ids, comp_mask)
        lp_ref, _         = token_logp(logits_ref, enc.input_ids, comp_mask)

        # entropy
        probs_new = F.softmax(logits_new, -1)
        entropy_tok = -(probs_new * F.log_softmax(logits_new, -1)).sum(-1)

        # JS divergence
        probs_ref = F.softmax(logits_ref, -1)
        m = 0.5 * (probs_new + probs_ref)
        js_tok = 0.5 * (
            (probs_new * (F.log_softmax(logits_new, -1) -
                          torch.log(m))).sum(-1) +
            (probs_ref * (F.log_softmax(logits_ref, -1) -
                          torch.log(m))).sum(-1)
        )

        # expand advantage to final token
        adv_tok = torch.zeros_like(lp_new)
        adv_tok[:, -1] = adv

        # ------------- losses ----------------------------------------------
        wmask   = w * comp_mask
        pg_loss = -masked_mean(wmask * adv_tok * lp_new, comp_mask)
        js_loss =  masked_mean(wmask * js_tok, comp_mask)
        ent_bonus = -masked_mean(wmask * entropy_tok, comp_mask)

        loss = pg_loss + self.args.beta_js * js_loss \
               + self.args.ent_coef * ent_bonus
        loss = loss / self.args.gradient_accumulation_steps
        loss.backward()

        # ------------- optimise -------------------------------------------
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       self.args.gradient_clip)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        torch.cuda.empty_cache(); gc.collect()
        return loss.item()

    # -----------------------------------------------------------------------
    def update_weights(self, weights: dict):
        with torch.no_grad():
            sd = self.model.state_dict()
            for k, v in weights.items():
                if k in sd and sd[k].shape == v.shape:
                    sd[k].copy_(torch.from_numpy(v).to(self.device))
