import torch, pathlib
from typing import Dict, Any, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel
from peft.tuners.lora import LoraLayer
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
try:                from torch.utils.checkpoint import CheckpointImpl
except ImportError: from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, apply_activation_checkpointing

def compute_gae(rewards, values, gamma=1.0, gae_lambda=1.0):
    assert len(rewards) == len(values)
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            nextnonterminal = 0  # Assume a complete episode
            nextvalues = 0  # Does not matter
        else:
            nextnonterminal = 1.0
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    return advantages

def build_critic_cls(base_cls, base_pretrain_cls, value_head_prefix):
    class CriticModel(base_pretrain_cls):
        supports_gradient_checkpointing = True
        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_cls(config))
            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, torch.nn.Linear(config.hidden_size, 1, bias=False))

        def forward(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.Tensor]=None, return_output=False, **_) -> torch.Tensor:
            position_ids = attention_mask.long().cumsum(-1) - 1; position_ids.masked_fill_(attention_mask == 0, 1)
            outputs = getattr(self, self.base_model_prefix)(input_ids, attention_mask=attention_mask, position_ids=position_ids)
            last_hidden_states = outputs["last_hidden_state"]; values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)
            if return_output:   return (values, outputs)
            else:               return values
    return CriticModel

def get_critic_model(pretrain_or_model: str, device: torch.device, torch_dtype, use_flash_attention_2: bool=False, value_head_prefix: str="value_head"):
    config = AutoConfig.from_pretrained(pretrain_or_model, trust_remote_code=True)
    config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"
    base_class = AutoModel._model_mapping[type(config)]
    critic_cls = build_critic_cls(base_class, base_class.__base__, value_head_prefix)
    model = critic_cls.from_pretrained(pretrain_or_model, config=config, trust_remote_code=True, torch_dtype=torch_dtype, device_map=device)
    value_head = getattr(model, value_head_prefix)
    value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
    return model

def _load_base(name: str, dtype, device, **kwargs): 
    with torch.device(device): 
        return AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, trust_remote_code=True, **kwargs)

def _freeze(model, ignore_substr: Optional[str] = None):
    for n, p in model.named_parameters():
        if ignore_substr and ignore_substr in n: continue
        p.requires_grad_(False)

def _build_lora(model, lora_cfg: Dict[str, Any], task_type: str):
    return get_peft_model(model, LoraConfig(
        r=lora_cfg.get("lora_rank", 32), lora_alpha=lora_cfg.get("lora_alpha", 32), lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias="none", task_type=task_type, target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    ))

def build_peft_model(base_name: str, device: torch.device, lora_cfg: Dict[str, Any]|None, initial_lora_path: Optional[str]=None, freeze_base: bool=True, critic_model: bool=False, value_head_prefix: str="value_head") -> Tuple[torch.nn.Module, "transformers.PreTrainedTokenizer"]:
    task_type = "TOKEN_CLS" if critic_model else "CAUSAL_LM"
    base = get_critic_model(base_name, device, torch_dtype=torch.bfloat16, value_head_prefix=value_head_prefix) if critic_model else _load_base(base_name, torch.bfloat16, device)
    if freeze_base: _freeze(base, None if not critic_model else value_head_prefix)
    model = _build_lora(base, lora_cfg or {}, task_type).to(device)
    if initial_lora_path: _load_lora_state(model, initial_lora_path)
    tok = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    return model, tok

def _json_safe(obj):
    if isinstance(obj, set): return list(obj) # turn sets into lists
    raise TypeError # let json handle the rest
    
def enable_full_activation_ckpt(model):
    def checkpoint_everything(mod):
        if isinstance(mod, LoraLayer): return False
        for _, child in mod.named_modules():
            if isinstance(child, LoraLayer): return False
        return True
    apply_activation_checkpointing(model, checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(m, checkpoint_impl=CheckpointImpl.NO_REENTRANT), check_fn=checkpoint_everything)  # "always recompute"