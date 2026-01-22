import os
import torch
from typing import Dict, Any, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel
from peft.tuners.lora import LoraLayer
from peft import LoraConfig, get_peft_model, PeftModel
from huggingface_hub import snapshot_download
try:                from torch.utils.checkpoint import CheckpointImpl
except ImportError: from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, apply_activation_checkpointing


def build_actor_critic_cls(base_cls, value_head_prefix):
    class SharedActorCriticModel(base_cls):
        supports_gradient_checkpointing = True
        def __init__(self, config: AutoConfig):
            super().__init__(config)
            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, torch.nn.Linear(config.hidden_size, 1, bias=False))
        def values(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.Tensor]=None, **kwargs) -> torch.Tensor:
            out = getattr(self, self.base_model_prefix)(input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True, **kwargs)
            return getattr(self, self.value_head_prefix)(out["last_hidden_state"]).squeeze(-1)
    return SharedActorCriticModel

def get_actor_critic_model(pretrain_or_model: str, device: torch.device, torch_dtype, use_flash_attention_2: bool=False, value_head_prefix: str="value_head"):
    config = AutoConfig.from_pretrained(pretrain_or_model, trust_remote_code=True)
    config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"
    base_class = AutoModelForCausalLM._model_mapping[type(config)]
    critic_cls = build_actor_critic_cls(base_class, value_head_prefix)
    model = critic_cls.from_pretrained(pretrain_or_model, config=config, trust_remote_code=True, torch_dtype=torch_dtype, device_map=device)
    value_head = getattr(model, value_head_prefix)
    value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
    return model

def _freeze(model, ignore_substr: Optional[str] = None):
    for n, p in model.named_parameters():
        if ignore_substr and ignore_substr in n: continue
        p.requires_grad_(False)

def _resolve_checkpoint_path(path: str, revision: Optional[str] = None) -> str:
    if os.path.exists(path): return path
    try:
        resolved = snapshot_download(repo_id=path, revision=revision, local_files_only=True)
        print(f"[_resolve_checkpoint_path] ✅ Found cached HF repo '{path}' at {resolved}")
        return resolved
    except Exception:
        resolved = snapshot_download(repo_id=path, revision=revision)
        print(f"[_resolve_checkpoint_path] ✅ Downloaded HF repo '{path}' to {resolved}")
        return resolved

def _load_lora_state(model, lora_path: str, revision: Optional[str] = None):
    resolved_path = _resolve_checkpoint_path(lora_path, revision)
    adapter_name = getattr(model, "actor_adapter_name", "default")
    model.load_adapter(resolved_path, adapter_name=adapter_name, is_trainable=True)
    model.set_adapter(adapter_name)
    print(f"[build_peft_model] ✅ Loaded LoRA adapter '{adapter_name}' from {resolved_path}")

def build_peft_model(base_name: str, device: torch.device, lora_cfg: Dict[str, Any]|None, checkpoint_cfg: Dict[str, Any]|None, freeze_base: bool=True, value_head: bool=False, value_head_prefix: str="value_head") -> Tuple[torch.nn.Module, "transformers.PreTrainedTokenizer"]:
    lora_cfg = lora_cfg or {}
    if value_head: base = get_actor_critic_model(base_name, device, torch_dtype=torch.bfloat16, value_head_prefix=value_head_prefix)  
    else: 
        with torch.device(device): base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
    base.config.attn_implementation = "flash_attention_2"; print(f"[build_peft_model] ✅ FlashAttention 2 enabled for {base_name}")
    if freeze_base: _freeze(base, None if not value_head else value_head_prefix)
    model = get_peft_model(base, LoraConfig(r=lora_cfg.get("lora_rank", 32), lora_alpha=lora_cfg.get("lora_alpha", 32), lora_dropout=lora_cfg.get("lora_dropout", 0.05), 
                                            bias="none", target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]))).to(device)
    model.actor_adapter_name = getattr(model, "active_adapter", "default")
    tok = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True, use_fast=True)
    tok.pad_token = tok.eos_token
    if value_head:
        model.critic_adapter_name = "critic"
        model.add_adapter(adapter_name=model.critic_adapter_name, peft_config=LoraConfig(r=lora_cfg.get("lora_rank", 32), lora_alpha=lora_cfg.get("lora_alpha", 32), lora_dropout=lora_cfg.get("lora_dropout", 0.05), 
                                                                                         bias="none", target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])))
        model.set_adapter(model.actor_adapter_name)
    if checkpoint_cfg.get('path', False): _load_lora_state(model, checkpoint_cfg['path'], checkpoint_cfg.get('revision'))
    return model, tok

def enable_full_activation_ckpt(model):
    def checkpoint_everything(mod):
        if isinstance(mod, LoraLayer): return False
        for _, child in mod.named_modules():
            if isinstance(child, LoraLayer): return False
        return True
    apply_activation_checkpointing(model, checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(m, checkpoint_impl=CheckpointImpl.NO_REENTRANT), check_fn=checkpoint_everything)  # "always recompute"
