import torch, pathlib
from safetensors.torch import load_file as safe_load
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

def build_lora_model(model, r=8, alpha=32, dropout=0.05):
    # Target common linear proj layers; adapt as needed for Qwen
    target = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # target = ["q_proj", "k_proj", "v_proj", "o_proj", "embed_tokens", "lm_head"]
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        bias="none", target_modules=target,
        task_type="CAUSAL_LM"
    )
    return get_peft_model(model, cfg)

def lora_state_dict(model):
    """Return *LoRA-only* weights on CPU / fp32 for Ray transport."""
    sd = get_peft_model_state_dict(model, adapter_name="default")
    return {k: v.detach().cpu() for k, v in sd.items()}

def load_lora_state(model, state):
    set_peft_model_state_dict(model, state, adapter_name="default")


def load_lora_state(peft_model, ckpt_dir: str | pathlib.Path):
    """
    ckpt_dir = directory produced by model.save_pretrained(...)

    Handles:
      • adapter_model.safetensors  (current PEFT default)
      • adapter_model.bin          (older PEFT)
      • pytorch_model.bin          (legacy fallback)
    """
    ckpt_dir = pathlib.Path(ckpt_dir)
    candidates = [
        ckpt_dir / "adapter_model.safetensors",
        ckpt_dir / "adapter_model.bin",
        ckpt_dir / "pytorch_model.bin",
    ]
    for path in candidates:
        if path.exists():
            print(f"[loader] found LoRA adapter → {path.name}")
            lora_sd = safe_load(str(path)) if path.suffix == ".safetensors" else torch.load(path, map_location="cpu")
            set_peft_model_state_dict(peft_model, lora_sd, adapter_name="default")
            return
    raise FileNotFoundError(f"No adapter_model.* found in {ckpt_dir}")