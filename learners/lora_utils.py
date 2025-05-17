import torch, pathlib
from safetensors.torch import load_file as safe_load
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict


def build_lora_model(model, args): 
    cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        use_rslora=args.use_rslora,
    )
    return get_peft_model(model, cfg)



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