from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

def build_lora_model(model, r=8, alpha=32, dropout=0.05):
    # Target common linear proj layers; adapt as needed for Qwen
    target = ["q_proj", "k_proj", "v_proj", "o_proj"]
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
