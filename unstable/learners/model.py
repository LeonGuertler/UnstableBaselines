from typing import Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


def build_critic_cls(base_cls, base_pretrain_cls, value_head_prefix):
    class CriticModel(base_pretrain_cls):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_cls(config))

            self.value_head_prefix = value_head_prefix
            setattr(
                self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False)
            )

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
            **_,
        ) -> torch.Tensor:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]
            values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(
                -1
            )

            if return_output:
                return (values, outputs)
            else:
                return values

    return CriticModel


def get_critic_model(
    pretrain_or_model: str,
    device: torch.device,
    torch_dtype,
    use_flash_attention_2: bool = False,
    value_head_prefix: str = "value_head",
):
    config = AutoConfig.from_pretrained(pretrain_or_model, trust_remote_code=True)
    config._attn_implementation = (
        "flash_attention_2" if use_flash_attention_2 else "eager"
    )

    base_class = AutoModel._model_mapping[type(config)]
    critic_cls = build_critic_cls(base_class, base_class.__base__, value_head_prefix)

    model = critic_cls.from_pretrained(
        pretrain_or_model,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    value_head = getattr(model, value_head_prefix)
    value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
    return model
