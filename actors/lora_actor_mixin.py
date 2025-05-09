from learners.lora_utils import load_lora_state


class LoRAHotSwapMixin:
    def update_weights(self, lora_state):
        """Called remotely by Collector."""
        load_lora_state(self.llm.model, lora_state)