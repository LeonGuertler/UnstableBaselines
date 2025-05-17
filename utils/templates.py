import re
from typing import Tuple, Dict



def apply_default_template(observation: str) -> str:
    return (
        "You are playing a two-player zero-sum game. Make valid moves to win. "
        "You should first reason about your next move, and then submit the move enclosed by \\boxed{}."
        f"Observation: {observation}\n"
    )

def apply_qwen3_template(observation: str) -> str:
    return (
        f"<|im_start|>user\nYou are playing a two-player zero-sum game. Make valid actions to win.\nObservation: {observation}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def extract_action_and_format_feedback(raw_action: str) -> Tuple[str, Dict[str, bool]]:
    # Find the first instance of \boxed{...} and extract its contents
    match = re.search(r"\\boxed\{(.*?)\}", raw_action)
    if match:
        action = match.group(1)
    else:
        action = ""
    format_feedback = {"has_think": False, "has_answer": False, "order_correct": False}
    return action, format_feedback



OBSERVATION_FORMATTING = {
    "default": apply_default_template,
    "qwen3": apply_qwen3_template
}

ACTION_EXTRACTION = {
    "default": extract_action_and_format_feedback
}



# other prompt/output utils
def truncate_after_boxed(raw_text: str) -> str:
    match = re.search(r"\\boxed\{.*?\}", raw_text) # Match \boxed{...} including the prefix
    if match:
        return raw_text[:match.end()]
    else:
        return raw_text
