import re
from typing import Tuple, Dict



def apply_default_template(observation: str) -> str:
    return (
        "You are playing a two-player zero-sum game. Make valid moves to win. "
        "You should first reason about your next move, and then submit the move enclosed by \\boxed{}."
        f"Observation: {observation}\n"
    )

def qwen3_template(observation: str) -> str:
    return (
        f"<|im_start|>user\nYou are playing a two-player zero-sum game. Make valid actions to win.\nObservation: {observation}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def qwen3_template_single_player(observation: str) -> str:
    return (
        f"<|im_start|>user\nYou are playing a single player game. Make valid actions to win.\nObservation: {observation}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def qwen3_template_reasoning(observation: str) -> str:
    return (
        "<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
        f"Question: {observation}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def extract_action_and_format_feedback(raw_action: str) -> Tuple[str, Dict[str, bool]]:
    matches = re.findall(r"\\boxed\{(.*?)\}", raw_action)
    
    if matches:
        action = matches[-1]  # Take the last one
        if "[" not in action:
            action = f"[{action.strip()}]"
        has_think = 1
    else:
        action = raw_action
        has_think = 0

    format_feedback = {"has_think": bool(has_think), "has_answer": False, "order_correct": False}
    return action, format_feedback
    


OBSERVATION_FORMATTING = {
    "default": apply_default_template,
    "qwen3": qwen3_template,
    "qwen3-game": qwen3_template,
    "qwen3-zs": qwen3_template,
    "qwen3-single-player": qwen3_template_single_player,
    "qwen3-reasoning": qwen3_template_reasoning,
}

ACTION_EXTRACTION = {
    "default": extract_action_and_format_feedback,
    "qwen3-zs": extract_action_and_format_feedback,
    "qwen3-single-player": extract_action_and_format_feedback,
}
