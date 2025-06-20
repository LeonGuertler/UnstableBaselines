import re
from typing import Tuple, Dict, Callable


def format_template(system: str = "", user: str = "", assistant: str = "") -> str: return f"{system}{user}{assistant}"
TEMPLATE_PARTS = {
    "default": {
        "user": lambda obs: f"You are playing a two-player zero-sum game. Make valid moves to win. You should first reason about your next move, and then submit the move enclosed by \\boxed{{}}.\nObservation: {obs}\n"
    },
    "qwen3-zs": {
        "user": lambda obs: f"<|im_start|>user\nYou are playing a two-player zero-sum game. Make valid actions to win.\nObservation: {obs}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n"
    },
    "qwen3-reasoning": {
        "user": lambda obs: f"<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{{}}.\nQuestion: {obs}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n"
    },
    "gemma3-zs": {
        "user": lambda obs: f"<bos><start_of_turn>user\nYou are playing a two-player zero-sum game. Make valid actions to win.\nObservation: {obs}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<end_of_turn>\n",
        "assistant": "<start_of_turn>model\n"
    },
    "llama-instruct-zs": {
        "system": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are playing a two-player zero-sum game. Make valid actions to win.<|eot_id|>",
        "user": lambda obs: f"<|start_header_id|>user<|end_header_id|>\n\nCurrent Observation: {obs}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|eot_id|>\n",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>"
    },
}
def apply_template(template_name: str, observation: str) -> str:
    parts = TEMPLATE_PARTS.get(template_name)
    return format_template(system=parts.get("system", ""), user=parts["user"](observation), assistant=parts.get("assistant", ""))


def extract_action_and_format_feedback(raw_action: str) -> Tuple[str, Dict[str, bool]]:
    matches = re.findall(r"\\boxed\{(.*?)\}", raw_action)
    if matches:
        last_match = matches[-1].strip()
        if last_match:  # non-empty boxed
            action = f"[{last_match}]" if "[" not in last_match else last_match
            has_think = 1
        else:  # empty boxed
            action = raw_action
            has_think = 0
    else:  # no boxed at all
        action = raw_action
        has_think = 0

    format_feedback = {"has_think": bool(has_think), "has_answer": False, "order_correct": False}
    return action, format_feedback


def iter_from_uid(uid: str) -> int:
    """ Extracts numerical iteration from UID strings like 'ckpt-123'. Defaults to 0 if no number is found """
    match = re.search(r"(\d+)$", uid)
    return int(match.group(1)) if match else 0


def extract_action(action: str, action_space=None) -> str:
    if action_space: match = action_space.search(action)
    else: match = re.search(r"\[(.*?)\]", action)
    return match.group(1).strip().lower() if match else None


OBSERVATION_FORMATTING: Dict[str, Callable[[str], str]] = {key: (lambda key=key: lambda observation: apply_template(key, observation))() for key in TEMPLATE_PARTS}
ACTION_EXTRACTION = {"default": extract_action_and_format_feedback}