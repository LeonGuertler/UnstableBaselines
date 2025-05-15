import re
from typing import Tuple, Dict



def apply_default_template(observation: str) -> str:
    return (
        "You are playing a two-player zero-sum game. Make valid moves to win. "
        "You should first reason about your next move, and then submit the move enclosed by \\boxed{}."
        f"Observation: {observation}\n"
    )


def extract_action_and_format_feedback(raw_action: str) -> Tuple[str, Dict[str, bool]]:
    # Find the first instance of \boxed{...} and extract its contents
    match = re.search(r"\\boxed\{(.*?)\}", raw_action)
    if match:
        action = match.group(1)
        has_think = 1
    else:
        action = raw_action
        has_think = 0
    format_feedback = {"has_think": has_think, "has_answer": False, "order_correct": False}
    return action, format_feedback



OBSERVATION_FORMATTING = {
    "default": apply_default_template
}

ACTION_EXTRACTION = {
    "default": extract_action_and_format_feedback
}
