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
        "\nPlease reason step by step, and output your action wrapped in \\boxed{}.\n"
        "For example, if your action is 'Lorem ipsum dolor sit amet', you should output \\boxed{'Lorem ipsum dolor sit amet'}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def xml_template(observation: str) -> str:
    return (
        f"You are playing a two-player zero-sum game. Make valid actions to win.\nObservation: {observation}"
        "\nIt is your turn to act next. Please reason step by step, and output the message you want to share with the other player wrapped in <public></public> tags.\n"
        "Example output:\n Understood! I should probably [...].\n <public>Hi! Have you ever[...]</public>\n"
    )

def qwen3_xml_template(observation: str) -> str:
    return (
        f"<|im_start|>user\nYou are playing a two-player zero-sum game. Make valid actions to win.\nObservation: {observation}"
        "\nIt is your turn to act next. Please reason step by step, and output the message you want to share with the other player wrapped in <public></public> tags.\n"
        "Example output:\n Understood! I should probably [...].\n <public>Hi! Have you ever[...]</public>\n"
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
    
def extract_action_and_format_feedback_xml(raw_action: str) -> Tuple[str, Dict[str, bool]]:
    matches = re.findall(r"<public>(.*?)</public>", raw_action, re.DOTALL)
    
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
    "xml": xml_template,
    "qwen3_xml": qwen3_xml_template
}

ACTION_EXTRACTION = {
    "default": extract_action_and_format_feedback,
    "xml": extract_action_and_format_feedback_xml
}
