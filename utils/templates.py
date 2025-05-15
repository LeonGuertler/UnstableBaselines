import re
from typing import Tuple, Dict



# def apply_default_template(observation: str) -> str:
#     return (
#         f"A conversation between User and Assistant. You are playing a two-player zero-sum game. Make valid moves to win. "
#         f"Make sure you enclose the final action you are submitting in squared brackets. "
#         f"The Assistant first thinks about the reasoning process in the mind and then provides the move. "
#         f"The reasoning process is enclosed within <think> </think>. Everything outside the think tags will be submitted to the environment.\n"
#         f"User: {observation}\nAssistant: <think>"
#     )
# def apply_default_template(observation: str) -> str:
#     return (
#         f"You are playing a two-player zero-sum game. Make valid moves to win. "
#         f"Make sure you enclose the final action you are submitting in squared brackets. "
#         f"You should first reason about your next move. "
#         f"The reasoning process is enclosed within <think> </think>. Everything outside the think tags will be submitted to the environment.\n"
#         f"Observation: {observation}\nAction: <think>"
#     )
# def extract_action_and_format_feedback(raw_action: str) -> Tuple[str, Dict[str, bool]]:
#     action = ""; format_reward = 0
#     if "</think>" in raw_action:
#         action = raw_action.split("</think>")[-1].strip()
#         # only award the format reward if reasoning makes up 80% of the tootal reply length
#         if len(action) <= 0.2*len(raw_action) and len(raw_action)>1000:
#             format_reward = 1

#     # else:
#     #     action = raw_action

#     format_feedback = {"has_think": format_reward, "has_answer": False, "order_correct": False}
#     return action, format_feedback

def apply_default_template(observation: str) -> str:
    return (
        "You are playing a two-player zero-sum game. Make valid moves to win. "
        "You should first reason about your next move, and then submit the move enclosed by \\boxed{}."
        f"Observation: {observation}\n"
    )
# def extract_action_and_format_feedback(raw_action: str) -> Tuple[str, Dict[str, bool]]:
#     # find first instance of \boxed{} and return the contens (if boxed, set has_think to 1)
#     action = ""; format_reward = 0
#     if "<action>" in raw_action:
#         action = raw_action.split("<action>")[-1]
#         if "</action>" in action:
#             action = action.split("</action>")[0]
#         format_reward = 1
#     else:
#         action = raw_action

#     format_feedback = {"has_think": format_reward, "has_answer": False, "order_correct": False}
#     return action, format_feedback

def extract_action_and_format_feedback(raw_action: str) -> Tuple[str, Dict[str, bool]]:
    # Find the first instance of \boxed{...} and extract its contents
    match = re.search(r"\\boxed\{(.*?)\}", raw_action)
    if match:
        action = match.group(1)
        has_think = 1
    else:
        action = raw_action
        has_think = 0

    format_feedback = {
        "has_think": has_think,
        "has_answer": False,
        "order_correct": False
    }

    return action, format_feedback

    

OBSERVATION_FORMATTING = {
    "default": apply_default_template
}

ACTION_EXTRACTION = {
    "default": extract_action_and_format_feedback
}
