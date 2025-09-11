import re
from typing import Tuple, Dict, Callable


def get_algorithm_config(algorithm: str) -> dict:
    import yaml
    try: return yaml.safe_load(open(f"configs/{algorithm}.yaml"))
    except FileNotFoundError: raise ValueError(f"Algorithm {algorithm} not found")


def get_learner_cls(algorithm: str) -> type:
    import unstable.algorithms
    match algorithm:
        case "reinforce": return unstable.algorithms.reinforce.REINFORCELearner
        case "a2c": return unstable.algorithms.a2c.A2CLearner
        case "ppo": return unstable.algorithms.ppo.PPOLearner
        case "grpo": return unstable.algorithms.grpo.GRPOLearner
        case _: raise ValueError(f"Algorithm {algorithm} not found")

def get_model_registry_cls(model_registry_strategy: str) -> type:
    import unstable.common.model_samplers
    match model_registry_strategy:
        case "default": return unstable.common.model_samplers.ModelRegistry
        case _: raise ValueError(f"Model registry strategy {model_registry_strategy} not found")

def get_model_sampler_cls(model_sampling_strategy: str) -> type:
    import unstable.common.model_samplers
    match model_sampling_strategy:
        case "default": return unstable.common.model_samplers.BaseModelSampler
        case "mirror": return unstable.common.model_samplers.BaseModelSampler
        case "fixed": return unstable.common.model_samplers.FixedOpponentModelSampler
        case _: raise ValueError(f"Model sampling strategy {model_sampling_strategy} not found")

def get_action_sampler_cls(action_sampling_strategy: str) -> type:
    import unstable.common.action_samplers
    match action_sampling_strategy:
        case "default": return unstable.common.action_samplers.BaseActionSampler
        case "majority_voting": return unstable.common.action_samplers.MajorityVotingActionSampler
        case _: raise ValueError(f"Action sampling strategy {action_sampling_strategy} not found")

def get_env_sampler_cls(env_sampling_strategy: str) -> type:
    import unstable.common.env_samplers
    match env_sampling_strategy:
        case "random": return unstable.common.env_samplers.UniformRandomEnvSampler
        case _: raise ValueError(f"Env sampling strategy {env_sampling_strategy} not found")

def get_replay_buffer_cls(replay_buffer_strategy: str) -> type:
    import unstable.common.buffers
    match replay_buffer_strategy:
        case "step_buffer": return unstable.common.buffers.StepBuffer
        case "episode_buffer": return unstable.common.buffers.EpisodeBuffer
        case _: raise ValueError(f"Replay buffer strategy {replay_buffer_strategy} not found")


def get_reward_transformation_cls(reward_transformation: str) -> type:
    import unstable.common.reward_transformations
    match reward_transformation:
        case "role_advantage": return unstable.common.reward_transformations.RoleAdvantageByEnvFormatter
        case "format_reward": return unstable.common.reward_transformations.RewardForFormat
        case "invalid_move_penalty": return unstable.common.reward_transformations.PenaltyForInvalidMove
        case "normalize_by_env": return unstable.common.reward_transformations.NormalizeRewardsByEnv
        case "group_relative_advantage": return unstable.common.reward_transformations.GroupRelativeAdvantage
        case _: raise ValueError(f"Reward transformation {reward_transformation} not found")

def format_template(system: str = "", user: str = "", assistant: str = "") -> str: return f"{system}{user}{assistant}"
TEMPLATE_PARTS = {
    "default": {
        "user": lambda obs: f"You are playing a two-player zero-sum game. Make valid moves to win. You should first reason about your next move, and then submit the move enclosed by \\boxed{{}}.\nObservation: {obs}\n"
    },
    "qwen3-zs": {
        "user": lambda obs: f"<|im_start|>user\nYou are playing a two-player zero-sum game. Make valid actions to win.\nObservation: {obs}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n"
    },
    "qwen3-sp": {
        "user": lambda obs:  f"<|im_start|>user\nYou are playing a single-player game. Make valid actions to solve it completely.\nObservation: {obs}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n"
    },
    "qwen3-reasoning": {
        "user": lambda obs: f"<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{{}}.\nQuestion: {obs}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n<think>"
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

    format_feedback = {"correct_answer_format": bool(has_think)}
    return action, format_feedback

OBSERVATION_FORMATTING: Dict[str, Callable[[str], str]] = {key: (lambda key=key: lambda observation: apply_template(key, observation))() for key in TEMPLATE_PARTS}
ACTION_EXTRACTION = {"default": extract_action_and_format_feedback}
DEFAULT_LORA_CFG = {"lora_rank": 32, "lora_alpha": 32, "lora_dropout": 0.0, "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj", "up_proj","down_proj"]}
