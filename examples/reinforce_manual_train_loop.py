import os
os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time, ray

import unstable.common.reward_transformations as retra
from unstable.common.env_samplers import UniformRandomEnvSampler
from unstable.common._types import TrainEnvSpec, EvalEnvSpec
from unstable.common.model_samplers import BaseModelSampler, ModelRegistry
from unstable.common.game_scheduler import GameScheduler
from unstable.common.buffers import StepBuffer
from unstable.common.trackers import Tracker
from unstable.algorithms import REINFORCELearner


# CONFIG
CONFIG = {
    "collection_workers": 20,
    "evaluation_workers": 2,
    "learner": {
        "num_training_steps": 200,
        "model_name": "Qwen/Qwen3-1.7B-Base",
        "batch_size": 20,
        "mini_batch_size": 1,
        "learning_rate": 1e-5,
        "grad_clip": 0.2,
        "max_train_len": 3000,
        "max_generation_len": 4096,
        "lora_cfg": {
            "lora_rank": 32, "lora_alpha": 32, "lora_dropout": 0.0,
            "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj", "up_proj","down_proj"]
        },
    },
    "replay_buffer": {
        "size": 20*2,
    },
    "vllm_config": {
        "model_name": "Qwen/Qwen3-1.7B-Base",
        "temperature": 0.6, 
        "max_tokens": 4090,
        "max_parallel_seq": 128, "max_loras": 8,
        "max_model_len": 8192,
        "lora_config": {
            "lora_rank": 32, "lora_alpha": 32, "lora_dropout": 0.0,
            "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj", "up_proj","down_proj"]
        },
    }
}


# INITIALIZATION
ray.init(namespace="unstable")  

# ENVIRONMENT SAMPLER
env_sampler = UniformRandomEnvSampler(
    train_env_specs=[
        TrainEnvSpec(env_id="SimpleTak-v0-train", num_players=2, num_actors=2, prompt_template="qwen3-zs"), # if num_players == num_actors, it's mirror self-play and no opponents will be sampled
    ],
    eval_env_specs=[
        EvalEnvSpec(env_id="SimpleTak-v0-train", num_players=2, prompt_template="qwen3-zs"),
        EvalEnvSpec(env_id="KuhnPoker-v0-train", num_players=2, prompt_template="qwen3-zs"),
])
tracker = Tracker.options(name="Tracker").remote(
    run_name=f"Test-{CONFIG['learner']['model_name'].split('/')[-1]}-{env_sampler.env_list()}-{int(time.time())}", 
    wandb_project="UnstableBaselines"
) 

# MODEL SAMPLER
model_registry = ModelRegistry.options(name="ModelRegistry").remote(tracker=tracker)
ray.get(model_registry.add_checkpoint.remote(uid="base", path=None, iteration=0))
ray.get(model_registry.add_fixed.remote(name="google/gemini-2.0-flash-lite-001"))
model_sampler = BaseModelSampler(model_registry=model_registry) 

# REPLAY BUFFER
game_scheduler = GameScheduler.options(name="GameScheduler").remote(model_sampler=model_sampler, env_sampler=env_sampler, logging_dir=ray.get(tracker.get_log_dir.remote()), eval_action_sampler="majority_voting")
step_buffer = StepBuffer.options(name="Buffer").remote(
    max_buffer_size=CONFIG['replay_buffer']['size'], tracker=tracker, vllm_config=CONFIG['vllm_config'], game_scheduler=game_scheduler,
    final_reward_transformation=retra.ComposeFinalRewardTransforms([retra.RoleAdvantageByEnvFormatter()]),
    step_reward_transformation=retra.ComposeStepRewardTransforms([retra.RewardForFormat(1.5), retra.PenaltyForInvalidMove(1.0, -1.0)]),
    sampling_reward_transformation=retra.ComposeSamplingRewardTransforms([retra.NormalizeRewardsByEnv(True)]),
)

# LEARNING ALGORITHM
learner = REINFORCELearner.options(num_gpus=1, name="Learner").remote(
    **CONFIG['learner'],
    buffer=step_buffer,
    tracker=tracker,
    model_registry=model_registry
)


# RUN TRAINING
try:
    step_buffer.collect.remote(num_train_workers=CONFIG['collection_workers'], num_eval_workers=CONFIG['evaluation_workers'])
    ray.get(learner.train.remote(CONFIG['learner']['num_training_steps']))
finally:
    ray.kill(step_buffer, no_restart=True)
    ray.shutdown()
