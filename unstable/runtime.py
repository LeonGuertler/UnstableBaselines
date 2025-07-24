import ray, time
from typing import List, Sequence, Optional

import unstable
import unstable.reward_transformations as retra


_DEFAULT_LORA_CFG = {"lora_rank": 32, "lora_alpha": 32, "lora_dropout": 0.0, "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj", "up_proj","down_proj"]}
_ENV_SAMPLERS = {"random": unstable.samplers.env_samplers.UniformRandomEnvSampler}
_OPP_SAMPLERS = {"none": unstable.samplers.model_samplers.BaseModelSampler, "mirror": unstable.samplers.model_samplers.BaseModelSampler, "fixed": unstable.samplers.model_samplers.FixedOpponentModelSampler}
_STEP_BUFFER_ALGOS = ["reinforce"]
_EPISODE_BUFFER_ALGOS = ["a2c"]
_ALGOS = {"reinforce": unstable.REINFORCELearner, "a2c": unstable.A2CLearner}
def _default_vllm_cfg(model_name: str, lora_cfg: dict, max_generation_len: int) -> dict: return {"model_name": model_name, "temperature": 0.6, "max_tokens": max_generation_len, "max_parallel_seq": 128, "max_loras": 8, "lora_config": lora_cfg, "max_model_len": 8192}

class _UBRun:
    def __init__(self, *, collector, learner): self.collector, self.learner = collector, learner
    def start(self, learning_steps: int = 200, num_collection_workers: int = 256, num_eval_workers: int = 16):
        try:
            self.collector.collect.remote(num_train_workers=num_collection_workers, num_eval_workers=num_eval_workers)
            ray.get(self.learner.train.remote(learning_steps))
        finally:
            ray.kill(collector, no_restart=True); ray.shutdown()

class _UBEval:
    def __init__(self, *, evaluator): self.evaluator = evaluator
    def evaluate(self, num_eval_workers: int=16):
        try: ray.get(self.evaluator.evaluate.remote(num_eval_workers=num_eval_workers))
        # try: self.evaluator.evaluate.remote(num_eval_workers=num_eval_workers)
        finally: ray.kill(self.evaluator, no_restart=True); ray.shutdown()

def build(*, model_name: str, train_envs: Sequence[unstable.TrainEnvSpec], eval_envs: Optional[Sequence[unstable.EvalEnvSpec]]=None, env_sampling_strategy: str = "random", opponent_sampling_strategy: str = "none", fixed_opponents: Sequence[str] = ["google/gemini-2.0-flash-lite-001"], algorithm: str = "reinforce", max_train_len: Optional[int]=None, max_generation_len: int=4096, batch_size: int=384, mini_batch_size: int=1, learning_rate: float=1e-5, gradient_clipping: float=0.2, activation_checkpointing: bool=True, gradient_checkpointing: bool=True, use_trainer_cache: bool = False, buffer_size: Optional[int]=None, lora_config: Optional[dict]=None, vllm_config: Optional[dict]=None, wandb_project: str="UnstableBaselines"):
    ray.init(namespace="unstable") # Ray init
    
    # env sampler
    assert env_sampling_strategy in _ENV_SAMPLERS, f"env_sampling_strategy='{env_sampling_strategy}' not found. Please use one of: {list(_ENV_SAMPLERS.keys())}"
    env_sampler = _ENV_SAMPLERS[env_sampling_strategy](train_env_specs=train_envs, eval_env_specs=eval_envs)

    # tracker
    tracker = unstable.Tracker.options(name="Tracker").remote(run_name=f"UnstableBaselines-{model_name.split('/')[-1]}-{env_sampler.env_list()}-{int(time.time())}", wandb_project="UnstableBaselines") 

    # initialize model registry
    model_registry = unstable.ModelRegistry.options(name="ModelRegistry").remote(tracker=tracker)
    ray.get(model_registry.add_checkpoint.remote(uid="base", path=None, iteration=0))
    for f_opp_name in fixed_opponents: ray.get(model_registry.add_fixed.remote(name=f_opp_name))

    # initialize opponent sampler
    assert opponent_sampling_strategy in _OPP_SAMPLERS, f"opponent_sampling_strategy='{opponent_sampling_strategy}' not found. Please use one of: {list(_OPP_SAMPLERS.keys())}"
    model_sampler = _OPP_SAMPLERS[opponent_sampling_strategy](model_registry=model_registry)

    # build game scheduler
    game_scheduler = unstable.GameScheduler.options(name="GameScheduler").remote(model_sampler=model_sampler, env_sampler=env_sampler, logging_dir=ray.get(tracker.get_log_dir.remote()))

    # build buffer TODO maybe move the reward transformations outside
    buffer_size = buffer_size or batch_size*2
    if algorithm in _STEP_BUFFER_ALGOS: buffer = unstable.StepBuffer.options(name="Buffer").remote(max_buffer_size=buffer_size, tracker=tracker, final_reward_transformation=retra.ComposeFinalRewardTransforms([retra.RoleAdvantageByEnvFormatter()]), step_reward_transformation=retra.ComposeStepRewardTransforms([retra.RewardForFormat(1.5), retra.PenaltyForInvalidMove(1.0, -1.0)]), sampling_reward_transformation=retra.ComposeSamplingRewardTransforms([retra.NormalizeRewardsByEnv(True)]))
    elif algorithm in _EPISODE_BUFFER_ALGOS: buffer = unstable.EpisodeBuffer.options(name="Buffer").remote(max_buffer_size=buffer_size, tracker=tracker, final_reward_transformation=retra.ComposeFinalRewardTransforms([retra.RoleAdvantageByEnvFormatter()]), step_reward_transformation=retra.ComposeStepRewardTransforms([retra.RewardForFormat(1.5), retra.PenaltyForInvalidMove(1.0, -1.0)]), sampling_reward_transformation=retra.ComposeSamplingRewardTransforms([retra.NormalizeRewardsByEnv(True)]))
    else: raise NotImplementedError(f"The algorithm used ({algorithm}) has not been allocated to a specific buffer type.")

    # initialize the collector
    _lora_cfg = lora_config or _DEFAULT_LORA_CFG
    collector = unstable.Collector.options(name="Collector").remote(vllm_config=vllm_config or _default_vllm_cfg(model_name, _lora_cfg, max_generation_len), tracker=tracker, buffer=buffer, game_scheduler=game_scheduler)

    # initialize the learner
    assert algorithm in _ALGOS, f"algorithm='{algorithm}' not found. Please use one of: {list(_ALGOS.keys())}"
    learner = _ALGOS[algorithm].options(num_gpus=1, name="Learner").remote(model_name=model_name, lora_cfg=_lora_cfg, batch_size=batch_size, mini_batch_size=mini_batch_size, learning_rate=learning_rate, grad_clip=gradient_clipping, buffer=buffer, tracker=tracker, model_registry=model_registry, activation_checkpointing=activation_checkpointing, gradient_checkpointing=gradient_checkpointing, use_trainer_cache=use_trainer_cache)
    match algorithm:
        case "reinforce":   ray.get(learner.initialize_algorithm.remote(max_train_len=max_train_len, max_generation_len=max_generation_len))
        case "a2c":         ray.get(learner.initialize_algorithm.remote(infer_mini_batch_size=16, critic_learning_rate=5e-5, normalize_adv=True, max_train_len=max_train_len, max_generation_len=max_generation_len)) # TODO find better solution
        case _:             ray.get(learner.initialize_algorithm.remote())

    return _UBRun(collector=collector, learner=learner)


def build_evaluator(*, model_name: str, eval_envs: Sequence[unstable.EvalEnvSpec], max_generation_len: int = 4096, num_runs_per_env: int=16, max_parallel_seq: int = 128):
    vllm_config = {"model_name": model_name, "temperature": 0.6, "max_tokens": max_generation_len, "max_parallel_seq": max_parallel_seq, "max_model_len": 8192, "max_loras": None, "lora_config": {"lora_rank": None}}
    ray.init(namespace="unstable") # Ray init
    tracker = unstable.Tracker.options(name="Tracker").remote(run_name=f"EvalRun-{model_name.split('/')[-1]}-{int(time.time())}", wandb_project="UnstableBaselines-eval")

    evaluator = unstable.Evaluator.options(name="Evaluator").remote(tracker=tracker, vllm_config=vllm_config, eval_env_specs=eval_envs, num_runs_per_env=num_runs_per_env)
    return _UBEval(evaluator=evaluator)

    # env_sampler = usntable.samplers.env_sampler.EvalEnvSampler(eval_env_specs=eval_env_specs)
    # tracker = unstable.Tracker.options(name="Tracker").remote(run_name=f"EvalRun-{model_name.split('/')[-1]}-{env_sampler.eval_env_list()}={int(time.time())}", wandb_project="UnstableBaselines-eval")
    # game_scheduler = unstable.GameScheduler.options(name="GameScheduler").remote(model_sampler=None, env_sampler=env_sampler, logging_dir=ray.get(tracker.get_log_dir.remote()))
