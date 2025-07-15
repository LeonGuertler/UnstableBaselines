import time, ray, unstable
import unstable.reward_transformations as retra

COLLECTION_WORKERS = 384
EVALUATION_WORKERS = 16
ITERATIONS = 200
MODEL_NAME = "Qwen/Qwen3-1.7B-Base"
BATCH_SIZE = 384
MINI_BATCH_SIZE = 1
BUFFER_SIZE = 384*2
LR = 1e-5
GRAD_CLIP = 0.2
MAX_TRAIN_SEQ_LEN = None
MAX_GENERATION_LENGTH = 4096 

lora_config = {
    "lora_rank": 32, "lora_alpha": 32, "lora_dropout": 0.0,
    "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj", "up_proj","down_proj"]
}
vllm_config = {
    "model_name": MODEL_NAME, "temperature": 0.6, "max_tokens": MAX_GENERATION_LENGTH,
    "max_parallel_seq": 128, "max_loras": 8, "lora_config": lora_config,
    "max_model_len": 8192
}

# Ray init
ray.init(namespace="unstable")  

# initialize environment scheduler
env_sampler = unstable.samplers.env_samplers.UniformRandomEnvSampler(
    train_env_specs=[
        unstable.TrainEnvSpec(env_id="SimpleTak-v0-train", num_players=2, num_actors=2, prompt_template="qwen3-zs"), # if num_players == num_actors, it's mirror self-play and no opponents will be sampled
    ],
    eval_env_specs=[
        unstable.EvalEnvSpec(env_id="SimpleTak-v0-train", num_players=2, prompt_template="qwen3-zs"),
        unstable.EvalEnvSpec(env_id="KuhnPoker-v0-train", num_players=2, prompt_template="qwen3-zs"),
])

# Tracker
tracker = unstable.Tracker.options(name="Tracker").remote(
    run_name=f"Test-{MODEL_NAME.split('/')[-1]}-{env_sampler.env_list()}-{int(time.time())}", 
    wandb_project="UnstableBaselines"
) 

# initialize model registry
model_registry = unstable.ModelRegistry.options(name="ModelRegistry").remote(tracker=tracker)
ray.get(model_registry.add_checkpoint.remote(uid="base", path=None, iteration=0))
ray.get(model_registry.add_fixed.remote(name="google/gemini-2.0-flash-lite-001"))

# initialize model sampler
model_sampler = unstable.samplers.model_samplers.BaseModelSampler(model_registry=model_registry) 

# build game scheduler
game_scheduler = unstable.GameScheduler.options(name="GameScheduler").remote(model_sampler=model_sampler, env_sampler=env_sampler, logging_dir=ray.get(tracker.get_log_dir.remote()))

# Data Buffer
step_buffer = unstable.EpisodeBuffer.options(name="Buffer").remote(
    max_buffer_size=BUFFER_SIZE, tracker=tracker,
    final_reward_transformation=retra.ComposeFinalRewardTransforms([retra.RoleAdvantageByEnvFormatter()]),
    step_reward_transformation=retra.ComposeStepRewardTransforms([retra.RewardForFormat(1.5), retra.PenaltyForInvalidMove(1.0, -1.0)]),
    sampling_reward_transformation=retra.ComposeSamplingRewardTransforms([retra.NormalizeRewardsByEnv(True)]),
)

# initialize the collector
collector = unstable.Collector.options(name="Collector").remote(
    vllm_config=vllm_config, tracker=tracker, buffer=step_buffer, game_scheduler=game_scheduler,
)

# initialize the learner
learner = unstable.A2CLearner.options(num_gpus=1, name="Learner").remote(
    model_name=MODEL_NAME,
    lora_cfg=lora_config,
    batch_size=BATCH_SIZE,
    mini_batch_size=MINI_BATCH_SIZE,
    learning_rate=LR,
    grad_clip=GRAD_CLIP,
    buffer=step_buffer,
    tracker=tracker,
    model_registry=model_registry,
    activation_checkpointing=True,
    gradient_checkpointing=True,
    use_trainer_cache=False
)
ray.get(learner.initialize_algorithm.remote(infer_mini_batch_size=32, critic_learning_rate=5e-5, normalize_adv=True, max_train_len=MAX_TRAIN_SEQ_LEN, max_generation_len=MAX_GENERATION_LENGTH))


try:
    collector.collect.remote(num_train_workers=COLLECTION_WORKERS, num_eval_workers=EVALUATION_WORKERS)
    ray.get(learner.train.remote(ITERATIONS))
finally:
    ray.kill(collector, no_restart=True)
    ray.shutdown()