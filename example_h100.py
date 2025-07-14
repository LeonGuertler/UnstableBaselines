import time, ray, unstable
import unstable.reward_transformations as retra

# always uses 1 learner and the remainder of the GPUS as actors
COLLECTION_WORKERS = 384 #768
EVALUATION_WORKERS = 16
ITERATIONS = 200
MODEL_NAME = "Qwen/Qwen3-4B"
# MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
BATCH_SIZE = 384
MINI_BATCH_SIZE = 1
BUFFER_SIZE = 384*2
LR = 1e-5
GRAD_CLIP = 0.2
MAX_TRAIN_SEQ_LEN = None
MAX_GENERATION_LENGTH = 4096 
SAMPLE_MODE = "mirror"

# vRAM optimization
ACTIVATION_CHECKPOINTING = True
GRADIENT_CHECKPOINTING = True
USE_TRAINER_CACHE = False


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
        unstable.TrainEnvSpec(env_id="SimpleTak-v0-train", num_players=2, num_actors=2, prompt_template="qwen3-reasoning"),
        # unstable.TrainEnvSpec(env_id="SimpleTak-v0-train", num_players=2, num_actors=2, prompt_template="qwen3-zs"),
    ],
    eval_env_specs=[
        unstable.EvalEnvSpec(env_id="SimpleTak-v0-train", num_players=2, prompt_template="qwen3-reasoning"),
        unstable.EvalEnvSpec(env_id="KuhnPoker-v0-train", num_players=2, prompt_template="qwen3-reasoning"),
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
model_sampler = unstable.samplers.model_samplers.FixedOpponentModelSampler(model_registry=model_registry) # TODO extend, but now ok for mirror self-play. unstable.model_sampler. # pass model registry here as well

# build game scheduler
game_scheduler = unstable.GameScheduler.options(name="GameScheduler").remote(model_sampler=model_sampler, env_sampler=env_sampler, logging_dir=ray.get(tracker.get_log_dir.remote()))





# Data Buffer
step_buffer = unstable.StepBuffer.options(name="Buffer").remote(
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
learner = unstable.REINFORCELearner.options(num_gpus=1, name="Learner").remote(
    model_name=MODEL_NAME,
    lora_cfg=lora_config,
    batch_size=BATCH_SIZE,
    mini_batch_size=MINI_BATCH_SIZE,
    learning_rate=LR,
    grad_clip=GRAD_CLIP,
    buffer=step_buffer,
    tracker=tracker,
    model_registry=model_registry,
    activation_checkpointing=ACTIVATION_CHECKPOINTING,
    gradient_checkpointing=GRADIENT_CHECKPOINTING,
    use_trainer_cache=USE_TRAINER_CACHE
)
ray.get(learner.initialize_algorithm.remote(max_train_len=MAX_TRAIN_SEQ_LEN, max_generation_len=MAX_GENERATION_LENGTH))



try:
    collector.collect.remote(num_train_workers=COLLECTION_WORKERS, num_eval_workers=EVALUATION_WORKERS)
    ray.get(learner.train.remote(ITERATIONS))
finally:
    ray.kill(collector, no_restart=True)
    ray.shutdown()




