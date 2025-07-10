import time, ray, unstable
import unstable.reward_transformations as retra

NUM_LEARNERS = 1
NUM_ACTORS = 7
COLLECTION_WORKERS = 384
EVALUATION_WORKERS = 2
ITERATIONS = 200
MODEL_NAME = "Qwen/Qwen3-1.7B-Base"
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

TRAINING_ENVS = [
    ("SimpleTak-v0-train", 2, "qwen3-zs"), 
    # ("LiarsDice-v0-train", 2, "qwen3-zs"), 
    # ("Nim-v0-train", 2, "qwen3-zs"), 
    # ("KuhnPoker-v0-train", 2, "qwen3-zs"), 
    # ("SimpleNegotiation-v0-train", 2, "qwen3-zs")
    # ("PigDice-v0-train", 2, "qwen3-zs")
    # ("Othello-v0-train", 2, "qwen3-zs")
    # ("Snake-v0-train", 2, "qwen3-zs")
    # ("Chopsticks-v0-train", 2, "qwen3-zs")
    # ("GameOfPureStrategy-v0-train", 2, "qwen3-zs")
]
EVALUATION_ENVS = [
    ("SimpleTak-v0-train", 2, "qwen3-zs"), 
    # ("LiarsDice-v0-train", 2, "qwen3-zs"), 
    ("Nim-v0-train", 2, "qwen3-zs"), 
    ("KuhnPoker-v0-train", 2, "qwen3-zs"), 
    # ("SimpleNegotiation-v0-train", 2, "qwen3-zs")
    # ("PigDice-v0-train", 2, "qwen3-zs")
    # ("Othello-v0-train", 2, "qwen3-zs")
    # ("Snake-v0-train", 2, "qwen3-zs")
    # ("Chopsticks-v0-train", 2, "qwen3-zs")
    # ("GameOfPureStrategy-v0-train", 2, "qwen3-zs")
]

# WANDB_RUN_NAME = f"Reward-Ablation--exp1-{MODEL_NAME.split('/')[-1]}-{[t[0] for t in TRAINING_ENVS]}-{int(time.time())}"
WANDB_RUN_NAME = f"Debugging-run-{MODEL_NAME.split('/')[-1]}-{[t[0] for t in TRAINING_ENVS]}-{int(time.time())}"


ray.init(namespace="unstable") # Ray init 
tracker = unstable.Tracker.options(name="Tracker").remote(run_name=WANDB_RUN_NAME, wandb_project="UnstableBaselines") # Tracker

# Data Buffer
step_buffer = unstable.StepBuffer.options(name="StepBuffer").remote(
    max_buffer_size=BUFFER_SIZE, tracker=tracker,
    final_reward_transformation=retra.ComposeFinalRewardTransforms([retra.RoleAdvantageByEnvFormatter()]),
    step_reward_transformation=retra.ComposeStepRewardTransforms([retra.RewardForFormat(1.5), retra.PenaltyForInvalidMove(1.0, -1.0)]),
    sampling_reward_transformation=retra.ComposeSamplingRewardTransforms([retra.NormalizeRewardsByEnv(True)]),
)

# Model Pool
model_pool = unstable.ModelPool.options(name="ModelPool").remote(tracker=tracker, sample_mode=SAMPLE_MODE, max_active_lora=5)
ray.get(model_pool.add_checkpoint.remote(path=None, iteration="-1"))
ray.get(model_pool.add_fixed.remote(name="google/gemini-2.0-flash-lite-001"))

# Collector
collector = unstable.Collector.options(name="Collector").remote(
    num_actors=NUM_ACTORS, 
    tracker=tracker, 
    vllm_config=vllm_config, 
    step_buffer=step_buffer, 
    model_pool=model_pool,
    training_envs=TRAINING_ENVS, 
    evaluation_envs=EVALUATION_ENVS, 
    evaluation_opponent="google/gemini-2.0-flash-lite-001",
    action_extraction="default"
)

# Algorithm and Learner
algorithm = unstable.algorithms.Reinforce()
learner = unstable.StandardLearner.options(num_gpus=NUM_LEARNERS, name="Learner").remote(
    num_learners=NUM_LEARNERS, 
    tracker=tracker, 
    model_name=MODEL_NAME, 
    step_buffer=step_buffer, 
    model_pool=model_pool, 
    algorithm=algorithm, 
    batch_size=BATCH_SIZE, 
    mini_batch_size=MINI_BATCH_SIZE, 
    learning_rate=LR, 
    grad_clip=GRAD_CLIP, 
    batch_delay_buffer=1.5, 
    lora_cfg=lora_config,
    activation_checkpointing=ACTIVATION_CHECKPOINTING,
    gradient_checkpointing=GRADIENT_CHECKPOINTING,
    use_trainer_cache=USE_TRAINER_CACHE,
    max_train_len=MAX_TRAIN_SEQ_LEN,
    max_generation_len=MAX_GENERATION_LENGTH,
)


try:
    collector.collect.remote(num_workers=COLLECTION_WORKERS, num_eval_workers=EVALUATION_WORKERS)
    ray.get(learner.train.remote(ITERATIONS))
finally:
    ray.kill(collector, no_restart=True)
    ray.shutdown()

