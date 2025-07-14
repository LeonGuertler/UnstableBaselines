import ray, unstable, time
import unstable.reward_transformations as retra

ray.init(namespace="unstable")

MODEL_NAME = "Qwen/Qwen3-4B-base"
TRAINING_ENVS = [
    # ("Wordle-v0-train", 1, "qwen3-sp"),
    # ("Mastermind-v0-train", 1, "qwen3-sp"), # similar to wordle
    # ("FrozenLake-v0-random-train", 1, "qwen3-sp"),
    # ("Minesweeper-v0-small-train", 1, "qwen3-sp"),
    # ("Sokoban-v0-train", 1, "qwen3-sp"),
    # ("Sudoku-v0-train", 1, "qwen3-sp"),
    # ("TowerOfHanoi-v0-train", 1, "qwen3-sp"),
    ("2048-v0-super-easy-train", 1, "qwen3-sp"),
    ("2048-v0-very-easy-train", 1, "qwen3-sp"),
    # ("2048-v0-easy-train", 1, "qwen3-sp"),
    # ("2048-v0-train", 1, "qwen3-sp"),

]
EVALUATION_ENVS = [
    # ("Wordle-v0-train", 1, "qwen3-sp"),
    # ("Mastermind-v0-train", 1, "qwen3-sp"), # similar to wordle
    ("FrozenLake-v0-random-train", 1, "qwen3-sp"),
    # ("Minesweeper-v0-small-train", 1, "qwen3-sp"),
    # ("Sokoban-v0-train", 1, "qwen3-sp"),
    # ("Sudoku-v0-train", 1, "qwen3-sp"),
    ("TicTacToe-v0-train", 2, "qwen3-zs"),
    ("2048-v0-super-easy-train", 1, "qwen3-sp"),
]

tracker = unstable.Tracker.options(name="Tracker").remote(run_name=f"sp-experiment1-{MODEL_NAME.split('/')[-1]}-{[t[0] for t in TRAINING_ENVS]}-{int(time.time())}", wandb_project="UnstableBaselines")

step_buffer = unstable.StepBuffer.options(name="StepBuffer").remote(
    max_buffer_size=384*2, 
    tracker=tracker,
    final_reward_transformation=retra.ComposeFinalRewardTransforms([retra.RoleAdvantageByEnvFormatter()]),
    step_reward_transformation=retra.ComposeStepRewardTransforms([retra.RewardForFormat(1.5), retra.PenaltyForInvalidMove(1.0, -1.0)]),
    sampling_reward_transformation=retra.ComposeSamplingRewardTransforms([retra.NormalizeRewardsByEnv(True)]),
)

model_pool = unstable.ModelPool.options(name="ModelPool").remote(sample_mode="mirror", max_active_lora=3, tracker=tracker)
ray.get(model_pool.add_checkpoint.remote(path=None, iteration=-1)) # set initial checkpoint as no LoRA

lora_cfg = {
    "lora_rank": 32, "lora_alpha": 32, "lora_dropout": 0.0,
    "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj", "up_proj","down_proj"]
}
collector = unstable.Collector.options(name="Collector").remote(
    num_actors=1, 
    step_buffer=step_buffer, 
    model_pool=model_pool, 
    tracker=tracker,
    vllm_config={
        "model_name": MODEL_NAME, 
        "max_parallel_seq": 128,
        "max_tokens": 4096, 
        "max_loras": 5, 
        "lora_config": lora_cfg, 
        "max_model_len": 8192
    },
    training_envs=TRAINING_ENVS, # (env-id, num players, prompt template)
    evaluation_envs=EVALUATION_ENVS,
    evaluation_opponent="google/gemini-2.0-flash-lite-001",
)

learner = unstable.StandardLearner.options(num_gpus=1, name="Learner").remote(
    model_name=MODEL_NAME, 
    step_buffer=step_buffer,
    model_pool=model_pool,
    tracker=tracker,
    algorithm=unstable.algorithms.Reinforce(),
    batch_size=384,
    mini_batch_size=1,
    learning_rate=1e-5,
    grad_clip=0.2,
    lora_cfg=lora_cfg,
    activation_checkpointing=True,
    gradient_checkpointing=True,
    use_trainer_cache=False,
    max_train_len=None, # always train on the full sequence
    max_generation_len=4096, # important for Dr. GRPO
)

# start the collection and training loops
collector.collect.remote(num_workers=384, num_eval_workers=16)  
ray.get(learner.train.remote(200)) # total update steps