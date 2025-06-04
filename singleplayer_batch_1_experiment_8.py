import ray, time
import unstable
import unstable.reward_transformations as retra

# set the relevant parameters
NUM_ACTORS = 3
NUM_LEARNERS = 1

MODEL_NAME = "Qwen/Qwen3-0.6B-base"

BATCH_SIZE = 384
BUFFER_SIZE = BATCH_SIZE*2
GRADIENT_ACCUMULATION_STEPS = 384

LR = 5e-5
GRAD_CLIP = 0.2


lora_config = {
    "lora_rank": 32,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]		
    # "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
}

vllm_config = {
    "model_name": MODEL_NAME, 
    "temperature": 0.7,
    "max_tokens": 4096,
    "max_parallel_seq": 384,
    "max_loras": 5,
    "lora_config": lora_config
}
TRAINING_ENVS = [
    # ("Wordle-v0-train", 1, "qwen3-single-player"), 
    # ("FrozenLake-v0-train-random", 1, "qwen3-single-player"), 
    # ("Sokoban-v0-train", 1, "qwen3-single-player"), 
    # ("Mastermind-v0-train", 1, "qwen3-single-player"), 
    ("Minesweeper-v0-train", 1, "qwen3-single-player"), 
]

# No need for evaluation envs in single player mode
EVALUATION_ENVS = [
    ("Wordle-v0-train", 1, "qwen3-single-player"), 
    ("FrozenLake-v0-train", 1, "qwen3-single-player"), 
    # ("Sokoban-v0-train", 1, "qwen3-single-player"), 
    ("Mastermind-v0-train", 1, "qwen3-single-player"), 
    ("Minesweeper-v0-train", 1, "qwen3-single-player"), 
]

WANDB_RUN_NAME = f"SinglePlayer-Batch-1-Experiment-8--{MODEL_NAME.split('/')[-1]}-[{','.join([t[0] for t in TRAINING_ENVS])}]-{int(time.time())}"


ray.init()

# initialize the tracker to keep wandb up to date and print as necessary
tracker = unstable.WandBTracker.options(name="Tracker").remote(wandb_run_name=WANDB_RUN_NAME, exploration_env_id=["Wordle-v0-train"]) # , exploration_env_id=["SimpleTak-v0-train"]) for exploration metrics

# build the reward transformations to be used
# final_reward_transformation = retra.ComposeFinalRewardTransforms([
#     retra.RoleAdvantageByEnvFormatter(), # normalize rewards for role advantage # TODO worth moving to step?
# ])
final_reward_transformation= None
step_reward_transformation = retra.ComposeStepRewardTransforms([
    retra.RewardForThinkTags(reward=1.5), # +0.25 for using the correct format
    retra.PenaltyForInvalidMove(reward= 1.0, penalty= -1.0), 
])
# step_reward_transformation = None
sampling_reward_transformation = retra.ComposeSamplingRewardTransforms([
    retra.NormalizeRewardsByEnv(z_score=True) # normalize the sampled batch
])
# sampling_reward_transformation = None

# initialize the StepBuffer (used to hold and sample from collected traces)
step_buffer = unstable.StepBuffer.remote(
    max_buffer_size = BUFFER_SIZE,
    tracker=tracker,
    final_reward_transformation = final_reward_transformation, #final_reward_transformation,
    step_reward_transformation = step_reward_transformation, #step_reward_transformation,
    sampling_reward_transformation = sampling_reward_transformation, #sampling_reward_transformation,
)


# initialize and populate the Opponent pool
model_pool = unstable.ModelPool.remote(
    tracker=tracker,
    # sample_mode="adaptive-trueskill",
    sample_mode="fixed",
    max_active_lora=5, # how many lora checkpoints to sample from
)
ray.get(model_pool.add_checkpoint.remote(path=None, iteration="-1")) # add base checkpoint
# model_pool.add_checkpoint.remote(path="", iteration="-1") # add previous checkpoint
# ray.get(model_pool.add_fixed.remote(name="google/gemini-2.0-flash-lite-001")) # add fixed opponents
# ray.get(model_pool.add_fixed.remote(name="google/gemini-2.0-flash-001")) # add fixed opponents
# ray.get(model_pool.add_fixed.remote(name="deepseek/deepseek-r1-distill-llama-8b")) # add fixed opponents
# ray.get(model_pool.add_fixed.remote(name="qwen/qwen3-8b")) # add fixed opponents
# ray.get(model_pool.add_fixed.remote(name="qwen/qwen-2.5-7b-instruct")) # add fixed opponents
# ray.get(model_pool.add_fixed.remote(name="deepseek/deepseek-r1-0528-qwen3-8b")) # add fixed opponents
# ray.get(model_pool.add_fixed.remote(name="qwen/qwen-turbo")) # add fixed opponents
# ray.get(model_pool.add_fixed.remote(name="qwen/qwen3-14b")) # add fixed opponents
# ray.get(model_pool.add_fixed.remote(name="meta-llama/llama-3.3-70b-instruct")) # add fixed opponents
# ray.get(model_pool.add_fixed.remote(name="qwen/qwen3-30b-a3b")) # add fixed opponents


# build collector
collector = unstable.Collector.options(name="Collector").remote(
    num_actors=NUM_ACTORS,
    tracker=tracker,
    vllm_config=vllm_config,
    step_buffer=step_buffer,
    model_pool=model_pool,
    training_envs=TRAINING_ENVS,
    evaluation_envs=EVALUATION_ENVS,
    evaluation_opponent="google/gemini-2.0-flash-lite-001"
)



# build learner
learner = unstable.Learner.options(num_gpus=NUM_LEARNERS, name="Learner").remote(
    num_learners=NUM_LEARNERS,
    tracker=tracker,
    model_name=MODEL_NAME,
    step_buffer=step_buffer,
    model_pool=model_pool,
    algorithm=unstable.algorithms.Reinforce(), # if you use something else, this is the place to pass stuff
    batch_size=BATCH_SIZE,
    gradient_accum_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LR,
    grad_clip=GRAD_CLIP,
    batch_delay_buffer=1.5,
    lora_cfg=lora_config,
)



try:
    collector.collect.remote(num_workers=134, num_eval_workers=0)
    ray.get(learner.train.remote(200))
finally:
    ray.kill(collector, no_restart=True)
    ray.shutdown()



# # TODO split up the rewards when saving the training data 
# # TODO buffer needs to build the output dir and communicate it to everybody else
# # TODO buffer needs to hold the logging object (such that both the collector and learner can log stuff via it)

# # TODO log trueskills
# # TODO log opponent frequency 
# # TODO add grad_norm back into tracking

# # TODO (long-term) keep pool of learning lora weights for stability


# # TODO Rich overview. Top right, elo table with sampling frequency. Bottom right, utilization metrics, Top left maybe buffer size, total games played etc; bottom left ?