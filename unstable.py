import os, time, random, argparse, threading
import numpy as np
from typing import List, Dict

import ray, torch
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig
from ray.air import RunConfig

import textarena as ta

# local imports
from actors import VLLMActor
import reward_transformations as retra
from learners.single_node_distributed import train_loop_per_worker
from trajectory_buffer import Trajectory, Step, StepBuffer, WandBTracker

# import utils
from utils.resources import validate_requested_gpus, reserve_resources_for_learners
from utils.templates import OBSERVATION_FORMATTING, ACTION_EXTRACTION
from utils.local_files import initialize_local_folder_structure
from utils.local_textarena_modules import FirstLastObservationWrapper
from utils.misc import truncate_after_boxed



@ray.remote(num_gpus=1)
class RayActor(VLLMActor):
    def __init__(self, args):
        super().__init__(args)

@ray.remote
class Collector:
    def __init__(self, args): 
        self.args = args
        self.actor_group: List[ray.actor.ActorHandle] = []
        self.lora_paths: List[Optional[str]] = [args.initial_lora_path]

    def initialize(self, num_actors: int):
        self.actor_group = [RayActor.remote(self.args) for _ in range(num_actors)]

    def get_actor(self):
        return random.choice(self.actor_group)

    def get_current_lora(self):
        return self.lora_paths[-1]
    
    def get_previous_lora(self):
        return self.lora_paths[-self.args.self_play_opponent_lag] if len(self.lora_paths) > self.args.self_play_opponent_lag else self.lora_paths[0]

    def get_random_lora(self): # get random lora weights from within the opponent delay window
        return random.choice(self.lora_paths[:-self.args.self_play_opponent_lag]) if len(self.lora_paths) > self.args.self_play_opponent_lag else self.lora_paths[0]

    def add_new_lora_paths(self, new_path: str):
        self.lora_paths.append(new_path)


def make_env(env_id: str):
    env = ta.make(env_id); env = FirstLastObservationWrapper(env)
    env.reset(num_players=2); env.state.error_allowance = 0
    return env, env.env_id

@ray.remote(num_cpus=0.1)
def collect_episode_once(args, player_id: int, buffer, tracker, actor, collector):
    env, env_id = make_env(env_id=args.train_env_id)
    traj = Trajectory()
    done, steps = False, 0
    lora_paths = {player_id: collector.get_current_lora, 1-player_id: collector.get_previous_lora}
    fixed_opponent = ta.agents.OpenRouterAgent(model_name=args.eval_model_name)
    while not done and steps < args.max_env_steps:
        pid, obs = env.get_observation()
        formatted_prompt = OBSERVATION_FORMATTING[args.observation_format_template](observation=obs)
        lora_path = ray.get(lora_paths[pid].remote())
        action = ray.get(actor.submit_prompt.remote(prompt=formatted_prompt, lora_path=lora_path))
        action = truncate_after_boxed(action) # extract trunc act
        extracted_action, format_feedback = ACTION_EXTRACTION[args.action_extraction_template](raw_action=action) # extract environment action
        done, _ = env.step(action=extracted_action)
        
        traj.pid.append(pid); traj.obs.append(formatted_prompt)
        traj.actions.append(action); traj.format_feedbacks.append(format_feedback)
        steps += 1

    traj.final_rewards = env.close() if done else {0: 0, 1: 0}
    # add an invlid move format reward
    if list(traj.final_rewards.values()) in [[0,-1], [-1,0]]:
        for i in range(len(traj.pid)):
            if i == len(traj.pid)-1:
                traj.format_feedbacks[i]["invalid_move"] = 1
            else:
                traj.format_feedbacks[i]["invalid_move"] = 0

    traj.num_turns = steps
    print(f"GAME FINISHED< ADDING TO BUFFER. num steps: {steps}")
    ray.get(buffer.add_trajectory.remote(traj, current_checkpoint_pid=player_id))
    ray.get(tracker.add_trajectory.remote(traj, current_checkpoint_pid=player_id, env_id=env_id))


@ray.remote(num_cpus=0.1)
def run_eval_episode(args, player_id: int, tracker, actor, collector):
    env, env_id = make_env(env_id=args.eval_env_id)

    episode_info = []
    done, steps = False, 0
    models = {player_id: actor, 1-player_id: ta.agents.OpenRouterAgent(model_name=args.eval_model_name)}

    while not done and steps < args.max_env_steps_eval:
        pid, obs = env.get_observation()
        
        # check which agent to use
        agent = models[pid]
        if isinstance(agent, ta.agents.OpenRouterAgent):
            action = agent(obs)
            raw_action = action
            model_name = args.eval_model_name
        else:
            model_name = "current_ckpt"
            formatted_prompt = OBSERVATION_FORMATTING[args.observation_format_template](observation=obs)
            lora_path = ray.get(collector.get_current_lora.remote())
            raw_action = ray.get(agent.submit_prompt.remote(prompt=formatted_prompt, lora_path=lora_path))
            action, format_feedback = ACTION_EXTRACTION[args.action_extraction_template](raw_action=raw_action)
        
        done, info = env.step(action=action) # submit to env
        step_info = {
            "pid": pid, "model_name": model_name, "observation": obs, "full_action": raw_action, 
            "submitted_action": action, "done": done, "info": info, "step": steps
        }
        episode_info.append(step_info)
        steps += 1
    # store the full episode in a csv file
    ray.get(tracker.add_eval_episode.remote(episode_info=episode_info, final_reward=env.close() if done else {0:0, 1:0}, current_ckpt_pid=player_id, env_id=env_id))

# todo add logging for the number of train env eval currently running
def start_actor_loop(args, collector, buffer, tracker):
    def clean_futures(futures):
        ready, not_ready = ray.wait(futures, timeout=0, num_returns=len(futures))
        return list(not_ready)

    def loop():
        collection_outstanding = []
        evaluation_outstanding = []
        while True:
            # Clean up finished futures
            collection_outstanding = clean_futures(collection_outstanding)
            evaluation_outstanding = clean_futures(evaluation_outstanding)

            # Replenish collection
            if len(collection_outstanding) < args.num_collection_workers:
                actor = ray.get(collector.get_actor.remote())
                player_id = int(np.random.uniform()<0.5)
                future = collect_episode_once.remote(args=args, player_id=player_id, buffer=buffer, tracker=tracker, actor=actor, collector=collector)
                collection_outstanding.append(future)

            # Replenish evaluation
            if len(evaluation_outstanding) < args.num_evaluation_workers:
                actor = ray.get(collector.get_actor.remote())
                player_id = int(np.random.uniform()<0.5)
                future = run_eval_episode.remote(args=args, player_id=player_id, tracker=tracker, actor=actor, collector=collector)
                evaluation_outstanding.append(future)

            time.sleep(0.05)
    thread = threading.Thread(target=loop, daemon=True)
    thread.start()


def parse_eval_env_id(arg): # If passed as a comma-separated string, split it
    if ',' in arg:
        return arg.split(',')
    return arg

def main():
    # base args
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=5e-6)

    # general configs
    ap.add_argument("--gradient_accumulation_steps", type=int, default=64)
    ap.add_argument("--gradient_checkpointing", action="store_true") 
    ap.add_argument("--bf16_training", action="store_true") 
    ap.add_argument("--gradient_clip", type=float, default=1.0)

    # reward design
    ap.add_argument("--format_reward_think", type=float, default=0.25)
    ap.add_argument("--format_reward_valid_move", type=float, default=1.0)
    ap.add_argument("--format_penalty_invalid_move", type=float, default=-1.0)

    # faster running vars
    ap.add_argument("--num_actors", type=int, default=3)
    ap.add_argument("--num_learners", type=int, default=1)
    ap.add_argument("--num_collection_workers", type=int, default=384)
    ap.add_argument("--num_evaluation_workers", type=int, default=4)
    ap.add_argument("--max_vllm_seq", type=int, default=384)

    # collection params
    # ap.add_argument("--train_env_id", default="TicTacToe-v0")
    # ap.add_argument("--train_env_id", default="TicTacToe-v0")
    ap.add_argument("--train_env_id", type=parse_eval_env_id, default="TicTacToe-v0", help="Single env as string or multiple envs as comma-separated string")
    ap.add_argument("--max_env_steps", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--observation_format_template", type=str, default="default")
    ap.add_argument("--action_extraction_template", type=str, default="default")
    ap.add_argument("--self_play_opponent_lag", type=int, default=7)
    ap.add_argument("--use_all_data", type=bool, default=False, help="Whether to use traces from both players or only the current player")

    # eval params
    ap.add_argument("--eval_env_id", type=parse_eval_env_id, default="TicTacToe-v0", help="Single env as string or multiple envs as comma-separated string")
    # ap.add_argument("--eval_env_id", default="TicTacToe-v0")
    ap.add_argument("--max_env_steps_eval", type=int, default=64)
    ap.add_argument("--eval_model_name", type=str, default="google/gemini-2.0-flash-lite-001")

    # directory and local logging args 
    ap.add_argument("--output_dir", type=str, default="outputs/")
    ap.add_argument("--save_strategy", type=str, default="best", choices=["steps"])
    ap.add_argument("--save_every_n_update_steps", type=int, default=50)
    ap.add_argument("--log_training_data", type=bool, default=True)

    # wandb & tracking params
    ap.add_argument("--wandb", action="store_true") 
    ap.add_argument("--wandb_project_name", type=str, default="UnstableBaselines")
    # ap.add_argument("--ema_tau", type=float, default=0.01)
    ap.add_argument("--ma_range", type=int, default=100)


    # lora
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=int, default=0.0)
    ap.add_argument("--initial_lora_path", type=str, default=None)
    ap.add_argument("--vllm_max_loras", type=int, default=4)


    args = ap.parse_args() 
    args.max_buffer_size = args.batch_size*3 # default TODO maybe move at some point
    args = initialize_local_folder_structure(args=args)


    # build the reward transformations to be used
    final_reward_transformation = retra.ComposeFinalRewardTransforms([
        retra.WinDrawLossFormatter(), # turn the rewards into (1,-1), (-1,1), (0,0)
        retra.RoleAdvantageFormatter(), # normalize rewards for role advantage # TODO worth moving to step?
    ])
    step_reward_transformation = retra.ComposeStepRewardTransforms([
        retra.RewardForThinkTags(reward=args.format_reward_think), # +0.25 for correct <think></think> tags
        retra.PenaltyForInvalidMove(reward=args.format_reward_valid_move, penalty=args.format_penalty_invalid_move), 
    ])
    sampling_reward_transformation = retra.ComposeSamplingRewardTransforms([
        retra.NormalizeRewards() # normalize the sampled batch
    ])

    # check whether the gpu counts are correct
    total_gpus, _ = validate_requested_gpus(args=args)
    ray.init(num_gpus=total_gpus)

    buffer = StepBuffer.remote(
        args=args,
        final_reward_transformation=final_reward_transformation,
        step_reward_transformation=step_reward_transformation,
        sampling_reward_transformation=sampling_reward_transformation
    )

    tracker = WandBTracker.remote(args=args)
    collector = Collector.remote(args=args)
    ray.get(collector.initialize.remote(num_actors=args.num_actors))
    start_actor_loop(args=args, collector=collector, buffer=buffer, tracker=tracker)

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker, 
        scaling_config=ScalingConfig(num_workers=args.num_learners, use_gpu=True, resources_per_worker={"CPU": 2}), 
        train_loop_config={"args": args, "buffer": buffer, "collector": collector}, 
        run_config=RunConfig(storage_path=os.path.join(os.getcwd(), args.output_dir_checkpoints))
    )
    
    threading.Thread(target=trainer.fit, daemon=True).start() # Start training in a background thread so the driver can keep running.


    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Shutting down.")

if __name__ == "__main__":
    main()



# TODO add a single gpu debugging mode frfr
# TODO asserts


# TODO optimize by grouping same lora paths to same gpus
# TODO add better reward stats (optimally somehow log the transformed rewards to wandb as well)
# TODO seperate the logs for everything (and actually log to files) for easier debuggin