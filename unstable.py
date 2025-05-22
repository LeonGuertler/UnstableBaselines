import os, time, random, argparse, threading, traceback
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv

import ray, torch
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig
from ray.air import RunConfig

import textarena as ta

# local imports
from actors import VLLMActor
import reward_transformations as retra
from learners.single_node_distributed import train_loop_per_worker
from core import Trajectory, Step
from wandb_tracker import WandBTracker
from trajectory_buffer import StepBuffer

# import utils
from utils.arguments import get_args
from utils.asserts import assert_args
from utils.local_files import initialize_local_folder_structure
from utils.templates import OBSERVATION_FORMATTING, ACTION_EXTRACTION #, truncate_after_boxed

load_dotenv()

@ray.remote(num_gpus=1)
class RayActor(VLLMActor):
    def __init__(self, args):
        super().__init__(args)


@ray.remote
class Collector:
    def __init__(self, args): 
        self.args = args
        self.actor_group: List[ray.actor.ActorHandle] = []
        self.lora_paths: List[Optional[str]] = [None if (args.initial_lora_path is None or args.initial_lora_path.lower()=="none") else args.initial_lora_path]

    def initialize(self, num_actors: int):
        self.actor_group = [RayActor.remote(self.args) for _ in range(num_actors)]

    def get_actor(self):
        return random.choice(self.actor_group)

    def get_current_lora(self):
        return self.lora_paths[-1]
    
    def get_previous_lora(self):
        return self.lora_paths[-self.args.self_play_opponent_lag_lower] if len(self.lora_paths) > self.args.self_play_opponent_lag_lower else self.lora_paths[0]

    def sample_prev_lora(self): # get random lora weights from within the opponent delay window
        lower=self.args.self_play_opponent_lag_lower; upper=self.args.self_play_opponent_lag_upper
        return random.choice(self.lora_paths[-min(upper, len(self.lora_paths)):-lower]) if len(self.lora_paths) > lower else self.lora_paths[0] 

    def add_new_lora_paths(self, new_path: str):
        self.lora_paths.append(new_path)


# def make_env(env_id: str):
#     env = ta.make(env_id); env = ta.wrappers.FirstLastObservationWrapper(env)
#     env.reset(num_players=2); env.state.error_allowance = 0
#     return env, env.env_id

def make_env(env_id: str, num_players: int):
    env = ta.make(env_id)
    env = ta.wrappers.GameMessageObservationWrapper(env) # TODO should be adjustable by environment
    env.reset(num_players=num_players)
    # ; env.state.error_allowance = 0
    return env

@ray.remote(num_cpus=0.1)
def collect_episode_once(args, player_id: int, env_id: str, num_players: int, buffer, tracker, actor, collector):
    env = make_env(env_id=env_id, num_players=num_players)
    traj = Trajectory()
    done, turns = False, 0

    while not done:
        pid, obs = env.get_observation()

        if pid == player_id: # current model moves
            lora_path = ray.get(collector.get_current_lora.remote()) # get current lora path
            formatted_prompt = OBSERVATION_FORMATTING[args.observation_format_template](observation=obs) # format observation
            action = ray.get(actor.submit_prompt.remote(prompt=formatted_prompt, lora_path=lora_path)) # get model action
            extracted_action, format_feedback = ACTION_EXTRACTION[args.action_extraction_template](raw_action=action) # extract environment action

            done, info = env.step(action=extracted_action) # step in the environment

            # add to trajectory
            traj.pid.append(pid)
            traj.obs.append(formatted_prompt)
            traj.actions.append(action)
            format_feedback["invalid_move"] = 0 # change to 1 for final move if invalid
            traj.format_feedbacks.append(format_feedback)

        else: # sample opponent action based on what is specified
            if args.opponent_type == "self_play": # sample from previous checkpoints
                lora_path = ray.get(collector.sample_prev_lora.remote()) # get current lora path
                formatted_prompt = OBSERVATION_FORMATTING[args.observation_format_template](observation=obs) # format observation
                action = ray.get(actor.submit_prompt.remote(prompt=formatted_prompt, lora_path=lora_path)) # get model action
                action, _ = ACTION_EXTRACTION[args.action_extraction_template](raw_action=action) # extract environment action

            elif args.opponent_type == "fixed": # sample from fixed opponent
                fixed_opponent_agent = ta.agents.OpenRouterAgent(model_name=random.choice(args.fixed_opponents)) # TODO check if this is expensive (compute wise)
                # TODO - given this is running a thread, we need to make sure we are actually sampling opponents, might always be the same. Maybe pass seed into thread
                action = fixed_opponent_agent(obs)

            else:
                raise NotImplementedError

            done, info = env.step(action=action) # step in the env
        turns += 1 # increment turn counter
    
    traj.final_rewards = env.close() # get final game rewards
    if info["end_by_invalid"] and pid==player_id:  traj.format_feedbacks[-1]["invalid_move"] = 1 # adjust final move to invalid as necessary
    traj.num_turns = turns
    print(f"GAME FINISHED< ADDING TO BUFFER. num turns: {turns} [steps by our model: {len(traj.pid)}]")
    ray.get(buffer.add_trajectory.remote(traj, player_id=player_id, env_id=env_id))
    ray.get(tracker.add_trajectory.remote(traj, player_id=player_id, env_id=env_id))


@ray.remote(num_cpus=0.1)
def run_eval_episode(args, player_id: int, env_id: str, num_players:int, tracker, actor, collector):
    env = make_env(env_id=env_id, num_players=num_players)

    episode_info = []
    done, turns = False, 0

    opponent_agent = ta.agents.OpenRouterAgent(model_name=args.eval_model_name)

    while not done:
        pid, obs = env.get_observation()

        if pid == player_id: # our model moves
            model_name = "current_ckpt"
            lora_path = ray.get(collector.get_current_lora.remote()) # get current lora path
            formatted_prompt = OBSERVATION_FORMATTING[args.observation_format_template](observation=obs) # format observation
            full_action = ray.get(actor.submit_prompt.remote(prompt=formatted_prompt, lora_path=lora_path)) # get model action
            action, format_feedback = ACTION_EXTRACTION[args.action_extraction_template](raw_action=full_action) # extract environment action

        else: # get action from fixed opponent
            model_name = args.eval_model_name
            action = opponent_agent(obs)
            full_action = action

        # step the env
        done, info = env.step(action=action)
        step_info = {
            "pid": pid, "model_name": model_name, "observation": obs, "full_action": full_action, 
            "submitted_action": action, "done": done, "info": info, "step": turns
        }
        episode_info.append(step_info)
        turns += 1

    # store the full episode in a csv file
    ray.get(tracker.add_eval_episode.remote(episode_info=episode_info, final_reward=env.close(), current_ckpt_pid=player_id, env_id=env_id))


# TODO (maybe) add logging for the number of train env eval currently running
def start_actor_loop(args, collector, buffer, tracker):
    def clean_futures(futures):
        ready, not_ready = ray.wait(futures, timeout=0, num_returns=len(futures))
        for obj_ref in ready:
            try:
                ray.get(obj_ref)
            except Exception as e:
                print(f"[FUTURE ERROR]: {e}")
                print(traceback.format_exc())
        return list(not_ready)

    def get_next_env_id(args, _type="train"):
        env_id, num_players = random.choice(args.train_env_id if _type=="train" else args.train_env_id)
        player_id = np.random.randint(num_players)
        return env_id, num_players, player_id

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
                env_id, num_players, player_id = get_next_env_id(args=args, _type="train")
                future = collect_episode_once.remote(args=args, player_id=player_id, env_id=env_id, num_players=num_players, buffer=buffer, tracker=tracker, actor=actor, collector=collector)
                collection_outstanding.append(future)

            # Replenish evaluation
            if len(evaluation_outstanding) < args.num_evaluation_workers:
                actor = ray.get(collector.get_actor.remote())
                env_id, num_players, player_id = get_next_env_id(args=args, _type="eval")
                future = run_eval_episode.remote(args=args, player_id=player_id, env_id=env_id, num_players=num_players, tracker=tracker, actor=actor, collector=collector)
                evaluation_outstanding.append(future)

            time.sleep(0.05)
    thread = threading.Thread(target=loop, daemon=True)
    thread.start()


def main():
    args = get_args()
    args = initialize_local_folder_structure(args=args)
    assert_args(args=args) # assert everything

    # build the reward transformations to be used
    final_reward_transformation = retra.ComposeFinalRewardTransforms([
        # retra.WinDrawLossFormatter(), # turn the rewards into (1,-1), (-1,1), (0,0) # TODO can be removed with TextArena v0.6.9
        retra.RoleAdvantageByEnvFormatter(), # normalize rewards for role advantage # TODO worth moving to step?
    ])
    step_reward_transformation = retra.ComposeStepRewardTransforms([
        retra.RewardForThinkTags(reward=args.format_reward_think), # +0.25 for correct <think></think> tags
        retra.PenaltyForInvalidMove(reward=args.format_reward_valid_move, penalty=args.format_penalty_invalid_move), 
    ])
    sampling_reward_transformation = retra.ComposeSamplingRewardTransforms([
        retra.NormalizeRewardsByEnv() # normalize the sampled batch
    ])

    ray.init(num_gpus=args.num_actors+args.num_learners)

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
        train_loop_config={"args": args, "buffer": buffer, "collector": collector, "tracker": tracker}, 
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