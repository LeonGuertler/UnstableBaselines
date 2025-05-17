import os, time, random, threading
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
from utils.asserts import assert_args
from utils.arguments import get_args
from utils.local_files import initialize_local_folder_structure
from utils.local_textarena_modules import FirstLastObservationWrapper, LLMObservationWrapper
from utils.templates import OBSERVATION_FORMATTING, ACTION_EXTRACTION, truncate_after_boxed


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

    def get_random_lora(self): # get random lora weights from within the opponent delay window
        lower=self.args.self_play_opponent_lag_lower; upper=self.args.self_play_opponent_lag_upper
        return random.choice(self.lora_paths[-min(upper, len(self.lora_paths)):-lower]) if len(self.lora_paths) > lower else self.lora_paths[0] 

    def add_new_lora_paths(self, new_path: str):
        self.lora_paths.append(new_path)


# def make_env(env_id: str):
#     env = ta.make(env_id); env = FirstLastObservationWrapper(env)
#     env.reset(num_players=2); env.state.error_allowance = 0
#     return env, env.env_id

def make_env(env_id: str):
    env = ta.make(env_id); 
    env = LLMObservationWrapper(env)
    env = ClipCharactersActionWrapper(env, max_num_characters=250)
    env.reset(num_players=2); 
    # env.state.error_allowance = 0
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
    ray.get(buffer.add_trajectory.remote(traj, current_checkpoint_pid=player_id, env_id=env_id))
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
                future = collect_episode_once.remote(args=args, player_id=int(np.random.uniform()<0.5), buffer=buffer, tracker=tracker, actor=ray.get(collector.get_actor.remote()), collector=collector)
                collection_outstanding.append(future)

            # Replenish evaluation
            if len(evaluation_outstanding) < args.num_evaluation_workers:
                future = run_eval_episode.remote(args=args, player_id=int(np.random.uniform()<0.5), tracker=tracker, actor=ray.get(collector.get_actor.remote()), collector=collector)
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

    

    
    
    ray.init(num_gpus=args.num_actors+args.num_learners, log_to_driver=args.debugging)

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
# TODO seperate the logs for everything (and actually log to files) for easier debuggin


"""
TODO:
    1. proper checkpointing (with strategy)
    2. average results for collection and eval
    3. role advantage estimation by environment
    4. multi-gpu TorchTrainer
    5. seperate the logs for everything (and actually log to files) for easier debuggin
    6. all the necessary asserts
    7. Play against n-a; n-b checkpoints (randomly selected)
    8. Organize .sh scripts

"""