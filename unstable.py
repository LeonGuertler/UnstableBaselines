import os, re, gc, time, wandb, random, argparse, asyncio, threading
import numpy as np 
from datetime import datetime
from typing import List, Dict

import ray
import textarena as ta

# local imports
from actors import VLLMActor
from learners import PPOLearner, REACTORLearner, REINFORCELearner
from trajectory_buffer import Trajectory, Step, StepBuffer, WandBTracker
from prompt_action_templates import OBSERVATION_FORMATTING, ACTION_EXTRACTION
from utils import validate_requested_gpus, average_weights, reserve_resources_for_learners
from reward_transformations import *


@ray.remote(num_gpus=1, num_cpus=1)
class RayLearner(REINFORCELearner):
    def __init__(self, args):
        super().__init__(args)

@ray.remote(num_gpus=1)
class RayActor(VLLMActor):
    def __init__(self, args):
        super().__init__(args)


class Collector:
    def __init__(self, args): 
        self.args = args
        self.group_0: List[ray.actor.ActorHandle] = []
        self.group_1: List[ray.actor.ActorHandle] = []
        self.current_group_id = 0

    def initialize(self, num_actors: int):
        assert num_actors % 2 == 0, f"expected an even number of actors for play against prev checkpoint"
        self.group_0 = [VLLMActor.remote(self.args) for _ in range(num_actors // 2)]
        self.group_1 = [VLLMActor.remote(self.args) for _ in range(num_actors // 2)]

    def get_current_and_prev_client(self):
        # return two actors. One with the previous checkoint and one with the current one
        current_group = self.group_0 if self.current_group_id == 0 else self.group_1
        prev_group = self.group_1 if self.current_group_id == 0 else self.group_0
        return random.choice(current_group), random.choice(prev_group)

    def update_all_weights(self, weights: Dict):
        current_group = self.group_0 if self.current_group_id == 0 else self.group_1
        ray.get([client.update_weights.remote(weights) for client in current_group])
        self.current_group_id = 1 - self.current_group_id  # flip roles



def make_env(env_id: str):
    env = ta.make(args.train_env_id); env = ta.wrappers.FirstLastObservationWrapper(env)
    env.reset(num_players=2); env.state.error_allowance = 0
    return env

@ray.remote(num_cpus=0.1)
def collect_episode_once(args, current_ckpt_player_id: int, env_id: str, buffer, tracker, actor1, actor2):
    env = make_env(env_id=args.train_env_id)

    traj = Trajectory()
    done, steps = False, 0
    actors = {current_ckpt_player_id: actor1, 1 - current_ckpt_player_id: actor2}

    while not done and steps < args.max_env_steps:
        pid, obs = env.get_observation()
        formatted_prompt = OBSERVATION_FORMATTING[args.observation_format_template](observation=obs)
        action = ray.get(actors[pid].submit_prompt.remote(formatted_prompt))

        # extract environment action
        extracted_action, format_feedback = ACTION_EXTRACTION[args.action_extraction_template](raw_action=action)
        done, _ = env.step(action=extracted_action)
        if args.use_all_data or pid == current_ckpt_player_id:
            traj.pid.append(pid)
            traj.obs.append(formatted_prompt)
            traj.actions.append(action)
            traj.format_feedbacks.append(format_feedback)
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
    ray.get(buffer.add_trajectory.remote(traj, current_checkpoint_pid=current_ckpt_player_id))
    ray.get(tracker.add_trajectory.remote(traj, current_checkpoint_pid=current_ckpt_player_id))


@ray.remote(num_cpus=0.1)
def run_eval_episode(player_id: int, env_id: str, tracker, actor):
    env = make_env(env_id=args.eval_env_id)

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
            formatted_prompt = apply_r1_template(observation=obs)
            raw_action = ray.get(agent.submit_prompt.remote(formatted_prompt))
            action, format_feedback = extract_action_and_format_feedback(raw_action=raw_action)
        
        done, info = env.step(action=action) # submit to env
        step_info = {
            "pid": pid, "model_name": model_name, "observation": obs, "full_action": raw_action, 
            "submitted_action": action, "done": done, "info": info, "step": steps
        }
        episode_info.append(step_info)
        steps += 1

    # store the full episode in a csv file
    ray.get(tracker.add_eval_episode.remote(episode_info=episode_info, final_reward=env.close() if done else {0:0, 1:0}, current_ckpt_pid=player_id))





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
                a1, a2 = collector.get_current_and_prev_client()
                player_id = int(np.random.uniform()<0.5)
                future = collect_episode_once.remote(args=args, current_ckpt_player_id=player_id, buffer=buffer, tracker=tracker, actor1=a1, actor2=a2)
                collection_outstanding.append(future)

            # Replenish evaluation
            if len(evaluation_outstanding) < args.num_evaluation_workers:
                a1, _ = collector.get_current_and_prev_client()
                player_id = int(np.random.uniform()<0.5)
                future = run_eval_episode.remote(args=args, player_id=player_id, tracker=tracker, actor=a1)
                evaluation_outstanding.append(future)

            time.sleep(0.05)
    threading.Thread(target=loop, daemon=True).start()



# TODO won't work well as it is because optimizer states are not synced
@ray.remote
def train_loop(learners, buffer, collector, args):
    num_learners = len(learners)
    while True:
        if ray.get(buffer.size.remote()) >= args.batch_size * 2:
            trajs = ray.get(buffer.get_batch.remote(args.batch_size))
            split_size = len(trajs) // num_learners

            # split batch among learners
            batches = [trajs[i*split_size : (i+1)*split_size] for i in range(num_learners - 1)]
            batches.append(trajs[(num_learners - 1)*split_size:])  # remainder goes to last
            update_futures = [learners[i].update.remote(batches[i]) for i in range(num_learners)]
            weights_list = ray.get(update_futures)

            avg_weights = average_weights(weights_list)
            collector.update_all_weights(avg_weights)

            if num_learners != 1:
                weight_futures = [learner.update_weights.remote(avg_weights) for learner in learners]
                ray.get(weight_futures) # Wait for all

            print("✅ All updates + syncing done:", time.time())
            avg_weights = None
            print("Weights updated at", time.time())
        else:
            time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--total_iters", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)

    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--base_port", type=int, default=8000)


    # reward args
    ap.add_argument("--normalize_role_advantage", action="store_true")
    ap.add_argument("--reward_strategy", type=str, default="win-loss", choices=["win-loss", "raw"])
    ap.add_argument("--format_reward_think", type=float, default=0.25)
    ap.add_argument("--format_reward_action", type=float, default=0.1)
    ap.add_argument("--format_reward_order", type=float, default=0.1)
    ap.add_argument("--format_reward_invalid_move", type=float, default=-1.0)


    ap.add_argument("--gradient_accumulation_steps", type=int, default=64)
    ap.add_argument("--gradient_checkpointing", action="store_true") 
    ap.add_argument("--bf16_training", action="store_true") 
    ap.add_argument("--ppo_epochs", type=int, default=1)

    ap.add_argument("--reward_scale", type=float, default=1.0)
    ap.add_argument("--ppo_clip_lower", type=float, default=0.2)
    ap.add_argument("--ppo_clip_upper", type=float, default=0.2)
    ap.add_argument("--ppo_value_clip", type=float, default=0.2)
    ap.add_argument("--vf_coef", type=float, default=1.0)
    ap.add_argument("--kl_penalty_coef", type=float, default=0.01)
    ap.add_argument("--gradient_clip", type=float, default=1.0)


    # faster running vars
    ap.add_argument("--num_actors", type=int, default=3)
    ap.add_argument("--num_learners", type=int, default=1)
    ap.add_argument("--num_collection_workers", type=int, default=384)
    ap.add_argument("--num_evaluation_workers", type=int, default=4)


    ap.add_argument("--observation_format_template", type=str, default="default")
    ap.add_argument("--action_extraction_template", type=str, default="default")


    # collection params
    ap.add_argument("--train_env_id", default="TicTacToe-v0")
    ap.add_argument("--max_env_steps", type=int, default=32)
    ap.add_argument("--use_all_data", action="store_true") # i.e. use prev checkpoint perspective as well

    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=2048)


    # eval params
    ap.add_argument("--eval_env_id", default="TicTacToe-v0")
    ap.add_argument("--max_env_steps_eval", type=int, default=64)
    ap.add_argument("--eval_model_name", type=str, default="google/gemini-2.0-flash-lite-001")



    # REACTOR vars
    ap.add_argument("--beta_js", type=float, default=0.1)
    ap.add_argument("--ent_coef", type=float, default=0.001)
    ap.add_argument("--sd_power", type=float, default=0.5)
    ap.add_argument("--huber_delta", type=float, default=1.0)
    ap.add_argument("--baseline_tau", type=float, default=0.01)
    

    ap.add_argument("--normalize_rewards", type=bool, default=True)
    ap.add_argument("--log_training_data", type=bool, default=True)


    # directory args 
    ap.add_argument("--output_dir", type=str, default="outputs/")
    ap.add_argument("--save_strategy", type=str, default="best", choices=["steps"])
    ap.add_argument("--save_every_n_update_steps", type=int, default=50)

    # wandb args
    ap.add_argument("--wandb", action="store_true") 
    ap.add_argument("--wandb_project_name", type=str, default="UnstableBaselines")


    args = ap.parse_args() 
    args.max_buffer_size = args.batch_size*3
    args.wandb_name = f"{args.model_name}-{args.train_env_id}-run-{int(time.time())}"
    args.run_folder = os.path.join(args.output_dir, f"{datetime.now().strftime('%Y%m%d-%H:%M:%S')}-{args.model_name.replace('/', '-')}-{args.train_env_id}")

    # create necessary folders
    os.makedir(args.output_dir, exist_ok=True)
    os.makedir(args.run_folder, exist_ok=True)

    # create train/eval/checkpoint folders
    args.output_dir_train = os.path.join(args.run_folder, "training_data")
    args.output_dir_eval = os.path.join(args.run_folder, "eval_data")
    args.output_dir_checkpoints = os.path.join(args.run_folder, "checkpoints")


    # tracking params
    ap.add_argument("--ema_tau", type=float, default=0.01)
    ap.add_argument("--ma_range", type=int, default=100)


    # check whether the gpu counts are correct
    total_gpus, total_cpus = validate_requested_gpus(args=args)
    ray.init(num_gpus=total_gpus)
    pg = reserve_resources_for_learners(args.num_learners)
    learners = [RayLearner.options(placement_group=pg, placement_group_bundle_index=i, num_cpus=2, num_gpus=1).remote(args=args) for i in range(args.num_learners)]

    buffer = StepBuffer.remote(args=args)
    tracker = WandBTracker.remote(args=args)
    collector = Collector(args=args)
    collector.initialize(num_actors=args.num_actors)
    start_eval_loop(args, tracker, collector)
    start_collection_loop(args, collector, buffer, tracker) #, max_outstanding=total_cpus)
    train_loop.remote(learners, buffer, collector, args)

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Shutting down.")

if __name__ == "__main__":
    main()



# TODO at start, print num threads for collection and evaluation (and give estimate if cpu is enough)
# TODO add a moving-average tracker and add tau/ma for both the wandb tracking