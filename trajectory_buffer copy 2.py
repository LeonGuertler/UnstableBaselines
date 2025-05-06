import numpy as np
import os, csv, ray, time, random, wandb
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from utils import REWARD_TRANSFORMATIONS

@dataclass
class Trajectory:
    pid: List[int] = field(default_factory=list); obs: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list); final_rewards: Dict[int, float] = field(default_factory=dict)
    num_turns: int = field(default_factory=int); format_feedbacks: List[Dict] = field(default_factory=list)

@dataclass
class Step:
    pid: int; obs: str; act: str; reward: float

@ray.remote
class StepBuffer:
    def __init__(self, args):
        self.args = args
        self.max_buffer_size = args.max_buffer_size
        self.reward_strategy = args.reward_strategy
        self.normalize_role_advantage = args.normalize_role_advantage
        self.role_advantage = {0:0, 1:0}

        self.steps: List[Step] = []

        self.training_steps = 0
        self.total_train_samples = 0

        self.format_reward_think = args.format_reward_think
        self.format_reward_action = args.format_reward_action
        self.format_reward_order = args.format_reward_order
        self.format_reward_invalid_move = args.format_reward_invalid_move

        self.output_dir_train = args.output_dir_train


    def add_trajectory(self, trajectory: Trajectory, current_checkpoint_pid: Optional[int] = None):
        # adjust the reward
        if self.reward_strategy in REWARD_TRANSFORMATIONS:
            transformed_rewards = REWARD_TRANSFORMATIONS[self.reward_strategy](raw_rewards=trajectory.final_rewards)
        else:
            raise Exception(f"Unrecognized reward strategy: {self.reward_strategy}")

        # keep track of role advantage and normalize
        self.role_advantage[0] = 0.99 * self.role_advantage[0] + 0.01*transformed_rewards[0]
        self.role_advantage[1] = 0.99 * self.role_advantage[1] + 0.01*transformed_rewards[1]

        if self.normalize_role_advantage:
            transformed_rewards[0] = transformed_rewards[0] - self.role_advantage[0]
            transformed_rewards[1] = transformed_rewards[1] - self.role_advantage[1]

        n = len(trajectory.pid)
        for i in range(n):
            # add the format reward if necessary
            reward = transformed_rewards.get(trajectory.pid[i], 0.0)
            if trajectory.format_feedbacks[i]["has_think"]:
                reward += self.format_reward_think
            if trajectory.format_feedbacks[i]["has_answer"]:
                reward += self.format_reward_answer
            if trajectory.format_feedbacks[i]["order_correct"]:
                reward += self.format_reward_order
            if trajectory.format_feedbacks[i]["invalid_move"]:
                reward += self.format_reward_invalid_move
            else:
                reward -= self.format_reward_invalid_move

            self.steps.append(Step(pid=trajectory.pid[i], obs=trajectory.obs[i], act=trajectory.actions[i], reward=reward))

        print(f"BUFFER SIZE: {len(self.steps)}")

        if len(self.steps) > self.max_buffer_size:
            self.steps = self.steps[-self.max_buffer_size:]


    def get_batch(self, batch_size: int) -> List[Step]:
        batch = random.sample(self.steps, batch_size)
        for b in batch:
            self.steps.remove(b)


        # if self.args.normalize_rewards:
        # normalize the rewards
        rewards = [step.reward for step in batch]
        mean = np.mean(rewards)
        std = np.std(rewards)

        for step in batch:
            step.reward = (step.reward-mean)/(std+1e-8)

        if self.args.log_training_data:
            # store training data as csv file
            csv_path = os.path.join(self.output_dir_train, f"train_data_step_{self.training_steps}.csv")
            with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['pid', 'obs', 'act', 'reward'])  # header
                for step in batch:
                    writer.writerow([step.pid, step.obs, step.act, step.reward])

        # info for logging
        self.training_steps += 1
        self.total_train_samples += batch_size
        return batch

    def size(self) -> int:
        return len(self.steps)

    def clear(self):
        self.steps.clear()


@ray.remote
class WandBTracker:
    def __init__(self, args):
        self.args = args 
        self.tau = args.ema_tau
        self.ma_range = args.ma_range
        self.output_dir_eval = args.output_dir_eval

        self.wandb_name = args.wandb_name 
        wandb.init(project=args.wandb_project_name, name=self.wandb_name, config=args)

        # Metric containers
        self.ema_metrics = {}
        self.ma_metrics = {}

        # Core counters
        self.eval_ep_count = 0
        self.num_trajectories = 0

    def update_metric(self, name, value):
        self.ema_metrics[name] = (1 - self.tau) * self.ema_metrics.get(name, 0.0) + self.tau * value # EMA
        if name not in self.ma_metrics: # MA
            self.ma_metrics[name] = deque(maxlen=self.ma_range) 
        self.ma_metrics[name].append(value)

    def log_metrics(self, prefix):
        ema_tag = f"{prefix} (EMA - tau={self.tau})"
        ma_tag  = f"{prefix} (MA - range={self.ma_range})"
        wandb_dict = {}

        for k in self.ema_metrics:
            wandb_dict[f"{ema_tag}/{k}"] = self.ema_metrics[k]
        for k in self.ma_metrics:
            if self.ma_metrics[k]:
                wandb_dict[f"{ma_tag}/{k}"] = sum(self.ma_metrics[k]) / len(self.ma_metrics[k])

        # Special counts
        if prefix == "eval":
            wandb_dict[f"{ema_tag}/Num Games"] = self.eval_ep_count
            wandb_dict[f"{ma_tag}/Num Games"] = self.eval_ep_count
        elif prefix == "collection":
            wandb_dict[f"{ema_tag}/Num Trajectories"] = self.num_trajectories
            wandb_dict[f"{ma_tag}/Num Trajectories"] = self.num_trajectories

        wandb.log(wandb_dict)

    def add_eval_episode(self, episode_info: list, final_reward: dict, current_ckpt_pid: int):
        print("ADDING EVAL EPISODE")
        reward_current = final_reward[current_ckpt_pid]
        reward_other = final_reward[1 - current_ckpt_pid]

        # Determine outcome
        outcome_metric = "Draw Rate"
        if reward_current > reward_other:
            outcome_metric = "Win Rate"
        elif reward_current < reward_other:
            outcome_metric = "Loss Rate"

        # Update outcome metrics
        for metric in ["Win Rate", "Loss Rate", "Draw Rate"]:
            self.update_metric(metric, int(metric == outcome_metric))
        self.update_metric("Game Length", len(episode_info)) # Turn count

        # Save CSV
        if episode_info:
            filename = os.path.join(self.output_dir_eval, f"episode-{self.eval_ep_count}-{outcome_metric.split()[0].lower()}.csv")
            with open(filename, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(episode_info[0].keys()))
                writer.writeheader()
                writer.writerows(episode_info)
            try:
                wandb.save(filename)
            except Exception as exc:
                print(f"Exception when pushing eval details to wandb: {exc}")

        self.eval_ep_count += 1
        self.log_metrics("eval")

    def add_trajectory(self, trajectory: Trajectory, current_checkpoint_pid: int):
        raw_current = trajectory.final_rewards[current_checkpoint_pid]
        raw_prev = trajectory.final_rewards[1 - current_checkpoint_pid]

        # Outcome
        self.update_metric("Win Rate",  int(raw_current > raw_prev))
        self.update_metric("Loss Rate", int(raw_current < raw_prev))
        self.update_metric("Draw Rate", int(raw_current == raw_prev))

        # Invalid game
        self.update_metric("Invalid Move Rate", int(list(trajectory.final_rewards.values()) in [[0,-1], [-1,0]]))

        # Game structure
        n_turns = len(trajectory.pid)
        self.update_metric("Game Length", n_turns)

        for i in range(n_turns):
            self.update_metric("Format Success Rate", int(trajectory.format_feedbacks[i]["has_think"]))
            self.update_metric("Format Invalid Move Rate", int(trajectory.format_feedbacks[i]["invalid_move"]))
            self.update_metric("Response Length (avg char)", len(trajectory.actions[i]))

        self.num_trajectories += 1
        self.log_metrics("collection")
