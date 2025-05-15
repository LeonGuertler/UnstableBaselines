import numpy as np
import os, ray, random, wandb
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable

# local imports
from utils.local_files import write_eval_data_to_file, write_training_data_to_file


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
    def __init__(self, args, final_reward_transformation: Optional[Callable]=None, step_reward_transformation: Optional[Callable]=None, sampling_reward_transformation: Optional[Callable]=None):
        self.args = args
        self.final_reward_transformation = final_reward_transformation
        self.step_reward_transformation = step_reward_transformation
        self.sampling_reward_transformation = sampling_reward_transformation

        self.steps: List[Step] = []
        self.training_steps = 0

    def add_trajectory(self, trajectory: Trajectory, current_checkpoint_pid: Optional[int] = None):
        transformed_rewards = self.final_reward_transformation(trajectory.final_rewards) # apply final rewards transformations
        n = len(trajectory.pid)
        for i in range(n):
            if current_checkpoint_pid==trajectory.pid[i] or self.args.use_all_data:
                reward = transformed_rewards[trajectory.pid[i]]
                step_reward = self.step_reward_transformation(trajectory=trajectory, step_index=i, base_reward=reward) # apply step reward transformations
                self.steps.append(Step(pid=trajectory.pid[i], obs=trajectory.obs[i], act=trajectory.actions[i], reward=step_reward))
        print(f"BUFFER SIZE: {len(self.steps)}, added {n} steps")

        excess_num_samples = len(self.steps) - self.args.max_buffer_size
        if excess_num_samples > 0:
            randm_sampled = random.sample(self.steps, excess_num_samples)
            for b in randm_sampled:
                self.steps.remove(b)

    def get_batch(self, batch_size: int) -> List[Step]:
        batch = random.sample(self.steps, batch_size)
        for b in batch:
            self.steps.remove(b)
        batch = self.sampling_reward_transformation(batch) # apply sampling reward transformations
        if self.args.log_training_data: # store training data as csv file
            filename = os.path.join(self.args.output_dir_train, f"train_data_step_{self.training_steps}.csv")
            write_training_data_to_file(batch=batch, filename=filename)
        self.training_steps += 1
        return batch

    def size(self) -> int:
        return len(self.steps)

    def clear(self):
        self.steps.clear()


@ray.remote
class WandBTracker:
    def __init__(self, args):
        self.args = args 
        self.ma_range = args.ma_range

        self.wandb_name = args.wandb_name 
        wandb.init(project=args.wandb_project_name, name=self.wandb_name, config=args)
        self.metrics = {"collection": {}, "evaluation": {}} # Metric containers
        self.eval_ep_count = {}; self.num_trajectories = {} # Core counters
        self.std_metrics = ["Player Rewards", "Game Length"]

    def update_metric(self, name, value, prefix, env_id):
        if env_id not in self.metrics[prefix]:
            self.metrics[prefix][env_id] = {}
        if name not in self.metrics[prefix][env_id]:
            self.metrics[prefix][env_id][name] = deque(maxlen=self.ma_range)
        self.metrics[prefix][env_id][name].append(value)

    def log_metrics(self, prefix):
        for env_id in self.metrics[prefix]:
            ma_tag  = f"{prefix} '{env_id}' (MA - range={self.ma_range})"
            wandb_dict = {f"{ma_tag}/Num Trajectories": self.num_trajectories[env_id] if prefix=="collection" else self.eval_ep_count[env_id]}
            for name in self.metrics[prefix][env_id]:
                if self.metrics[prefix][env_id][name]:
                    wandb_dict[f"{ma_tag}/{name}"] = np.mean(self.metrics[prefix][env_id][name])
                    if name in self.std_metrics: wandb_dict[f"{ma_tag}/{name} (std)"] = np.std(self.metrics[prefix][env_id][name])
            wandb.log(wandb_dict)

    def add_eval_episode(self, episode_info: list, final_reward: dict, current_ckpt_pid: int, env_id: str):
        if env_id not in self.eval_ep_count: self.eval_ep_count[env_id] = 0
        reward_current = final_reward[current_ckpt_pid]
        reward_other = final_reward[1-current_ckpt_pid]

        # Determine outcome
        outcome_metric = "Draw Rate"
        if reward_current > reward_other:
            outcome_metric = "Win Rate"
        elif reward_current < reward_other:
            outcome_metric = "Loss Rate"

        # Update outcome metrics
        for metric in ["Win Rate", "Loss Rate", "Draw Rate"]:
            self.update_metric(metric, int(metric == outcome_metric), "evaluation", env_id)
        self.update_metric("Game Length", len(episode_info), "evaluation", env_id) # Turn count

        # Save CSV
        self.log_metrics("evaluation")
        if episode_info:
            filename = os.path.join(self.args.output_dir_eval, f"{env_id}-episode-{self.eval_ep_count[env_id]}-{outcome_metric.split()[0].lower()}.csv")
            write_eval_data_to_file(episode_info=episode_info, filename=filename)
            wandb.save(filename)
        self.eval_ep_count[env_id] += 1

    def add_trajectory(self, trajectory: Trajectory, current_checkpoint_pid: int, env_id: str):
        if env_id not in self.num_trajectories: self.num_trajectories[env_id] = 0
        raw_current = trajectory.final_rewards[current_checkpoint_pid]
        raw_prev = trajectory.final_rewards[1-current_checkpoint_pid]

        self.update_metric("Win Rate",  int(raw_current > raw_prev), "collection", env_id)
        self.update_metric("Loss Rate", int(raw_current < raw_prev), "collection", env_id)
        self.update_metric("Draw Rate", int(raw_current == raw_prev), "collection", env_id)
        self.update_metric("Invalid Move Rate", int(list(trajectory.final_rewards.values()) in [[0,-1], [-1,0]]), "collection", env_id)
        self.update_metric("Player Rewards", trajectory.final_rewards[current_checkpoint_pid], "collection", env_id)
        self.update_metric(f"Player Rewards (pid={current_checkpoint_pid})", trajectory.final_rewards[current_checkpoint_pid], "collection", env_id)

        # Game structure
        n_turns = len(trajectory.pid)
        self.update_metric("Game Length", n_turns, "collection", env_id)
        for i in range(n_turns):
            self.update_metric("Format Success Rate", int(trajectory.format_feedbacks[i]["has_think"]), "collection", env_id)
            self.update_metric("Format Invalid Move Rate", int(trajectory.format_feedbacks[i]["invalid_move"]), "collection", env_id)
            self.update_metric("Response Length (avg char)", len(trajectory.actions[i]), "collection", env_id)
        self.num_trajectories[env_id] += 1
        self.log_metrics("collection")
