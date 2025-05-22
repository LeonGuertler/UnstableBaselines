import re
import math
import hashlib
import numpy as np
import os, ray, random, wandb
from collections import deque
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable

# local imports
from utils.local_files import write_eval_data_to_file, write_training_data_to_file


@dataclass
class Trajectory:
    pid: List[int] = field(default_factory=list); obs: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list); final_rewards: Dict[int, float] = field(default_factory=dict)
    infos: List[Dict] = field(default_factory=list); num_turns: int = field(default_factory=int);
    format_feedbacks: List[Dict] = field(default_factory=list); board_states: List[str] = field(default_factory=list);
    extracted_actions: List[str] = field(default_factory=list)

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
        
        transformed_rewards = self.final_reward_transformation(trajectory.final_rewards) if self.final_reward_transformation is not None else trajectory.final_rewards
        n = len(trajectory.pid)
        for i in range(n):
            if current_checkpoint_pid==trajectory.pid[i] or self.args.use_all_data:
                reward = transformed_rewards[trajectory.pid[i]]
                step_reward = self.step_reward_transformation(trajectory=trajectory, step_index=i, base_reward=reward) if self.step_reward_transformation is not None else reward
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
        batch = self.sampling_reward_transformation(batch) if self.sampling_reward_transformation is not None else batch
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
        self.metrics = {"collection": {"all": {}}, "evaluation": {"all": {}}} # Metric containers
        self.eval_ep_count = {"all":0}; self.num_trajectories = {"all":0}; self.checkpoint_player_turns = {"all":0}; self.exploration_counters = {} # Core counters
        self.std_metrics = ["Player Rewards", "Game Length", "Response Length (avg char)", "Observation Length (avg char)"]

    def update_metric(self, name, value, prefix, env_id):
        if env_id not in self.metrics[prefix]:
            self.metrics[prefix][env_id] = {}
        if name not in self.metrics[prefix][env_id]:
            self.metrics[prefix][env_id][name] = deque(maxlen=self.ma_range)
        self.metrics[prefix][env_id][name].append(value)

        if name not in self.metrics[prefix]["all"]:
            self.metrics[prefix]["all"][name] = deque(maxlen=self.ma_range)
        self.metrics[prefix]["all"][name].append(value)

    def log_metrics(self, prefix):
        for env_id in self.metrics[prefix]:
            ma_tag  = f"{prefix} '{env_id}' (MA - range={self.ma_range})"
            wandb_dict = {
                f"{ma_tag}/Num Trajectories": self.num_trajectories[env_id] if prefix=="collection" else self.eval_ep_count[env_id],
                f"{ma_tag}/Checkpoint Player Turns": self.checkpoint_player_turns[env_id] if env_id in self.checkpoint_player_turns else 0
            }
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
            foldername = os.path.join(self.args.output_dir_eval, env_id)
            os.makedirs(foldername, exist_ok=True)
            filename = os.path.join(foldername, f"episode-{self.eval_ep_count[env_id]}-{outcome_metric.split()[0].lower()}.csv")
            write_eval_data_to_file(episode_info=episode_info, filename=filename)
            wandb.save(filename)
        self.eval_ep_count[env_id] += 1
        self.eval_ep_count["all"] += 1

    def add_trajectory(self, trajectory: Trajectory, current_checkpoint_pid: int, env_id: str):
        n_turns = len(trajectory.pid)
        checkpoint_player_turn_count = sum(1 for pid in trajectory.pid if pid == current_checkpoint_pid) 
        raw_current = trajectory.final_rewards[current_checkpoint_pid]
        raw_prev = trajectory.final_rewards[1-current_checkpoint_pid]
        trajectory_exploration_counters = {"states": {}, "actions": {}}
        if env_id not in self.num_trajectories: self.num_trajectories[env_id] = 0; self.checkpoint_player_turns[env_id] = 0
        if env_id in self.args.exploration_env_id and env_id not in self.exploration_counters: self.exploration_counters[env_id] = {"all": {"states": {}, "actions": {}, "trajectories": {}}, "last_100": {"actions": deque(maxlen=100)}}

        self.update_metric("Win Rate",  int(raw_current > raw_prev), "collection", env_id)
        self.update_metric("Loss Rate", int(raw_current < raw_prev), "collection", env_id)
        self.update_metric("Draw Rate", int(raw_current == raw_prev), "collection", env_id)
        self.update_metric("Invalid Move Rate", int(list(trajectory.final_rewards.values()) in [[0,-1], [-1,0]]), "collection", env_id)
        self.update_metric("Player Rewards", trajectory.final_rewards[current_checkpoint_pid], "collection", env_id)
        self.update_metric(f"Player Rewards (pid={current_checkpoint_pid})", trajectory.final_rewards[current_checkpoint_pid], "collection", env_id)
        self.update_metric("Game Length", n_turns, "collection", env_id)        

        for i in range(n_turns):
            if current_checkpoint_pid==trajectory.pid[i] or self.args.use_all_data:
                self.update_metric("Format Success Rate", int(trajectory.format_feedbacks[i]["has_think"]), "collection", env_id)
                self.update_metric("Format Invalid Move Rate", int(trajectory.format_feedbacks[i]["invalid_move"]), "collection", env_id)
                self.update_metric("Response Length (avg char)", len(trajectory.actions[i]), "collection", env_id)
                self.update_metric("Observation Length (avg char)", len(trajectory.obs[i]), "collection", env_id)

                if env_id in self.args.exploration_env_id:
                    # Store states
                    state = hashlib.md5(trajectory.board_states[i].encode()).hexdigest()
                    trajectory_exploration_counters["states"][state] = trajectory_exploration_counters["states"].get(state, 0) + 1
                    self.exploration_counters[env_id]["all"]["states"][state] = self.exploration_counters[env_id]["all"]["states"].get(state, 0) + 1

                    # Store actions
                    action = '[]' if "reason" in trajectory.infos[i] and "Invalid Move" in trajectory.infos[i]['reason'] else re.compile(r"\[\s*(\d+)\s*\]").search(trajectory.extracted_actions[i]).group(1)
                    trajectory_exploration_counters["actions"][action] = trajectory_exploration_counters["actions"].get(action, 0) + 1
                    self.exploration_counters[env_id]["all"]["actions"][action] = self.exploration_counters[env_id]["all"]["actions"].get(action, 0) + 1
                    self.exploration_counters[env_id]["last_100"]["actions"].append(action)

                    # Update exploration-specific metrics
                    self.update_metric("State Entropy (Trajectory)", self._entropy(trajectory_exploration_counters["states"]), "collection", env_id)
                    self.update_metric("Unique States Visited (Trajectory)", len(trajectory_exploration_counters["states"]), "collection", env_id)
                    self.update_metric("State Entropy (Global)", self._entropy(self.exploration_counters[env_id]["all"]["states"]), "collection", env_id)
                    self.update_metric("Unique States Visited (Global)", len(self.exploration_counters[env_id]["all"]["states"]), "collection", env_id)
                    self.update_metric("Action Entropy (Trajectory)", self._entropy(trajectory_exploration_counters["actions"]), "collection", env_id)
                    self.update_metric("Unique Actions (Trajectory)", len(trajectory_exploration_counters["actions"]), "collection", env_id)
                    last_100_action_counts = dict(Counter(self.exploration_counters[env_id]["last_100"]["actions"]))
                    self.update_metric("Action Entropy (Last 100)", self._entropy(last_100_action_counts), "collection", env_id)
                    self.update_metric("Unique Actions (Last 100)", len(last_100_action_counts), "collection", env_id)
                    self.update_metric("Action Entropy (Global)", self._entropy(self.exploration_counters[env_id]["all"]["actions"]), "collection", env_id)
                    self.update_metric("Unique Actions (Global)", len(self.exploration_counters[env_id]["all"]["actions"]), "collection", env_id)

        if env_id in self.args.exploration_env_id:
            trajectory_signature = hashlib.md5(str(trajectory.board_states).encode()).hexdigest()
            self.exploration_counters[env_id]["all"]["trajectories"][trajectory_signature] = self.exploration_counters[env_id]["all"]["trajectories"].get(trajectory_signature, 0) + 1
            self.update_metric("Unique Trajectories", len(self.exploration_counters[env_id]["all"]["trajectories"]), "collection", env_id)

        print('POOOL', last_100_action_counts)
        self.num_trajectories[env_id] += 1
        self.num_trajectories["all"] += 1
        self.checkpoint_player_turns[env_id] += checkpoint_player_turn_count
        self.checkpoint_player_turns["all"] += checkpoint_player_turn_count  
        self.log_metrics("collection")

    @staticmethod
    def _entropy(counts: Dict[str, int]) -> float:
        """Return the Shannon entropy (natural log) of a mapping of counts.

        Parameters
        ----------
        counts : Dict[str, int]
            A mapping from items to how often they were observed.

        Returns
        -------
        float
            The entropy value.
        """
        total = sum(counts.values())
        return -sum((c / total) * math.log(c / total) for c in counts.values()) if total > 0.0 else 0.0
