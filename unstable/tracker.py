import re, math, ray, hashlib
import os, wandb, datetime
import numpy as np
from collections import Counter, deque
from typing import List, Dict, Optional, Tuple, Callable

# local imports
from unstable.core import Trajectory, Step
from unstable.utils.local_files import write_eval_data_to_file


@ray.remote
class WandBTracker:
    def __init__(
        self, 
        wandb_run_name: str,
        ma_range: int = 100,
        eval_games_per_update_step: int = 10,
        output_dir: Optional[str] = None,
        wandb_project_name: str= "UnstableBaselines"
    ):
        self.ma_range = ma_range
        self.wandb_name = wandb_run_name
        self._build_output_dir(output_dir=output_dir) 
        self.eval_games_per_update_step = eval_games_per_update_step

        wandb.init(project=wandb_project_name, name=self.wandb_name)
        self.metrics = {"collection": {"all": {}}, "evaluation": {"all": {}}} # Metric containers
        self.eval_iter_metrics = {} # use iteration as key, when full, log and clear to save space
        self.eval_ep_count = {"all":0}; self.num_trajectories = {"all":0} # Core counters
        self.mean_metrics = ["Player Rewards", "Game Length", "Response Length (avg char)", "Observation Length (avg char)"]


    def _build_output_dir(self, output_dir):
        self.output_dir = os.path.abspath(os.path.join("outputs", str(datetime.datetime.now().strftime('%Y-%m-%d')), str(datetime.datetime.now().strftime('%H-%M-%S')), self.wandb_name)) if not output_dir else output_dir
        os.makedirs(self.output_dir)

        self.output_dir_train = os.path.join(self.output_dir, "training_data"); os.makedirs(self.output_dir_train, exist_ok=True)
        self.output_dir_eval = os.path.join(self.output_dir, "eval_data"); os.makedirs(self.output_dir_eval, exist_ok=True)
        self.output_dir_checkpoints = os.path.join(self.output_dir, "checkpoints"); os.makedirs(self.output_dir_checkpoints, exist_ok=True)

    def get_checkpoints_dir(self):
        return self.output_dir_checkpoints

    def get_train_dir(self):
        return self.output_dir_train

    def update_metric(self, name, value, prefix, env_id, mean=False):
        if mean and name not in self.mean_metrics: self.mean_metrics.append(name)
        if env_id not in self.metrics[prefix]: self.metrics[prefix][env_id] = {}
        if mean and name not in self.metrics[prefix][env_id]: 
            self.metrics[prefix][env_id][name] = deque(maxlen=self.ma_range)
        if mean: 
            self.metrics[prefix][env_id][name].append(value) 
        else: self.metrics[prefix][env_id][name] = value

        if mean and name not in self.metrics[prefix]["all"]: self.metrics[prefix]["all"][name] = deque(maxlen=self.ma_range)
        if mean: 
            self.metrics[prefix]["all"][name].append(value)
        else: self.metrics[prefix]["all"][name] = value

    def update_iteration_eval_metric(self, name, value, env_id, ckpt_iteration):
        if ckpt_iteration not in self.eval_iter_metrics: self.eval_iter_metrics[ckpt_iteration] = {}
        if env_id not in self.eval_iter_metrics[ckpt_iteration]: self.eval_iter_metrics[ckpt_iteration][env_id] = {}
        if name not in self.eval_iter_metrics[ckpt_iteration][env_id]: self.eval_iter_metrics[ckpt_iteration][env_id][name] = []
        self.eval_iter_metrics[ckpt_iteration][env_id][name].append(value)

    def log_metrics(self, prefix):
        for env_id in self.metrics[prefix]:
            tag  = f"{prefix} '({env_id}')"
            wandb_dict = {
                f"{tag}/Num Trajectories": self.num_trajectories[env_id] if prefix in ["collection", "exploration"] else self.eval_ep_count[env_id],
            }
            for name in self.metrics[prefix][env_id]:
                # if self.metrics[prefix][env_id][name]: # Does not log win_rate == 0
                wandb_dict[f"{tag}/{name}"] = np.mean(self.metrics[prefix][env_id][name]) if name in self.mean_metrics else self.metrics[prefix][env_id][name]
                if name in self.mean_metrics: wandb_dict[f"{tag}/{name} (std)"] = np.std(self.metrics[prefix][env_id][name])
            wandb.log(wandb_dict)

    def add_eval_episode(self, episode_info: list, final_reward: dict, current_ckpt_pid: int, env_id: str, ckpt_iteration: int):
        # check num players
        if len(final_reward) == 1: # single player env, just report the final reward
            self.update_iteration_eval_metric("Reward", final_reward[0], env_id, ckpt_iteration)
            self.update_iteration_eval_metric("Game Length", len(episode_info), env_id, ckpt_iteration)
            if len(self.eval_iter_metrics[ckpt_iteration][env_id]["Reward"]) >= self.eval_games_per_update_step: # log it
                wandb_dict = {f"Eval '{env_id}'/{name}": np.mean(self.eval_iter_metrics[ckpt_iteration][env_id][name]) for name in self.eval_iter_metrics[ckpt_iteration][env_id]}
                wandb_dict[f"Eval '{env_id}'/ckpt-iteration"] = ckpt_iteration
                wandb.log(wandb_dict)
            return

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
            self.update_iteration_eval_metric(metric, int(metric==outcome_metric), env_id, ckpt_iteration)
        self.update_metric("Game Length", len(episode_info), "evaluation", env_id) # Turn count
        self.update_iteration_eval_metric("Game Length",  len(episode_info), env_id, ckpt_iteration)  # Turn count

        # Save CSV
        self.log_metrics("evaluation")
        if episode_info:
            foldername = os.path.join(self.output_dir_eval, env_id)
            os.makedirs(foldername, exist_ok=True)
            filename = os.path.join(foldername, f"episode-{self.eval_ep_count[env_id]}-{outcome_metric.split()[0].lower()}.csv")
            write_eval_data_to_file(episode_info=episode_info, filename=filename)
            wandb.save(filename)
        self.eval_ep_count[env_id] += 1
        self.eval_ep_count["all"] += 1

        # check if we should log the iteration based results
        # if len(self.eval_iter_metrics[ckpt_iteration][env_id]["Game Length"]) >= self.args.eval_games_per_update_step: # log it
        #     wandb_dict = {f"Eval '{env_id}'/{name}": np.mean(self.eval_iter_metrics[ckpt_iteration][env_id][name]) for name in self.eval_iter_metrics[ckpt_iteration][env_id]}
        #     wandb_dict[f"Eval '{env_id}'/ckpt-iteration"] = ckpt_iteration
        #     wandb.log(wandb_dict)

    def add_trajectory(self, trajectory: Trajectory, player_id: int, env_id: str):
        n_turns = len(trajectory.pid)
        player_turns = sum(1 for pid in trajectory.pid if pid == player_id) 

        trajectory_counters = {"states": {}, "actions": {}}
        if env_id not in self.num_trajectories: self.num_trajectories[env_id] = 0

        if len(trajectory.final_rewards) == 1:
            raw_current = 1 #trajectory.final_rewards[player_id]
            raw_prev = 1 #trajectory.final_rewards[1-player_id]
        else:
            raw_current = trajectory.final_rewards[player_id]
            raw_prev = trajectory.final_rewards[1-player_id]

        self.update_metric("Win Rate",  int(raw_current > raw_prev), "collection", env_id)
        self.update_metric("Loss Rate", int(raw_current < raw_prev), "collection", env_id)
        self.update_metric("Draw Rate", int(raw_current == raw_prev), "collection", env_id)
        self.update_metric("Invalid Move Rate", int(list(trajectory.final_rewards.values()) in [[0,-1], [-1,0]]), "collection", env_id)
        self.update_metric("Player Rewards", trajectory.final_rewards[player_id], "collection", env_id)
        self.update_metric(f"Player Rewards (pid={player_id})", trajectory.final_rewards[player_id], "collection", env_id)

        # Game structure
        n_turns = len(trajectory.pid)
        self.update_metric("Game Length", n_turns, "collection", env_id)
        for i in range(n_turns):
            if player_id==trajectory.pid[i]:
                self.update_metric("Format Success Rate", int(trajectory.format_feedbacks[i]["has_think"]), "collection", env_id)
                self.update_metric("Format Invalid Move Rate", int(trajectory.format_feedbacks[i]["invalid_move"]), "collection", env_id)
                self.update_metric("Response Length (avg char)", len(trajectory.actions[i]), "collection", env_id)
                self.update_metric("Observation Length (avg char)", len(trajectory.obs[i]), "collection", env_id)

        self.num_trajectories[env_id] += 1
        self.num_trajectories["all"] += 1 
        self.log_metrics("collection")

    def log_learner(self, wandb_dict):
        wandb.log(wandb_dict)

    def log_trueskill(self, step, uid, mu, sigma):
        wandb.log({f"trueskill/{uid}/mu": mu, f"trueskill/{uid}/sigma": sigma, "learner/step": step})

    def log_matchup_counts(self, step, counts: dict):
        # counts = {("ckpt-200","ckpt-195"): 17, ("ckpt-200","gemini"): 9, …}
        for (u1, u2), n in counts.items():
            wandb.log({f"matchups/{u1}_vs_{u2}": n, "learner/step": step})

    def log_action_n_grams(self, action_n_grams: dict, env_id: str):
        def _normalize_ngrams(ngrams):
            total = sum(ngrams.values())
            return {k: v / total for k, v in ngrams.items()}
        def _compute_entropy(prob_dist):
            return -sum(p * math.log2(p) for p in prob_dist if p > 0)
        wandb_dict = {
            **{f"exploration ({env_id})/{n}-gram entropy": _compute_entropy(_normalize_ngrams(action_n_grams[n]).values()) for n in action_n_grams},
            **{f"exploration ({env_id})/unique {n}-grams": len(action_n_grams[n]) for n in action_n_grams}
            }
        wandb.log(wandb_dict)

    def log_pairwise_n_gram_distances(self, dist_matrix: np.ndarray, uids: List[str]):
        wandb_dict = {}
        for i in range(dist_matrix.shape[0]):
            distances = [dist_matrix[i, j] for j in range(dist_matrix.shape[1]) if i != j]
            if distances:  # Only compute min if we have distances
                wandb_dict[f'opponent sampling/min dist.{uids[i]}'] = np.min(distances)
                wandb_dict[f'opponent sampling/max dist.{uids[i]}'] = np.max(distances)
            else:
                wandb_dict[f'opponent sampling/min dist.{uids[i]}'] = 0.0  # or np.nan
                wandb_dict[f'opponent sampling/max dist.{uids[i]}'] = 0.0  # or np.nan
        wandb.log(wandb_dict)

    @staticmethod
    def _entropy(counts: Dict[str, int]) -> float:
        total = sum(counts.values())
        return -sum((c / total) * math.log(c / total) for c in counts.values()) if total > 0.0 else 0.0
