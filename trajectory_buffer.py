import ray, time, random, wandb
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

        self.use_wandb = args.wandb 
        if self.use_wandb: # initialize wandb
            wandb.init(project=args.wandb_project_name, name=f"{args.model_name}-{args.train_env_id}-run-{int(time.time())}", config=args)


        self.episode_stats = {
            "invalid_move_rate":0, "count":0, "raw_avg_current_ckpt_reward":0, "raw_avg_prev_ckpt_reward":0,
            "current_checkpoint_win_rate":0, "current_checkpoint_loss_rate":0, "current_checkpoint_draw_rate":0,
            "avg_current_ckpt_reward":0, "avg_prev_ckpt_reward":0,
            "raw_avg_pid_0_current_ckpt_reward":0, "raw_avg_pid_1_current_ckpt_reward":0,
            "avg_pid_0_current_ckpt_reward":0, "avg_pid_1_current_ckpt_reward":0, "num_turns": 0
        }
        self.turn_stats = {
            "avg_response_length":0, "count":0, "pid_0_freq":0, "pid_1_freq":0,
            "format_think":0, "format_answer":0, "format_order":0
        }


    def add_trajectory(self, trajectory: Trajectory, current_checkpoint_pid: Optional[int] = None):
        self.episode_stats["count"] += 1

        # check if invalid move
        if list(trajectory.final_rewards.values()) in [[0,-1], [-1,0]]:
            self.episode_stats["invalid_move_rate"] += 1

        raw_current_ckpt_reward = trajectory.final_rewards[current_checkpoint_pid]
        raw_prev_ckpt_reward = trajectory.final_rewards[1-current_checkpoint_pid]

        self.episode_stats["raw_avg_current_ckpt_reward"] += raw_current_ckpt_reward
        self.episode_stats["raw_avg_prev_ckpt_reward"] += raw_prev_ckpt_reward

        if raw_current_ckpt_reward > raw_prev_ckpt_reward:
            self.episode_stats["current_checkpoint_win_rate"] += 1
        elif raw_current_ckpt_reward < raw_prev_ckpt_reward:
            self.episode_stats["current_checkpoint_loss_rate"] += 1
        else:
            self.episode_stats["current_checkpoint_draw_rate"] += 1
        self.episode_stats[f"raw_avg_pid_{current_checkpoint_pid}_current_ckpt_reward"] += raw_current_ckpt_reward


        # adjust the reward
        if self.reward_strategy in REWARD_TRANSFORMATIONS:
            transformed_rewards = REWARD_TRANSFORMATIONS[self.reward_strategy](raw_rewards=trajectory.final_rewards)
        else:
            raise Exception(f"Unrecognized reward strategy: {self.reward_strategy}")

        # keep track of role advantage and normalize
        self.role_advantage[0] = 0.95 * self.role_advantage[0] + 0.1*transformed_rewards[0]
        self.role_advantage[1] = 0.95 * self.role_advantage[1] + 0.1*transformed_rewards[1]

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

            self.steps.append(Step(pid=trajectory.pid[i], obs=trajectory.obs[i], act=trajectory.actions[i], reward=reward))

            # print("outside if")
            if trajectory.pid[i] == current_checkpoint_pid:
                # print("inside if")
                self.turn_stats["avg_response_length"] += len(trajectory.actions[i])
                self.turn_stats[f"pid_{trajectory.pid[i]}_freq"] += 1
                self.turn_stats["count"] += 1

                # track format rewards
                self.turn_stats["format_think"] += int(trajectory.format_feedbacks[i]["has_think"])
                self.turn_stats["format_answer"] += int(trajectory.format_feedbacks[i]["has_answer"])
                self.turn_stats["format_order"] += int(trajectory.format_feedbacks[i]["order_correct"])


        self.episode_stats["avg_current_ckpt_reward"] += transformed_rewards[current_checkpoint_pid]
        self.episode_stats["avg_prev_ckpt_reward"] += transformed_rewards[1-current_checkpoint_pid]
        self.episode_stats[f"avg_pid_{current_checkpoint_pid}_current_ckpt_reward"] += transformed_rewards[current_checkpoint_pid]
        self.episode_stats["num_turns"] += trajectory.num_turns

        if len(self.steps) > self.max_buffer_size:
            self.steps = self.steps[-self.max_buffer_size:]


    def get_batch(self, batch_size: int) -> List[Step]:
        batch = random.sample(self.steps, batch_size)
        for b in batch:
            self.steps.remove(b)

        # info for logging
        self.training_steps += 1
        self.total_train_samples += batch_size
        return batch

    def size(self) -> int:
        return len(self.steps)

    def clear(self):
        self.steps.clear()

    def log_training_info_to_wandb(self):
        if self.use_wandb:
            wandb_dict = {}
            episode_stats = self.episode_stats.copy()
            turn_stats = self.turn_stats.copy()

            episode_count = episode_stats["count"] 
            turn_count = turn_stats["count"] 

            # calculate stats
            avg_game_length = self.episode_stats["num_turns"] / episode_count
            invalid_move_rate = self.episode_stats["invalid_move_rate"] / episode_count
            raw_avg_current_ckpt_reward = self.episode_stats["raw_avg_current_ckpt_reward"] / episode_count
            raw_avg_prev_ckpt_reward = self.episode_stats["raw_avg_prev_ckpt_reward"] / episode_count
            current_checkpoint_win_rate = self.episode_stats["current_checkpoint_win_rate"] / episode_count
            current_checkpoint_loss_rate = self.episode_stats["current_checkpoint_loss_rate"] / episode_count
            current_checkpoint_draw_rate = self.episode_stats["current_checkpoint_draw_rate"] / episode_count
            avg_current_ckpt_reward = self.episode_stats["avg_current_ckpt_reward"] / episode_count
            avg_prev_ckpt_reward = self.episode_stats["avg_prev_ckpt_reward"] / episode_count
            raw_avg_pid_0_current_ckpt_reward = self.episode_stats["raw_avg_pid_0_current_ckpt_reward"] / episode_count
            raw_avg_pid_1_current_ckpt_reward = self.episode_stats["raw_avg_pid_1_current_ckpt_reward"] / episode_count
            raw_avg_current_ckpt_reward_delta = raw_avg_pid_0_current_ckpt_reward - raw_avg_pid_1_current_ckpt_reward
            avg_pid_0_current_ckpt_reward = self.episode_stats["avg_pid_0_current_ckpt_reward"] / episode_count
            avg_pid_1_current_ckpt_reward = self.episode_stats["avg_pid_1_current_ckpt_reward"] / episode_count
            avg_current_ckpt_reward_delta = avg_pid_0_current_ckpt_reward - avg_pid_1_current_ckpt_reward


            avg_response_chars = self.turn_stats["avg_response_length"] / turn_count
            pid_0_frequency = self.turn_stats["pid_0_freq"] / turn_count
            pid_1_frequency = self.turn_stats["pid_1_freq"] / turn_count
            pid_freq_delta = pid_0_frequency - pid_1_frequency
            correct_format_think_rate = self.turn_stats["format_think"] / turn_count
            correct_format_answer_rate = self.turn_stats["format_answer"] / turn_count
            correct_format_order_rate = self.turn_stats["format_order"] / turn_count


            wandb_dict["general/Game Length (avg)"] = avg_game_length
            wandb_dict["general/Invalid Move Rate"] = invalid_move_rate
            wandb_dict["general/Current Checkpoint Win Rate"] = current_checkpoint_win_rate
            wandb_dict["general/Current Checkpoint Loss Rate"] = current_checkpoint_loss_rate
            wandb_dict["general/Current Checkpoint Draw Rate"] = current_checkpoint_draw_rate

            wandb_dict["rewards/Format Success Rate (Think)"] = correct_format_think_rate
            wandb_dict["rewards/Format Success Rate (Answer)"] = correct_format_answer_rate
            wandb_dict["rewards/Format Success Rate (Order)"] = correct_format_order_rate

            wandb_dict["rewards/Raw Current Checkpoint Reward (avg)"] = raw_avg_current_ckpt_reward
            wandb_dict["rewards/Raw Previous Checkpoint Reward (avg)"] = raw_avg_prev_ckpt_reward
            wandb_dict["rewards/Raw Current Checkpoint Reward Delta (avg)"] = raw_avg_current_ckpt_reward_delta
            wandb_dict["rewards/Current Checkpoint Reward (avg)"] = avg_current_ckpt_reward
            wandb_dict["rewards/Previous Checkpoint Reward (avg)"] = avg_prev_ckpt_reward
            wandb_dict["rewards/Current Checkpoint Reward Delta (avg)"] = avg_current_ckpt_reward_delta

            wandb_dict["rewards (by pid)/Player 0 - Raw Current Checkpoint Reward (avg)"] = raw_avg_pid_0_current_ckpt_reward
            wandb_dict["rewards (by pid)/Player 1 - Raw Current Checkpoint Reward (avg)"] = raw_avg_pid_1_current_ckpt_reward
            wandb_dict["rewards (by pid)/Raw Current Checkpoint Reward Delta (avg)"] = raw_avg_current_ckpt_reward_delta
            wandb_dict["rewards (by pid)/Player 0 - Current Checkpoint Reward (avg)"] = avg_pid_0_current_ckpt_reward
            wandb_dict["rewards (by pid)/Player 1 - Current Checkpoint Reward (avg)"] = avg_pid_1_current_ckpt_reward
            wandb_dict["rewards (by pid)/Current Checkpoint Reward Delta (avg)"] = avg_current_ckpt_reward_delta

            wandb_dict["general/Respone Length (avg char)"] = avg_response_chars
            wandb_dict["general/Player 0 Turn Frequency (avg)"] = pid_0_frequency
            wandb_dict["general/Player 1 Turn Frequency (avg)"] = pid_1_frequency
            wandb_dict["general/Player Turn Frequency Delta (avg)"] = pid_freq_delta

            wandb_dict["general/Buffer Size"] = len(self.steps)
            wandb_dict["general/Training Steps"] = self.training_steps
            wandb_dict["general/Training Samples"] = self.total_train_samples

            wandb.log(wandb_dict)
            self.episode_stats = {
                "invalid_move_rate":0, "count":0, "raw_avg_current_ckpt_reward":0, "raw_avg_prev_ckpt_reward":0,
                "current_checkpoint_win_rate":0, "current_checkpoint_loss_rate":0, "current_checkpoint_draw_rate":0,
                "avg_current_ckpt_reward":0, "avg_prev_ckpt_reward":0,
                "raw_avg_pid_0_current_ckpt_reward":0, "raw_avg_pid_1_current_ckpt_reward":0,
                "avg_pid_0_current_ckpt_reward":0, "avg_pid_1_current_ckpt_reward":0, "num_turns": 0
            }
            self.turn_stats = {
                "avg_response_length":0, "count":0, "pid_0_freq":0, "pid_1_freq":0,
                "format_think":0, "format_answer":0, "format_order":0
            }