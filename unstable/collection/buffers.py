
import os, ray, tree, random, tree
from threading import Lock
from typing import List, Optional, Union

from unstable.utils.logger import setup_logger
from unstable.utils._types import PlayerTrajectory, Step
from unstable.collection.trackers import BaseTracker
from unstable.utils.misc import write_training_data_to_file
from unstable.collection.reward_transformations import ComposeFinalRewardTransforms, ComposeStepRewardTransforms, ComposeSamplingRewardTransforms


class BaseBuffer:
    def __init__(self, max_buffer_size: int, tracker: BaseTracker, final_reward_transformation: Optional[ComposeFinalRewardTransforms], step_reward_transformation: Optional[ComposeStepRewardTransforms], sampling_reward_transformation: Optional[ComposeSamplingRewardTransforms], buffer_strategy: str = "random"): ...
    def add_player_trajectory(self, player_traj: PlayerTrajectory, env_id: str): ...
    def get_batch(self, batch_size: int): ...



@ray.remote
class StepBuffer(BaseBuffer):
    def __init__(
        self, max_buffer_size: int, tracker: BaseTracker, 
        final_reward_transformation: Optional[ComposeFinalRewardTransforms], 
        step_reward_transformation: Optional[ComposeStepRewardTransforms], 
        sampling_reward_transformation: Optional[ComposeSamplingRewardTransforms], 
        buffer_strategy: str = "random",
        remove_invalid_move_trajectory=True
    ):
        self.max_buffer_size, self.buffer_strategy = max_buffer_size, buffer_strategy
        self.final_reward_transformation = final_reward_transformation
        self.step_reward_transformation = step_reward_transformation
        self.sampling_reward_transformation = sampling_reward_transformation
        self.remove_invalid_move_trajectory = remove_invalid_move_trajectory
        self.collect = True
        self.steps: List[Step] = []
        self.training_steps = 0
        self.tracker = tracker
        self.local_storage_dir = ray.get(self.tracker.get_train_dir.remote())
        self.logger = setup_logger("step_buffer", ray.get(tracker.get_log_dir.remote())) # setup logging
        self.mutex = Lock()

    def add_player_trajectory(self, player_traj: PlayerTrajectory, env_id: str):
        reward = self.final_reward_transformation(reward=player_traj.final_reward, pid=player_traj.pid, env_id=env_id) if self.final_reward_transformation else player_traj.final_reward
        # Optionally, if episode ended due to invalid move, only keep the last step
        indices = [len(player_traj.obs) - 1] if self.remove_invalid_move_trajectory and player_traj.game_info.get("invalid_move") and len(player_traj.obs) > 0 else list(range(len(player_traj.obs)))
        for idx in indices:
            step_reward = self.step_reward_transformation(player_traj=player_traj, step_index=idx, reward=reward) if self.step_reward_transformation else reward
            with self.mutex:
                self.steps.append(Step(pid=player_traj.pid, obs=player_traj.obs[idx], prompt=player_traj.prompts[idx], prompt_ids=player_traj.prompt_ids[idx], completion=player_traj.completions[idx], completion_ids=player_traj.completion_ids[idx], completion_logprobs=player_traj.completion_logprobs[idx], reward=step_reward, env_id=env_id, step_info={"raw_reward": player_traj.final_reward, "env_reward": reward, "step_reward": step_reward}))
        self.logger.info(f"Buffer size: {len(self.steps)}, added {len(indices)} steps")
        # downsample if necessary
        excess_num_samples = max(0, len(self.steps) - self.max_buffer_size); self.logger.info(f"Excess Num Samples: {excess_num_samples}")
        if excess_num_samples > 0:
            self.logger.info(f"Downsampling buffer because of excess samples")
            with self.mutex: 
                randm_sampled = random.sample(self.steps, excess_num_samples)
                for b in randm_sampled:
                    self.steps.remove(b)
                self.logger.info(f"Buffer size after downsampling: {len(self.steps)}")

    def get_batch(self, batch_size: int) -> List[Step]:
        with self.mutex: 
            batch = random.sample(self.steps, batch_size)
            for b in batch: self.steps.remove(b)
        batch = self.sampling_reward_transformation(batch) if self.sampling_reward_transformation is not None else batch
        self.logger.info(f"Sampling {len(batch)} samples from buffer.")
        try: write_training_data_to_file(batch=batch, filename=os.path.join(self.local_storage_dir, f"train_data_step_{self.training_steps}.csv"))
        except Exception as exc: self.logger.error(f"Exception when trying to write training data to file: {exc}")
        self.training_steps += 1
        return batch

    def stop(self):                 self.collect = False
    def size(self) -> int:          return len(self.steps)
    def continue_collection(self):  return self.collect
    def clear(self):                
        with self.mutex: 
            self.steps.clear()


@ray.remote
class EpisodeBuffer(BaseBuffer):
    def __init__(
        self, max_buffer_size: int, tracker: BaseTracker,
        final_reward_transformation: Optional[ComposeFinalRewardTransforms],
        step_reward_transformation: Optional[ComposeStepRewardTransforms],
        sampling_reward_transformation: Optional[ComposeSamplingRewardTransforms],
        buffer_strategy: str = "random",
        flatten: bool = False
    ):
        self.max_buffer_size, self.buffer_strategy = max_buffer_size, buffer_strategy
        self.final_reward_transformation = final_reward_transformation
        self.step_reward_transformation = step_reward_transformation
        self.sampling_reward_transformation = sampling_reward_transformation
        self.flatten = flatten
        self.collect = True
        self.training_steps = 0
        self.tracker = tracker
        self.local_storage_dir = ray.get(self.tracker.get_train_dir.remote())
        self.logger = setup_logger("episode_buffer", ray.get(tracker.get_log_dir.remote()))
        self.episodes: List[List[Step]] = []
        self.mutex = Lock()

    def _get_full_groups(self) -> tuple:
        """Returns (full_group_list, max_group_size). Full groups are groups whose size equals max_group_size."""
        groups: dict = {}
        for ep in self.episodes:
            seed = ep[0].step_info.get("seed") if ep and ep[0].step_info else None
            key = seed if seed is not None else id(ep)
            groups.setdefault(key, []).append(ep)
        if not groups:
            return [], 0
        max_group_size = max(len(g) for g in groups.values())
        return [g for g in groups.values() if len(g) == max_group_size], max_group_size

    def add_player_trajectory(self, player_traj: PlayerTrajectory, env_id: str):
        episode = []
        final_reward = self.final_reward_transformation(reward=player_traj.final_reward, pid=player_traj.pid, env_id=env_id) if self.final_reward_transformation else player_traj.final_reward
        for idx in list(range(len(player_traj.obs))):
            step_reward = final_reward if idx == len(player_traj.obs) - 1 else 0.0
            step_reward = self.step_reward_transformation(player_traj=player_traj, step_index=idx, reward=step_reward) if self.step_reward_transformation else step_reward
            episode.append(Step(pid=player_traj.pid, obs=player_traj.obs[idx], prompt=player_traj.prompts[idx], prompt_ids=player_traj.prompt_ids[idx], completion=player_traj.completions[idx], completion_ids=player_traj.completion_ids[idx], completion_logprobs=player_traj.completion_logprobs[idx], reward=step_reward, env_id=env_id, step_info={"raw_reward": player_traj.final_reward, "env_reward": final_reward, "step_reward": step_reward, "seed": player_traj.game_info.get("seed")}))
        if len(episode) > 0:
            with self.mutex:
                self.episodes.append(episode)
                full_groups, max_group_size = self._get_full_groups()
                usable_episodes = len(full_groups) * max_group_size
                self.logger.info(f"BUFFER full-group episodes: {usable_episodes} ({len(full_groups)} groups of size {max_group_size})")
                excess_num_samples = max(0, usable_episodes - self.max_buffer_size)
                while excess_num_samples > 0:
                    full_groups, max_group_size = self._get_full_groups()
                    if not full_groups: break
                    group_to_remove = random.choice(full_groups)
                    for ep in group_to_remove: self.episodes.remove(ep)
                    full_groups, max_group_size = self._get_full_groups()
                    usable_episodes = len(full_groups) * max_group_size
                    excess_num_samples = max(0, usable_episodes - self.max_buffer_size)

    def get_batch(self, batch_size: int) -> Union[List[List[Step]], List[Step]]:
        with self.mutex:
            full_groups, max_group_size = self._get_full_groups()
            assert len(full_groups) * max_group_size >= batch_size
            random.shuffle(full_groups)
            episode_count = 0
            sampled_episodes = []
            for group in full_groups:
                sampled_episodes.extend(group)
                episode_count += len(group)
                if episode_count >= batch_size: break
            for ep in sampled_episodes: self.episodes.remove(ep)
        sampled_episodes = self.sampling_reward_transformation(sampled_episodes) if self.sampling_reward_transformation is not None else sampled_episodes
        try: write_training_data_to_file(batch=tree.flatten(sampled_episodes), filename=os.path.join(self.local_storage_dir, f"train_data_step_{self.training_steps}.csv"))
        except Exception as exc: self.logger.error(f"Exception when trying to write training data to file: {exc}")
        self.logger.info(f"Sampling {len(sampled_episodes)} episodes from buffer.")
        self.training_steps += 1
        return tree.flatten(sampled_episodes) if self.flatten else sampled_episodes

    def stop(self):                 self.collect = False
    def size(self) -> int:
        full_groups, max_group_size = self._get_full_groups()
        return len(full_groups) * max_group_size
    def continue_collection(self):  return self.collect
    def clear(self):
        with self.mutex: 
            self.episodes.clear()
