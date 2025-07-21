
import os, ray, tree, random
from threading import Lock
from typing import List, Dict, Optional, Tuple, Callable

# local imports
from unstable.utils.logging import setup_logger
from unstable._types import PlayerTrajectory, Step
# from unstable.core import BaseTracker
from unstable.trackers import BaseTracker
from unstable.utils import write_training_data_to_file
from unstable.reward_transformations import ComposeFinalRewardTransforms, ComposeStepRewardTransforms, ComposeSamplingRewardTransforms


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
        buffer_strategy: str = "random"
    ):
        self.max_buffer_size, self.buffer_strategy = max_buffer_size, buffer_strategy
        self.final_reward_transformation = final_reward_transformation
        self.step_reward_transformation = step_reward_transformation
        self.sampling_reward_transformation = sampling_reward_transformation
        self.collect = True
        self.steps: List[Step] = []
        self.training_steps = 0
        self.tracker = tracker
        self.local_storage_dir = ray.get(self.tracker.get_train_dir.remote())
        self.logger = setup_logger("step_buffer", ray.get(tracker.get_log_dir.remote())) # setup logging
        self.mutex = Lock()

    def add_player_trajectory(self, player_traj: PlayerTrajectory, env_id: str):
        reward = self.final_reward_transformation(reward=player_traj.final_reward, pid=player_traj.pid, env_id=env_id) if self.final_reward_transformation else player_traj.final_reward
        for idx in range(len(player_traj.obs)):
            step_reward = self.step_reward_transformation(player_traj=player_traj, step_index=idx, reward=reward) if self.step_reward_transformation else reward
            with self.mutex: 
                self.steps.append(Step(pid=player_traj.pid, obs=player_traj.obs[idx], act=player_traj.actions[idx], reward=step_reward, env_id=env_id, step_info={"raw_reward": player_traj.final_reward, "env_reward": reward, "step_reward": step_reward}))
        self.logger.info(f"Buffer size: {len(self.steps)}, added {len(player_traj.obs)} steps")
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
        buffer_strategy: str = "random"
    ):
        self.max_buffer_size, self.buffer_strategy = max_buffer_size, buffer_strategy
        self.final_reward_transformation = final_reward_transformation
        self.step_reward_transformation = step_reward_transformation
        self.sampling_reward_transformation = sampling_reward_transformation
        self.collect = True
        self.training_steps = 0
        self.tracker = tracker
        self.local_storage_dir = ray.get(self.tracker.get_train_dir.remote())
        self.logger = setup_logger("step_buffer", ray.get(tracker.get_log_dir.remote()))  # setup logging
        self.episodes: List[List[Step]] = []
        self.mutex = Lock()

    def add_player_trajectory(self, player_traj: PlayerTrajectory, env_id: str):
        episode = []
        reward = self.final_reward_transformation(reward=player_traj.final_reward, pid=player_traj.pid, env_id=env_id) if self.final_reward_transformation else player_traj.final_reward
        for idx in range(len(player_traj.obs)):
            step_reward = self.step_reward_transformation(player_traj=player_traj, step_index=idx, reward=reward) if self.step_reward_transformation else reward
            episode.append(Step(pid=player_traj.pid, obs=player_traj.obs[idx], act=player_traj.actions[idx], reward=step_reward, env_id=env_id, step_info={"raw_reward": player_traj.final_reward, "env_reward": reward, "step_reward": step_reward, "format_feedbacks": player_traj.format_feedbacks[idx]}))
        with self.mutex:
            self.episodes.append(episode)
            excess_num_samples = max(0, len(tree.flatten(self.episodes)) - self.max_buffer_size)
            self.logger.info(f"BUFFER NUM of STEP {len(tree.flatten(self.episodes))}")
            while excess_num_samples > 0:
                randm_sampled = random.sample(self.episodes, 1)
                for b in randm_sampled: self.episodes.remove(b)
                excess_num_samples = max(0, len(tree.flatten(self.episodes)) - self.max_buffer_size)
        
    def get_batch(self, batch_size: int) -> List[List[Step]]:
        with self.mutex:
            assert len(tree.flatten(self.episodes)) >= batch_size
            step_count = 0
            sampled_episodes = []
            random.shuffle(self.episodes)
            for ep in self.episodes:
                sampled_episodes.append(ep)
                step_count += len(ep)
                if step_count >= batch_size: break
            for ep in sampled_episodes: self.episodes.remove(ep)
        self.logger.info(f"Sampling {len(sampled_episodes)} episodes from buffer.")
        self.training_steps += 1
        return sampled_episodes

    def stop(self):                 self.collect = False
    def size(self) -> int:          return len(tree.flatten(self.episodes))
    def continue_collection(self):  return self.collect
    def clear(self):
        with self.mutex: 
            self.episodes.clear()
