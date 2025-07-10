import os, ray, csv, random, logging, tree
from typing import List, Dict, Optional, Tuple, Callable
from threading import Lock

# local imports
from unstable.utils.logging import setup_logger
from unstable.core import Trajectory, Step, BaseTracker
from unstable.reward_transformations import (
    ComposeFinalRewardTransforms,
    ComposeStepRewardTransforms,
    ComposeSamplingRewardTransforms,
)


def write_eval_data_to_file(episode_info, filename):
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(episode_info[0].keys()))
        writer.writeheader()
        writer.writerows(episode_info)


def write_training_data_to_file(batch, filename: str):
    with open(filename, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["pid", "obs", "act", "reward", "env_id", "step_info"]
        )  # header
        for step in batch:
            writer.writerow(
                [step.pid, step.obs, step.act, step.reward, step.env_id, step.step_info]
            )


@ray.remote
class StepBuffer:
    def __init__(
        self,
        max_buffer_size: int,
        tracker: BaseTracker,
        final_reward_transformation: Optional[ComposeFinalRewardTransforms],
        step_reward_transformation: Optional[ComposeStepRewardTransforms],
        sampling_reward_transformation: Optional[ComposeSamplingRewardTransforms],
        buffer_strategy: str = "random",
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
        self.logger = setup_logger(
            "step_buffer", ray.get(tracker.get_log_dir.remote())
        )  # setup logging

    def add_trajectory(self, trajectory: Trajectory, player_id: int, env_id: str):
        import pickle

        pickle.dump(trajectory, open("./tmp_traj.pkl", "wb"))
        try:
            transformed_rewards = (
                self.final_reward_transformation(
                    trajectory.final_rewards, env_id=env_id
                )
                if self.final_reward_transformation
                else trajectory.final_rewards
            )
        except Exception as exc:
            self.logger.exception(
                f"reward-transformation blew up  env={env_id} -\n\n{exc}\n\n"
            )
            raise
        n = len(trajectory.pid)
        for i in range(n):  # these are already just steps by our model
            reward = transformed_rewards[trajectory.pid[i]]
            step_reward = (
                self.step_reward_transformation(
                    trajectory=trajectory, step_index=i, base_reward=reward
                )
                if self.step_reward_transformation is not None
                else reward
            )
            step_info = {
                "raw_reward": trajectory.final_rewards[trajectory.pid[i]],
                "transformed_end_of_game_reward": transformed_rewards[
                    trajectory.pid[i]
                ],
                "step_reward": step_reward,
            }
            self.steps.append(
                Step(
                    pid=trajectory.pid[i],
                    obs=trajectory.obs[i],
                    act=trajectory.actions[i],
                    reward=step_reward,
                    env_id=env_id,
                    step_info=step_info,
                )
            )
        self.logger.info(f"BUFFER SIZE: {len(self.steps)}, added {n} steps")
        pickle.dump(self.steps, open("./tmp_steps.pkl", "wb"))

        excess_num_samples = max(0, len(self.steps) - self.max_buffer_size)
        self.logger.info(f"Excess Num Samples: {excess_num_samples}")
        if excess_num_samples > 0:
            self.logger.info(f"Downsampling buffer because of excess samples")
            randm_sampled = random.sample(self.steps, excess_num_samples)
            for b in randm_sampled:
                self.steps.remove(b)
            self.logger.info(f"Buffer size after downsampling: {len(self.steps)}")

    def get_batch(self, batch_size: int) -> List[Step]:
        try:
            batch = random.sample(self.steps, batch_size)
        except ValueError as exc:
            self.logger.error(
                "requested %s samples - only %s in buffer", batch_size, len(self.steps)
            )
            raise
        for b in batch:
            self.steps.remove(b)
        batch = (
            self.sampling_reward_transformation(batch)
            if self.sampling_reward_transformation is not None
            else batch
        )
        self.logger.info(f"Sampling {len(batch)} samples from buffer.")
        try:
            write_training_data_to_file(
                batch=batch,
                filename=os.path.join(
                    self.local_storage_dir, f"train_data_step_{self.training_steps}.csv"
                ),
            )
        except Exception as exc:
            self.logger.error(
                f"Exception when trying to write training data to file: {exc}"
            )
        self.training_steps += 1
        self.logger.info(f"returning batch of size {len(batch)}")
        return batch

    def clear(self):
        self.steps.clear()

    def stop(self):
        self.collect = False

    def size(self) -> int:
        return len(self.steps)

    def continue_collection(self):
        return self.collect


@ray.remote
class EpisodeBuffer:
    def __init__(
        self,
        max_buffer_size: int,
        tracker: BaseTracker,
        final_reward_transformation: Optional[ComposeFinalRewardTransforms],
        step_reward_transformation: Optional[ComposeStepRewardTransforms],
        sampling_reward_transformation: Optional[ComposeSamplingRewardTransforms],
        buffer_strategy: str = "random",
    ):
        self.max_buffer_size, self.buffer_strategy = max_buffer_size, buffer_strategy
        self.final_reward_transformation = final_reward_transformation
        self.step_reward_transformation = step_reward_transformation
        self.sampling_reward_transformation = sampling_reward_transformation
        self.collect = True
        self.training_steps = 0
        self.tracker = tracker
        self.local_storage_dir = ray.get(self.tracker.get_train_dir.remote())
        self.logger = setup_logger(
            "step_buffer", ray.get(tracker.get_log_dir.remote())
        )  # setup logging
        self.episodes: List[List[Step]] = []
        self.mutex = Lock()

    def add_trajectory(self, trajectory: Trajectory, player_id: int, env_id: str):
        try:
            transformed_rewards = (
                self.final_reward_transformation(
                    trajectory.final_rewards, env_id=env_id
                )
                if self.final_reward_transformation
                else trajectory.final_rewards
            )
        except Exception as exc:
            self.logger.exception(
                f"reward-transformation blew up  env={env_id} -\n\n{exc}\n\n"
            )
            raise
        n = len(trajectory.pid)
        episode = []
        for i in range(n):  # these are already just steps by our model
            reward = transformed_rewards[trajectory.pid[i]]
            step_reward = (
                self.step_reward_transformation(
                    trajectory=trajectory, step_index=i, base_reward=reward
                )
                if self.step_reward_transformation is not None
                else reward
            )
            step_info = {
                "raw_reward": trajectory.final_rewards[trajectory.pid[i]],
                "transformed_end_of_game_reward": transformed_rewards[
                    trajectory.pid[i]
                ],
                "step_reward": step_reward,
            }
            episode.append(
                Step(
                    pid=trajectory.pid[i],
                    obs=trajectory.obs[i],
                    act=trajectory.actions[i],
                    reward=step_reward,
                    env_id=env_id,
                    step_info=step_info,
                )
            )
        with self.mutex:
            self.episodes.append(episode)
            excess_num_samples = max(
                0, len(tree.flatten(self.episodes)) - self.max_buffer_size
            )
            self.logger.info(f"BUFFER NUM of STEP {len(tree.flatten(self.episodes))}")
            while excess_num_samples > 0:
                randm_sampled = random.sample(self.episodes, 1)
                for b in randm_sampled:
                    self.episodes.remove(b)
                excess_num_samples = max(
                    0, len(tree.flatten(self.episodes)) - self.max_buffer_size
                )

    def get_batch(self, batch_size: int) -> List[List[Step]]:
        with self.mutex:
            assert len(tree.flatten(self.episodes)) >= batch_size
            print(len(self.episodes), len(tree.flatten(self.episodes)), batch_size)
            step_count = 0
            sampled_episodes = []
            random.shuffle(self.episodes)
            for ep in self.episodes:
                sampled_episodes.append(ep)
                step_count += len(ep)
                if step_count >= batch_size:
                    break
                print(step_count)
            for ep in sampled_episodes:
                self.episodes.remove(ep)
        # batch = self.sampling_reward_transformation(batch) if self.sampling_reward_transformation is not None else batch
        self.logger.info(f"Sampling {len(sampled_episodes)} episodes from buffer.")
        # try: write_training_data_to_file(batch=batch, filename=os.path.join(self.local_storage_dir, f"train_data_step_{self.training_steps}.csv"))
        # except Exception as exc: self.logger.error(f"Exception when trying to write training data to file: {exc}")
        self.training_steps += 1
        # self.logger.info(f"returning batch of size {len(batch)}")
        return sampled_episodes

    def clear(self):
        with self.mutex:
            self.episodes.clear()

    def stop(self):
        self.collect = False

    def size(self) -> int:
        return len(tree.flatten(self.episodes))

    def continue_collection(self):
        return self.collect
