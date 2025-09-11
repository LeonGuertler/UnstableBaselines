import numpy as np
from typing import List, Optional
from collections import defaultdict
from unstable.common._types import Step, PlayerTrajectory


### Final Reward
class FinalRewardTransform:
    def __call__(self, reward: float, pid: int, env_id: Optional[str] = None) -> float: raise NotImplementedError

class ComposeFinalRewardTransforms:
    def __init__(self, transforms: List[FinalRewardTransform]): self.transforms = transforms
    def __call__(self, reward: float, pid: int, env_id: Optional[str] = None) -> float:
        for transform in self.transforms: reward = transform(reward, pid, env_id)
        return reward

class RoleAdvantageFormatter(FinalRewardTransform):
    def __init__(self, role_adv: float=0.0, tau: float=0.001): self.role_adv, self.tau = role_adv, tau
    def __call__(self, reward: float, pid: int, env_id: Optional[str] = None) -> float:
        self.role_adv[pid] = (1-self.tau) * self.role_adv[pid] + self.tau * reward
        reward -= self.role_adv[pid]
        return reward

class RoleAdvantageByEnvFormatter(FinalRewardTransform):
    def __init__(self, default_role_adv: float=0.0, tau: float=0.001): self.default_role_adv, self.tau, self.role_advantage_dict = default_role_adv, tau, {}
    def __call__(self, reward: float, pid: int, env_id: Optional[str] = None) -> float:
        if env_id not in self.role_advantage_dict: self.role_advantage_dict[env_id] = {}
        self.role_advantage_dict[env_id][pid] = (1-self.tau) *  self.role_advantage_dict[env_id].get(pid, self.default_role_adv) + self.tau * reward
        reward -= self.role_advantage_dict[env_id][pid]
        return reward

### Step-wise Reward
class StepRewardTransform:
    def __call__(self, player_traj: PlayerTrajectory, step_index: int, reward: float) -> float: raise NotImplementedError

class ComposeStepRewardTransforms:
    def __init__(self, transforms: List[StepRewardTransform]): self.transforms = transforms
    def __call__(self, player_traj: PlayerTrajectory, step_index: int, reward: float) -> float:
        for transform in self.transforms: reward = transform(player_traj, step_index, reward)
        return reward

class RewardForFormat(StepRewardTransform):
    def __init__(self, reward: float=0, penalty: float=0): self.reward, self.penalty = reward, penalty
    def __call__(self, player_traj: PlayerTrajectory, step_index: int, reward: float) -> float:
        reward += (self.reward if player_traj.format_feedbacks[step_index].get("correct_answer_format") else self.penalty)
        return reward

class PenaltyForInvalidMove(StepRewardTransform):
    def __init__(self, reward: float=0, penalty: float=0): self.reward, self.penalty = reward, penalty
    def __call__(self, player_traj: PlayerTrajectory, step_index: int, reward: float) -> float:
        reward += (self.penalty if player_traj.format_feedbacks[step_index].get("invalid_move") else self.reward)
        return reward

### Step-wise Training Batch Reward
class SamplingRewardTransform:
    def __call__(self, steps: List[Step], env_id: Optional[str] = None) -> List[Step]: raise NotImplementedError

class ComposeSamplingRewardTransforms:
    def __init__(self, transforms: List[SamplingRewardTransform]):  self.transforms = transforms
    def __call__(self, steps: List[Step]) -> List[Step]:
        for transform in self.transforms: steps = transform(steps)
        return steps

class NormalizeRewards(SamplingRewardTransform):
    def __init__(self, z_score: bool=False): self.z_score = z_score
    def __call__(self, steps: List[Step], env_id: Optional[str] = None) -> List[Step]:
        rewards = [step.reward for step in steps]
        mean, std = np.mean(rewards), np.std(rewards)+1e-8
        for step in steps: step.reward = (step.reward-mean)/(std if self.z_score else 1)
        return steps

class NormalizeRewardsByEnv(SamplingRewardTransform):
    def __init__(self, z_score: bool = False): self.z_score = z_score 
    def __call__(self, steps: List[Step], env_id: Optional[str] = None) -> List[Step]:
        env_buckets = defaultdict(list)
        for step in steps: env_buckets[step.env_id].append(step) # bucket by env
        for env_steps in env_buckets.values():
            r = np.asarray([s.reward for s in env_steps], dtype=np.float32)
            normed = ((r-r.mean())/r.std()+1e-8) if self.z_score else r-r.mean()
            for s, nr in zip(env_steps, normed): s.reward = float(nr) # write back
        return steps

# Episode Training Batch Reward
class EpisodeSamplingRewardTransform:
    def __call__(self, episodes: List[List[Step]], env_id: Optional[str] = None) -> List[List[Step]]: raise NotImplementedError

class ComposeEpisodeSamplingRewardTransforms:
    def __init__(self, transforms: List[EpisodeSamplingRewardTransform]): self.transforms = transforms
    def __call__(self, episodes: List[List[Step]], env_id: Optional[str] = None) -> List[List[Step]]:
        for transform in self.transforms: episodes = transform(episodes, env_id)
        return episodes

class GroupRelativeAdvantage(EpisodeSamplingRewardTransform):
    def __call__(self, episodes: List[List[Step]], env_id: Optional[str] = None) -> List[List[Step]]:
        print([len(episode) for episode in episodes])
        episode_returns = np.array([episode[-1].reward for episode in episodes])
        mean_return = episode_returns.mean(); std_return = episode_returns.std()+1e-8
        for episode in episodes:
            for step in episode: step.reward = (step.reward - mean_return) / std_return
        return episodes
