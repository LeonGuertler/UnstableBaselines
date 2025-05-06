import numpy as np
from typing import Dict, List, Tuple, Union
from trajectory_buffer import Trajectory, Step

FINAL_REWARDS_FORMAT = Dict[int, Union[float, int]]


# Final reward Transformations
class FinalRewardTransform:
    def __call__(self, x: FINAL_REWARDS_FORMAT) -> FINAL_REWARDS_FORMAT:
        raise NotImplementedError

class ComposeFinalRewardTransforms:
    def __init__(self, transforms: List[FinalRewardTransform]):
        self.transforms = transforms

    def __repr__(self):
        return f"{self.__class__.__name__}({self.transforms})"

    def register(self, transform: FinalRewardTransform):
        self.transforms.append(transform)

    def __call__(self, x: FINAL_REWARDS_FORMAT) -> FINAL_REWARDS_FORMAT:
        for t in self.transforms:
            x = t(x)
        return x 


class WinDrawLossFormatter(FinalRewardTransform):
    def __init__(self, win_reward: float=1.0, draw_reward: float=0.0, loss_reward: float=-1.0):
        self.win_reward = win_reward
        self.draw_reward = draw_reward
        self.loss_reward = loss_reward

    def __call__(self, x: FINAL_REWARDS_FORMAT) -> FINAL_REWARDS_FORMAT:
        x = x.copy()
        assert len(x)==2, f"WinDrawLossFormatter only works for two-player games. Recieved final reward: {x}"
        if x[0]<x[1]:
            x[0] = self.loss_reward
            x[1] = self.win_reward
        elif x[0]>x[1]:
            x[0] = self.win_reward
            x[1] = self.loss_reward
        else:
            x[0] = self.draw_reward
            x[1] = self.draw_reward
        return x

class RoleAdvantage(FinalRewardTransform):
    def __init__(self, role_advantage: Dict[int, float]={0:0.0, 1:0.0}, tau: float=0.01):
        self.role_advantage = role_advantage
        self.tau = tau
    
    def __call__(self, x: FINAL_REWARDS_FORMAT) -> FINAL_REWARDS_FORMAT:
        x = x.copy()
        for pid in x.keys():
            self.role_advantage[pid] = (1-self.tau) * self.role_advantage[pid] + self.tau * x[pid]
            x[pid] -= self.role_advantage[pid]
        return x


# Step reward Transformations
class StepRewardTransform:
    def __call__(self, trajectory: Trajectory, step_index: int, base_reward: float) -> float:
        raise NotImplementedError

class ComposeStepRewardTransforms:
    def __init__(self, transforms: List[StepRewardTransform]):
        self.transforms = transforms

    def __repr__(self):
        return f"{self.__class__.__name__}({self.transforms})"

    def register(self, transform: StepRewardTransform):
        self.transforms.append(transform)

    def __call__(self, trajectory: Trajectory, step_index: int, base_reward: float) -> float:
        for t in self.transforms:
            base_reward = t(trajectory, step_index, base_reward)
        return base_reward


class RewardForThink(StepRewardTransform):
    def __init__(self, reward: float=0, penalty: float=0):
        self.reward = reward
        self.penalty = penalty

    def __call__(self, trajectory: Trajectory, step_index: int, base_reward: float) -> float:
        if trajectory.format_feedbacks[step_index].get("has_think"):
            base_reward += self.reward
        else:
            base_reward += self.penalty
        return base_reward


class PenaltyForInvalidMove(StepRewardTransform):
    def __init__(self, reward: float=0, penalty: float=0):
        self.reward = reward
        self.penalty = penalty

    def __call__(self, trajectory: Trajectory, step_index: int, base_reward: float) -> float:
        if trajectory.format_feedbacks[step_index].get("invalid_move"):
            base_reward += self.penalty
        else:
            base_reward += self.reward
        return base_reward


# Sampling reward Transformations
class SamplingRewardTransform:
    def __call__(self, x: List[Step]) -> List[Step]:
        raise NotImplementedError

class ComposeSamplingRewardTransforms:
    def __init__(self, transforms: List[SamplingRewardTransform]):
        self.transforms = transforms

    def __repr__(self):
        return f"{self.__class__.__name__}({self.transforms})"

    def register(self, transform: SamplingRewardTransform):
        self.transforms.append(transform)

    def __call__(self, x: List[Step]) -> List[Step]:
        for t in self.transforms:
            x = t(x)
        return x

class NormalizeRewards(SamplingRewardTransform):
    def __call__(self, steps: List[Step]) -> List[Step]:
        rewards = [step.reward for step in steps]
        mean = np.mean(rewards)
        std = np.std(rewards) + 1e-8  # avoid divide-by-zero

        for step in steps:
            step.reward = (step.reward - mean) / std
        return steps







