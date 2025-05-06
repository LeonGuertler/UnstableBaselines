from typing import List
from trajectory_buffer import Trajectory


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