import numpy as np
from typing import List
from trajectory_buffer import Step

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
            step.reward = (step.reward - mean) #/ std
        return steps


