from typing import Dict, List, Tuple, Union
FINAL_REWARDS_FORMAT = Dict[int, Union[float, int]]

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

class RoleAdvantageFormatter(FinalRewardTransform):
    def __init__(self, role_advantage: Dict[int, float]={0:0.0, 1:0.0}, tau: float=0.01):
        self.role_advantage = role_advantage
        self.tau = tau
    
    def __call__(self, x: FINAL_REWARDS_FORMAT) -> FINAL_REWARDS_FORMAT:
        x = x.copy()
        for pid in x.keys():
            self.role_advantage[pid] = (1-self.tau) * self.role_advantage[pid] + self.tau * x[pid]
            x[pid] -= self.role_advantage[pid]
        return x