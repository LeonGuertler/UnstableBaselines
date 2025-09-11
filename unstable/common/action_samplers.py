import math
import random
from typing import List
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from unstable.common._types import Action


class BaseActionSampler:
    def __init__(self, agent) -> None:
        self._agent = agent

    def __call__(self, observation: str) -> Action:
        return self.sample_action(observation)

    def sample_action(self, observation: str) -> Action:
        raw, extracted, format_feedback = None, None, None
        if hasattr(self._agent, "act_full"): raw, extracted, _, format_feedback = self._agent.act_full(observation)
        else: raw = extracted = self._agent(observation)
        return Action(extracted_action=extracted, raw=raw, format_feedback=format_feedback, sampler_info={})


class MajorityVotingActionSampler(BaseActionSampler):
    def __init__(self, agent, k: int = 10) -> None:
        if k <= 0: raise ValueError(f"k must be a positive integer (got {k})")
        super().__init__(agent)
        self._k = k

    def sample_action(self, observation: str) -> Action:
        actions: List[str] = []; raw_actions: List[str] = []; format_feedbacks: List[str] = []
        with ThreadPoolExecutor(max_workers=self._k) as executor:
            if hasattr(self._agent, "act_full"): futures = [executor.submit(self._agent.act_full, observation) for _ in range(self._k)]
            else: futures = [executor.submit(self._agent, observation) for _ in range(self._k)]
            for fut in as_completed(futures):
                if hasattr(self._agent, "act_full"): raw, extracted, _, format_feedback = fut.result()
                else: raw = extracted = fut.result(); format_feedback = None
                actions.append(extracted); raw_actions.append(raw); format_feedbacks.append(format_feedback)
        counts = Counter(actions)
        most_common = [action for action, cnt in counts.most_common() if cnt == counts.most_common()[0][1]]
        winner = actions.index(random.choice(most_common))
        return Action(extracted_action=actions[winner], raw=raw_actions[winner], format_feedback=format_feedbacks[winner], sampler_info={'confidence': counts[actions[winner]] / self._k, 'entropy': self._entropy(counts)})

    def _entropy(self, counts: Counter) -> float: return -sum([p * math.log(p) for p in [counts[action] / self._k for action in counts]])
