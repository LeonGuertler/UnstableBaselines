import random
from typing import Protocol, List, Any
from unstable._types import TrainEnvSpec, EvalEnvSpec, GameInformation


class BaseEnvSampler: 
    def __init__(self, train_env_specs: List[TrainEnvSpec], eval_env_specs: List[EvalEnvSpec]|None = None, rng_seed: int|None = 489):
        assert train_env_specs, "Need at least one training environment"
        self._train, self._eval = train_env_specs, eval_env_specs
        self._rng = random.Random(rng_seed)
    def env_list(self) -> str: return ",".join([tes.env_id for tes in self._train])
    def sample(self, kind: str = "train") -> TrainEnvSpec | EvalEnvSpec: ...
    def update(self, avg_actor_reward: float, avg_opponent_reward: float|None) -> None: ...

class UniformRandomEnvSampler(BaseEnvSampler):
    def sample(self, kind: str = "train") -> TrainEnvSpec | EvalEnvSpec:
        if kind not in ("train", "eval"): raise ValueError(f"kind must be 'train' or 'eval' (got {kind!r})")
        return self._rng.choice(self._train if kind == "train" else self._eval)
    def update(self, avg_actor_reward: float, avg_opponent_reward: float|None) -> None: return # No-op for the uniform strategy - but the hook is here so the scheduler can call it unconditionally.

class RewardSampler(BaseEnvSampler):
    def __init__(self, train_env_specs: List[TrainEnvSpec], env_chains: List[List[str]], eval_env_specs: List[EvalEnvSpec] | None = None, rng_seed: int | None = 489):
        super().__init__(train_env_specs, eval_env_specs, rng_seed)

        self.env_chains = [tuple(chain) for chain in env_chains] # [(A1, A2), (B1, B2, B3)] 
        self.env_id_to_spec = {env.env_id: env for env in train_env_specs} # e.g. {...-v0: TrainEnvSpec(..), ... }
        self.chain_indices = {chain: 0 for chain in self.env_chains}
        self.total_reward = {env.env_id: 0.0 for env in train_env_specs}
        self.num_episodes = {env.env_id: 0 for env in train_env_specs}

    def sample(self, kind: str = "train") -> TrainEnvSpec | EvalEnvSpec:
        if kind == "eval":
            return self._rng.choice(self._eval)

        candidates = []
        for chain in self.env_chains:
            idx = self.chain_indices[chain]
            if idx < len(chain):
                candidates.append(chain[idx])

        if not candidates:
            candidates = list(self.env_id_to_spec.keys())

        env_id = self._rng.choice(candidates)
        print(f"[RewardSampler] Sampled environment: {env_id}")
        return self.env_id_to_spec[env_id]


    def update(self, env_id: str, avg_actor_reward: float | None, avg_opponent_reward: float | None) -> None:
        if avg_actor_reward is None:
            return

        self.total_reward[env_id] += avg_actor_reward
        self.num_episodes[env_id] += 1

        avg_reward = self.total_reward[env_id] / self.num_episodes[env_id]
        num_plays = self.num_episodes[env_id]

        print(f"[RewardSampler] Update for {env_id} → avg_reward={avg_reward:.3f}, episodes={num_plays}")

        for chain in self.env_chains:
            idx = self.chain_indices[chain]
            if idx >= len(chain):
                continue
            current_env_id = chain[idx]
            if current_env_id == env_id and avg_reward >= 0.2 and num_plays >= 100:
                self.chain_indices[chain] = min(idx + 1, len(chain))
                print(f"[RewardSampler] → Progressed {chain} to {self.chain_indices[chain]} (unlocked {chain[self.chain_indices[chain]-1]})")

