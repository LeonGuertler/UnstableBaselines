import random
from typing import Protocol, List, Any, Dict
from unstable._types import TrainEnvSpec, EvalEnvSpec, GameInformation


class BaseEnvSampler: 
    def __init__(self, train_env_specs: List[TrainEnvSpec], eval_env_specs: List[EvalEnvSpec]|None = None, rng_seed: int|None = 489):
        assert train_env_specs, "Need at least one training environment"
        self._train, self._eval = train_env_specs, eval_env_specs
        self._rng = random.Random(rng_seed)
    def env_list(self) -> str: return ",".join([tes.env_id for tes in self._train])
    def sample(self, kind: str = "train") -> TrainEnvSpec | EvalEnvSpec: ...
    def update(self, env_id: str, avg_actor_reward: float, avg_opponent_reward: float|None) -> None: ...

class UniformRandomEnvSampler(BaseEnvSampler):
    def sample(self, kind: str = "train") -> TrainEnvSpec | EvalEnvSpec:
        if kind not in ("train", "eval"): raise ValueError(f"kind must be 'train' or 'eval' (got {kind!r})")
        return self._rng.choice(self._train if kind == "train" else self._eval)
    def update(self, env_id: str, avg_actor_reward: float, avg_opponent_reward: float|None) -> None: return # No-op for the uniform strategy - but the hook is here so the scheduler can call it unconditionally.

class CurriculumEnvSampler(BaseEnvSampler):
    def __init__(self, train_env_specs: List[TrainEnvSpec], eval_env_specs: List[EvalEnvSpec] | None = None, rng_seed: int | None = 489,
                 reward_threshold: float = 0.2, episode_threshold: int = 100, base_prob_weight: float = 0.7, decay_factor: float = 0.2):
        super().__init__(train_env_specs, eval_env_specs, rng_seed)

        self.env_id_to_spec = {env.env_id: env for env in train_env_specs}
        self.env_chains = self._infer_env_chains_from_registry()
        self.chain_progress = {chain: 0 for chain in self.env_chains} 
        self.total_reward = {env.env_id: 0.0 for env in train_env_specs}
        self.num_episodes = {env.env_id: 0 for env in train_env_specs}
        self.reward_threshold = reward_threshold
        self.episode_threshold = episode_threshold
        self.base_prob_weight = base_prob_weight
        self.decay_factor = decay_factor
        self.passed_envs = set()

    def _infer_env_chains_from_registry(self) -> list[tuple[str]]:
        from collections import defaultdict
        from textarena.envs import registration
        allowed_env_ids = set(self.env_id_to_spec.keys())
        chains = defaultdict(list)
        for env_id in registration.ENV_REGISTRY.keys():
            if env_id.endswith("-train") and env_id in allowed_env_ids:
                prefix = "-".join(env_id.split("-")[:2])
                chains[prefix].append(env_id)
        return [tuple(chain) for chain in chains.values() if len(chain) > 1]

    def _calculate_chain_probabilities(self, chain: tuple) -> Dict[str, float]:
        focus_on = self.chain_progress[chain]
        probs = {}
        for i, env_id in enumerate(chain):
            if i < focus_on: probs[env_id] = self.base_prob_weight * (self.decay_factor ** (focus_on - i))
            elif i == focus_on: probs[env_id] = self.base_prob_weight
            else: probs[env_id] = self.base_prob_weight * (self.decay_factor ** (i - focus_on))
        return probs
    
    def _get_sampling_probabilities(self) -> Dict[str, float]:
        all_probs = {}
        chained_envs = set()
        for chain in self.env_chains: chain_probs = self._calculate_chain_probabilities(chain); all_probs.update(chain_probs); chained_envs.update(chain)
        return all_probs

    def _log_current_state(self):
        """Log current state for debugging."""
        print(f"[RewardSampler] Current chain progress: {self.chain_progress}")
        print(f"[RewardSampler] Passed environments: {self.passed_envs}")
        for chain in self.env_chains: chain_probs = self._calculate_chain_probabilities(chain); print(f"[RewardSampler] Chain {chain} probabilities: {chain_probs}")

    def sample(self, kind: str = "train") -> TrainEnvSpec | EvalEnvSpec:
        if kind == "eval":
            return self._rng.choice(self._eval)
        
        prob_dict = self._get_sampling_probabilities()
        if not prob_dict: print("[RewardSampler] WARNING: No probabilities calculated, falling back to uniform sampling."); return self._rng.choice(list(self.env_id_to_spec.values()))
        
        env_ids, weights = list(prob_dict.keys()), list(prob_dict.values())
        total_weight = sum(weights)
        if total_weight > 0: weights = [w / total_weight for w in weights]
        else: print("[RewardSampler] WARNING: total_weight is not > 0."); weights = [1.0 / len(weights)] * len(weights) # another fallback

        env_id = self._rng.choices(env_ids, weights=weights)[0]
        print(f"[RewardSampler] Sample {env_id} with probability {prob_dict[env_id]:.3f}")
        self._log_current_state()
        
        return self.env_id_to_spec[env_id]
    
    def _update_chain_progression(self, passed_env_id: str):
        for chain in self.env_chains:
            if passed_env_id in chain:
                focus_on = self.chain_progress[chain]
                env_index = chain.index(passed_env_id)
                if env_index == focus_on:
                    new_focus = min(focus_on + 1, len(chain) - 1)
                    self.chain_progress[chain] = new_focus
                    print(f"[RewardSampler] Chain {chain} progressed from index {focus_on} -> index {new_focus}")
                    if new_focus < len(chain): print(f"[RewardSampler] New focus: {chain[new_focus]}")
                    else: print(f"[RewardSampler] Chain {chain} completed!")
                break

    def update(self, env_id: str, avg_actor_reward: float | None, avg_opponent_reward: float | None) -> None:
        if avg_actor_reward is None:
            return

        self.total_reward[env_id] += avg_actor_reward
        self.num_episodes[env_id] += 1

        avg_reward = self.total_reward[env_id] / self.num_episodes[env_id]
        num_plays = self.num_episodes[env_id]

        print(f"[RewardSampler] Update for {env_id} → avg_reward={avg_reward:.3f}, episodes={num_plays}")

        if avg_reward >= self.reward_threshold and num_plays >= self.episode_threshold:
            if env_id not in self.passed_envs:
                self.passed_envs.add(env_id)
                print(f"[RewardSampler] {env_id} passed criteria!")
                self._update_chain_progression(env_id)