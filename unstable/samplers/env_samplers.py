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
                 reward_threshold: float = 0.2, episode_threshold: int = 100, base_prob_weight: float = 0.7, decay_factor: float = 0.2,
                 plateau_patience: int | None = 500, plateau_window: int = 100, plateau_threshold: float = 0.01):
        """
        Environment sampler that progresses through chains based on reward thresholds or plateau detection.
        
        Args:
            plateau_patience: If specified, will progress after this many episodes of reward plateau
            plateau_window: Number of episodes to use for plateau detection (default: 50)
            plateau_threshold: Maximum improvement rate to consider as plateau (default: 0.01 = 1%)
        """
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
        
        # Plateau detection mechanism
        self.plateau_patience = plateau_patience
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold
        self.plateau_progressed = set()  # Track which envs were progressed due to plateau
        
        # Track recent rewards for plateau detection
        self.recent_rewards = {env.env_id: [] for env in train_env_specs}
        self.plateau_episodes = {env.env_id: 0 for env in train_env_specs}  # Episodes since plateau started

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

    def _is_reward_plateaued(self, env_id: str) -> bool:
        """Check if reward has plateaued for this environment."""
        if len(self.recent_rewards[env_id]) < self.plateau_window:
            return False
        
        recent = self.recent_rewards[env_id][-self.plateau_window:]
        
        # Split into first and second half to check improvement
        mid = len(recent) // 2
        first_half = recent[:mid]
        second_half = recent[mid:]
        
        if len(first_half) == 0 or len(second_half) == 0:
            return False
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        # Calculate improvement rate
        if abs(first_avg) < 1e-6:  # Avoid division by zero
            improvement_rate = second_avg - first_avg
        else:
            improvement_rate = (second_avg - first_avg) / abs(first_avg)
        
        # Consider plateaued if improvement is below threshold
        return improvement_rate < self.plateau_threshold

    def _log_current_state(self):
        """Log current state for debugging."""
        print(f"[RewardSampler] Current chain progress: {self.chain_progress}")
        print(f"[RewardSampler] Passed environments: {self.passed_envs}")
        if self.plateau_patience:
            print(f"[RewardSampler] Plateau-progressed environments: {self.plateau_progressed}")
            # Show plateau status for current focus environments
            for chain in self.env_chains:
                focus_env = chain[self.chain_progress[chain]]
                if focus_env in self.plateau_episodes:
                    plateau_eps = self.plateau_episodes[focus_env]
                    print(f"[RewardSampler] {focus_env} plateau episodes: {plateau_eps}")
        for chain in self.env_chains: 
            chain_probs = self._calculate_chain_probabilities(chain)
            print(f"[RewardSampler] Chain {chain} probabilities: {chain_probs}")

    def sample(self, kind: str = "train") -> TrainEnvSpec | EvalEnvSpec:
        if kind == "eval":
            return self._rng.choice(self._eval)
        
        prob_dict = self._get_sampling_probabilities()
        if not prob_dict: 
            print("[RewardSampler] WARNING: No probabilities calculated, falling back to uniform sampling.")
            return self._rng.choice(list(self.env_id_to_spec.values()))
        
        env_ids, weights = list(prob_dict.keys()), list(prob_dict.values())
        total_weight = sum(weights)
        if total_weight > 0: 
            weights = [w / total_weight for w in weights]
        else: 
            print("[RewardSampler] WARNING: total_weight is not > 0.")
            weights = [1.0 / len(weights)] * len(weights)

        env_id = self._rng.choices(env_ids, weights=weights)[0]
        print(f"[RewardSampler] Sample {env_id} with probability {prob_dict[env_id]:.3f}")
        self._log_current_state()
        
        return self.env_id_to_spec[env_id]
    
    def _update_chain_progression(self, passed_env_id: str, reason: str = "reward_threshold"):
        for chain in self.env_chains:
            if passed_env_id in chain:
                focus_on = self.chain_progress[chain]
                env_index = chain.index(passed_env_id)
                if env_index == focus_on:
                    new_focus = min(focus_on + 1, len(chain) - 1)
                    self.chain_progress[chain] = new_focus
                    print(f"[RewardSampler] Chain {chain} progressed from index {focus_on} -> index {new_focus} (reason: {reason})")
                    if new_focus < len(chain): 
                        print(f"[RewardSampler] New focus: {chain[new_focus]}")
                        # Reset plateau tracking for new environment
                        new_env = chain[new_focus]
                        self.plateau_episodes[new_env] = 0
                    else: 
                        print(f"[RewardSampler] Chain {chain} completed!")
                break

    def update(self, env_id: str, avg_actor_reward: float | None, avg_opponent_reward: float | None) -> None:
        if avg_actor_reward is None:
            return

        self.total_reward[env_id] += avg_actor_reward
        self.num_episodes[env_id] += 1

        # Update recent rewards for plateau detection
        self.recent_rewards[env_id].append(avg_actor_reward)
        # Keep only recent rewards (2x window size for better trend analysis)
        if len(self.recent_rewards[env_id]) > self.plateau_window:
            self.recent_rewards[env_id] = self.recent_rewards[env_id][-self.plateau_window:]

        avg_reward = self.total_reward[env_id] / self.num_episodes[env_id]
        num_plays = self.num_episodes[env_id]

        # Update plateau tracking
        if self.plateau_patience is not None and len(self.recent_rewards[env_id]) >= self.plateau_window:
            if self._is_reward_plateaued(env_id):
                self.plateau_episodes[env_id] += 1
            else:
                self.plateau_episodes[env_id] = 0  # Reset if not plateaued

        print(f"[RewardSampler] Update for {env_id} → avg_reward={avg_reward:.3f}, episodes={num_plays}")
        if self.plateau_patience is not None and len(self.recent_rewards[env_id]) >= self.plateau_window:
            print(f"[RewardSampler] {env_id} plateau episodes: {self.plateau_episodes[env_id]}")

        # Check if environment should progress
        should_progress = False
        reason = ""
        
        # Original reward-based progression
        if avg_reward >= self.reward_threshold and num_plays >= self.episode_threshold:
            should_progress = True
            reason = "reward_threshold"
        
        # Plateau-based progression
        elif (self.plateau_patience is not None and 
              self.plateau_episodes[env_id] >= self.plateau_patience and 
              env_id not in self.passed_envs):
            should_progress = True
            reason = "plateau"
            self.plateau_progressed.add(env_id)
            print(f"[RewardSampler] {env_id} progressed due to reward plateau after {self.plateau_episodes[env_id]} plateau episodes")

        if should_progress and env_id not in self.passed_envs:
            self.passed_envs.add(env_id)
            print(f"[RewardSampler] {env_id} passed criteria! (reason: {reason})")
            self._update_chain_progression(env_id, reason)