import random, math
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
                 window_size: int = 50, min_episodes: int = 0, temperature: float = 0.03, smoothing_factor: float = 0.01,
                 forward_boost_factor: float = 3.0, decay_factor: float = 0.4):
        super().__init__(train_env_specs, eval_env_specs, rng_seed)
        self.env_id_to_spec = {env.env_id: env for env in train_env_specs}
        self.env_chains = self._infer_env_chains_from_registry()
        self.window_size = window_size
        self.min_episodes = min_episodes
        self.temperature = temperature
        self.smoothing_factor = smoothing_factor
        self.forward_boost_factor = forward_boost_factor
        self.decay_factor = decay_factor
        
        self.reward_history = {env.env_id: [] for env in train_env_specs}
        self.num_episodes = {env.env_id: 0 for env in train_env_specs}
        
        self.chain_progress = {chain: 0 for chain in self.env_chains}

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

    def _calculate_rate_of_change(self, env_id: str) -> float:
        history = self.reward_history[env_id]
        if len(history) < self.min_episodes: return 0.0
        recent_rewards = history[-self.window_size:]
        if len(recent_rewards) < 2: return 0.0
        n = len(recent_rewards)
        x_values = list(range(n))
        x_mean = sum(x_values) / n
        y_mean = sum(recent_rewards) / n
        num = sum((x_values[i] - x_mean) * (recent_rewards[i] - y_mean) for i in range(n))
        den = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        if abs(den) < 1e-8: return 0.0
        slope = num / den
        return slope / (1.0 + self.smoothing_factor)

    def _update_chain_progress(self):
        for chain in self.env_chains:
            max_rate = -float('inf')
            best_idx = -1
            for i, env_id in enumerate(chain):
                rate = self._calculate_rate_of_change(env_id)
                if rate > max_rate:
                    max_rate = rate
                    best_idx = i
            if best_idx >= 0:
                self.chain_progress[chain] = best_idx

    def _softmax(self, values: List[float]) -> List[float]:
        if not values: return []
        scaled_values = [v / self.temperature for v in values]
        max_val = max(scaled_values)
        exp_values = [math.exp(v - max_val) for v in scaled_values]
        sum_exp = sum(exp_values)
        if sum_exp == 0: return [1.0 / len(values)] * len(values)
        return [exp_val / sum_exp for exp_val in exp_values]

    def _calculate_forward_looking_scores(self, chain: tuple) -> Dict[str, float]:
        """Calculate raw scores for environments based on forward-looking curriculum logic."""
        focus_env_idx = self.chain_progress[chain] # for the chain, get the current 'focus' index that had the best rate of change.
        focus_env_id = chain[focus_env_idx]
        focus_rate = self._calculate_rate_of_change(focus_env_id)
        scores = {}
        for i, env_id in enumerate(chain): scores[env_id] = 1.0
        
        if focus_rate > 0:
            # Positive rate: boost harder environments proportionally to the rate
            for i in range(focus_env_idx + 1, len(chain)):
                distance = i - focus_env_idx
                boost = self.forward_boost_factor * focus_rate * (self.decay_factor ** (distance - 1))
                scores[chain[i]] += boost
            
            # Slightly reduce score of easier environments
            for i in range(focus_env_idx):
                distance = focus_env_idx - i
                reduction = 0.5 * focus_rate * (self.decay_factor ** (distance - 1))
                scores[chain[i]] -= reduction
        
        elif focus_rate <= 0:
            # Negative rate: boost easier environments proportionally to the absolute rate
            for i in range(focus_env_idx):
                distance = focus_env_idx - i
                boost = self.forward_boost_factor * abs(focus_rate) * (self.decay_factor ** (distance - 1))
                scores[chain[i]] += boost
            
            # Give current environment some boost too (recovery)
            scores[focus_env_id] += abs(focus_rate) * 0.5
            
            # Reduce score of harder environments
            for i in range(focus_env_idx + 1, len(chain)):
                distance = i - focus_env_idx
                reduction = abs(focus_rate) * self.forward_boost_factor * (self.decay_factor ** (distance - 1))
                scores[chain[i]] -= reduction
        
        # If rate is near zero, scores remain mostly at base values (neutral state)
        
        return scores

    def _get_sampling_probs(self) -> Dict[str, float]:
        self._update_chain_progress()
        chain_scores = []
        for chain in self.env_chains: chain_scores.append(max(self._calculate_rate_of_change(eid) for eid in chain))
        chain_probs = self._softmax(chain_scores)
        env_probs = {}
        for chain, p_chain in zip(self.env_chains, chain_probs):
            depth_scores = self._calculate_forward_looking_scores(chain)
            depth_vals = [depth_scores[eid] for eid in chain]
            depth_probs = self._softmax(depth_vals)
            for eid, p_env in zip(chain, depth_probs):
                env_probs[eid] = env_probs.get(eid, 0.0) + p_chain * p_env
        total = sum(env_probs.values())
        if total == 0: uniform = 1.0 / len(env_probs); return {eid: uniform for eid in env_probs}
        return {eid: p / total for eid, p in env_probs.items()}

    def _log_current_state(self):
        print(f"[ExplorativeCurriculumEnvSampler] Environment chains: {self.env_chains}")
        print(f"[ExplorativeCurriculumEnvSampler] Chain progress: {self.chain_progress}")
        
        for chain in self.env_chains:
            print(f"[ExplorativeCurriculumEnvSampler] Chain {chain}:")
            focus_idx = self.chain_progress[chain]
            chain_scores = self._calculate_forward_looking_scores(chain)
            chain_score_values = [chain_scores[env_id] for env_id in chain]
            chain_probs = self._softmax(chain_score_values)
            
            for i, env_id in enumerate(chain):
                rate = self._calculate_rate_of_change(env_id)
                episodes = self.num_episodes[env_id]
                recent_avg = 0.0
                if self.reward_history[env_id]:
                    recent_rewards = self.reward_history[env_id][-min(10, len(self.reward_history[env_id])):]
                    recent_avg = sum(recent_rewards) / len(recent_rewards)
                
                status = ""
                if i == focus_idx:
                    if rate > 0:
                        status = f"FOCUS (PROGRESSING +{rate:.4f})"
                    elif rate <= 0:
                        status = f"FOCUS (NO GROWTH {rate:.4f})"
                
                print(f"[ExplorativeCurriculumEnvSampler]   {env_id}: rate={rate:.4f}, episodes={episodes}, "
                      f"recent_avg_rewards={recent_avg:.3f}, chain_prob={chain_probs[i]:.3f}, score={chain_scores[env_id]:.3f} {status}")
        
        prob_dict = self._get_sampling_probs()
        print(f"[ExplorativeCurriculumEnvSampler] Final sampling probabilities (softmax within chains, normalized): {prob_dict}")

    def sample(self, kind: str = "train") -> TrainEnvSpec | EvalEnvSpec:
        if kind == "eval":
            return self._rng.choice(self._eval)
        
        prob_dict = self._get_sampling_probs()
        if not prob_dict:
            print("[ExplorativeCurriculumEnvSampler] WARNING: No probabilities calculated, falling back to uniform sampling.")
            return self._rng.choice(list(self.env_id_to_spec.values()))
        
        env_ids = list(prob_dict.keys())
        weights = list(prob_dict.values())
        total_weight = sum(weights)
        
        if total_weight <= 0:
            print("[ExplorativeCurriculumEnvSampler] WARNING: Invalid weights, falling back to uniform sampling.")
            weights = [1.0 / len(weights)] * len(weights)
        
        env_id = self._rng.choices(env_ids, weights=weights)[0]
        print(f"[ExplorativeCurriculumEnvSampler] Sampled {env_id} with probability {prob_dict[env_id]:.3f}")
        self._log_current_state()
        
        return self.env_id_to_spec[env_id]

    def update(self, env_id: str, avg_actor_reward: float | None, avg_opponent_reward: float | None) -> None:
        if avg_actor_reward is None: return
        self.reward_history[env_id].append(avg_actor_reward)
        self.num_episodes[env_id] += 1
        if len(self.reward_history[env_id]) > self.window_size: self.reward_history[env_id] = self.reward_history[env_id][-self.window_size:] 
        rate_of_change = self._calculate_rate_of_change(env_id)
        print(f"[ExplorativeCurriculumEnvSampler] Update for {env_id} → reward={avg_actor_reward:.3f}, "
              f"episodes={self.num_episodes[env_id]}, rate_of_change={rate_of_change:.4f}")