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
                 reward_threshold: float = 0.5, episode_threshold: int = 1000, base_prob_weight: float = 0.7, decay_factor: float = 0.5,
                 stuck_duration: int | None = 200, stuck_window: int = 50, stuck_threshold: float = 0.05):
        """
        Environment sampler that progresses through chains based on reward thresholds or stuck detection.
        
        Args:
            stuck_duration: If specified, will progress after this many episodes of reward stuck
            stuck_window: Number of episodes to use for stuck detection (default: 50)
            stuck_threshold: Maximum improvement rate to consider as stuck (default: 0.01 = 1%)
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
        
        # Stuck detection mechanism
        self.stuck_duration = stuck_duration
        self.stuck_window = stuck_window
        self.stuck_threshold = stuck_threshold
        self.stuck_progressed = set()  # Track which envs were progressed due to stuck
        self.recent_rewards = {env.env_id: [] for env in train_env_specs}
        self.stuck_episodes = {env.env_id: 0 for env in train_env_specs}  # Episodes since stuck started

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
    
    def _get_sampling_probs(self) -> Dict[str, float]:
        all_probs = {}
        chained_envs = set()
        for chain in self.env_chains: chain_probs = self._calculate_chain_probabilities(chain); all_probs.update(chain_probs); chained_envs.update(chain)
        return all_probs

    def _is_reward_stucked(self, env_id: str) -> bool:
        """Check if reward has stucked for this environment."""
        if len(self.recent_rewards[env_id]) < self.stuck_window: return False
        
        recent = self.recent_rewards[env_id][-self.stuck_window:]
        mid = len(recent) // 2
        if len(recent[:mid]) == 0 or len(recent[mid:]) == 0: return False
        first_avg, second_avg = sum(recent[:mid]) / len(recent[:mid]), sum(recent[mid:]) / len(recent[mid:])
        
        if abs(first_avg) < 1e-6: improvement_rate = second_avg - first_avg
        else: improvement_rate = (second_avg - first_avg) / abs(first_avg)
        return improvement_rate < self.stuck_threshold

    def _log_current_state(self):
        """Log current state for debugging."""
        print(f"[RewardSampler] Current chain progress: {self.chain_progress}")
        print(f"[RewardSampler] Passed environments: {self.passed_envs}")
        if self.stuck_duration:
            print(f"[RewardSampler] Stuck-progressed environments: {self.stuck_progressed}")
            # Show stuck status for current focus environments
            for chain in self.env_chains:
                focus_env = chain[self.chain_progress[chain]]
                if focus_env in self.stuck_episodes:
                    stuck_eps = self.stuck_episodes[focus_env]
                    print(f"[RewardSampler] {focus_env} stuck episodes: {stuck_eps}")
        for chain in self.env_chains: 
            chain_probs = self._calculate_chain_probabilities(chain)
            print(f"[RewardSampler] Chain {chain} probabilities: {chain_probs}")

    def sample(self, kind: str = "train") -> TrainEnvSpec | EvalEnvSpec:
        if kind == "eval":
            return self._rng.choice(self._eval)
        
        prob_dict = self._get_sampling_probs()
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
                        # Reset stuck tracking for new environment
                        new_env = chain[new_focus]
                        self.stuck_episodes[new_env] = 0
                    else: 
                        print(f"[RewardSampler] Chain {chain} completed!")
                break

    def update(self, env_id: str, avg_actor_reward: float | None, avg_opponent_reward: float | None) -> None:
        if avg_actor_reward is None:
            return

        self.total_reward[env_id] += avg_actor_reward
        self.num_episodes[env_id] += 1

        # Update recent rewards for stuck detection
        self.recent_rewards[env_id].append(avg_actor_reward)
        if len(self.recent_rewards[env_id]) > self.stuck_window:
            self.recent_rewards[env_id] = self.recent_rewards[env_id][-self.stuck_window:]

        avg_reward = self.total_reward[env_id] / self.num_episodes[env_id]
        num_plays = self.num_episodes[env_id]
        print(f"[RewardSampler] Update for {env_id} → avg_reward={avg_reward:.3f}, episodes={num_plays}")

        # Update stuck tracking
        if self.stuck_duration is not None and len(self.recent_rewards[env_id]) >= self.stuck_window:
            if self._is_reward_stucked(env_id):
                self.stuck_episodes[env_id] += 1
            else:
                self.stuck_episodes[env_id] = 0  # Reset if not stucked
            print(f"[RewardSampler] {env_id} stuck episodes: {self.stuck_episodes[env_id]}")
        
        # Checks if need progressing
        if avg_reward >= self.reward_threshold and num_plays >= self.episode_threshold:
            should_progress, reason = True, "reward_threshold"
        elif (self.stuck_duration is not None and self.stuck_episodes[env_id] >= self.stuck_duration and num_plays >= self.episode_threshold and env_id not in self.passed_envs):
            should_progress, reason = True, "stuck"
            self.stuck_progressed.add(env_id)
            print(f"[RewardSampler] {env_id} progressed due to reward stuck after {self.stuck_episodes[env_id]} stuck episodes")
        else:
            should_progress, reason = False, ""

        if should_progress and env_id not in self.passed_envs:
            self.passed_envs.add(env_id)
            print(f"[RewardSampler] {env_id} passed criteria! (reason: {reason})")
            self._update_chain_progression(env_id, reason)


class ExploitativeCurriculumEnvSampler(BaseEnvSampler):
    def __init__(self, train_env_specs: List[TrainEnvSpec], eval_env_specs: List[EvalEnvSpec] | None = None, rng_seed: int | None = 489,
                 window_size: int = 50, min_episodes: int = 50, temperature: float = 0.001, smoothing_factor: float = 0.01):
        super().__init__(train_env_specs, eval_env_specs, rng_seed)
        self.env_id_to_spec = {env.env_id: env for env in train_env_specs}
        self.env_chains = self._infer_env_chains_from_registry()
        self.window_size = window_size
        self.min_episodes = min_episodes
        self.temperature = temperature
        self.smoothing_factor = smoothing_factor
        self.reward_history = {env.env_id: [] for env in train_env_specs}
        self.num_episodes = {env.env_id: 0 for env in train_env_specs}

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

    def _softmax(self, values: List[float]) -> List[float]:
        if not values: return []
        scaled_values = [v / self.temperature for v in values]
        max_val = max(scaled_values)
        exp_values = [math.exp(v - max_val) for v in scaled_values]
        sum_exp = sum(exp_values)
        if sum_exp == 0: return [1.0 / len(values)] * len(values)
        return [exp_val / sum_exp for exp_val in exp_values]

    def _get_sampling_probs(self) -> Dict[str, float]:
        all_probs = {}
        for chain in self.env_chains:
            rates_of_change = []
            for env_id in chain:
                rate = self._calculate_rate_of_change(env_id)
                rates_of_change.append(rate)
            chain_probs = self._softmax(rates_of_change)
            for env_id, prob in zip(chain, chain_probs):
                all_probs[env_id] = prob
        total_prob = sum(all_probs.values())
        if total_prob > 0: all_probs = {env_id: prob / total_prob for env_id, prob in all_probs.items()}
        else: num_envs = len(all_probs); all_probs = {env_id: 1.0 / num_envs for env_id in all_probs.keys()}
        return all_probs

    def _log_current_state(self):
        print(f"[SoftmaxCurriculumEnvSampler] Environment chains: {self.env_chains}")
        for chain in self.env_chains:
            print(f"[SoftmaxCurriculumEnvSampler] Chain {chain}:")
            for env_id in chain:
                rate = self._calculate_rate_of_change(env_id=env_id)
                episodes = self.num_episodes[env_id]
                recent_avg = 0.0
                if self.reward_history[env_id]:
                    recent_rewards = self.reward_history[env_id][-min(10, len(self.reward_history[env_id])):]
                    recent_avg = sum(recent_rewards) / len(recent_rewards)
                print(f"[SoftmaxCurriculumEnvSampler]   {env_id}: rate={rate:.4f}, episodes={episodes}, recent_avg_rewards={recent_avg:.3f}")
        prob_dict = self._get_sampling_probs()
        print(f"[SoftmaxCurriculumEnvSampler] Sampling probabilities: {prob_dict}")

    def sample(self, kind: str = "train") -> TrainEnvSpec | EvalEnvSpec:
        if kind == "eval": return self._rng.choice(self._eval)
        prob_dict = self._get_sampling_probs()
        if not prob_dict: 
            print("[SoftmaxCurriculumEnvSampler] WARNING: No probabilities calculated, falling back to uniform sampling.")
            return self._rng.choice(list(self.env_id_to_spec.values()))
        env_ids = list(prob_dict.keys())
        weights = list(prob_dict.values())
        total_weight = sum(weights)
        if total_weight <= 0:
            print("[SoftmaxCurriculumEnvSampler] WARNING: Invalid weights, falling back to uniform sampling.")
            weights = [1.0 / len(weights)] * len(weights)
        
        env_id = self._rng.choices(env_ids, weights=weights)[0]
        print(f"[SoftmaxCurriculumEnvSampler] Sampled {env_id} with probability {prob_dict[env_id]:.3f}")
        self._log_current_state()
        
        return self.env_id_to_spec[env_id]

    def update(self, env_id: str, avg_actor_reward: float | None, avg_opponent_reward: float | None) -> None:
        if avg_actor_reward is None: return
        self.reward_history[env_id].append(avg_actor_reward)
        self.num_episodes[env_id] += 1
        # Keep only the most recent rewards within window
        if len(self.reward_history[env_id]) > self.window_size: self.reward_history[env_id] = self.reward_history[env_id][-self.window_size:]
        rate_of_change = self._calculate_rate_of_change(env_id)
        print(f"[SoftmaxCurriculumEnvSampler] Update for {env_id} → reward={avg_actor_reward:.3f}, "
              f"episodes={self.num_episodes[env_id]}, rate_of_change={rate_of_change:.4f}")
        
class ExplorativeCurriculumEnvSampler(BaseEnvSampler):
    def __init__(self, train_env_specs: List[TrainEnvSpec], eval_env_specs: List[EvalEnvSpec] | None = None, rng_seed: int | None = 489,
                 window_size: int = 50, min_episodes: int = 50, temperature: float = 0.1, smoothing_factor: float = 0.01,
                 forward_boost_factor: float = 4.0, decay_factor: float = 0.4):
        """
        Environment sampler that looks ahead in the curriculum chain based on rate of change.
        
        When an environment shows positive rate of change, it increases sampling probability
        for more difficult environments in the chain. When rate of change slows or becomes
        negative, it shifts probability back to the current or easier environments.
        
        Args:
            forward_boost_factor: Multiplier for probability boost given to harder environments (default: 2.0)
            decay_factor: How much probability decreases for environments further in the chain (default: 0.5)
        """
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
        
        # Track chain progress for each chain
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
        # Update chain progress first
        self._update_chain_progress()
        all_probs = {}
        for chain in self.env_chains:
            chain_scores = self._calculate_forward_looking_scores(chain)
            chain_score_values = [chain_scores[env_id] for env_id in chain]
            chain_probs = self._softmax(chain_score_values)
            for env_id, prob in zip(chain, chain_probs):
                all_probs[env_id] = prob
        total_prob = sum(all_probs.values())
        if total_prob > 0: all_probs = {env_id: prob / total_prob for env_id, prob in all_probs.items()}
        else: num_envs = len(all_probs); all_probs = {env_id: 1.0 / num_envs for env_id in all_probs.keys()}
        return all_probs

    def _log_current_state(self):
        print(f"[ForwardLookingCurriculumEnvSampler] Environment chains: {self.env_chains}")
        print(f"[ForwardLookingCurriculumEnvSampler] Chain progress: {self.chain_progress}")
        
        for chain in self.env_chains:
            print(f"[ForwardLookingCurriculumEnvSampler] Chain {chain}:")
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
                
                print(f"[ForwardLookingCurriculumEnvSampler]   {env_id}: rate={rate:.4f}, episodes={episodes}, "
                      f"recent_avg_rewards={recent_avg:.3f}, chain_prob={chain_probs[i]:.3f}, score={chain_scores[env_id]:.3f} {status}")
        
        prob_dict = self._get_sampling_probs()
        print(f"[ForwardLookingCurriculumEnvSampler] Final sampling probabilities (softmax within chains, normalized): {prob_dict}")

    def sample(self, kind: str = "train") -> TrainEnvSpec | EvalEnvSpec:
        if kind == "eval":
            return self._rng.choice(self._eval)
        
        prob_dict = self._get_sampling_probs()
        if not prob_dict:
            print("[ForwardLookingCurriculumEnvSampler] WARNING: No probabilities calculated, falling back to uniform sampling.")
            return self._rng.choice(list(self.env_id_to_spec.values()))
        
        env_ids = list(prob_dict.keys())
        weights = list(prob_dict.values())
        total_weight = sum(weights)
        
        if total_weight <= 0:
            print("[ForwardLookingCurriculumEnvSampler] WARNING: Invalid weights, falling back to uniform sampling.")
            weights = [1.0 / len(weights)] * len(weights)
        
        env_id = self._rng.choices(env_ids, weights=weights)[0]
        print(f"[ForwardLookingCurriculumEnvSampler] Sampled {env_id} with probability {prob_dict[env_id]:.3f}")
        self._log_current_state()
        
        return self.env_id_to_spec[env_id]

    def update(self, env_id: str, avg_actor_reward: float | None, avg_opponent_reward: float | None) -> None:
        if avg_actor_reward is None: return
        self.reward_history[env_id].append(avg_actor_reward)
        self.num_episodes[env_id] += 1
        if len(self.reward_history[env_id]) > self.window_size: self.reward_history[env_id] = self.reward_history[env_id][-self.window_size:] 
        rate_of_change = self._calculate_rate_of_change(env_id)
        print(f"[ForwardLookingCurriculumEnvSampler] Update for {env_id} → reward={avg_actor_reward:.3f}, "
              f"episodes={self.num_episodes[env_id]}, rate_of_change={rate_of_change:.4f}")