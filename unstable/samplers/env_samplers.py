import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random, math
from typing import Protocol, List, Any, Dict
from unstable._types import TrainEnvSpec, EvalEnvSpec
from collections import deque, defaultdict


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

class CurriculumPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_envs: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.num_envs = num_envs
        # policy network
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_envs),
            nn.Softmax(dim=-1)
        )
        # value network for advantage estimation
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        action_probs = self.policy_net(state)
        value = self.value_net(state)
        return action_probs, value
    
class CurriculumEnvSampler(BaseEnvSampler):
    def __init__(self, train_env_specs: List[TrainEnvSpec], eval_env_specs: List[EvalEnvSpec] | None = None, 
                 rng_seed: int | None = 489, window_size: int = 50,
                 learning_rate: float = 0.001, hidden_dim: int = 128, update_frequency: int = 10,
                 log_frequency: int = 50):
        super().__init__(train_env_specs, eval_env_specs, rng_seed)

        self.env_id_to_spec = {env.env_id: env for env in train_env_specs}
        self.env_id_to_idx = {env_id: idx for idx, env_id in enumerate(self.env_id_to_spec.keys())}
        self.idx_to_env_id = {idx: env_id for env_id, idx in self.env_id_to_idx.items()}
        
        self.env_chains = self._infer_env_chains_from_registry()
        self.reward_history = {env.env_id: deque(maxlen=window_size) for env in train_env_specs}
        self.num_episodes = {env.env_id: 0 for env in train_env_specs}

        # setup policy
        self.feature_dim = self._calculate_feature_dim()
        self.policy = CurriculumPolicy(input_dim=self.feature_dim, hidden_dim=hidden_dim, num_envs=len(train_env_specs))
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=learning_rate)

        # buffer for policy training
        self.update_frequency, self.experience_buffer, self.update_counter = update_frequency, [], 0

        # metrics for reward calculation
        self.last_state_features, self.last_action_idx, self.global_reward_tracker = None, None, deque(maxlen=1000)

        # for logging and visualization
        self.log_frequency = log_frequency
        self.sampling_history = deque(maxlen=1000)
        self.probability_history = {env_id: deque(maxlen=200) for env_id in self.env_id_to_spec.keys()}
        self.policy_loss_history = deque(maxlen=100)
        self.value_loss_history = deque(maxlen=100)
        self.exploration_rate_history = deque(maxlen=200)
        self.total_samples = 0
        self.exploration_samples = 0
        self.env_sample_counts = {env_id: 0 for env_id in self.env_id_to_spec.keys()}

    def _calculate_feature_dim(self) -> int:
        # Features per env: [avg_reward, rate_of_change, episodes_ratio, chain_pos]
        # Global features: [global_progress, exploration_bonus]
        return len(self.env_id_to_spec) * 4 + 2

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
        hist = list(self.reward_history[env_id])
        if len(hist) < 2: return 0.0
        n = len(hist)
        x_vals = list(range(n))
        x_mean = sum(x_vals) / n
        y_mean = sum(hist) / n
        num = sum((x_vals[i] - x_mean) * (hist[i] - y_mean) for i in range(n))
        den = sum((x_vals[i] - x_mean) ** 2 for i in range(n))
        if abs(den) < 1e-8: return 0.0
        return num / den

    def _get_chain_position(self, env_id: str) -> float:
        for chain in self.env_chains:
            if env_id in chain:
                return chain.index(env_id) / (len(chain) - 1) if len(chain) > 1 else 0.0
        return 0.5
    
    def _extract_state_features(self) -> torch.Tensor:
        features = []

        for env_id in self.env_id_to_spec.keys():
            # avg reward (normalized)
            hist = list(self.reward_history[env_id])
            avg_reward = np.mean(hist) if hist else 0.0
            # rate_of_change
            rate = self._calculate_rate_of_change(env_id)
            # epsidoe_ratio
            total_episodes = sum(self.num_episodes.values())
            episode_ratio = self.num_episodes[env_id] / max(total_episodes, 1)
            # chain position
            chain_pos = self._get_chain_position(env_id)
            # append all
            features.extend([avg_reward, rate, episode_ratio, chain_pos])
        
        # gloabl features
        all_rewards = [r for history in self.reward_history.values() for r in history] # flatten
        global_progress = np.mean(all_rewards) if all_rewards else 0.0
        
        # exploration bonus
        total_episodes = sum(self.num_episodes.values())
        exploration_entropy = -sum(
            (count / max(total_episodes, 1)) * np.log(max(count / max(total_episodes, 1), 1e-8))
            for count in self.num_episodes.values()
        )

        features.extend([global_progress, exploration_entropy])

        return torch.FloatTensor(features).unsqueeze(0)
    
    def _calculate_policy_reward(self, env_id: str, avg_actor_reward: float) -> float:
        reward = avg_actor_reward
        # give a reward signal for rate_of_change on harder envs
        rate_of_change = self._calculate_rate_of_change(env_id)
        if rate_of_change > 0:
            chain_pos = self._get_chain_position(env_id)
            bonus = rate_of_change * (1.0 + chain_pos)
            reward += bonus
        # give a reward signal for underexplored envs
        total_episodes = sum(self.num_episodes.values())
        if total_episodes > 0:
            env_ratio = self.num_episodes[env_id] / total_episodes
            bonus = max(0, 0.1 - env_ratio) * 10
            reward += bonus
        return reward
    
    def sample(self, kind: str = "train", verbose: bool = True) -> TrainEnvSpec | EvalEnvSpec:
        if kind == "eval": return self._rng.choice(self._eval)

        state_features = self._extract_state_features()
        with torch.no_grad():
            action_probs, _ = self.policy(state_features)
            action_probs = action_probs.squeeze()
        
        for env_id, prob in zip(self.env_id_to_spec.keys(), action_probs):
            self.probability_history[env_id].append(prob.item())

        exploration_rate = max(0.05, 0.3 * np.exp(-self.total_samples / 1000))
        self.exploration_rate_history.append(exploration_rate)

        if torch.rand(1).item() < exploration_rate:
            action_idx = torch.randint(0, len(self.env_id_to_spec), (1,)).item()
            self.exploration_samples += 1
            exploration_used = True
        else:
            action_idx = torch.multinomial(action_probs, 1).item()
            exploration_used = False
        
        env_id = self.idx_to_env_id[action_idx]

        self.total_samples += 1
        self.env_sample_counts[env_id] += 1
        self.sampling_history.append({
            'env_id': env_id,
            'probability': action_probs[action_idx].item(),
            'exploration_used': exploration_used,
            'total_samples': self.total_samples
        })

        self.last_state_features = state_features
        self.last_action_idx = action_idx

        if verbose or self.total_samples % self.log_frequency == 0:
            self._log_detailed_state(action_probs, env_id, exploration_used)
        
        print(f"[Policy] Sampled {env_id} with probability {action_probs[action_idx]:.3f} "
              f"({'exploration' if exploration_used else 'policy'})")
        
        return self.env_id_to_spec[env_id]
    
    def update(self, env_id: str, avg_actor_reward: float | None, avg_opponent_reward: float | None) -> None:
        if avg_actor_reward is None: return
        self.reward_history[env_id].append(avg_actor_reward)
        self.num_episodes[env_id] += 1
        self.global_reward_tracker.append(avg_actor_reward)

        if self.last_state_features is not None and self.last_action_idx is not None:
            policy_reward = self._calculate_policy_reward(env_id, avg_actor_reward)
            experience = (self.last_state_features.squeeze(), self.last_action_idx, policy_reward)
            self.experience_buffer.append(experience)

        self.update_counter += 1
        if self.update_counter >= self.update_frequency:
            self._update_policy()
            self.update_counter = 0 # reset

        rate_of_change = self._calculate_rate_of_change(env_id)
        print(f"[Policy] Update for {env_id} → reward={avg_actor_reward:.3f}, "
              f"episodes={self.num_episodes[env_id]}, rate_of_change={rate_of_change:.4f}")
        
        if self.total_samples > 0 and self.total_samples % (self.log_frequency * 2) == 0:
            self._log_analytics_summary()

    def _update_policy(self) -> None:
        if len(self.experience_buffer) == 0: return

        states = torch.stack([experience[0] for experience in self.experience_buffer])
        actions = torch.LongTensor([experience[1] for experience in self.experience_buffer])
        rewards = torch.FloatTensor([experience[2] for experience in self.experience_buffer])

        # Calculate future rewards
        returns, running_return = [], 0
        for reward in reversed(rewards):
            running_return = reward + 0.99 * running_return 
            returns.append(running_return)
        returns.reverse()
        returns = torch.FloatTensor(returns)
        # now normalize
        if len(returns) > 1: returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # forward pass
        action_probs, values = self.policy(states)
        values = values.squeeze()

        # calc advantages
        advantages = returns - values.detach()

        # Policy Loss using Reinforce with Baseline
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1))).squeeze()
        policy_loss = -(log_probs * advantages).mean()

        # value loss
        value_loss = nn.MSELoss()(values, returns)

        # total loss
        total_loss = policy_loss + 0.5 * value_loss

        # take a step
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        # log the policy and value losses
        self.policy_loss_history.append(policy_loss.item())
        self.value_loss_history.append(value_loss.item())

        # clear the experience buffer after every update
        self.experience_buffer.clear()

        print(f"[Policy] Updated - Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}")

    def _log_detailed_state(self, action_probs: torch.Tensor, sampled_env: str, exploration_used: bool):
        print(f"\n----- [Policy] Detailed State (Sample #{self.total_samples}) -----")
        print("[Policy] Current Action Probabilities:")
        for env_id, prob in zip(self.env_id_to_spec.keys(), action_probs):
            sample_count = self.env_sample_counts[env_id]
            sample_ratio = sample_count / max(self.total_samples, 1)
            recent_avg_reward = np.mean(list(self.reward_history[env_id])[-10:]) if self.reward_history[env_id] else 0.0
            
            marker = ">>> " if env_id == sampled_env else "    "
            print(f"{marker}{env_id}: prob={prob:.3f}, samples={sample_count} ({sample_ratio:.1%}), "
                  f"recent_reward={recent_avg_reward:.3f}")

        exploration_ratio = self.exploration_samples / max(self.total_samples, 1)
        current_exploration_rate = self.exploration_rate_history[-1] if self.exploration_rate_history else 0.0
        print(f"[Policy] Exploration: rate={current_exploration_rate:.3f}, "
              f"used={exploration_ratio:.1%} ({self.exploration_samples}/{self.total_samples})")

        if self.policy_loss_history:
            recent_policy_loss = np.mean(list(self.policy_loss_history)[-5:])
            recent_value_loss = np.mean(list(self.value_loss_history)[-5:])
            print(f"[Policy] Recent Training: policy_loss={recent_policy_loss:.4f}, "
                  f"value_loss={recent_value_loss:.4f}")
        print("-" * 60)

    def _log_analytics_summary(self):
        print(f"\n----- [Policy] Analytics Summary (Sample #{self.total_samples}) -----")
        print("[Policy] Environment Selection Distribution:")
        for env_id in self.env_id_to_spec.keys():
            count = self.env_sample_counts[env_id]
            percentage = count / max(self.total_samples, 1) * 100
            recent_probs = list(self.probability_history[env_id])[-20:] if self.probability_history[env_id] else [0.0]
            avg_recent_prob = np.mean(recent_probs) * 100
            prob_std = np.std(recent_probs) * 100
            
            episodes = self.num_episodes[env_id]
            recent_rewards = list(self.reward_history[env_id])[-10:] if self.reward_history[env_id] else []
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            
            print(f"  {env_id}: {count} samples ({percentage:.1f}%), "
                  f"avg_prob={avg_recent_prob:.1f}%±{prob_std:.1f}%, "
                  f"episodes={episodes}, recent_reward={avg_reward:.3f}")

        if len(self.policy_loss_history) >= 10:
            recent_policy_losses = list(self.policy_loss_history)[-10:]
            policy_trend = "↓" if recent_policy_losses[-1] < recent_policy_losses[0] else "↑"
            print(f"[Policy] Training Trend: policy_loss {policy_trend} "
                  f"({recent_policy_losses[0]:.4f} → {recent_policy_losses[-1]:.4f})")

        if len(self.exploration_rate_history) >= 20:
            early_exploration = np.mean(list(self.exploration_rate_history)[:10])
            recent_exploration = np.mean(list(self.exploration_rate_history)[-10:])
            print(f"[Policy] Exploration Decay: {early_exploration:.3f} → {recent_exploration:.3f}")

        sample_counts = list(self.env_sample_counts.values())
        if sample_counts:
            gini_coefficient = self._calculate_gini_coefficient(sample_counts)
            print(f"[Policy] Selection Diversity: Gini={gini_coefficient:.3f} (0=uniform, 1=concentrated)")
        
        print("-" * 80)

    def _calculate_gini_coefficient(self, values):
        if not values or all(v == 0 for v in values): return 0.0
        sorted_values = sorted(values)
        n = len(sorted_values)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n