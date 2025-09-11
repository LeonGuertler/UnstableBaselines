PPO
===

**Proximal Policy Optimization** uses a clipped surrogate objective to optimize the policy. 
It learns a token-wise value function and uses generalized advantage estimation (GAE) to estimate the advantage. 
We optimize the following PPO objective:

.. math::

   \mathcal{L}_{\text{PPO}}(\theta)_i
   = \frac{1}{T} \sum_{t=0}^{T} \Big[
        \min\!\big(
          r_{i,t}(\theta) \, A_{i,t},\;
          \text{clip}(r_{i,t}(\theta), 1{-}\epsilon, 1{+}\epsilon) \, A_{i,t}
        \big)
      \Big]

with policy ratio:

.. math::

   r_{i,t}(\theta)
   = \frac{\pi_{\theta}(a_{i,t} \mid s_{i,t})}
          {\pi_{\theta_{\text{old}}}(a_{i,t} \mid s_{i,t})}

Benchmarks
""""""""""

Hyperparameters
"""""""""""""""

**infer_mini_batch_size: int (default: 4)**
  Batch size for inference.
**learning_rate: float (default: 1e-5)**
  Learning rate for the policy.
**critic_learning_rate: float (default: 1e-4)**
  Learning rate for the critic.
**clip_ratio: float (default: 0.2)**
  Clip ratio for the policy.
**clip_value: float (default: 0.2)**
  Clip value for the critic.
**entropy_coeff: float (default: 0.0)**
  Entropy coefficient.
**beta: float (default: 0.0)**
  Beta coefficient to weight the KL divergence to the reference model. When larger than 0.0, a referene model must be loaded.
**gamma: float (default: 0.99)**
  Discount factor.
**gae_lambda: float (default: 0.95)**
  Lambda for the GAE.
**normalize_adv: bool (default: False)**
  Whether to normalize the advantage.
**max_generation_len: int (default: None)**
  Maximum generation length.
**max_train_len: int (default: None)**
  Maximum training length. Can be used to save vram by truncating the training data.
**actor_grad_accumulation_steps: int (default: 4)**
  Actor gradient accumulation steps.
**critic_grad_accumulation_steps: int (default: 2)**
  Critic gradient accumulation steps.
**actor_lr_scheduler_type: str (default: "linear")**
  Actor learning rate scheduler type.
**actor_lr_warmup_ratio: float (default: 0.025)**
  Actor learning rate warmup ratio.
**critic_lr_scheduler_type: str (default: "linear")**
  Critic learning rate scheduler type.
**critic_lr_warmup_ratio: float (default: 0.025)**
  Critic learning rate warmup ratio.
