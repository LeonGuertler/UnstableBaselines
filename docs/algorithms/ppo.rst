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

Hyperparameters
"""""""""""""""

**epochs: int (default: 2)**
  Number of epochs to train the policy.
**local_batch_size: int (default: 384)**
  Per-GPU batch size.
**micro_batch_size: int (default: 1)**
  The micro batch size used during training.
**learning_rate: float (default: 1e-5)**
  Learning rate for the policy.
**lr_scheduler_type: str (default: "constant")**
  Learning rate scheduler type.
**lr_warmup_ratio: float (default: 0.01)**
  Learning rate warmup ratio.
**clip_ratio: float (default: 0.2)**
  Clip ratio for the policy.
**grad_clip: float (default: 0.2)**
  Gradient clipping value for the policy.
**entropy_coeff: float (default: 0.0)**
  Entropy coefficient.
**beta: float (default: 0.01)**
  Beta coefficient to weight the KL divergence to the reference model. When larger than 0.0, a referene model must be loaded.
**infer_micro_batch_size: int (default: 4)**
  Batch size for inference.
**critic_learning_rate: float (default: 1e-4)**
  Learning rate for the critic.
**clip_value: float (default: 0.2)**
  Clipping value for the critic.
**gamma: float (default: 0.99)**
  Discount factor.
**gae_lambda: float (default: 0.95)**
  Lambda for the GAE.
**normalize_adv: bool (default: False)**
  Whether to normalize the advantage.
