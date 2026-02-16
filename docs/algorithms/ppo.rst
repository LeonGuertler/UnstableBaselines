PPO
===

**Proximal Policy Optimization** uses a clipped surrogate objective to optimize the policy.
It learns a step-wise value function and uses generalized advantage estimation (GAE) to estimate the advantage for an improved bias-variance trade-off and credit assignment.
We optimize the following combined objective:

.. math::

   \mathcal{L}(\theta)
   = \mathcal{L}_{\text{policy}}
   + c_v \, \mathcal{L}_{\text{value}}
   + \beta_{\text{KL}} \, \hat{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})
   - c_{\text{ent}} \, \hat{H}(\pi_\theta)

with clipped policy loss:

.. math::

   \mathcal{L}_{\text{policy}}(\theta)
   = -\frac{1}{N} \sum_{i=1}^{N}
     \min\!\big(
       A_i \, r_i(\theta),\;
       A_i \, \text{clamp}(r_i(\theta),\, \epsilon_{\text{lo}},\, \epsilon_{\text{hi}})
     \big)

importance ratio:

.. math::

   r_i(\theta)
   = \frac{\pi_\theta(a_i \mid s_i)}{\pi_{\theta_{\text{old}}}(a_i \mid s_i)},
   \quad
   \log \pi_\theta(a_i \mid s_i)
   = \sum_{t=1}^{T_i}
     \log \pi_\theta(a_{i,t} \mid s_{i,t})

and clipped value loss:

.. math::

   \mathcal{L}_{\text{value}}(\theta)
   = \frac{1}{2N} \sum_{i=1}^{N}
     \max\!\big(
       (V_\theta(s_i) - G_i)^2,\;
       (\bar{V}_i - G_i)^2
     \big)

   \bar{V}_i
   = V_{\text{old}}(s_i)
   + \text{clamp}\!\big(V_\theta(s_i) - V_{\text{old}}(s_i),\, -\epsilon_v,\, \epsilon_v\big)


.. admonition:: Actor-Critic Architecture
   :class: tip

   We add a value head on top of the base model and use separate LoRA adapters for the actor and the critic. During training, we dynamically swap between the two adapters for efficient parameter-sharing without interference, allowing both networks to share the same backbone and avoiding the need to load a second model.

Hyperparameters
"""""""""""""""

**epochs: int (default: 2)**
  Number of epochs to train the policy.
**local_batch_size: int (default: 256)**
  Per-GPU batch size.
**micro_batch_size: int (default: 1)**
  The micro batch size used during training.
**learning_rate: float (default: 1e-5)**
  Learning rate for the policy.
**lr_scheduler_type: str (default: "constant")**
  Learning rate scheduler type.
**lr_warmup_ratio: float (default: 0.01)**
  Learning rate warmup ratio.
**upper_clip_ratio: float (default: 0.4)**
  Upper bound for the asymmetric clipping of the importance ratio.
**lower_clip_ratio: float (default: 0.2)**
  Lower bound for the asymmetric clipping of the importance ratio.
**grad_clip: float (default: 0.2)**
  Gradient clipping value for the policy.
**entropy_coeff: float (default: 0.0)**
  Entropy coefficient.
**value_coeff: float (default: 0.5)**
  Coefficient for the value function loss.
**beta: float (default: 0.0)**
  Beta coefficient to weight the KL divergence to the reference model. When larger than 0.0, a reference model must be loaded.
**infer_micro_batch_size: int (default: 4)**
  Batch size for value inference before training.
**critic_learning_rate: float (default: 1e-5)**
  Learning rate for the critic.
**clip_value: float (default: 0.2)**
  Clipping value for the critic.
**gamma: float (default: 0.99)**
  Discount factor.
**gae_lambda: float (default: 0.95)**
  Lambda for the GAE.
**normalize_adv: bool (default: True)**
  Whether to normalize the advantage.
