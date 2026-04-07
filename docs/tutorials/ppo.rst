PPO Self-Play on SimpleTak
==========================

**Proximal Policy Optimization (PPO)** learns a value function alongside the policy and uses Generalized Advantage Estimation (GAE) to reduce variance in the gradient signal.
The clipped surrogate objective prevents large policy updates that could destabilize training.

Full experiment: ``experiments/ppo/``

.. code-block:: bash

   python experiments/ppo/run.py

.. math::

   A_t = \sum_{k=0}^{T-t} (\gamma \lambda)^k \delta_{t+k}, \quad \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)

.. math::

   \mathcal{L}(\theta)
   = -\frac{1}{N} \sum_{i=1}^{N}
     \min\!\left(
       r_i \, A_i,\;
       \mathrm{clip}(r_i,\, 1 - \varepsilon_{\text{lo}},\, 1 + \varepsilon_{\text{hi}}) \, A_i
     \right)

.. admonition:: Actor-Critic Architecture
   :class: tip

   The value head is added on top of the base model with a separate LoRA adapter for the critic. Actor and critic share the same backbone — no second model is loaded. During training the adapters are swapped dynamically.

Key config
""""""""""

.. code-block:: yaml

   learner:
     type: "ppo"
     learning_rate: 0.00001
     local_batch_size: 256
     epochs: 2
     gamma: 0.99
     gae_lambda: 0.95
     value_coeff: 0.5
     upper_clip_ratio: 0.4
     lower_clip_ratio: 0.2
     normalize_adv: true

   replay_buffer:
     type: "episode_buffer"
     flatten: true
     reward_transformations:
       final:
         role_advantage: {}
       step:
         format_reward: {reward: 0.1, penalty: -0.1}
         invalid_move_penalty: {reward: 0.2, penalty: -0.2}
       sampling: {}

PPO requires ``episode_buffer`` because GAE needs the full episode trajectory to compute returns before training.

Hyperparameters
"""""""""""""""

See :doc:`../algorithms/ppo` for the full hyperparameter reference.
