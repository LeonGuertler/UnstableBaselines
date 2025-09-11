GRPO
====

**Group Relative Policy Optimization** neglects the critic and computes group-wise relative advantages. 

.. math::

   \mathcal{L}_{\text{PPO}}(\theta)_i
   = \frac{1}{T} \sum_{t=0}^{T} \Big[
        \min\!\big(
          r_{i,t}(\theta) \, A_{i,t},\;
          \text{clip}(r_{i,t}(\theta), 1{-}\epsilon, 1{+}\epsilon) \, \hat{A}_{i,t}
        \big)
      \Big]

with:

.. math::

   r_{i,t}(\theta)
   = \frac{\pi_{\theta}(a_{i,t} \mid s_{i,t})}
          {\pi_{\theta_{\text{old}}}(a_{i,t} \mid s_{i,t})};
         \quad
   \hat{A}_{i,t} = \frac{R_i - \text{mean}(R_{1:G})}{\text{std}(R_{1:G}) + \epsilon}

The group-wise relative advantage is computed based on sampled episodes in a batch. 

Hyperparameters
"""""""""""""""

Benchmarks
""""""""""
