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

.. admonition:: Rolling Advantage Estimation
   :class: tip

   We recommend using Rolling Advantage Estimation (RAE), introduced in [1], to use a rolling baseline to optimize the advantage for improved training convergence, stability, and multi-environment support! RAE maintains separate baselines :math:`b_{G,p}` for each game 
   :math:`G \in \mathcal{G}` and role :math:`p \in \{0,1\}`, where each baseline 
   estimates the expected return :math:`\mathbb{E}[R_p(\tau)]` for that role in that game. 
   We update these baselines using exponential moving average (EMA) with a decay 
   rate :math:`\alpha \in [0,1]`:

   .. math::

      \begin{align}
      b_{G,p} &\leftarrow \alpha b_{G,p} + (1-\alpha) R_p(\tau) 
         && \text{(update baseline)} \\
      A_{G,p}(\tau) &= R_p(\tau) - b_{G,p} 
         && \text{(compute advantage)}
      \end{align}
   
   [1] SPIRAL

Hyperparameters
"""""""""""""""

**epochs: int (default: 2)**
  Number of epochs to train the policy.
**batch_size: int (default: 384)**
  Batch size for the policy.
**mini_batch_size: int (default: 1)**
  Mini batch size for the policy.
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
