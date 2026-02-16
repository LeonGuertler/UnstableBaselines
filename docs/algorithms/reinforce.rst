REINFORCE
=========

**REINFORCE** is a policy gradient algorithm that directly maximizes the expected return of a completion by the policy.
We optimize the following loss with off-policy importance sampling correction:

.. math::

   \mathcal{L}_{\text{REINFORCE}}(\theta)
   = -\frac{1}{N} \sum_{i=1}^{N}
     A_i \,
     \pi_\theta(a_i \mid s_i)

where:

.. math::

   \log \pi_\theta(a_i \mid s_i)
   = \sum_{t=1}^{T_i}
     \log \pi_\theta(a_{i,t} \mid s_{i,t})

.. admonition:: Rolling Advantage Estimation
   :class: tip

   We recommend using Rolling Advantage Estimation (RAE), introduced in [1], to use a rolling baseline to optimize the advantage for improved training convergence and stability! RAE maintains separate baselines :math:`b_{G,p}` for each game 
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
   
   [1] Liu, B., Guertler, L., Yu, S., Liu, Z., Qi, P., Balcells, D., ... & Jaques, N. (2025). SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning. arXiv preprint arXiv:2506.24119.

Hyperparameters
"""""""""""""""
**epochs: int (default: 2)**
  Number of epochs to train the policy.
**local_batch_size: int (default: 384)**
  Per-GPU batch size.
**micro_batch_size: int (default: 1)**
  The micro batch size used during training.
**learning_rate: float (default: 1e-6)**
  Learning rate for the policy.
**lr_scheduler_type: str (default: "constant")**
  Learning rate scheduler type.
**lr_warmup_ratio: float (default: 0.01)**
  Learning rate warmup ratio.
**grad_clip: float (default: 0.2)**
  Gradient clipping value for the policy.
**kl_coef: float (default: 0.0)**
  Coefficient for KL divergence penalty against the reference model.
