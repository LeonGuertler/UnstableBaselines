REINFORCE
=========

**REINFORCE** is a policy gradient algorithm that directly maximizes the expected return of a completion by the policy: 

.. math::

   \nabla_\theta J(\theta)_i =
   \mathbb{E}_{\tau \sim \pi_\theta}\left[
      R_i
      \sum_{t=0}^T
      \nabla_\theta \log \pi_\theta(a_{i,t} \mid s_{i,t})\, 
   \right]

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
   
   [1] SPIRAL

Hyperparameters
"""""""""""""""
**batch_size: int (default: 384)**
  Number of epochs to train the policy.
**mini_batch_size: int (default: 1)**
  Mini batch size for the policy.
**learning_rate: float (default: 1e-5)**
  Learning rate for the policy.
**lr_scheduler_type: str (default: "constant")**
  Learning rate scheduler type.
**lr_warmup_ratio: float (default: 0.01)**
  Learning rate warmup ratio.
**grad_clip: float (default: 0.2)**
  Gradient clipping value for the policy.
