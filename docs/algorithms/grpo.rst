GRPO
====

**Group Relative Policy Optimization (GRPO)** optimizes the policy with a clipped surrogate objective and importance sampling correction:

.. math::

   \mathcal{L}(\theta)
   = -\frac{1}{N} \sum_{i=1}^{N}
     \min\!\left(
       r_i \, A_i,\;
       \mathrm{clip}(r_i,\, 1 - \varepsilon,\, 1 + \varepsilon) \, A_i
     \right)

where :math:`r_i` is the importance ratio between the current and collection policy.

.. admonition:: Rolling Advantage Estimation
   :class: tip

   We recommend using Rolling Advantage Estimation (RAE), introduced in [1], which maintains a separate EMA baseline :math:`b_{G,p}` per game :math:`G` and role :math:`p`:

   .. math::

      \begin{align}
      b_{G,p} &\leftarrow \alpha\, b_{G,p} + (1-\alpha)\, R_p(\tau) \\
      A_{G,p}(\tau) &= R_p(\tau) - b_{G,p}
      \end{align}

   [1] Liu et al., *SPIRAL*, arXiv:2506.24119.

Hyperparameters
"""""""""""""""

**epochs** *int* (default: ``2``) — Passes over the training batch per update.

**local_batch_size** *int* (default: ``384``) — Steps per gradient update.

**micro_batch_size** *int* (default: ``1``) — Steps per micro-batch; controls gradient accumulation.

**learning_rate** *float* (default: ``1e-6``)

**lr_scheduler_type** *str* (default: ``"constant"``)

**lr_warmup_ratio** *float* (default: ``0.01``)

**grad_clip** *float* (default: ``0.2``) — Maximum gradient norm.

**clip_eps** *float* (default: ``0.2``) — Importance ratio clip range :math:`\varepsilon`.

**kl_coef** *float* (default: ``0.0``) — KL penalty coefficient against the frozen base model.

**entropy_coeff** *float* (default: ``0.0``) — Entropy bonus coefficient.

**loss** *str* (default: ``"drgrpo"``) — Loss normalization mode (``"grpo"`` or ``"drgrpo"``).

**max_generation_len** *int* — Max response tokens; denominator in ``drgrpo`` mode.

**temperature** *float* (default: ``1.0``) — Logit temperature during training. Should match vLLM sampling temperature.
