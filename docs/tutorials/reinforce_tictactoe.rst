GRPO Self-Play on TicTacToe
===========================

This tutorial walks through training a small language model to play TicTacToe using GRPO and self-play.
It is a good starting point for understanding how the framework fits together: how games are collected,
how rewards are shaped, and how the learner is configured.

Full experiment: ``experiments/reinforce/``

.. code-block:: bash

   python experiments/reinforce/run.py


Model and environment
"""""""""""""""""""""

This experiment implements self-play via the ``MirrorModelSampler`` and trains ``Qwen3-1.7B-Base`` with a LoRA adapter (rank 32). 
LoRA means the base
model weights are never modified — only the adapter is trained and checkpointed. This makes
checkpoints small and cheap to hot-swap into vLLM during collection. At the start of each game,
the ``MirrorModelSampler`` samples an opponent checkpoint from the most recent ``opponent_top_k`` saved checkpoints.
Because games are collected asynchronously, many games are in flight at once — each against a
slightly different version of the policy.

The environment is ``TicTacToe-v0-train`` from TextArena. Each game has 2 players and 2 actors,
meaning both roles are played by the same vLLM instance. The model is prompted to reason before
each move using the ``qwen3-zs`` zero-shot template, and generation is capped at 16384 tokens to
allow for chain-of-thought.

Reward shaping
""""""""""""""

Raw game outcomes (win/loss/draw) are not used directly as training signals. Instead, rewards
are shaped at three stages configured under ``replay_buffer.reward_transformations``:

**Step-level** (``step``) — applied as each step is collected:

- ``format_reward`` (±0.1): gives a small positive reward when the model produces a well-formed
  move and a small penalty otherwise. This incentivises the model to stay on-format even early
  in training when it has not yet learned to win games.
- ``invalid_move_penalty`` (±0.2): a larger penalty for moves rejected by the environment as illegal.
  This is more important than format: an invalid move wastes a turn.

**Episode-level** (``final``) — applied once per episode at the end:

- ``role_advantage``: assigns the win/loss/draw signal per player role. In a 2-player game the
  two roles receive opposite signals, so the model learns both to win and to prevent the opponent
  from winning.

**Batch-level** (``sampling``) — applied when sampling a training batch from the buffer:

- ``normalize_by_env`` with ``z_score=True``: z-score normalizes advantages within each environment
  type. This ensures that different games in the buffer don't dominate the gradient just because
  they happen to have larger raw reward magnitudes.

.. admonition:: Why normalize at sampling time?
   :class: note

   Normalizing at sampling time (rather than at collection time) means the normalization statistics
   reflect the current buffer contents. As the buffer fills with increasingly competent play,
   the baseline shifts accordingly.

Buffer and batching
"""""""""""""""""""

The experiment uses a ``StepBuffer`` — each step is stored and trained on individually, rather
than waiting for full episodes. This gives faster feedback and works well for games like TicTacToe
where every move is meaningful.

The buffer holds up to 768 steps. Once full, the learner is called with a batch of
``local_batch_size=64`` steps, split into micro-batches of size 1 for gradient accumulation.
With ``epochs=1``, each buffered step is used for exactly one gradient update.

Key config
""""""""""

.. code-block:: yaml

   learner:
     type: "grpo"
     loss: "grpo"           # normalize by sequence length
     learning_rate: 0.00001
     local_batch_size: 64
     micro_batch_size: 1
     epochs: 1
     max_generation_len: 16384
     grad_clip: 0.2
     lora_cfg:
       lora_rank: 32
       lora_alpha: 32

   model_sampler:
     type: "mirror"
     opponent_top_k: 20      # sample opponent from 20 most recent checkpoints
     opponent_temperature: 0.1

   replay_buffer:
     type: "step_buffer"
     max_buffer_size: 768
     reward_transformations:
       final:
         role_advantage: {}
       step:
         format_reward: {reward: 0.1, penalty: -0.1}
         invalid_move_penalty: {reward: 0.2, penalty: -0.2}
       sampling:
         normalize_by_env: {z_score: true}
