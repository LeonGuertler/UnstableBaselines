Train a reasoning model to play tic tac toe with PPO
=====================================================

In this tutorial, we train a small language model to play Tic Tac Toe using Proximal Policy Optimization and Generalized Advantage Estimation.
We use self-play with a mirror model sampler, so the model learns by playing against copies of itself.

Setup
"""""

Make sure you have Unstable Baselines installed:

.. code-block:: bash

   pip install unstable-rl

Quick start
"""""""""""

The fastest way to get started is with the ``examples/run.py`` script:

.. code-block:: bash

   python examples/run.py \
       --name "tictactoe-ppo" \
       --algorithm ppo \
       --envs "TicTacToe-v0-train" \
       --model "Qwen/Qwen3-1.7B-Base" \
       --template "qwen3-zs"

This loads the default PPO config, overrides the environment and model, and starts training.

Custom config in Python
"""""""""""""""""""""""

For more control, load and modify the config directly:

.. code-block:: python

   from unstable import train, get_algorithm_config

   config = get_algorithm_config("ppo")

   # Run settings
   config["run"] = "tictactoe-ppo"
   config["model_name"] = "Qwen/Qwen3-1.7B-Base"
   config["training_iterations"] = 500

   # Use the Tic Tac Toe environment
   config["env_sampler"]["train"] = [
       {
           "id": "TicTacToe-v0-train",
           "num_players": 2,
           "num_actors": 2,
           "prompt_template": "qwen3-zs",
       }
   ]
   config["env_sampler"]["eval"] = [
       {
           "id": "TicTacToe-v0-train",
           "num_players": 2,
           "prompt_template": "qwen3-zs",
           "fixed_opponent": "google/gemini-2.0-flash-lite-001",
       }
   ]

   # Tune hyperparameters for Tic Tac Toe
   config["learner"]["learning_rate"] = 1e-5
   config["learner"]["local_batch_size"] = 128
   config["learner"]["total_training_steps"] = 500

   checkpoint_path = train(config)

Reward shaping
""""""""""""""

The replay buffer applies reward transformations to the raw game outcomes.
For Tic Tac Toe, you might want to penalize invalid moves and give a small reward for correct formatting:

.. code-block:: python

   config["replay_buffer"]["reward_transformations"] = {
       "final": {},
       "step": {
           "format_reward": {"reward": 0.1},
           "invalid_move_penalty": {"reward": 0.1, "penalty": -0.1},
       },
       "sampling": {},
   }

Evaluation
""""""""""

After training, evaluate the checkpoint against a fixed opponent:

.. code-block:: bash

   python -m unstable.eval \
       --checkpoint <checkpoint_path> \
       --env "TicTacToe-v0-train" \
       --opponent "google/gemini-2.0-flash-lite-001" \
       --num_runs 128

Results
""""""""""


