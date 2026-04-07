Structuring Experiments
=======================

The ``experiments/`` directory contains ready-to-run examples that apply Unstable Baselines to specific tasks.
Each experiment is self-contained: it has its own config, launcher script, and notes, so you can reproduce,
fork, or compare runs without touching the core library.

.. code-block:: text

   experiments/
     ppo/                  — PPO self-play on SimpleTak
     reinforce/            — GRPO self-play on TicTacToe
     reinforce-math/       — GRPO single-turn math reasoning

Experiment layout
"""""""""""""""""

We recommend to follow the same three-file layout for every experiment:

``config.yaml``
   A full, standalone training configuration. Start from the closest base config in ``unstable/config/``
   and override what you need. All fields are explicit — there are no hidden defaults layered on top.

``run.py``
   A minimal Python launcher. It loads ``config.yaml``, optionally overrides values in Python, and
   calls ``train()``. Running it is as simple as:

   .. code-block:: bash

      python experiments/<name>/run.py

   For most experiments ``run.py`` is just a few lines:

   .. code-block:: python

      import os, yaml
      from unstable.train import train

      if __name__ == "__main__":
          config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
          with open(config_path) as f:
              config = yaml.safe_load(f)
          train(config)

``README.md``
   Can be used to document the hypothesis, setup choices, and results as the experiment evolves.
   Treat it as a shared lab notebook with your coding agent — update it as you run ablations.

Adding a new experiment
"""""""""""""""""""""""

Copy the closest base config from ``unstable/config/`` into a new subdirectory and adjust from there:

.. code-block:: bash

   mkdir experiments/my-experiment
   cp unstable/config/grpo.yaml experiments/my-experiment/config.yaml

Then create a ``run.py`` from the template above and a ``README.md`` describing your goal.
Register the experiment in ``experiments/README.md`` so it appears in the index.

.. admonition:: Tip — override in Python, not in YAML
   :class: tip

   For quick iterations, prefer overriding hyperparameters in ``run.py`` rather than editing
   ``config.yaml``. That way your YAML stays close to the base config and your changes are visible
   at a glance:

   .. code-block:: python

      config['learner']['learning_rate'] = 3e-5
      config['learner']['epochs'] = 2
      train(config)

Custom learner classes
""""""""""""""""""""""

If your experiment needs a learner algorithm that doesn't belong in the core library, define it in
the experiment directory and inject it via ``run.py``.

   .. code-block:: python

      from my_learner import MyLearner

      config['learner']['type'] = MyLearner
      train(config)
