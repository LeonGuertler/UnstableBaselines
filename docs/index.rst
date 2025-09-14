.. raw:: html

    <div align="center">
        <img style="width: 250px; margin-top: 100px;" src="_static/ub.png" />
    </div>

    <h3 align="center", style="margin-bottom: 50px;">An Async, Online, Multi-Turn, Multi-Agent RL library for training reasoning models on TextArena games.</h3>



Unstable Baselines is a **lightweight reinforcement-learning research library** focused on self-play for text-based games. 
Through its deep integration with TextArena, it supports wide range of single and multi-player games. 
The interface is simple and hackable. All learning algorithms run on a single GPU, making it easy to experiment, extend and customize: 


.. code-block:: python

   from unstable import train, get_algorithm_config

   config = get_algorithm_config("reinforce")
   config['learner']['learning_rate'] = 1e-5
   config['learner']['grad_clip'] = 0.2
   config['replay_buffer']['max_buffer_size'] = 800
   checkpoint_path = train(config)

----

.. admonition:: Why "Unstable"?
   :class: tip

   Our project is meant for rapid prototying of new research ideas. 

.. note::

   Feel free to extend this documentation and open a PR on the `GitHub repository <https://github.com/LeonGuertler/UnstableBaselines>`_.


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Introduction

   introduction/overview
   introduction/installation
   introduction/quickstart
   api/index

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Algorithms

    algorithms/reinforce
    algorithms/ppo
    algorithms/grpo

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorials

   tutorials/ppo

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Our Projects

    TextArena <https://github.com/LeonGuertler/TextArena>
    Unstable Baselines <https://github.com/LeonGuertler/UnstableBaselines>
