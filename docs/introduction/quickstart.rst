Quickstart
==========

Python
""""""

To run experiments directly in Python and use your own custom components, simply import the get_algorithm_config function to build your own config. 
Then, pass your modified components and run the train function:

.. code-block:: python

   from unstable import train, get_algorithm_config

   class MyModelSampler(BaseModelSampler):
        ...

   config = get_algorithm_config("reinforce")
   config['learner']['learning_rate'] = 1e-5
   config['learner']['grad_clip'] = 0.2
   config['replay_buffer']['max_buffer_size'] = 800
   config['model_sampler']['type'] = MyModelSampler
   checkpoint_path = train(config)