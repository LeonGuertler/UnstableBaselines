Action Sampler
~~~~~~~~~~~~~~

Action samplers define how an agent selects actions from a large language model.  

.. raw:: html

    <div align="center">
        <img style="width: 620px;" src="../_static/actionsampler.png" />
    </div>


API Reference
"""""""""""""

.. py:class:: BaseActionSampler(agent)
   :module: unstable.collection.action_samplers
   :noindex:

   Abstract base class for action samplers.  
   Provides a consistent callable interface over agents implementing either :code:`act_full(observation)` or a simple callable API :code:`agent(observation)`.

   :param agent: The agent or policy object used to generate actions.
   :type agent: Any

   **Methods**

   .. py:method:: sample_action(observation: str) -> Action

      Query the agent for an action given the textual observation.

      :param observation: The environment observation provided to the agent.
      :type observation: str

MajorityVotingActionSampler
---------------------------

.. py:class:: MajorityVotingActionSampler(agent, k: int = 10)
   :module: unstable.collection.action_samplers
   :noindex:

   Action sampler implementing majority voting over multiple parallel agent invocations.

   :param agent: The language model to sample from.
   :type agent: Any
   :param k: Number of parallel samples used for majority voting. Must be a positive integer.
   :type k: int

   **Methods**

   .. py:method:: sample_action(observation: str) -> Action

      Sample :math:`k` independent actions from the agent and return the majority-voted result.

      :param observation: The environment observation.
      :type observation: str

   .. py:method:: _entropy(counts: Counter) -> float
      :noindex:

      Compute Shannon entropy of the empirical vote distribution.

      :param counts: Counter of action frequencies.
      :type counts: Counter
