Environment Sampler
~~~~~~~~~~~~~~~~~~~

Environment samplers define how training and evaluation environments are selected during data collection and evaluation. They provide a flexible interface for sampling from one or more configured environment specifications, with optional stochasticity or adaptive scheduling.
We currently provide a uniform random sampler implementation.

.. raw:: html

    <div align="center">
        <img style="width: 600px;" src="../_static/envsampler.png" />
    </div>


API Reference
"""""""""""""

.. py:class:: BaseEnvSampler(train_env_specs, eval_env_specs=None, rng_seed=489)

   Abstract base class for all environment samplers.

   :param train_env_specs: List of environment specifications used for training.
   :type train_env_specs: List[unstable.utils._types.TrainEnvSpec]
   :param eval_env_specs: Optional list of environment specifications used for evaluation.
   :type eval_env_specs: Optional[List[unstable.utils._types.EvalEnvSpec]]
   :param rng_seed: Optional integer seed for reproducible random sampling.
   :type rng_seed: Optional[int]

   **Methods**

   .. py:method:: env_list() -> str

      Return a comma-separated string listing the identifiers of all training environments.

   **Expected Methods**

   .. py:method:: sample(kind: str = "train") -> TrainEnvSpec | EvalEnvSpec
      :noindex:

      Sample one environment specification according to the implemented strategy.

      :param kind: Whether to sample from ``"train"`` or ``"eval"`` environments.
      :type kind: str

   .. py:method:: update(avg_actor_reward: float, avg_opponent_reward: float | None)
      :noindex:

      Update internal state based on observed training metrics. 

      :param avg_actor_reward: Average episodic reward achieved by the learning agent.
      :type avg_actor_reward: float
      :param avg_opponent_reward: Optional average reward of the opponent (if applicable).
      :type avg_opponent_reward: Optional[float]


UniformRandomEnvSampler
-----------------------

.. py:class:: UniformRandomEnvSampler(train_env_specs, eval_env_specs=None, rng_seed=489)
   :module: unstable.collection.env_samplers
   :noindex:

   Samples environments uniformly at random.

   :param train_env_specs: List of environment specifications used for training.
   :type train_env_specs: List[unstable.utils._types.TrainEnvSpec]
   :param eval_env_specs: Optional list of environment specifications used for evaluation.
   :type eval_env_specs: Optional[List[unstable.utils._types.EvalEnvSpec]]
   :param rng_seed: Random seed for reproducibility.
   :type rng_seed: Optional[int]

   **Methods**

   .. py:method:: sample(kind: str = "train") -> TrainEnvSpec | EvalEnvSpec

      Sample a single environment specification uniformly at random.

      :param kind: Sampling mode. Either ``"train"`` or ``"eval"``.
      :type kind: str

   .. py:method:: update(avg_actor_reward: float, avg_opponent_reward: float | None) -> None

      This method is a **no-op** for the uniform random strategy, since sampling probabilities are fixed. 

      :param avg_actor_reward: Average episodic reward achieved by the learning agent.
      :type avg_actor_reward: float
      :param avg_opponent_reward: Optional average reward of the opponent.
      :type avg_opponent_reward: Optional[float]


---
