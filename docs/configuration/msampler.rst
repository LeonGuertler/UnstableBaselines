Model Sampler
~~~~~~~~~~~~~

This compoenent chooses opponents for training/evaluation jobs.
It maintains a model registry that maintains metadata and TrueSkill ratings for all models (checkpoints and fixed baselines)

.. raw:: html

    <div align="center">
        <img style="width: 600px;" src="../_static/modelsampler.png" />
    </div>

API Reference
"""""""""""""

.. py:class:: BaseModelSampler(model_registry)
   :noindex:

   Abstract base for opponent sampling strategies. Samplers read from the registry to select opponents and push back results after games.

   :param model_registry: Actor that keeps metadata about each checkpoint and fixed opponent.
   :type model_registry: ray.actor.ActorHandle

   **Methods**

   .. py:method:: get_current_ckpt() -> tuple[str | None, str]

      Fetch the current checkpoint UID and its corresponding path/name.

   .. py:method:: update(game_info: GameInformation, job_info: Dict[str, Any]) -> None

      Push training/evaluation outcomes back to the registry.

      :param game_info: Holds per-episode results including :code:`final_rewards` per player ID.
      :type game_info: unstable.utils._types.GameInformation
      :param job_info: Dict with at least :code:`"models"` (list of dicts with :code:`uid`, :code:`pid`) and :code:`"env_id"`.
      :type job_info: Dict[str, Any]

   **Expected Methods**

   .. py:method:: sample_opponent() -> tuple[str, str, None, str]
      :noindex:

      Strategy-specific opponent sampling. 

Fixed Opponent Sampler
----------------------

.. py:class:: FixedOpponentModelSampler(model_registry, include_current_ckpt: bool = False)
   :noindex:

   Samples an opponent uniformly at random from fixed baselines. Optionally, the current checkpoint may be included in the candidate set.

   :param model_registry: Actor that keeps metadata about each checkpoint and fixed opponent.
   :type model_registry: ray.actor.ActorHandle
   :param include_current_ckpt: Whether to include the current checkpoint as a possible opponent.
   :type include_current_ckpt: bool

   **Methods**

   .. py:method:: sample_opponent() -> tuple[str, str, None, str]

      Return a randomly selected opponent. 


AsynchronousModelSampler
------------------------

.. py:class:: AsynchronousModelSampler(model_registry)
   :noindex:

   Samples an opponent uniformly at random from checkpoint models. 
   Designed to support asynchronous self-play where multiple recent checkpoints are active concurrently.

   :param model_registry: Actor that keeps metadata about each checkpoint and fixed opponent.
   :type model_registry: ray.actor.ActorHandle

   **Methods**

   .. py:method:: sample_opponent() -> tuple[str, str, None, str]

      Return a randomly selected recent checkpoint as opponent along with its metadata.

ModelRegistry (Ray actor)
-------------------------

.. py:class:: ModelRegistry(tracker, beta: float = 4.0, k: int = 1)
   :noindex:

   Stores models, keeps running TrueSkill ratings, and logs periodic snapshots for analysis.

   :param tracker: The experiment logging tracker. 
   :type tracker: ray.actor.ActorHandle
   :param beta: TrueSkill performance variance parameter (β). Higher values increase the magnitude of rating updates.
   :type beta: float
   :param k: Maximum number of active checkpoints kept in a rolling window. 
   :type k: int

   **Methods**

   .. py:method:: add_checkpoint(uid: str, path: str, iteration: int, inherit: bool = True) -> None

      Register a new checkpoint model. If :code:`inherit=True` and a current checkpoint exists, initialize the new rating from the current checkpoint’s μ and 2×σ; otherwise use the default TrueSkill prior.

      :param uid: Unique identifier for the checkpoint.
      :type uid: str
      :param path: Filesystem path or model identifier (LoRA path, HF name, etc.).
      :type path: str
      :param iteration: Training iteration when this checkpoint was produced.
      :type iteration: int
      :param inherit: Whether to initialize rating from the current checkpoint.
      :type inherit: bool

   .. py:method:: add_fixed(name: str, prior_mu: float = 25.0) -> None

      Register a fixed baseline model. 

      :param name: Name of the fixed baseline.
      :type name: str
      :param prior_mu: Prior TrueSkill μ for the baseline.
      :type prior_mu: float

   .. py:method:: update_ratings(uids: List[str], scores: List[float], env_id: str, dummy_uid: str = "fixed-env") -> None

      Update TrueSkill ratings after a match between the provided models.

      :param uids: Ordered list of participating model UIDs.
      :type uids: List[str]
      :param scores: Real-valued performance scores aligned with :code:`uids`.
      :type scores: List[float]
      :param env_id: Identifier of the environment/task where the match occurred (for logging).
      :type env_id: str
      :param dummy_uid: UID of the synthetic opponent for single-player updates.
      :type dummy_uid: str

   .. py:method:: get_all_models() -> Dict[str, ModelMeta]

      Return a deep copy of the internal registry of all models and their metadata.

   .. py:method:: get_current_ckpt() -> str | None

      Return the UID of the current checkpoint (or :code:`None` if none exists).

   .. py:method:: get_name_or_lora_path(uid: str) -> str

      Return the model’s :code:`path_or_name` for the given UID.

   **Static Methods**

   .. py:method:: _scores_to_ranks(scores: List[float]) -> List[int]
      :noindex:

      Convert scores to ranking indices (lower rank value is better). Ties receive the same rank.
