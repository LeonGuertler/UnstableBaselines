Game Scheduler
~~~~~~~~~~~~~~

The game scheduler is responsible for starting games for both, training and evaluation. The class balances work across GPU actors and streams results to the replay buffer.

.. raw:: html

    <div align="center">
        <img style="width: 600px;" src="../_static/gamescheduler.png" />
    </div>


API Reference
"""""""""""""

.. py:class:: GameScheduler(vllm_config, tracker, buffer, model_sampler, env_sampler, action_sampler: str = "default")
   :module: unstable.collection.game_scheduler
   :noindex:

   **Constructor parameters**

   :param vllm_config: Configuration passed to each :class:`VLLMActor` (see section **vLLM Configuration** below).
   :type vllm_config: Mapping[str, Any]
   :param tracker: Actor that provides logging and weights and biases integration.
   :type tracker: ray.actor.ActorHandle
   :param buffer: Actor that stores training trajectories and exposes.
   :type buffer: ray.actor.ActorHandle
   :param model_sampler: Component that provides the models for the next game.
   :type model_sampler: BaseModelSampler
   :param env_sampler: Component that provides the environment for the next game.
   :type env_sampler: BaseEnvSampler
   :param action_sampler: Name of the action-sampling strategy to use for evaluation. Default is "default" which samples a single action from the model.
   :type action_sampler: str, optional

   **Methods**

   .. py:method:: collect(num_train_workers: int, num_eval_workers: Optional[int] = None)

      Schedules training (and optionally evaluation) games until the buffer signals to stop.

      :param num_train_workers: Maximum number of concurrent training episodes.
      :type num_train_workers: int
      :param num_eval_workers: Maximum number of concurrent evaluation episodes.
                               If ``None``, no evaluation episodes are scheduled.
      :type num_eval_workers: Optional[int]

   .. rubric:: Internal helpers

   .. py:method:: _launch_jobs(max_train: int, max_eval: Optional[int])
   .. py:method:: _handle_finished_job(ref)
   .. py:method:: _next_train_job()
   .. py:method:: _post_train(meta: TaskMeta, game_information: GameInformation, player_trajs: List[PlayerTrajectory])
   .. py:method:: _next_eval_job()
   .. py:method:: _post_eval(meta: TaskMeta, game_information: GameInformation)


vLLM Configuration
""""""""""""""""""

**model_name** 
   HuggingFace or local model identifier.

**temperature** 
   Sampling temperature used during text generation (must be ≥ 0.0).

**max_tokens** 
   Maximum number of tokens to generate per sequence.

**max_parallel_seq** 
   Maximum number of sequences processed in parallel on a single actor.

**max_loras** 
   Maximum number of concurrently loaded LoRA adapters.

**max_model_len** 
   Maximum context window length for the underlying model.

**lora_config** 
   Optional configuration for LoRA fine-tuning adapters.

   **lora_rank** Rank of the LoRA projection matrices.

   **lora_alpha** Scaling factor for the LoRA updates.

   **lora_dropout** Dropout probability applied to LoRA layers (range 0.0–1.0).

   **target_modules** List of target submodules where LoRA adapters are applied.

      Typical target modules include:

      - ``q_proj``
      - ``k_proj``
      - ``v_proj``
      - ``o_proj``
      - ``gate_proj``
      - ``up_proj``
      - ``down_proj``
