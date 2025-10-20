Replay Buffers
~~~~~~~~~~~~~~


We provide two replay buffers for multi-agent game collection: 
StepBuffer stores and samples individual steps across episodes while the EpisodeBuffer stores and samples entire episodes.

.. raw:: html

    <div align="center">
        <img style="width: 600px;" src="../_static/replaybuffer.png" />
    </div>

Both buffers support three kinds of reward transformations:

- **Final reward transformation**: Applied once per trajectory and passed into step-level computations.
- **Step reward transformation** Applied per step, typically using the final/episode reward.
- **Sampling reward transformation** Applied *at sampling time* to the batch being returned.

API Reference
"""""""""""""

.. py:class:: BaseBuffer(max_buffer_size, tracker, final_reward_transformation, step_reward_transformation, sampling_reward_transformation, buffer_strategy: str = "random")

   Abstract base with a minimal interface common to both buffers.

   :param max_buffer_size: Maximum number of steps the buffer can hold in total. 
   :type max_buffer_size: int
   :param tracker: Tracker for logging.
   :type tracker: unstable.collection.trackers.BaseTracker
   :param final_reward_transformation: Transformation applied once per trajectory to produce an episode-level reward.
   :type final_reward_transformation: Optional[ComposeFinalRewardTransforms]
   :param step_reward_transformation: Per-step transformation producing the step reward.
   :type step_reward_transformation: Optional[ComposeStepRewardTransforms]
   :param sampling_reward_transformation: Transformation applied when sampling a batch.
   :type sampling_reward_transformation: Optional[ComposeSamplingRewardTransforms]
   :param buffer_strategy: Buffer when full. Currently ``"random"`` downsampling is implemented.
   :type buffer_strategy: str

   **Expected methods**

   .. py:method:: add_player_trajectory(player_traj, env_id)
      :noindex:

      Add one complete trajectory to the buffer.

   .. py:method:: get_batch(batch_size)
      :noindex:

      Sample and remove a batch from the buffer.

   .. py:method:: stop()
      :noindex:

      Signal collection to stop.

   .. py:method:: size() -> int
      :noindex:

      Current number of stored steps.

   .. py:method:: continue_collection() -> bool
      :noindex:

      Whether the buffer is still accepting data.

   .. py:method:: clear()
      :noindex:

      Remove all stored data.

StepBuffer
----------

.. py:class:: StepBuffer(max_buffer_size, tracker, final_reward_transformation, step_reward_transformation, sampling_reward_transformation, buffer_strategy: str = "random")
   :module: unstable.collection.buffers
   :noindex:

   Stores steps from incoming trajectories in a single flat list and samples individual steps.

   :param max_buffer_size: Upper bound on number of stored steps. When exceeded, the buffer downsamples.
   :type max_buffer_size: int
   :param tracker: Tracker for logging.
   :type tracker: ray.actor.ActorHandle
   :param final_reward_transformation: Applied once per trajectory to produce a reward signal passed to per-step logic.
   :type final_reward_transformation: Optional[ComposeFinalRewardTransforms]
   :param step_reward_transformation: Applied to each step (given the trajectory and index) to yield the stored step reward.
   :type step_reward_transformation: Optional[ComposeStepRewardTransforms]
   :param sampling_reward_transformation: Applied to the *sampled batch of steps* before returning it.
   :type sampling_reward_transformation: Optional[ComposeSamplingRewardTransforms]
   :param buffer_strategy: Downsampling policy. Currently only ``"random"`` is supported.
   :type buffer_strategy: str

   **Methods**

   .. py:method:: add_player_trajectory(player_traj: PlayerTrajectory, env_id: str)
      
      Add the trajectory to the buffer. For each step:

      1. Compute the final reward transformation if provided; otherwise it uses the final reward of the trajectory.
      2. Compute the step reward transformation if provided; otherwise it uses the final reward.

      After insertion, if the number of stored steps exceeds the maximum buffer size, the buffer downsamples.

      :param player_traj: The trajectory.
      :type player_traj: PlayerTrajectory
      :param env_id: Environment identifier.
      :type env_id: str

   .. py:method:: get_batch(batch_size: int) -> List[Step]

      Sample without replacement a set of steps and remove them from the buffer.

      :param batch_size: Number of steps to sample.
      :type batch_size: int

   .. py:method:: stop()

      Mark the buffer as closed to further collection. Does not clear data.

   .. py:method:: size() -> int

      Current number of stored steps.

   .. py:method:: continue_collection() -> bool

      Returns ``True`` until :meth:`stop` is called.

   .. py:method:: clear()

      Remove all stored steps.

EpisodeBuffer
-------------

.. py:class:: EpisodeBuffer(max_buffer_size, tracker, final_reward_transformation, step_reward_transformation, sampling_reward_transformation, buffer_strategy: str = "random")
   :module: unstable.collection.buffers
   :noindex:

   Stores entire episodes and samples batches of full episodes. 

   :param max_buffer_size: Upper bound on number of stored steps. When exceeded, the buffer downsamples.
   :type max_buffer_size: int
   :param tracker: Tracker for logging.
   :type tracker: ray.actor.ActorHandle
   :param final_reward_transformation: Applied once per trajectory to produce a reward signal passed to per-step logic.
   :type final_reward_transformation: Optional[ComposeFinalRewardTransforms]
   :param step_reward_transformation: Applied to each step (given the trajectory and index) to yield the stored step reward.
   :type step_reward_transformation: Optional[ComposeStepRewardTransforms]
   :param sampling_reward_transformation: Applied to the *sampled batch of steps* before returning it.
   :type sampling_reward_transformation: Optional[ComposeSamplingRewardTransforms]
   :param buffer_strategy: Downsampling policy. Currently only ``"random"`` is supported.
   :type buffer_strategy: str

   **Methods**

   .. py:method:: add_player_trajectory(player_traj: PlayerTrajectory, env_id: str)

      Add the trajectory to the buffer. For each step:

      1. Compute the final reward transformation if provided; otherwise it uses the final reward of the trajectory.
      2. Compute the step reward transformation if provided; otherwise it uses the final reward.

      After insertion, if the number of stored steps exceeds the maximum buffer size, the buffer downsamples entire episodes.

      :param player_traj: The source trajectory.
      :type player_traj: PlayerTrajectory
      :param env_id: Environment identifier.
      :type env_id: str

   .. py:method:: get_batch(batch_size: int) -> List[List[Step]]

      Sample without replacement a set of episodes and remove them from the buffer.

      :param batch_size: Number of episodes to sample.
      :type batch_size: int

   .. py:method:: stop()

      Mark the buffer as closed to further collection. Does not clear data.

   .. py:method:: size() -> int

      Current number of stored steps.

   .. py:method:: continue_collection() -> bool

      Returns ``True`` until :meth:`stop` is called.

   .. py:method:: clear()

      Remove all stored episodes.
