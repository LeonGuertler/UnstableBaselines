Configuration
=============

Unstable Baselines follows a modular architecture.
Every component can be easily configured, customized and extended.
You can find below the default configuration used in REINFORCE.
You can either adapt the values or pass over a new sub-class of the module.
All parameters of the component are passed as arguments to the constructor.

.. code-block:: python

   from unstable import train, get_algorithm_config

   class MyModelSampler(BaseModelSampler):
        ...

   config = get_algorithm_config("grpo")
   config['learner']['learning_rate'] = 1e-5
   config['learner']['grad_clip'] = 0.2
   config['replay_buffer']['max_buffer_size'] = 800
   config['model_sampler']['type'] = MyModelSampler
   checkpoint_path = train(config)

This way it is possible to isolate research on specific branches.
For example, if you want to experiment with new curriculum learning strategies, you can simply create a new environment sampler and pass it as a dictionary argument to the config.
That way, you can measure specifically the effect of the new strategy while leaving the rest of the configuration unchanged.


You can find all default configuration files in the `unstable/config <https://github.com/TextArena/UnstableBaselines/tree/main/unstable/config>`_-folder.

--------------------------------

**unstable.config.grpo.yaml**

.. code-block:: yaml
   :linenos:

   run: "default"
   project: "Test"
   model_name: &model_name "Qwen/Qwen3-1.7B-Base"
   collection_workers: 250
   evaluation_workers: 128
   evaluation_every_iterations: 20
   evaluation_runs_per_env: 256
   logging_dir: "outputs"
   training_iterations: &training_iterations 2000

   checkpoint:
     policy:
       uid: "base"
       path: null
     iteration: 0
     wandb_id: null

   env_sampler:
     type: "random"
     train:
       - id: "ConnectFour-v0-train"
         num_players: 2
         num_actors: 2
         prompt_template: "qwen3-zs"
     eval:
       - id: "SimpleTak-v0-train"
         num_players: 2
         prompt_template: "qwen3-zs"
         fixed_opponent: "google/gemini-2.5-flash-lite"

   action_sampler:
     type: "default"

   model_sampler:
     type: "mirror"
     opponent_top_k: 20
     opponent_temperature: 0.1

   learner:
     num_gpus: 1
     type: "grpo"
     total_training_steps: *training_iterations
     model_name: *model_name
     local_batch_size: 384
     micro_batch_size: 1
     learning_rate: 0.000001
     lr_scheduler_type: "constant"
     lr_warmup_ratio: 0.01
     epochs: 2
     grad_clip: 0.2
     max_train_len: null
     max_generation_len: &max_generation_len 4096
     lora_cfg: &lora_cfg
       lora_rank: 32
       lora_alpha: 32
       lora_dropout: 0.0
       target_modules:
         - "q_proj"
         - "k_proj"
         - "v_proj"
         - "o_proj"
         - "gate_proj"
         - "up_proj"
         - "down_proj"
     activation_checkpointing: false
     gradient_checkpointing: false
     use_trainer_cache: false

   replay_buffer:
     type: "step_buffer"
     max_buffer_size: 768
     reward_transformations:
       final:
         role_advantage: {}
       step:
         format_reward:
           reward: 0.0
           penalty: -0.1
         invalid_move_penalty:
           reward: 0.0
           penalty: -0.1
       sampling:
         normalize_by_env:
           z_score: true

   vllm_config:
     model_name: *model_name
     temperature: 0.7
     top_k: 50
     top_p: 0.95
     max_tokens: *max_generation_len
     max_parallel_seq: 100
     max_loras: 8
     max_model_len: 8192
     lora_config: *lora_cfg
