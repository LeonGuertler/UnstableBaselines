Configuration
=============

Unstable Baselines follows a modular architecture.
Every compoonent can be easily configured, customized and extended.
You can find below the default configuration used in GRPO. 
You can either adapt the values or pass over a new sub-class of the module. 
All parameters of the component are passed as arguments to the constructor.

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

This way it is possible to isolate research o specific branches. 
For example, if you want to experiment with new curriculum learning strategies, you can simply create a new enviroment sampler and pass it as a dictionary argument to the config.
That way, youc an measure specifically the effect of the new strategy while leaving the rest of the configuration unchanged.


You can find all default configuration files in the `unstable/config <https://www.openai.com>`_-folder.

--------------------------------

**unstable.config.grpo.yaml**

.. code-block:: yaml
   :linenos:

   run: "correct"
   project: "EXP1-Benchmarking"
   model_name: &model_name "Qwen/Qwen3-1.7B"
   collection_workers: 128
   evaluation_workers: 16
   logging_dir: "outputs"
   training_iterations: &training_iterations 1000

   checkpoint:
     uid: "base"
     path: null
     iteration: 0

   env_sampler:
     type: "random"
     train:
       - id: "Chopsticks-v0-train"
         num_players: 2
         num_actors: 2
         prompt_template: "qwen3-zs"
     eval:
       - id: "Chopsticks-v0-train"
         num_players: 2
         prompt_template: "qwen3-zs"

   action_sampler:
     type: "default"

   model_sampler:
     type: "mirror"
     registry:
       type: "default"
       fixed_opponents:
         - "google/gemini-2.0-flash-lite-001"

   learner:
     type: "grpo"
     total_training_steps: *training_iterations
     model_name: *model_name
     epochs: 2
     batch_size: 128
     mini_batch_size: 1
     learning_rate: 0.00001
     lr_scheduler_type: "constant"
     lr_warmup_ratio: 0.01
     clip_ratio: 0.2
     grad_clip: 0.2
     entropy_coeff: 0.0
     beta: 0.01
     max_train_len: 3072
     max_generation_len: &max_generation_len 2048
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

   replay_buffer:
     type: "step_buffer"
     max_buffer_size: 256
     reward_transformations:
       final:
         role_advantage: {}
       step:
         format_reward:
           reward: 1.5
         invalid_move_penalty:
           reward: 1.0
           penalty: -1.0
       sampling:
         normalize_by_env:
           z_score: true

   vllm_config:
     model_name: *model_name
     temperature: 0.6
     max_tokens: *max_generation_len
     max_parallel_seq: 128
     max_loras: 8
     max_model_len: 8192
     lora_config: *lora_cfg

