import ray, argparse
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from typing import Dict, Optional, Union

from unstable.utils._types import TrainEnvSpec, EvalEnvSpec
from unstable.collection.game_scheduler import GameScheduler
from unstable.collection.trackers import Tracker
from unstable.collection.reward_transformations import ComposeFinalRewardTransforms, ComposeStepRewardTransforms, ComposeSamplingRewardTransforms, ComposeEpisodeSamplingRewardTransforms
from unstable.utils.templates import (
    get_model_sampler_cls,
    get_reward_transformation_cls,
    get_env_sampler_cls,
    get_replay_buffer_cls,
    get_learner_cls,
    get_algorithm_config
)


def train(config: Optional[Union[Dict, str]] = 'grpo', interface: bool = False):
    # Configuration
    if isinstance(config, str): config = get_algorithm_config(config)
    num_collection_workers = config.get('collection_workers', 256)
    num_evaluation_workers = config.get('evaluation_workers', 16)
    
    # Initialization
    ray.init(config.get('head', None), namespace=config.get('project', 'UnstableBaselines'))
    learner_config = config['learner']; checkpoint_config = config['checkpoint']
    learner_gpus = learner_config.pop('num_gpus', 1)
    gpus_per_learner = learner_config.pop('gpus_per_learner', 1)
    learner_placement_group = placement_group(bundles=[{"GPU": gpus_per_learner, "CPU": 1} for _ in range(learner_gpus)], strategy="PACK")
    ray.get(learner_placement_group.ready())
    
    # Tracker
    tracker = Tracker.options(name="Tracker").remote(
        run_name=f"{config.get('run', 'Run')}",
        wandb_project=config.get('project', 'UnstableBaselines'), wandb_id=checkpoint_config.get('wandb_id', None), wandb_config=config,
        logging_dir=config.get('logging_dir', 'outputs'),
        collection_batch_size=config.get('replay_buffer', {}).get('max_buffer_size')
    )
    
    # Environment Sampler
    env_sampler_config = config['env_sampler']
    env_sampler = get_env_sampler_cls(env_sampler_config.pop('type'))(
        train_env_specs=[
            TrainEnvSpec(env_id=env['id'], num_players=env['num_players'], num_actors=env['num_actors'], prompt_template=env['prompt_template'], action_extraction_fn=env.get('action_extraction_fn', 'default'), group_size=env.get('group_size', 1))
            for env in env_sampler_config.pop('train')
        ],
        eval_env_specs=[
            EvalEnvSpec(env_id=env['id'], num_players=env['num_players'], prompt_template=env.get('prompt_template', 'qwen3-zs'),
                        action_extraction_fn=env.get('action_extraction_fn', 'default'),
                        fixed_opponent=env.get("fixed_opponent", "google/gemini-2.0-flash-lite-001"), kind=env.get("kind", "openrouter"),
                        temperature=env.get('temperature'), top_p=env.get('top_p'), top_k=env.get('top_k'), max_tokens=env.get('max_tokens'))
            for env in env_sampler_config.pop('eval')
    ], **env_sampler_config)
    
    # Model Sampler
    model_sampler_config = config['model_sampler']
    fixed_opponents = model_sampler_config.pop('fixed_opponents') if 'fixed_opponents' in model_sampler_config else []
    fixed_checkpoints = model_sampler_config.pop('fixed_checkpoints') if 'fixed_checkpoints' in model_sampler_config else []
    model_sampler = get_model_sampler_cls(model_sampler_config.pop('type')).options(name="ModelSampler").remote(tracker=tracker, **model_sampler_config)
    for fixed_opponent in fixed_opponents: ray.get(model_sampler.add_fixed.remote(name=fixed_opponent))
    for ckpt in fixed_checkpoints: ray.get(model_sampler.add_fixed_checkpoint.remote(uid=ckpt['uid'], path=ckpt['path']))
    policy_ckpt = checkpoint_config['policy']
    ray.get(model_sampler.add_checkpoint.remote(uid=policy_ckpt['uid'], path=policy_ckpt['path'], iteration=checkpoint_config['iteration'], eval=True))
    for ckpt in checkpoint_config.get('eval_checkpoints', []):
        ray.get(model_sampler.add_eval_checkpoint.remote(uid=ckpt['uid'], path=ckpt['path']))
    
    # Replay Buffer
    replay_buffer_config = config['replay_buffer']; reward_transformations = replay_buffer_config.pop('reward_transformations'); buffer_type = replay_buffer_config.pop('type')
    replay_buffer = get_replay_buffer_cls(buffer_type).options(name="Buffer").remote(tracker=tracker,
        final_reward_transformation=ComposeFinalRewardTransforms([get_reward_transformation_cls(k)(**v) for k,v in reward_transformations['final'].items()]),
        step_reward_transformation=ComposeStepRewardTransforms([get_reward_transformation_cls(k)(**v) for k,v in reward_transformations['step'].items()]),
        sampling_reward_transformation=ComposeSamplingRewardTransforms([get_reward_transformation_cls(k)(**v) for k,v in reward_transformations['sampling'].items()]) if buffer_type == 'step_buffer' else ComposeEpisodeSamplingRewardTransforms([get_reward_transformation_cls(k)(**v) for k,v in reward_transformations['sampling'].items()]),
        **replay_buffer_config
    )
    
    # Learning algorithm
    learner_type = learner_config.pop('type')
    learner_cls = learner_type if callable(learner_type) else get_learner_cls(learner_type)
    leaners = [
        learner_cls.options(num_gpus=gpus_per_learner, name=f"Learner-{i}", scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=learner_placement_group, placement_group_bundle_index=i)).remote(
            **learner_config,
            checkpoint_cfg=checkpoint_config,
            buffer=replay_buffer,
            tracker=tracker,
            model_sampler=model_sampler,
            rank=i,
            world_size=learner_gpus
        ) for i in range(learner_gpus)
    ]
    
    # Game Scheduler
    action_sampler_config = config['action_sampler']
    eval_config = config.get('evaluation', {})
    game_scheduler = GameScheduler.options(name="GameScheduler").remote(
        vllm_config=config['vllm_config'], tracker=tracker, buffer=replay_buffer, model_sampler=model_sampler, env_sampler=env_sampler, action_sampler=action_sampler_config.pop('type'),
        eval_every=config.get('evaluation_steps'), eval_runs=config.get('evaluation_runs', 64),
        max_concurrent_workers=config.get('max_concurrent_workers')
    )
    
    # Run
    try:
        game_scheduler.collect.remote(num_train_workers=num_collection_workers, num_eval_workers=num_evaluation_workers)
        ray.get([learner.train.remote(iterations=config['learner']['total_training_steps']) for learner in leaners])
        _, current_ckpt_lora_path = model_sampler.get_current_ckpt()
    finally: 
        ray.kill(game_scheduler, no_restart=True); ray.shutdown()
    return current_ckpt_lora_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="grpo", help="Algorithm. Either 'grpo', 'ppo', or a path to a custom config file.")
    parser.add_argument("--interface", action="store_true", help="Enable monitoring terminal interface")
    args = parser.parse_args()
    train(config=args.config, interface=args.interface)