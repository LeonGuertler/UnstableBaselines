# UnstableBaselines — Claude Context

## Project Overview

**UnstableBaselines** is an async, online RL training library for fine-tuning language models on TextArena games and reasoning tasks. The design philosophy is simplicity and hackability — easy to prototype new RL ideas.

Key traits:
- **LoRA-first**: All fine-tuning uses LoRA adapters (never full fine-tuning)
- **Async actor/learner**: Ray actors collect game data concurrently while a DeepSpeed learner updates the policy
- **vLLM inference**: LoRA hot-swapping via vLLM for fast multi-model inference
- **W&B tracking**: All runs logged to Weights & Biases

## Repository Layout

```
unstable/
  train.py                  # Entry point: python -m unstable.train --config <yaml>
  eval.py                   # Evaluation only
  learner/
    base.py                 # BaseLearner: model init, DeepSpeed, checkpointing, grad accumulation
    reinforce.py            # REINFORCELearner
    ppo.py                  # PPOLearner (actor-critic with GAE)
    reinforce_prm.py        # REINFORCEPRMLearner (Q-value critic for step rewards)
    models.py               # build_peft_model(), LoRA setup, value head
  collection/
    game_scheduler.py       # GameScheduler Ray actor: orchestrates workers, evals, checkpoint syncing
    actor.py                # VLLMActor: wraps vLLM, handles LoRA weight loading
    buffers.py              # StepBuffer, EpisodeBuffer (experience replay)
    model_samplers.py       # MirrorModelSampler, FixedOpponentModelSampler, WinRateModelSampler
    env_samplers.py         # UniformRandomEnvSampler
    action_samplers.py      # Default and majority-vote action extraction
    reward_transformations.py  # Composable transforms: FormatReward, InvalidMovePenalty,
                               # RoleAdvantage, NormalizeByEnv, GroupRelativeAdvantage, etc.
    trackers.py             # Tracker: W&B logging, checkpoint management
  config/
    ppo.yaml                # PPO baseline — SimpleTak-v0 self-play
    reinforce.yaml          # REINFORCE — AIME multi-turn reasoning
    reinforce-singleturn.yaml  # REINFORCE — math500 single-turn
    eval.yaml               # Eval-only for game environments
    eval-singleturn.yaml    # Eval-only for single-turn environments
  utils/
    templates.py            # Config loading, model/buffer/sampler factories, prompt templates
    _types.py               # Core dataclasses: Step, PlayerTrajectory, GameInformation, ModelMeta
    logger.py               # Logging setup
experiments/                # Experiment configs, notes, and run scripts (see experiments/README.md)
examples/                   # Usage examples
```

## Running Training

```bash
python -m unstable.train --config unstable/config/ppo.yaml
# or for an experiment:
python experiments/ppo/run.py
```

## Core Data Flow

1. `GameScheduler` spawns `VLLMActor` workers with the current LoRA checkpoint
2. Workers play games in TextArena environments, collecting `Step` objects
3. Steps go into `StepBuffer` or `EpisodeBuffer` with reward transformations applied
4. When buffer has enough data, learner's `update()` is called
5. Learner runs DeepSpeed backward pass, saves checkpoint
6. `GameScheduler` hot-swaps the new LoRA weights into vLLM

## Key Concepts

**Reward Transformations** are applied at three stages (configured in `replay_buffer.reward_transformations`):
- `step`: applied per step as collected (e.g., `format_reward`, `invalid_move_penalty`)
- `final`: applied at episode end (e.g., `role_advantage` — gives each player their win/loss signal)
- `sampling`: applied at batch sample time (e.g., `normalize_by_env`, `group_relative_advantage`)

**Model Samplers** control opponent selection:
- `mirror`: opponent samples from recent checkpoints (self-play)
- `fixed`: fixed external opponent (e.g., Gemini via API)
- `win_rate`: opponent selected based on win-rate tracking

**Prompt Templates** (in `utils/templates.py`): `qwen3-zs`, `qwen3-math`, `llama-zs`, `gemma-zs`, etc.

**Buffers**:
- `StepBuffer`: stores individual steps; used when each step has its own reward signal
- `EpisodeBuffer`: stores full episodes; supports `flatten=True` to unroll into steps after episode completion

## Adding a New Experiment

1. Create `experiments/<name>/` with `README.md`, `config.yaml`, and optionally `run.py`
2. Copy the closest base config from `unstable/config/` and modify
3. For a new learner algorithm, subclass `BaseLearner` and implement `_update()` and `_micro_batch_update_step()`
4. For a new reward transformation, implement the callable interface in `reward_transformations.py`
5. Register new learner/sampler types in `utils/templates.py`

## Experiments Index

See `experiments/README.md` for an overview of all experiments and their status.
