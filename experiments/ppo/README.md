# PPO — SimpleTak Self-Play

## Goal

Establish a PPO baseline for multi-turn, two-player game self-play using SimpleTak-v0. This is the primary benchmark for comparing new RL algorithms on game environments.

## Setup

- **Model**: Qwen3-1.7B-Base with LoRA (rank 32)
- **Algorithm**: PPO with GRPO-style loss, GAE advantage estimation (γ=0.7, λ=0.9)
- **Environment**: SimpleTak-v0, 2 players, self-play via `MirrorModelSampler`
- **Opponent sampling**: top-20 recent checkpoints (temperature 0.1)
- **Critic**: separate LoRA critic head, value_coeff=0.5, clip_value=0.2
- **Batch**: local_batch_size=256, 600 iterations
- **Reward shaping**: format_reward (+0.1), invalid_move_penalty (±0.2)
- **Evaluation**: every 25 steps, 256 rollouts vs. latest checkpoint

## Config

Base config: `unstable/config/ppo.yaml`

Experiment-specific overrides: `config.yaml`

## Running

```bash
python -m unstable.train --config experiments/ppo/config.yaml
```

## Notes

<!-- Add observations, ablations, and results here as the experiment progresses -->
