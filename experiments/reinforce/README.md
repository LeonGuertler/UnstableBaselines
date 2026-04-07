# REINFORCE — TicTacToe Self-Play

## Goal

Train a model to play TicTacToe via REINFORCE with self-play.

## Setup

- **Model**: Qwen3-1.7B-Base with LoRA (rank 32)
- **Algorithm**: REINFORCE with GRPO loss
- **Environment**: TicTacToe-v0-train, 2 players, 2 actors per game
- **Opponent sampling**: `MirrorModelSampler` (self-play)
- **Batch**: local_batch_size=64, 600 iterations
- **Generation**: max 16384 tokens (long chain-of-thought)
- **Reward shaping**:
  - `step`: format_reward (±0.1), invalid_move_penalty (±0.2)
  - `final`: role_advantage (win/loss by role)
  - `sampling`: normalize_by_env with z-score
- **Buffer**: StepBuffer (max 768), individual step rewards
- **Evaluation**: disabled (no evaluation workers)

## Config

Base config: `unstable/config/reinforce.yaml`

Experiment-specific overrides: `config.yaml`

## Running

```bash
python -m unstable.train --config experiments/reinforce/config.yaml
```

## Notes

<!-- Add observations, ablations, and results here as the experiment progresses -->
