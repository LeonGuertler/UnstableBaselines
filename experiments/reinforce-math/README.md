# REINFORCE — math Single-Turn

## Goal

REINFORCE baseline for single-turn mathematical reasoning on math500. Uses group-relative advantage (GRPO-style) to normalize rewards across groups of 8 samples per problem.

## Setup

- **Model**: Qwen3-1.7B-Base with LoRA (rank 32)
- **Algorithm**: REINFORCE (no KL, no entropy)
- **Environment**: math500-v0, single-player, single-turn
- **Prompt template**: `qwen3-math` (open-ended reasoning format)
- **Group size**: 8 samples per problem for advantage normalization
- **Batch**: local_batch_size=256, 600 iterations, max 4096 tokens
- **Reward shaping**:
  - `step`: format_reward (±0.1)
  - `sampling`: group_relative_advantage
- **Buffer**: EpisodeBuffer with flatten=True, group_size=8 (max 256)
- **Evaluation**: every 20 steps, 500 rollouts on math500-v0

## Config

Base config: `unstable/config/reinforce-singleturn.yaml`

Experiment-specific overrides: `config.yaml`

## Running

```bash
python -m unstable.train --config experiments/reinforce-math/config.yaml
```

## Notes

<!-- Add observations, ablations, and results here as the experiment progresses -->
