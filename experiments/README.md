# Experiments

Each subdirectory is a self-contained experiment with its own config, run script, and notes.

| Experiment | Algorithm | Environment | Status |
|---|---|---|---|
| [ppo](ppo/) | PPO | SimpleTak-v0 (self-play) | baseline |
| [reinforce](reinforce/) | REINFORCE | TicTacToe-v0-train (self-play) | baseline |
| [reinforce-math](reinforce-math/) | REINFORCE | math500 (single-turn) | baseline |

## Adding an Experiment

```
experiments/
  <name>/
    README.md    # hypothesis, setup, results
    config.yaml  # full config (copy from unstable/config/ and modify)
    run.py       # optional launcher script
```
