# Unstable Baselines Documentation

> **Version:** 0.1 · **Last Updated:** 2025-06-21

---

## Table of Contents

1. [Introduction](#introduction)

2. [Getting Started](#getting-started)
   * Installation
   * Quick Start

3. [Architecture Overview](#architecture-overview)

4. [Core Modules](#core-modules)
   * [Actor (`actor.py`)](#actor)
   * [Collector (`collector.py`)](#collector)
   * [Model Pool (`model_pool.py`)](#model-pool)
   * [Step Buffer (`buffer.py`)](#step-buffer)
   * [Learner (`learners/standard_learner.py`)](#learner)
   * [Tracker (`trackers.py`)](#tracker)
   * [Terminal Interface (`terminal_interface.py`)](#terminal-interface)
   * [Core Data Structures (`core.py`)](#core-data-structures)

5. [Reward Transformations](#reward-transformations)
   * Final Reward
   * Step Reward
   * Sampling Reward

6. [Algorithms](#algorithms)
   * Reinforce (`algorithms/reinforce.py`)
   * Extending with Custom Algorithms

7. [Utilities and Helpers](#utilities-and-helpers)
   * Templates (`utils/templates.py`)
   * Logging (`utils/logging.py`)

8. [Configuration Reference](#configuration-reference)

9. [Contributing](#contributing)

10. [Contact](#contact-and-support)

---

## Introduction

Brief overview of Unstable Baselines, goals, and main features.

---

## Getting Started

### Installation
```bash
# build TextArena v0.6.9 (until it’s on PyPI)
git clone https://github.com/LeonGuertler/TextArena.git
cd TextArena
git checkout v0.6.9
python3 setup.py sdist bdist_wheel
pip install -e .
cd ..

# install UnstableBaselines
pip install unstable-rl
```

### Quick Start
To get you started, in this short example we will run you through the process of training `Qwen3-1.7B-Base` via **mirror self-play** on _SimpleTak_ and evaluating it against `google/gemini-2.0-flash-lite-001` on _SimpleTak_ and _KuhnPoker_. We will be running the experiments on 3xRTX6000 ada. If you are limited to 24gb of vRam, you can reduce the `MAX_TRAIN_SEQ_LEN` to around _2500_; this means that the model will only be trained on the first 2500 prompt+answer tokens, but can still generate answer that are longer than that. Since (in our experience) models tend to shorten their reasoning throughout training, this works very well.


```python
import ray, unstable
import unstable.reward_transformations as retra

ray.init(namespace="unstable")

tracker = unstable.Tracker.options(name="Tracker").remote(run_name="demo", wandb_project="UB")

step_buffer = unstable.StepBuffer.options(name="StepBuffer").remote(
    max_buffer_size=768, 
    tracker=tracker,
    final_reward_transformation=retra.ComposeFinalRewardTransforms([retra.RoleAdvantageByEnvFormatter()]),
    step_reward_transformation=retra.ComposeStepRewardTransforms([retra.RewardForFormat(1.5), retra.PenaltyForInvalidMove(1.0, -1.0)]),
    sampling_reward_transformation=retra.ComposeSamplingRewardTransforms([retra.NormalizeRewardsByEnv(True)]),
)

model_pool = unstable.ModelPool.options(name="ModelPool").remote(sample_mode="mirror", max_active_lora=3, tracker=tracker)
ray.get(model_pool.add_checkpoint.remote(path=None, iteration=-1)) # set initial checkpoint as no LoRA

lora_cfg = {
    "lora_rank": 32, "lora_alpha": 32, "lora_dropout": 0.0,
    "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj", "up_proj","down_proj"]
}
collector = unstable.Collector.options(name="Collector").remote(
    num_actors=2, 
    step_buffer=step_buffer, 
    model_pool=model_pool, 
    tracker=tracker,
    vllm_config={
        "model_name": "Qwen/Qwen3-1.7B-base", 
        "max_parallel_seq": 128,
        "max_tokens": 4096, 
        "max_loras": 5, 
        "lora_config": lora_cfg, 
        "max_model_len": 8192
    },
    training_envs=[("SimpleTak-v0-train", 2, "qwen3-zs")], # (env-id, num players, prompt template)
    evaluation_envs=[("SimpleTak-v0-train", 2, "qwen3-zs"), ("KuhnPoker-v0-train", 2, "qwen3-zs")],
    evaluation_opponent="google/gemini-2.0-flash-lite-001",
)

learner = unstable.StandardLearner.options(num_gpus=1, name="Learner").remote(
    model_name="Qwen/Qwen3-1.7B-base", 
    step_buffer=step_buffer,
    model_pool=model_pool,
    tracker=tracker,
    algorithm=unstable.algorithms.Reinforce(),
    batch_size=384,
    mini_batch_size=1,
    learning_rate=1e-5,
    grad_clip=0.2,
    lora_cfg=lora_cfg,
    activation_checkpointing=False,
    gradient_checkpointing=False,
    max_train_len=None, # always train on the full sequence
    max_generation_len=4096, # important for Dr. GRPO
)

# start the collection and training loops
collector.collect.remote(num_workers=384, num_eval_workers=16)  
ray.get(learner.train.remote(200)) # total update steps
```
In a Nutshell, the **Collector** will maintain `384` and `16` in parallel running collection and evaluation games (respectively). Whenever a game finishes, the trajectory is passed to the **StepBuffer** and a new game is started. The **StepBuffer** splits each trajectory into steps and applies the specified reward transformations (on the game and step level first; and batch level once the Learner pulls the next batch).

The **Learner** will periodically (once every 0.2 seconds) check if the **StepBuffer** has accumulated enough data for training. If so, it'll request a full training batch from the **StepBuffer**, train on the data, and push the new set of LoRA weights to the **ModelPool**.

The **Collector** will keep collecting episodes until the Learner tells it to stop (in this case, after `200` update steps).

Since we set `num_eval_workers=16`, throughout training there are always 16 eval games running in parallel (using the most recent lora checkpoint). Running 200 learner steps took a total of ~12h on the 3xRTX6000 ada setup we used.
![Results (light)](https://raw.githubusercontent.com/LeonGuertler/UnstableBaselines/main/docs/results_plot_light.png#gh-light-mode-only)
![Results (dark)](https://raw.githubusercontent.com/LeonGuertler/UnstableBaselines/main/docs/results_plot_dark.png#gh-dark-mode-only)


As can be seen in the plots the Win-Rate against a fixed opponent (in this case `google/gemini-2.0-flash-lite-001`) improves significantly for both the training and evaluation environment, showing that at least some of learned reasoning patterns generalize to other tasks and problems.


---

## Architecture Overview <a id="architecture-overview"></a>

The runtime can be thought of as three asynchronous loops:
```
                                                ┌───────────────┐
                                                │               │
                                                │   Algorithm   │
                                                │               │
                                                └───────────────┘
                                                        ▲        
                                                        │ Get Loss &
                                                        │ update weights
                                                        ▼
    ┌───────────────┐                           ┌───────────────┐
    │               │    Register new lora      │               │
    │   Model Pool  │◀──────────────────────────│    Learner    │
    │               │       checkpoint          │               │
    └───────────────┘                           └───────────────┘
           ▲ │                                         ▲ │ 
           │ │ Sample                        If enough │ │ Check if enough
    Update │ │ Opponent                     data, pull │ │ data for training
 Trueskill │ │                          the next batch │ │ is available
           │ ▼                                         │ ▼
    ┌───────────────┐                           ┌───────────────┐
    │               │     Process and store     │               │
    │   Collector   │──────────────────────────▶│   StepBuffer  │
    │               │  collected Trajectories   │               │
    └───────────────┘                           └───────────────┘
           ▲ │
           │ │ Maintain
    return │ │ Pool of 
Trajectory │ │ n parallel
           │ │ workers
           │ ▼
     ┌─────────────┐
     │  run_game() │
     │  train\eval │
     └─────────────┘
```

* **Collector** instances roll games with the latest learner checkpoint vs. opponents sampled by the **ModelPool**.
* End‑of‑game rewards & formatted trajectories land in the **StepBuffer**.
* The **Learner** periodically drains a batch, performs a gradient step, saves a LoRA checkpoint and registers it with the **ModelPool**.
* The **Tracker** aggregates metrics; the **Terminal Interface** turns them into a live Rich dashboard.

---

---

# Core Modules

Below is a high‑level index of every core component in **Unstable Baselines**. Click any row (or the ▸ icon) to expand its full reference.

| Module                   | Source File                    | One‑line Purpose                                   |
| ------------------------ | ------------------------------ | -------------------------------------------------- |
| **VLLMActor**            | `actor.py`                     | GPU‑bound async text generation + LoRA hot‑swap    |
| **Collector**            | `collector.py`                 | Orchestrates episode rollout & trajectory capture  |
| **ModelPool**            | `model_pool.py`                | Checkpoint registry, ELO scores, opponent sampling |
| **StepBuffer**           | `buffer.py`                    | Replay buffer & prioritised sampling               |
| **Learner**              | `learners/standard_learner.py` | PPO / REINFORCE optimiser & weight sync            |
| **Tracker**              | `trackers.py`                  | Centralised metrics & experiment logging           |
| **TerminalInterface**    | `terminal_interface.py`        | Lightweight CLI dashboard                          |
| **Core Data Structures** | `core.py`                      | `Trajectory`, `EpisodeResult`, etc. schema         |

---

<details>
<summary><strong>VLLMActor (`actor.py`)</strong></summary>

## `VLLMActor` — *actor.py* <a id="actor"></a>

Asynchronous, Ray‑based wrapper around a single **vLLM** engine instance.
Receives text‑generation requests, batches them on a GPU, supports **LoRA** hot‑swapping, and reports rich throughput metrics.

### Parameters

| Name      | Type                    | Meaning                                              |
| --------- | ----------------------- | ---------------------------------------------------- |
| `cfg`     | `Dict[str, Any]`        | Parsed YAML/CLI configuration (selected keys below). |
| `tracker` | `ray.actor.ActorHandle` | Central metrics sink.                                |
| `name`    | `str`                   | Human‑readable tag used in logs & dashboards.        |

| **`cfg` keys consumed here**       | Purpose                                           |
| ---------------------------------- | ------------------------------------------------- |
| `model_name`                       | Base model (HF id or local path).                 |
| `max_loras`                        | Maximum resident LoRA adapters (GPU + CPU).       |
| `lora_config.lora_rank`            | Rank for each adapter.                            |
| `max_parallel_seq`                 | Upper bound on concurrent sequences per `step()`. |
| `max_model_len`                    | Context length.                                   |
| `temperature / top_p / max_tokens` | Sampling hyper‑parameters.                        |

### Attributes

* **`engine`** `vllm.LLMEngine` – underlying generator initialised from **EngineArgs**.
* **`sampling_params`** `vllm.SamplingParams` – immutable settings shared by every request.
* **`submit_prompt()`** – awaitable API entry‑point.
* **`_batch_loop()`** – background task that drains the queue and calls `engine.step()`.
* **`_report_loop()`** – background task that sends queue / TPS metrics to *Tracker* every 5 s.
* **`_tok_rate()`** – helper for rolling tokens‑per‑second.

### Runtime Lifecycle

1. **`submit_prompt`** – queues *(prompt, lora)* pair; returns an `asyncio.Future`.
2. **`_batch_loop`** – every 20 ms drains the queue, adds requests to vLLM, calls `engine.step()`, timestamps new tokens for TPS, fulfils finished futures.
3. **`_report_loop`** – every 5 s logs & forwards `{queued,running,tok_s}` to *Tracker*.
4. **Shutdown** – cancelling the Ray actor stops both background tasks gracefully.

### Public API Summary

| Method          | Signature                                                     | Purpose                                                                |
| --------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `submit_prompt` | `async (prompt: str, lora_path: Optional[str] = None) -> str` | Enqueue a generation job and await the resulting text.                 |
| `_tok_rate`     | `(window: float = 2.0) -> float`                              | Rolling tokens‑per‑second over *window* s (internal, handy for tests). |

</details>

<details>
<summary><strong>Collector (`collector.py`)</strong></summary>

## `Collector` — *collector.py* <a id="collector"></a>

Ray actor responsible for orchestrating self‑play **training** episodes and fixed‑opponent **evaluation** episodes. It routes finished trajectories to the learner’s **StepBuffer**, maintains ELO scores via **ModelPool**, and logs everything through **Tracker**.

### Responsibilities

* Spawns `num_actors` GPU workers (`VLLMActor`) and assigns episodes round‑robin.
* Samples training & evaluation environments/opponents.
* Submits remote `play_episode` tasks, tracks them in `flight`, and handles results.
* Streams trajectories to **StepBuffer**, pushes game outcomes to **ModelPool**, and records metrics via **Tracker**.

### Constructor Arguments

| Name                      | Type                                | Purpose                                        |
| ------------------------- | ----------------------------------- | ---------------------------------------------- |
| `num_actors`              | `int`                               | How many `VLLMActor` GPUs to spawn.            |
| `step_buffer`             | `ray.actor.ActorHandle`             | Remote buffer storing raw steps.               |
| `model_pool`              | `ray.actor.ActorHandle`             | Checkpoint registry & ELO logic.               |
| `tracker`                 | `BaseTracker`                       | Central experiment logger.                     |
| `vllm_config`             | `dict`                              | Config forwarded to each `VLLMActor`.          |
| `training_envs`           | `list[(env_id, players, template)]` | Candidate envs for self‑play.                  |
| `evaluation_envs`         | `list[(env_id, players, template)]` | Candidate envs for offline eval.               |
| `evaluation_opponent`     | `str`                               | Fixed opponent HF / OpenRouter model.          |
| `max_eval_games_per_ckpt` | `int`                               | Cap evaluation episodes per checkpoint × env.  |
| `filter_opponent_invalid` | `bool`                              | Drop games ended by opponent invalid.          |
| `action_extraction`       | `str`                               | Key selecting extraction/formatting functions. |

### Key Methods

| Method                                   | Purpose                                                                            |
| ---------------------------------------- | ---------------------------------------------------------------------------------- |
| `collect(num_workers, num_eval_workers)` | Main loop: keeps *num\_workers* train & *num\_eval\_workers* eval tasks in flight. |
| `_submit_train()`                        | Launches a training episode with a sampled opponent.                               |
| `_submit_eval(ckpt_uid)`                 | Launches an evaluation episode against the fixed opponent.                         |
| `_handle_finished(ref)`                  | Processes a completed `play_episode`; delegates to `_post_train/_post_eval`.       |
| `_post_train` / `_post_eval`             | Push trajectory / eval reward to downstream subsystems.                            |

### Episode Flow

1. **Spec creation** – build `PlaySpec` describing env, players, checkpoint paths & seeds.
2. **Remote rollout** – `play_episode.remote(spec, actor)` executes the full loop off‑process.
3. **Result handling** – finished futures are popped from `flight`; data streamed to buffers & loggers.
4. **Back‑pressure** – honours `StepBuffer.continue_collection()` to pause when buffer is near capacity.

### Practical Tips

* Increase `num_eval_workers` if evaluation becomes a bottleneck.
* Enable `filter_opponent_invalid` in competitive settings to ignore wins by opponent invalid move.
* Separate `training_envs` & `evaluation_envs` to avoid evaluator leakage.

</details>

<details>
<summary><strong>ModelPool (`model_pool.py`)</strong></summary>

## `ModelPool` — *model\_pool.py* <a id="model-pool"></a>

Central registry and rating system for **all opponents**: learner checkpoints and fixed baseline models.
Maintains **TrueSkill** ratings, exploration statistics, opponent sampling logic, and enforces a VRAM‑friendly cap on active LoRA adapters.

### Constructor Arguments

| Name              | Type                            | Purpose                                               |
| ----------------- | ------------------------------- | ----------------------------------------------------- |
| `sample_mode`     | `str`                           | Opponent selection strategy (see *Sampling Modes*).   |
| `max_active_lora` | `int`                           | Max number of checkpoint LoRAs flagged `active=True`. |
| `tracker`         | `ray.actor.ActorHandle \| None` | Optional tracker for dashboard snapshots.             |
| `lag_range`       | `(int,int)`                     | Low/high indices used by the *lagged* strategy.       |

### Responsibilities

* **Checkpoint registry** – `add_checkpoint()` logs a new UID, carries forward μ/σ.
* **Fixed opponents** – `add_fixed()` registers static baselines (no checkpoints).
* **Opponent sampling** – `sample(uid_me)` implements 6+ heuristics.
* **Rating updates** – `push_game_outcome()` calls `_update_ratings()` and `_register_game()`.
* **Exploration metrics** – Tracks state‑space coverage via `ExplorationTracker`.
* **LoRA pool maintenance** – `_maintain_active_pool()` flips `Opponent.active` flags to honor `max_active_lora`.
* **Snapshotting** – `snapshot()` pushes a JSON‑serialisable view to *Tracker* for later analysis.

### Key Methods

| Method                                                                 | Returns                | Summary                                            |
| ---------------------------------------------------------------------- | ---------------------- | -------------------------------------------------- |
| `current_uid()`                                                        | `str \| None`          | UID of the latest learner checkpoint.              |
| `latest_ckpt()`                                                        | `str \| None`          | Alias for `current_uid()`.                         |
| `ckpt_path(uid)`                                                       | `(path, kind) \| None` | Resolve a UID to (filesystem path, kind).          |
| `sample(uid_me)`                                                       | `str`                  | Choose an opponent UID according to `sample_mode`. |
| `push_game_outcome(uid_me, uid_opp, final_reward, action_seq, env_id)` | —                      | Update ratings & exploration, then snapshot state. |

### Sampling Modes

| Mode            | Logic                                                       |         |                                     |
| --------------- | ----------------------------------------------------------- | ------- | ----------------------------------- |
| `fixed`         | Uniform random among fixed baselines only.                  |         |                                     |
| `mirror`        | Returns the current learner checkpoint (self‑play).         |         |                                     |
| `lagged`        | Uniform among *active* past checkpoints inside `lag_range`. |         |                                     |
| `random`        | Uniform over fixed + active checkpoints.                    |         |                                     |
| `match-quality` | Softmax based on `TrueSkill.quality()` vs. `uid_me`.        |         |                                     |
| `ts-dist`       | Softmax over                                                | μ★–μopp | (smaller distance ⇒ higher weight). |
| `exploration`   | Placeholder: rank opponents by expected state diversity.    |         |                                     |

### Rating Update Formula

For a finished game with reward *r ∈ {‑1, 0, 1}* (win/draw/loss for *learner*):

```python
if r == 1:
    new_a, new_b = TS.rate_1vs1(a, b)      # learner wins
elif r == -1:
    new_b, new_a = TS.rate_1vs1(b, a)      # learner loses
else:
    new_a, new_b = TS.rate_1vs1(a, b, drawn=True)
```

μ/σ are then written back into `self._models`.

### Practical Tips

* **Keep `max_active_lora` small** (≤4) when GPUs are scarce; inactive checkpoints can still be sampled as *fixed* opponents via OpenRouter.
* Switch to **`match-quality`** after a few hundred games to keep training pairs evenly matched.
* Call **`add_fixed()`** early so baseline ratings converge before checkpoints appear.
* The **`exploration`** mode is experimental—PRs are welcome!

</details>

<details>
<summary><strong>StepBuffer (`buffer.py`)</strong></summary>

## `StepBuffer` — *buffer.py* <a id="step-buffer"></a>

High‑throughput **step‑level** replay buffer that lives on a Ray actor.
Stores `Step` objects emitted from complete game trajectories, applies configurable reward transformations, downsamples when full, and serves randomised **training batches** to the learner.

### Constructor Arguments

| Name                             | Type                                      | Purpose                                                |
| -------------------------------- | ----------------------------------------- | ------------------------------------------------------ |
| `max_buffer_size`                | `int`                                     | Hard cap on number of `Step` objects kept in memory.   |
| `tracker`                        | `BaseTracker`                             | Logger for buffer metrics & CSV dumps.                 |
| `final_reward_transformation`    | `ComposeFinalRewardTransforms \| None`    | Optional pipeline applied to end‑of‑game rewards.      |
| `step_reward_transformation`     | `ComposeStepRewardTransforms \| None`     | Optional function applied at each step (shaping).      |
| `sampling_reward_transformation` | `ComposeSamplingRewardTransforms \| None` | Optional post‑processing applied *only when sampling*. |
| `buffer_strategy`                | `str`                                     | Currently only `"random"` (uniform reservoir).         |

### Responsibilities

* **Trajectory ingestion** – `add_trajectory()` unrolls a `Trajectory` into individual `Step`s and stores them.
* **Reward shaping** – applies the supplied transformation pipelines at *final* and *step* granularity.
* **Capacity management** – once `len(steps) > max_buffer_size`, uniformly removes excess samples.
* **Batch provisioning** – `get_batch(batch_size)` uniform random‑samples *without replacement*, applies optional `sampling_reward_transformation`, and returns the list.
* **Book‑keeping** – CSV dumps of each batch and buffer‑size logging for easy debugging.

### Key Methods

| Method                                          | Returns      | Summary                                                       |
| ----------------------------------------------- | ------------ | ------------------------------------------------------------- |
| `add_trajectory(trajectory, player_id, env_id)` | —            | Flattens a finished trajectory into `Step`s and appends them. |
| `get_batch(batch_size)`                         | `List[Step]` | Pop *batch\_size* random steps; writes a CSV snapshot.        |
| `clear()`                                       | —            | Purge all stored steps.                                       |
| `stop()`                                        | —            | Set `collect=False` so Collector pauses ingestion.            |
| `size()`                                        | `int`        | Current number of stored steps.                               |
| `continue_collection()`                         | `bool`       | Helper polled by Collector for back‑pressure.                 |

### Reward Transformation Hooks

* **Final reward** – `ComposeFinalRewardTransforms` maps the *vector* of per‑player rewards to a new vector (e.g., win → +1 / loss → –1).
* **Step reward** – called for each step *i* with `(trajectory, step_index=i, base_reward)`; enables shaped rewards like dense progress signals.
* **Sampling reward** – run on the *batch* right before returning; useful for on‑policy advantages or normalisation.

### Capacity Workflow

```text
add_trajectory()
  ├── append new steps
  └── if len(steps) > max_buffer_size:
        random.sample(excess) → steps.remove()
```

This simple uniform reservoir keeps memory bounded while preserving sample diversity.

### Practical Tips

* **Disk snapshots** – batches are written to `<train_dir>/train_data_step_<N>.csv`; disable by monkey‑patching `write_training_data_to_file`.
* **Prioritised replay** – implement a new `buffer_strategy` (e.g., PER) and replace the random down‑sampling / sampling logic.
* When training becomes I/O‑bound, consider moving CSV writes onto a background thread or disabling them in production.

</details>

<details>
<summary><strong>Learner (`learners/standard_learner.py`)</strong></summary>

## `StandardLearner` — *learners/standard_learner.py* <a id="learner"></a>

Main **parameter‑updating** component. Consumes `Step` batches from **StepBuffer**, computes policy‑gradient losses via a pluggable `BaseAlgo` (e.g., PPO, REINFORCE) and writes **LoRA checkpoints** every *N* steps. Also registers each new checkpoint with **ModelPool** so it can be sampled as an opponent.

### Constructor Arguments

| Name                       | Type          | Purpose                                              |
| -------------------------- | ------------- | ---------------------------------------------------- |
| `model_name`               | `str`         | HF id or local path of the *base* model.             |
| `step_buffer`              | `StepBuffer`  | Source of training data batches.                     |
| `model_pool`               | `ModelPool`   | Destination for newly‑minted checkpoints.            |
| `algorithm`                | `BaseAlgo`    | Policy‑gradient implementation (PPO, etc.).          |
| `batch_size`               | `int`         | Number of `Step`s per learner update.                |
| `mini_batch_size`          | `int`         | Sub‑division for gradient accumulation.              |
| `max_generation_len`       | `int`         | Truncation length during rollouts.                   |
| `learning_rate`            | `float`       | AdamW learning rate (LoRA params only).              |
| `grad_clip`                | `float`       | Global **L2‑norm** gradient clip.                    |
| `batch_delay_buffer`       | `float`       | Multiplier controlling back‑pressure on buffer.      |
| `lora_cfg`                 | `dict`        | LoRA rank, α, dropout, etc.                          |
| `initial_lora_path`        | `str \| None` | Warm‑start from a prior adapter.                     |
| `num_learners`             | `int`         | How many concurrent learners share the buffer.       |
| `ckpt_root`                | `str`         | Directory for saving checkpoints.                    |
| `save_every`               | `int`         | Save+register every *N* learner steps.               |
| `activation_checkpointing` | `bool`        | Enable full activation CKPT to save VRAM.            |
| `gradient_checkpointing`   | `bool`        | Enable HF gradient CKPT.                             |
| `use_trainer_cache`        | `bool`        | Keep model KV cache during fwd pass (speed vs. RAM). |
| `max_train_len`            | `int \| None` | Hard limit on token count seen by loss fn.           |

### Training Loop (`train(iterations)`) — High‑level Steps

1. **Wait for data** – block until `StepBuffer.size() ≥ batch_size × batch_delay_buffer`.
2. **Fetch batch** – `get_batch(batch_size)` returns uniform random `Step`s.
3. **Gradient accumulation** – split into `mini_batch_size` chunks; call `algorithm.update()` under `torch.autocast(bfloat16)`.
4. **Clip & step** – global L2 clipping then `optimizer.step()`.
5. **Logging** – aggregate metrics, grad norm, LR; push to **Tracker**.
6. **Checkpoint** – every *save\_every* steps, write LoRA adapter to disk and `ModelPool.add_checkpoint()`.
7. **Stop‑condition** – once `self._step == iterations`, stop buffer collection.

### Attributes Exposed to Other Actors

| Attribute       | Type                               | Description                                        |
| --------------- | ---------------------------------- | -------------------------------------------------- |
| `device`        | `torch.device`                     | CUDA / CPU device resolved from Ray GPU placement. |
| `model`         | `transformers.PreTrainedModel`     | PEFT‑wrapped policy network.                       |
| `tokenizer`     | `transformers.PreTrainedTokenizer` | Matching tokenizer for `model`.                    |
| `_step`         | `int`                              | Learner update counter.                            |
| `_samples_seen` | `int`                              | Cumulative number of `Step`s consumed.             |

### PEFT & Memory Optimisations

* **LoRA‑only training** keeps GPU memory low; base weights are frozen by default.
* `enable_full_activation_ckpt()` wraps each module in `torch.utils.checkpoint` — expect \~20‑30 % slower fwd pass but ≤50 % VRAM.
* Set `torch.set_default_dtype(torch.bfloat16)` and `torch.set_float32_matmul_precision('high')` for Ampere+ GPUs.

### Practical Tips

* **Throughput** – choose `batch_delay_buffer ≈ 1.5–2.0`; higher values reduce idle GPU time.
* **Stability** – if loss spikes, reduce `learning_rate` or increase `grad_clip`.
* **Checkpoint hygiene** – old adapters can be pruned offline; `ModelPool` only keeps `max_active_lora` in VRAM.
* **Multiple learners** – set `num_learners > 1` only when you shard the buffer; otherwise they’ll compete for samples.

</details>

<details>
<summary><strong>Tracker (`trackers.py`)</strong></summary>

## `Tracker` — *trackers.py* <a id="tracker"></a>

Central **metrics bus** for the entire pipeline. Runs as a lightweight Ray
actor, buffers scalar logs in‑memory, aggregates them into means, and
periodically pushes the result to **Weights & Biases** (optional) and to
the interactive terminal UI.

### Constructor Arguments

| Name            | Type          | Purpose                                                                                               |
| --------------- | ------------- | ----------------------------------------------------------------------------------------------------- |
| `run_name`      | `str`         | Display name for the current experiment.                                                              |
| `wandb_project` | `str \| None` | If supplied, `wandb.init(project=…, name=run_name)` is called and every flush uploads a metrics dict. |

### Internal State

| Attribute          | Purpose                                                      |
| ------------------ | ------------------------------------------------------------ |
| `FLUSH_EVERY`      | Seconds between *automatic* flushes (default 64 s).          |
| `_m`               | `defaultdict(str→deque)` raw per‑key history (≤512 entries). |
| `_buffer`          | Current *aggregated* snapshot that will be flushed.          |
| `_n`               | Per‑prefix counters (e.g., number of trajectories logged).   |
| `_interface_stats` | Cached dict used by the **TerminalInterface**.               |
| `use_wandb`        | Bool gate so the actor works offline too.                    |

### Responsibilities

* **Aggregation** – store every scalar via `_put(k,v)`; compute means with `_agg(prefix)`.
* **Time‑based flushing** – `_flush_if_due()` fires when `time.monotonic() – _last_flush >= FLUSH_EVERY`.
* **Metric namespaces** – prefixes encode data sources:

  * `collection‑<env_id>/…` – training trajectories.
  * `evaluation‑<env_id>/…` – offline evaluation.
  * `inference/<actor>/…` – GPU token/sec + queue stats.
  * `learner/…` – loss, grad norm, samples seen.
* **Model‑pool introspection** – `log_model_pool()` dumps TrueSkill, exploration % and match counts into the dashboard.
* **Terminal feed** – `get_interface_info()` returns a compact dict used by the curses‑style UI.

### Key Public Methods

| Method                                               | Summary                                                          |
| ---------------------------------------------------- | ---------------------------------------------------------------- |
| `add_trajectory(traj, player_id, env_id)`            | Logs reward, win‑rate, formatting success, game length, etc.     |
| `add_eval_episode(rewards, player_id, env_id)`       | Logs evaluation reward & outcome.                                |
| `log_inference(actor, gpu_ids, stats)`               | Ingests throughput stats from every `VLLMActor`.                 |
| `log_learner(info)`                                  | Single‑call log for each learner step (losses, LR, grad norm).   |
| `log_model_pool(match_counts, ts_dict, exploration)` | Records pool‑level data (TrueSkill μ/σ, unique n‑gram coverage). |
| `get_interface_info()`                               | Returns dict consumed by **TerminalInterface**.                  |

### Flush Cycle

```text
┌ every scalar arrives via any log_* method ┐
│  _put(key, value)                        │
└──► _buffer.update(_agg(prefix))          │
            │                              │
            └──► _flush_if_due() ──► wandb.log(_buffer) every 64 s
```

### Practical Tips

* **Offline mode** – omit `wandb_project` to disable WANDB completely; metrics remain query‑able via `TerminalInterface`.
* **Custom scalars** – any key that starts with an existing prefix will
  be averaged automatically; no schema changes required.
* **Adjust cadence** – set `Tracker.FLUSH_EVERY = 30` before launching if
  you prefer faster WANDB updates.
* **Derived metrics** – compute heavy stats offline; push them via
  `log_model_pool()` rather than inside the tight game loop.

</details>

<details>
<summary><strong>TerminalInterface (`terminal_interface.py`)</strong></summary>

## 'Terminal Interface' - *terminal_interface.py* <a id="terminal-interface"></a>
*Documentation forthcoming…*

</details>

<details>
<summary><strong>Core Data Structures (`core.py`)</strong></summary> 

## Key Dataclasses <a id="core-data-structures"></a>

| Name                       | Fields                                                                                    | Purpose                                                          |
| -------------------------- | ----------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| **`Trajectory`**           | `pid, obs, actions, extracted_actions, infos, final_rewards, num_turns, format_feedbacks` | Full record of a *single* game episode from one agent’s POV.     |
| **`Step`**                 | `pid, obs, act, reward, env_id, step_info`                                                | Flattened, per‑turn training sample passed to **Learner**.       |
| **`Opponent`**             | `uid, kind, path_or_name, rating, active`                                                 | Metadata + TrueSkill rating for every opponent in **ModelPool**. |
| **`EpisodeResult`**        | `traj, end_by_opponent_invalid, action_seq, final_rewards`                                | Light‑weight wrapper returned by `play_episode()`.               |
| **`PlaySpec`** *(frozen)*  | `env_id, num_players, player_id, agent_specs, seed`                                       | Declarative description used to spawn a rollout.                 |
| **`AgentSpec`** *(frozen)* | `kind, model, prompt_template, action_extraction_fn`                                      | Specifies how each player should act inside `play_episode`.      |
| **`TaskMeta`**             | `type, env_id, player_id, seed, ckpt_uid, opponent_uid`                                   | Book‑keeping blob attached to every in‑flight rollout.           |

### Utility Classes

* **`BaseAlgo`** – abstract interface for policy‑gradient algorithms (`initialize`, `prepare_batch`, `update`).
* **`BaseTracker`** – filesystem helper that exposes output directories (train / eval / checkpoints / logs).
* **`ExplorationTracker`** – rolling window *n‑gram* coverage metric used by **ModelPool** to encourage diverse opponents.

### Example — Building a Custom Dataclass

Need a new structure (e.g., to log curiosity bonuses)? Simply import `dataclass` and extend:

```python
from dataclasses import dataclass

@dataclass
class CuriosityStep:
    pid: int
    obs: str
    act: str
    reward: float
    curiosity: float  # 👈 your extra field
```

`Learner.prepare_batch()` can then branch on `isinstance(step, CuriosityStep)`.

</details>

---

# Reward Transformations <a id="reward-transformations"></a>

Below utilities live under `unstable/reward_transformations/`. They let you
reshape sparse win‑loss rewards into *denser* learning signals or correct
for known biases (e.g., first‑player advantage).

<details>
<summary><strong>Final‑Reward Transforms (`transformation_final.py`)</strong></summary>

### API

* Every transform inherits from **`FinalRewardTransform`** and implements
  `__call__(x: Dict[int, float], env_id: str|None) -> Dict[int, float]`.
* A stack is built via **`ComposeFinalRewardTransforms([...])`**; transforms
  are applied *sequentially*.

### Built‑in Transforms

| Class                         | Effect                                             |
| ----------------------------- | -------------------------------------------------- |
| `WinDrawLossFormatter`        | Maps raw score *s* → `{‑1,0,1}` win/draw/loss.     |
| `RoleAdvantageFormatter`      | Subtracts an EMA of each role’s historical reward. |
| `RoleAdvantageByEnvFormatter` | Same, but tracked per‑environment ID.              |

### Custom Transform Example

```python
class ScaleRewardTransform(FinalRewardTransform):
    """Multiply every reward by *alpha*."""
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
    def __call__(self, x, env_id=None):
        return {pid: r * self.alpha for pid, r in x.items()}

# Register it:
transforms = ComposeFinalRewardTransforms([
    WinDrawLossFormatter(),
    ScaleRewardTransform(alpha=0.2),
])
```

This scales the usual `{‑1,0,1}` output down to `{‑0.2,0,0.2}`.

</details>

<details>
<summary><strong>Step‑Reward Transforms (`transformation_step.py`)</strong></summary>

### API

* Implement **`StepRewardTransform`** with `__call__(trajectory, step_index, base_reward) -> float`.
* Chain them with **`ComposeStepRewardTransforms([...])`**; each transform receives the output of the previous one.

### Built‑in Transforms

| Class                   | Effect                                                                           |
| ----------------------- | -------------------------------------------------------------------------------- |
| `RewardForFormat`       | Adds `reward` if the agent’s answer is well‑formatted; otherwise adds `penalty`. |
| `PenaltyForInvalidMove` | Adds `penalty` when the agent commits an invalid move; otherwise adds `reward`.  |

### Custom Transform Example

```python
class DiscountFutureRewards(StepRewardTransform):
    """Apply γ^t discount to every intermediate reward."""
    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma
    def __call__(self, trajectory, step_index, base_reward):
        return base_reward * (self.gamma ** step_index)

step_transforms = ComposeStepRewardTransforms([
    RewardForFormat(reward=0.05, penalty=-0.05),
    DiscountFutureRewards(gamma=0.97),
])
```

This first rewards/penalises formatting, then exponentially discounts by step index.

</details>

<details>
<summary><strong>Sampling‑Reward Transforms (`transformation_sampling.py`)</strong></summary>

### API

* Sub‑class **`SamplingRewardTransform`** and implement `__call__(steps: List[Step]) -> List[Step]`.
* A stack is applied via **`ComposeSamplingRewardTransforms([...])`** *after* the batch is drawn from **StepBuffer**.

### Built‑in Transforms

| Class                   | Effect                                                                 |
| ----------------------- | ---------------------------------------------------------------------- |
| `NormalizeRewards`      | Subtracts the mean reward across the batch (optionally divide by std). |
| `NormalizeRewardsByEnv` | Mean‑centres (and optionally z‑scores) rewards *per environment ID*.   |

### Custom Transform Example

```python
class ClampRewards(SamplingRewardTransform):
    """Clip rewards into [min_r, max_r]."""
    def __init__(self, min_r: float = -1.0, max_r: float = 1.0):
        self.min_r, self.max_r = min_r, max_r
    def __call__(self, steps, env_id=None):
        for s in steps:
            s.reward = max(self.min_r, min(self.max_r, s.reward))
        return steps

sampling_transforms = ComposeSamplingRewardTransforms([
    NormalizeRewardsByEnv(z_score=True),
    ClampRewards(min_r=-2, max_r=2),
])
```

This normalises rewards per env and then clamps extreme values.

</details>

---

## Algorithms <a id="algorithms"></a>

### Reinforce
Explanation, use-cases, and examples.

### Extending with Custom Algorithms
How to implement and integrate custom algorithms.

---

## Utilities and Helpers <a id="utilities-and-helpers"></a>

### Templates
Documentation for templates handling prompts and action extraction.

### Logging
Logging utility documentation.

---

## Configuration Reference <a id="configuration-reference"></a>

Below are recommended configurations for different VRAM capacities. We strongly recommend using **one GPU as a learner** and dedicating **remaining GPUs as actors** for inference.
Currently you will need at least 2 gpus to run the code (1 learner and 1 actor); we plan to relax this requirement in the future. Here are some rough guidelines on which parameter settings to use for what model size and hardware (all of these assume 4096 generation length and are tested on SimpleTak):

### Qwen3 1.7B
| VRAM | Activation Checkpointing | Gradient Checkpointing | Train Length Truncation |
| ---- | :----------------------: | :--------------------: | :---------------------: |
| 16GB | TOOD                     | TOOD                   | TOOD                    |
| 24GB | True                     | True                   | 3500                    |
| 48GB+| False                    | False                  | None                    |

### Llama3.2 3B
| VRAM | Activation Checkpointing | Gradient Checkpointing | Train Length Truncation |
| ---- | :----------------------: | :--------------------: | :---------------------: |
| 16GB | TOOD                     | TOOD                   | TOOD                    |
| 24GB | True                     | True                   | 3000                    |
| 48GB | True                     | True                   | None                    |
| 80GB+| False                    | False                  | None                    |

### Qwen3 4B
| VRAM | Activation Checkpointing | Gradient Checkpointing | Train Length Truncation |
| ---- | :----------------------: | :--------------------: | :---------------------: |
| 24GB | True                     | True                   | 1500                    |
| 48GB | True                     | True                   | None                    |
| 80GB+| False                    | False                  | None                    |

### Qwen3 8B
| VRAM  | Activation Checkpointing | Gradient Checkpointing | Train Length Truncation |
| ----- | :----------------------: | :--------------------: | :---------------------: |
| 48GB  | True                     | True                   | 3500 (also try 4k)      |
| 80GB+ | TOOD                     | TOOD                   | TOOD                    |

### Qwen3 14B
| VRAM  | Activation Checkpointing | Gradient Checkpointing | Train Length Truncation |
| ----- | :----------------------: | :--------------------: | :---------------------: |
| 48GB  | TOOD                     | TOOD                   | 1500                    |
| 80GB+ | TOOD                     | TOOD                   | TOOD                    |

* **Activation Checkpointing** significantly reduces VRAM usage (~50-60%) but incurs roughly 20-30% longer training times.
* **Gradient Checkpointing** slightly reduces memory with minimal impact on speed.
* **Train Length Truncation** controls maximum input token length, with shorter lengths substantially reducing VRAM requirements (only truncates for training; works surprisingly well in practice).
* We usually run everything with mini-batch-size = 1



---
 
## Contributing <a id="contributing"></a>
TODO Guidelines on contributing to the project.

---

## Contact <a id="contact"></a>
Contact information and channels for support and discussion.
