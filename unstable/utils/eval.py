import ray, random, os, csv, time, argparse
from typing import Optional, Union, Dict
import itertools
from collections import deque
from ray.exceptions import RayTaskError, RayActorError
from huggingface_hub import snapshot_download
import numpy as np

from unstable.collection.trackers import Tracker
from unstable.utils.logging import setup_logger
from unstable.collection.actor import VLLMActor
from unstable.utils._types import AgentSpec, GameSpec, TaskMeta, EvalEnvSpec
from unstable.utils.misc import write_game_information_to_file
from unstable.collection.game_scheduler import run_game
from unstable.utils.templates import get_algorithm_config


def eval(run_name: str = 'test', config: Optional[Union[Dict, str]] = 'eval',  ckpt_name_or_path: Optional[str] = None, ckpt_revision: Optional[str] = None, model: str = None):
    # Configuration
    if isinstance(config, str): config = get_algorithm_config(config)
    if model is not None: config['model_name'] = model
    ckpt_path = None if ckpt_name_or_path is not None and not os.path.exists(ckpt_name_or_path) else ckpt_name_or_path
    if ckpt_path is None and ckpt_name_or_path is not None:
        try: ckpt_path = snapshot_download(repo_id=ckpt_name_or_path, revision=ckpt_revision, local_files_only=True)
        except Exception: ckpt_path = snapshot_download(repo_id=ckpt_name_or_path, revision=ckpt_revision)
    # Helper functions
    def _launch_jobs(max_eval: int, _games_to_run):
        try:
            while _num_running("eval") < max_eval and _games_to_run:
                try:
                    game_spec = _games_to_run.popleft()
                    logger.info(f"received eval game_spec: {game_spec}")
                    actor: VLLMActor = next(_actor_iter)
                    ref = run_game.remote(game_spec, actor)
                    flight[ref] = TaskMeta("eval", game_spec.env_id)
                except Exception as exc: logger.info(f"Exception in eval game {game_spec}: {exc}")
        except Exception as exc: logger.info(f"Exception in _launch_jobs: {exc}")
    def _handle_finished_job(ref):
        try:
            meta = flight.pop(ref)
            try: game_information, player_trajs = ray.get(ref)
            except (RayTaskError, RayActorError) as err: logger.error(f"Remote episode failed for {meta.type} task: env={meta.env_id}: {err}", exc_info=True); return
            rewards = game_information.final_rewards
            eval_reward = rewards.get(game_information.eval_model_pid, float("nan"))
            opp_rewards = [r for pid, r in rewards.items() if pid != game_information.eval_model_pid]
            avg_opp_reward = float(np.mean(opp_rewards)) if opp_rewards else float("nan")
            num_turns = getattr(game_information, "num_turns", -1)
            row = {
                "run": run_name,
                "hf": ckpt_name_or_path if ckpt_name_or_path is not None else "base",
                "ckpt": ckpt_revision if ckpt_revision is not None else "iter-0",
                "game_idx": game_information.game_idx,
                "env_id": meta.env_id,
                "eval_model_pid": game_information.eval_model_pid,
                "eval_model_reward": eval_reward,
                "avg_opponent_reward": avg_opp_reward,
                "num_turns": num_turns,
                "info": game_information.game_info
            }
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fields)
                writer.writerow(row)
                f.flush()
            write = os.path.join(output_folder, game_information.env_id, ckpt_revision if ckpt_revision is not None else "iteration-0")
            if not os.path.exists(write): print('CREATING', write); os.makedirs(write, exist_ok=True)
            write_game_information_to_file(game_information, os.path.join(write, f"game_{game_information.game_idx}_info.csv"))
            logger.info("[EVAL] game=%d env=%s pid=%d -> eval=%.4f opp=%.4f T=%d", row["game_idx"], row["env_id"], row["eval_model_pid"], row["eval_model_reward"], row["avg_opponent_reward"], row["num_turns"])
        except Exception as exc: logger.info(f"Exception in _handle_finished_job: {exc}")
    # Setup
    try:
        ray.init(namespace=config.get('project', 'UnstableBaselines'))
        tracker = Tracker.options(name="Tracker").remote(
            run_name=f"{config.get('run', 'Run')}", 
            wandb_project=None
        )
        logger = setup_logger("evaluator", ray.get(tracker.get_log_dir.remote()))
        output_folder = config.get('output_dir', ray.get(tracker.get_eval_dir.remote()))
        output_folder = os.path.join(output_folder, run_name, config.get('model_name'))
        # game_log_dir = os.path.join(output_folder, f"{(ckpt_name_or_path or 'base').replace('/', '-')}", f"{ckpt_revision or 'iter-0'}")
        if not os.path.exists(output_folder): os.makedirs(output_folder, exist_ok=True)
        csv_path = os.path.join(config.get('output_dir', ray.get(tracker.get_eval_dir.remote())), "results.csv")
        eval_env_specs, num_runs_per_env = config['environments'], config.get('num_runs_per_env', 10)
        actors = [VLLMActor.options(num_gpus=1).remote(cfg=config['vllm_config'], tracker=tracker, name=f"Actor-{i}") for i in range(int(ray.available_resources().get("GPU", 0)))]
        _actor_iter = itertools.cycle(actors)
        _games_to_run = deque()
        for game_idx in range(num_runs_per_env*len(eval_env_specs)):
            _env_spec = EvalEnvSpec(**eval_env_specs[game_idx%len(eval_env_specs)])
            pids = list(range(_env_spec.num_players)); random.shuffle(pids); agent_specs = []
            for i, pid in enumerate(pids):
                if i == 0:  agent_specs.append(AgentSpec(pid=pid, kind="checkpoint", collect_data=False, lora_path=ckpt_path, prompt_template=_env_spec.prompt_template, action_extraction_fn=_env_spec.action_extraction_fn))
                else:       agent_specs.append(AgentSpec(pid=pid, kind="openrouter", lora_path=None, openrouter_name=_env_spec.fixed_opponent))
            _games_to_run.append(GameSpec(game_idx=game_idx, env_id=_env_spec.env_id, seed=game_idx, agent_specs=agent_specs, eval_model_pid=pids[0], eval_opponent_name=_env_spec.fixed_opponent, error_allowance=config.get('error_allowance', 0)))
        logger.info(_games_to_run)
        flight: Dict[ray.ObjectRef, TaskMeta] = {}
        _num_running = lambda typ: sum(meta.type == typ for meta in flight.values())
        logger.info(f"csv_path: {csv_path}")
        csv_fields = ["run", "hf", "ckpt", "game_idx", "env_id", "eval_model_pid", "eval_model_reward", "avg_opponent_reward", "num_turns", "info"]
        if not os.path.exists(csv_path):
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fields)
                writer.writeheader()
        logger.info("Evaluator initialized")
        if ckpt_name_or_path: logger.info(f"Using checkpoint repo '{ckpt_name_or_path}' @ {ckpt_revision} at local path: {ckpt_path}")
    except Exception as exc: logger.info(f"Exception in evaluator setup: {exc}")
    # Run
    while _games_to_run or flight:
        logger.info("entered collect loop")
        _launch_jobs(config['num_eval_workers'], _games_to_run)
        if not flight: time.sleep(0.01); continue
        done_ref, _ = ray.wait(list(flight), num_returns=1)
        _handle_finished_job(done_ref[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="test", help="Run name for logging.")
    parser.add_argument("--config", type=str, default="eval", help="Evaluation config file.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint repository name (huggingface) or local path to lora-adapters.")
    parser.add_argument("--revision", type=str, default=None, help="Checkpoint revision, if huggingface.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Base", help="Model for evaluation.")
    args = parser.parse_args()
    eval(run_name=args.run_name, config=args.config, ckpt_name_or_path=args.checkpoint, ckpt_revision=args.revision)
