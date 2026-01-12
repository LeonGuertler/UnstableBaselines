import ray, random, os, csv, time, argparse, fcntl
from typing import Optional, Union, Dict
from collections import deque
from ray.exceptions import RayTaskError, RayActorError
from huggingface_hub import snapshot_download
import numpy as np
import itertools

from unstable.collection.trackers import Tracker
from unstable.utils.logger import setup_logger
from unstable.collection.actor import VLLMActor
from unstable.utils._types import AgentSpec, GameSpec, TaskMeta, EvalEnvSpec
from unstable.utils.misc import write_game_information_to_file
from unstable.collection.game_scheduler import run_game
from unstable.utils.templates import get_algorithm_config


def _launch_jobs(max_eval: int, _games_to_run, _num_running, flight, actors, opponent_adapter_checkpoint, opponent_actors, logger):
    try:
        while _num_running("eval") < max_eval and _games_to_run:
            try:
                actor = next(actors); opponent_actor = next(opponent_actors) if opponent_adapter_checkpoint else None
                game_spec = _games_to_run.popleft()
                logger.info(f"received eval game_spec: {game_spec}")
                ref = run_game.remote(game_spec, actor if not opponent_adapter_checkpoint else {spec.pid: (actor if spec.pid == game_spec.eval_model_pid else opponent_actor) for spec in game_spec.agent_specs})
                flight[ref] = TaskMeta("eval", game_spec.env_id)
            except Exception as exc: logger.info(f"Exception in eval game {game_spec}: {exc}")
    except Exception as exc: logger.info(f"Exception in _launch_jobs: {exc}")


def _handle_finished_job(ref, flight, run_name, config, adapter_checkpoint, adapter_revision, opponent_model, opponent_adapter_checkpoint, csv_path, csv_fields, output_folder, logger):
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
            "model": config["model_name"],
            "adapter_checkpoint": adapter_checkpoint if adapter_checkpoint is not None else "base",
            "adapter_revision": adapter_revision if adapter_revision is not None else "iter-0",
            "eval_model_pid": game_information.eval_model_pid,
            "opponent_model": opponent_model,
            "opponent_adapter_checkpoint": opponent_adapter_checkpoint if opponent_adapter_checkpoint is not None else "base",
            "env_id": meta.env_id,
            "game_idx": game_information.game_idx,
            "num_turns": num_turns,
            "eval_model_reward": eval_reward,
            "avg_opponent_reward": avg_opp_reward,
            "info": game_information.game_info
        }
        with open(csv_path, "a", newline="") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                writer = csv.DictWriter(f, fieldnames=csv_fields)
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        write = os.path.join(output_folder, game_information.env_id, adapter_revision if adapter_revision is not None else "iteration-0")
        if not os.path.exists(write): print('CREATING', write); os.makedirs(write, exist_ok=True)
        write_game_information_to_file(game_information, os.path.join(write, f"game_{game_information.game_idx}_info.csv"))
        logger.info("[EVAL] game=%d env=%s pid=%d -> eval=%.4f opp=%.4f T=%d", row["game_idx"], row["env_id"], row["eval_model_pid"], row["eval_model_reward"], row["avg_opponent_reward"], row["num_turns"])
    except Exception as exc: logger.info(f"Exception in _handle_finished_job: {exc}")


def eval(run_name: str = 'test', config: Optional[Union[Dict, str]] = 'eval',  adapter_checkpoint: Optional[str] = None, adapter_revision: Optional[str] = None, opponent_model: str = None, opponent_adapter_checkpoint: str = None, env: str = None):
    # Configuration
    if isinstance(config, str): config = get_algorithm_config(config)
    if env is not None: config["environments"][0]['env_id'] = env
    if opponent_model is not None: config["environments"][0]["fixed_opponent"] = opponent_model
    ckpt_path = adapter_checkpoint if adapter_checkpoint is not None and os.path.exists(adapter_checkpoint) else None
    if ckpt_path is None and adapter_checkpoint is not None:
        try: ckpt_path = snapshot_download(repo_id=adapter_checkpoint, revision=adapter_revision, local_files_only=True)
        except Exception: ckpt_path = snapshot_download(repo_id=adapter_checkpoint, revision=adapter_revision)
    
    # Setup
    try:
        ray.init(namespace=config.get('project', 'UnstableBaselines'))
        tracker = Tracker.options(name="Tracker").remote(
            run_name=f"{config.get('run', 'Run')}", 
            wandb_project=None
        )
        logger = setup_logger("evaluator", ray.get(tracker.get_log_dir.remote()))
        
        # Actors
        actors = [VLLMActor.options(num_gpus=1).remote(cfg=config['vllm_config'], tracker=tracker, name=f"Actor-{i}") for i in range(config.get('num_actors', 1))]
        for actor in actors: ray.get(actor.ready.remote())
        actors = itertools.cycle(actors)
        opponent_vllm_config = config["vllm_config"].copy(); opponent_vllm_config["model_name"] = opponent_model
        opponent_actors = [VLLMActor.options(num_gpus=1).remote(cfg=opponent_vllm_config, tracker=tracker, name=f"Eval-Actor-{i}") if opponent_adapter_checkpoint else None for i in range(config.get('num_actors', 1))]
        for actor in opponent_actors: 
            if actor is not None: ray.get(actor.ready.remote())
        opponent_actors = itertools.cycle([actor for actor in opponent_actors if actor is not None])

        # Results Logging
        output_folder = config.get('output_dir', 'outputs')
        output_folder = os.path.join(output_folder, run_name, config.get('model_name'))
        if not os.path.exists(output_folder): os.makedirs(output_folder, exist_ok=True)
        csv_path = os.path.join(config.get('output_dir', 'outputs'), "results.csv")
        logger.info(f"csv_path: {csv_path}")
        csv_fields = ["run", "model", "adapter_checkpoint", "adapter_revision", "eval_model_pid", "opponent_model", "opponent_adapter_checkpoint", "env_id", "game_idx", "num_turns", "eval_model_reward", "avg_opponent_reward", "info"]
        with open(csv_path, "a+", newline="") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            if os.stat(csv_path).st_size == 0:
                writer = csv.DictWriter(f, fieldnames=csv_fields)
                writer.writeheader()
            fcntl.flock(f, fcntl.LOCK_UN)

        # Games
        eval_env_specs, num_runs_per_env = config['environments'], config.get('num_runs_per_env', 10)
        _games_to_run = deque()
        for game_idx in range(num_runs_per_env*len(eval_env_specs)):
            _env_spec = EvalEnvSpec(**eval_env_specs[game_idx%len(eval_env_specs)])
            pids = list(range(_env_spec.num_players)); random.shuffle(pids); agent_specs = []
            for i, pid in enumerate(pids):
                if i == 0:  agent_specs.append(AgentSpec(pid=pid, kind="checkpoint", collect_data=False, lora_path=ckpt_path, prompt_template=_env_spec.prompt_template, action_extraction_fn=_env_spec.action_extraction_fn))
                else:
                    if opponent_adapter_checkpoint: agent_specs.append(AgentSpec(pid=pid, kind="checkpoint", collect_data=False, lora_path=opponent_adapter_checkpoint , prompt_template=_env_spec.prompt_template, action_extraction_fn=_env_spec.action_extraction_fn))
                    else: agent_specs.append(AgentSpec(pid=pid, kind="openrouter", lora_path=None, openrouter_name=_env_spec.fixed_opponent))
            _games_to_run.append(GameSpec(game_idx=game_idx, env_id=_env_spec.env_id, seed=game_idx, agent_specs=agent_specs, eval_model_pid=pids[0], eval_opponent_name=_env_spec.fixed_opponent, error_allowance=config.get('error_allowance', 0)))
        logger.info(_games_to_run)
        flight: Dict[ray.ObjectRef, TaskMeta] = {}
        _num_running = lambda typ: sum(meta.type == typ for meta in flight.values())
        logger.info("Evaluator initialized")
        if adapter_checkpoint: logger.info(f"Using checkpoint repo '{adapter_checkpoint}' @ {adapter_revision} at local path: {ckpt_path}")
    except Exception as exc: logger.info(f"Exception in evaluator setup: {exc}")
    
    # Run
    while _games_to_run or flight:
        logger.info("entered collect loop")
        _launch_jobs(config['num_eval_workers'], _games_to_run, _num_running, flight, actors, opponent_adapter_checkpoint, opponent_actors, logger)
        if not flight: time.sleep(0.01); continue
        done_ref, _ = ray.wait(list(flight), num_returns=1)
        _handle_finished_job(done_ref[0], flight, run_name, config, adapter_checkpoint, adapter_revision, opponent_model, opponent_adapter_checkpoint, csv_path, csv_fields, output_folder, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="eval", help="Run name for logging.")
    parser.add_argument("--config", type=str, default="eval", help="Evaluation config file.")
    parser.add_argument("--model", type=str, default='Qwen/Qwen3-4B-Base', help="Base model name for evaluation.")
    parser.add_argument("--adapter_checkpoint", type=str, default=None, help="Checkpoint repository name (huggingface) or local path to lora-adapters.")
    parser.add_argument("--adapter_revision", type=str, default=None, help="Checkpoint revision, if huggingface.")
    parser.add_argument("--opponent_model", type=str, default='google/gemini-2.5-flash-lite', help="Opponent model name for evaluation. Can be hf or openrouter.")
    parser.add_argument("--opponent_adapter_checkpoint", type=str, default=None, help="Evaluation checkpoint repository name (huggingface) or local path.")
    parser.add_argument("--env", type=str, default=None, help="Environment ID for evaluation.")
    args = parser.parse_args()
    eval(run_name=args.name, config=args.config, adapter_checkpoint=args.adapter_checkpoint, adapter_revision=args.adapter_revision, opponent_model=args.opponent_model, opponent_adapter_checkpoint=args.opponent_adapter_checkpoint, env=args.env)