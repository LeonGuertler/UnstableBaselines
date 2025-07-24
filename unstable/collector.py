import os, re, csv, random, logging, itertools
from pathlib import Path
from collections import deque
import numpy as np
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Tuple, Protocol, Optional

import ray
from ray.exceptions import RayActorError, RayTaskError

import textarena as ta
assert ta.__version__ >= "0.6.16", f"TextArena package version is too old: {ta.__version__}. Required version is at least 0.6.16."

# local imports
from unstable.actor import VLLMActor
from unstable._types import GameSpec, AgentSpec, GameInformation, PlayerTrajectory, TaskMeta
from unstable.utils.logging import setup_logger
from unstable.utils.templates import ACTION_EXTRACTION, OBSERVATION_FORMATTING



class CallableActorWrapper:
    def __init__(self, actor: VLLMActor, lora_path: str|Path, obs_fmt_fn: Callable[[str],str], extract_fn: Callable[[str], Tuple[str, Dict[str, Any]]]) -> None:
        self._actor, self._lora, self._fmt, self._extract = actor, lora_path, obs_fmt_fn, extract_fn

    def __call__(self, observation: str) -> str: 
        _, extracted, _, _ = self.act_full(observation)
        return extracted

    def act_full(self, observation: str) -> Tuple[str, str, str, dict]:
        prompt = self._fmt(observation=observation)
        raw = ray.get(self._actor.submit_prompt.remote(prompt=prompt, lora_path=self._lora))
        extracted, format_feedback = self._extract(raw_action=raw)
        return raw, extracted, prompt, format_feedback

@ray.remote(num_cpus=0)
def run_game(game_spec: GameSpec, actor: VLLMActor):
    game_information = GameInformation(game_idx=game_spec.game_idx, eval_model_pid=game_spec.eval_model_pid, eval_opponent_name=game_spec.eval_opponent_name)
    agents = {agent_spec.pid: {
        "traj": PlayerTrajectory(pid=agent_spec.pid) if agent_spec.collect_data else None, 
        "name": agent_spec.lora_path if agent_spec.lora_path else agent_spec.openrouter_name,
        "model": CallableActorWrapper(actor=actor, lora_path=agent_spec.lora_path, obs_fmt_fn=OBSERVATION_FORMATTING[agent_spec.prompt_template], extract_fn=ACTION_EXTRACTION[agent_spec.action_extraction_fn]) if agent_spec.openrouter_name==None else ta.agents.OpenRouterAgent(agent_spec.openrouter_name)
    } for agent_spec in game_spec.agent_specs} # build agents
    env=ta.make(game_spec.env_id); env.reset(num_players=len(agents), seed=game_spec.seed); env.state.error_allowance=0; turn=0
    while True:
        pid, obs = env.get_observation()
        # get model (or opponent) action
        if agents[pid]["traj"] == None: raw = extracted = agents[pid]["model"](obs) # fix opponent
        else: raw, extracted, prompt, format_feedback = agents[pid]["model"].act_full(obs)
        done, step_info = env.step(extracted); turn+= 1 # execute the action & increment turn counter
        # general tracking
        game_information.pid.append(pid); game_information.obs.append(obs); game_information.full_actions.append(raw)
        game_information.extracted_actions.append(extracted); game_information.step_infos.append(step_info); game_information.names[pid] = agents[pid]["name"]
        # player specific trackering
        if agents[pid]["traj"] != None:
            agents[pid]["traj"].obs.append(obs); agents[pid]["traj"].actions.append(raw); agents[pid]["traj"].extracted_actions.append(extracted)
            format_feedback["invalid_move"] = False; agents[pid]["traj"].format_feedbacks.append(format_feedback); agents[pid]["traj"].step_infos.append(step_info)
        if done: break
    final_rewards, game_info = env.close()
    for pid in agents.keys():
        if agents[pid]["traj"]!=None: agents[pid]["traj"].final_reward=final_rewards[pid]; agents[pid]["traj"].game_info=game_info[pid]; agents[pid]["traj"].num_turns=turn
        if game_info[pid]["invalid_move"] and agents[pid]["traj"]!=None: agents[pid]["traj"].format_feedbacks[-1]["invalid_move"]=True
    game_information.final_rewards=final_rewards; game_information.num_turns=turn; game_information.game_info=game_info
    return game_information, [agents[pid]["traj"] for pid in agents.keys() if agents[pid]["traj"]!=None]


@ray.remote
class Collector:
    def __init__(self, vllm_config, tracker, buffer, game_scheduler):
        self.logger = setup_logger("collector", ray.get(tracker.get_log_dir.remote()))
        self.tracker, self.buffer, self.game_scheduler = tracker, buffer, game_scheduler
        # self.actors = [VLLMActor.options(num_gpus=1).remote(cfg=vllm_config, tracker=tracker, name=f"Actor-{i}") for i in range(int(ray.available_resources().get("GPU", 0))-1)]
        self.actors = [VLLMActor.options(num_gpus=1).remote(cfg=vllm_config, tracker=tracker, name=f"Actor-{i}") for i in range(int(ray.available_resources().get("GPU", 0)))]
        self._actor_iter = itertools.cycle(self.actors)

        # thead keeping
        self.flight: Dict[ray.ObjectRef, TaskMeta] = {}
        self._num_running = lambda typ: sum(meta.type == typ for meta in self.flight.values())
        self.logger.info("Collector initialized")
    
    def _launch_jobs(self, max_train: int, max_eval: Optional[int]):
        while self._num_running("train") < max_train: # submit new train game
            try:
                game_spec: GameSpec = ray.get(self.game_scheduler.next_train_job.remote()) # sample game spec
                self.logger.info(f"received train game_spec: {game_spec}")
                actor: VLLMActor = next(self._actor_iter) # get actor
                ref = run_game.remote(game_spec, actor)
                self.flight[ref] = TaskMeta("train", game_spec.env_id)
            except Exception as exc:
                self.logger.info(f"Exception in train game {game_spec}: {exc}")

        while max_eval!=None and self._num_running("eval") < max_eval:
            try:
                game_spec: GameSpec = ray.get(self.game_scheduler.next_eval_job.remote())
                self.logger.info(f"received eval game_spec: {game_spec}")
                actor: VLLMActor = next(self._actor_iter) # get actor
                ref = run_game.remote(game_spec, actor)
                self.flight[ref] = TaskMeta("eval", game_spec.env_id)
            except Exception as exc:
                self.logger.info(f"Exception in eval game {game_spec}: {exc}")

    def _handle_finished_job(self, ref):
        meta = self.flight.pop(ref)
        try: game_information, player_trajs = ray.get(ref)
        except (RayTaskError, RayActorError) as err: self.logger.error(f"Remote episode failed for {meta.type} task: env={meta.env_id}: {err}", exc_info=True); return
        self._post_train(meta, game_information, player_trajs) if meta.type=="train" else self._post_eval(meta, game_information)
    
    def _post_train(self, meta: TaskMeta, game_information: GameInformation, player_trajs: List[PlayerTrajectory]):
        for traj in player_trajs: self.buffer.add_player_trajectory.remote(traj, env_id=meta.env_id); self.tracker.add_player_trajectory.remote(traj, env_id=meta.env_id)
        self.game_scheduler.update.remote(game_info=game_information)

    def _post_eval(self, meta: TaskMeta, game_information: GameInformation):
        self.tracker.add_eval_game_information.remote(game_information=game_information, env_id=meta.env_id)
    
    def collect(self, num_train_workers: int, num_eval_workers: Optional[int]=None):
        self.logger.info("entered collect func")
        while ray.get(self.buffer.continue_collection.remote()):
            self.logger.info("entered colelct loop")
            self._launch_jobs(num_train_workers, num_eval_workers)
            if not self.flight: continue
            done_ref, _ = ray.wait(list(self.flight), num_returns=1)
            self._handle_finished_job(done_ref[0])



@ray.remote
class Evaluator:
    def __init__(self, vllm_config, tracker, eval_env_specs, num_runs_per_env, output_folder):
        try:
            self.logger = setup_logger("evaluator", ray.get(tracker.get_log_dir.remote()))
            self.tracker, self.eval_env_specs, self.num_runs_per_env = tracker, eval_env_specs, num_runs_per_env
            self.actors = [VLLMActor.options(num_gpus=1).remote(cfg=vllm_config, tracker=tracker, name=f"Actor-{i}", use_lora=False) for i in range(int(ray.available_resources().get("GPU", 0)))]
            self._actor_iter = itertools.cycle(self.actors)
            self._games_to_run = deque()
            # populate all games to be run
            for game_idx in range(num_runs_per_env*len(eval_env_specs)):
                _env_spec = eval_env_specs[game_idx%len(eval_env_specs)]
                pids = list(range(_env_spec.num_players)); random.shuffle(pids); agent_specs = []
                for i, pid in enumerate(pids):
                    if i == 0:  agent_specs.append(AgentSpec(pid=pid, kind="checkpoint", collect_data=False, lora_path=None, prompt_template=_env_spec.prompt_template, action_extraction_fn=_env_spec.action_extraction_fn)) # only one actor, rest fixed
                    else:       agent_specs.append(AgentSpec(pid=pid, kind="openrouter", lora_path=None, openrouter_name=_env_spec.fixed_opponent)) # sample opponent and add
                self._games_to_run.append(GameSpec(game_idx=game_idx, env_id=_env_spec.env_id, seed=game_idx, agent_specs=agent_specs, eval_model_pid=pids[0], eval_opponent_name=_env_spec.fixed_opponent)) # populate GameSpec

            self.logger.info(self._games_to_run)
            # thead keeping
            self.flight: Dict[ray.ObjectRef, TaskMeta] = {}
            self._num_running = lambda typ: sum(meta.type == typ for meta in self.flight.values())
            output_folder = output_folder or ray.get(self.tracker.get_eval_dir.remote())
            self.csv_path = os.path.join(output_folder, "eval_results.csv")
            self.logger.info(f"csv_path: {self.csv_path}")
            self.csv_fields = ["game_idx", "env_id", "eval_model_pid", "eval_model_reward", "avg_opponent_reward", "num_turns"]
            # if not self.csv_path.exists():
            if not os.path.exists(self.csv_path):
                # with self.csv_path.open("w", newline="") as f:
                with open(self.csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=self.csv_fields)
                    writer.writeheader()
            self.logger.info("Evaluator initialized")
        except Exception as exc: self.logger.info(f"Exception in evaluator setup: {exc}")


    def _launch_jobs(self, max_eval: int):
        try:
            while self._num_running("eval") < max_eval and self._games_to_run:
                try:
                    game_spec = self._games_to_run.popleft()
                    self.logger.info(f"received eval game_spec: {game_spec}")
                    actor: VLLMActor = next(self._actor_iter)
                    ref = run_game.remote(game_spec, actor)
                    self.flight[ref] = TaskMeta("eval", game_spec.env_id)
                except Exception as exc:
                    self.logger.info(f"Exception in eval game {game_spec}: {exc}")
        except Exception as exc: self.logger.info(f"Exception in _launch_jobs: {exc}")

    def _handle_finished_job(self, ref):
        try:
            meta = self.flight.pop(ref)
            try: game_information, player_trajs = ray.get(ref)
            except (RayTaskError, RayActorError) as err: self.logger.error(f"Remote episode failed for {meta.type} task: env={meta.env_id}: {err}", exc_info=True); return
            self._post_eval(meta, game_information)
        except Exception as exc: self.logger.info(f"Exception in _handle_finished_job: {exc}")
    

    def _post_eval(self, meta: TaskMeta, gi: "GameInformation"):
        try:
            rewards = gi.final_rewards
            eval_reward = rewards.get(gi.eval_model_pid, float("nan"))
            opp_rewards = [r for pid, r in rewards.items() if pid != gi.eval_model_pid]
            avg_opp_reward = float(np.mean(opp_rewards)) if opp_rewards else float("nan")
            num_turns = getattr(gi, "num_turns", -1)
            row = {"game_idx": gi.game_idx, "env_id": meta.env_id, "eval_model_pid": gi.eval_model_pid, "eval_model_reward": eval_reward, "avg_opponent_reward": avg_opp_reward, "num_turns": num_turns}
            # with self.csv_path.open("a", newline="") as f:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_fields)
                writer.writerow(row)
                f.flush()
            self.logger.info("[EVAL] game=%d env=%s pid=%d -> eval=%.4f opp=%.4f T=%d", row["game_idx"], row["env_id"], row["eval_model_pid"], row["eval_model_reward"], row["avg_opponent_reward"], row["num_turns"])
        except Exception as exc: self.logger.info(f"Exception in _post_eval: {exc}")

    def evaluate(self, num_eval_workers: int):
        self.logger.info("entered collect func")
        try:
            # while self._games_to_run:
            while self._games_to_run or self.flight:
                self.logger.info("entered colelct loop")
                self._launch_jobs(num_eval_workers)
                if not self.flight: time.sleep(0.01); continue
                done_ref, _ = ray.wait(list(self.flight), num_returns=1)
                self._handle_finished_job(done_ref[0])
            return None # for ray.get # presumeably, not actually sure whether this is necessary
        except Exception as exc: self.logger.info(f"Exception in evaluate(): {exc}")

