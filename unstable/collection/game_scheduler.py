import ray, random, itertools, os
from typing import Dict, List, Optional, Union
import ray
from ray.exceptions import RayActorError, RayTaskError
import textarena as ta
assert ta.__version__ >= "0.6.16", f"TextArena package version is too old: {ta.__version__}. Required version is at least 0.6.16."

from unstable.collection.actor import VLLMActor, CallableActorWrapper
from unstable.utils._types import GameSpec, GameInformation, PlayerTrajectory, TaskMeta, AgentSpec
from unstable.utils.logger import setup_logger
from unstable.utils.templates import ACTION_EXTRACTION, OBSERVATION_FORMATTING, get_action_sampler_cls
from unstable.utils.misc import write_game_information_to_file


@ray.remote(num_cpus=0)
def run_game(game_spec: GameSpec, actor: Union["VLLMActor", dict[int, "VLLMActor"]]):
    game_information = GameInformation(game_idx=game_spec.game_idx, env_id=game_spec.env_id, eval_model_pid=game_spec.eval_model_pid, eval_opponent_name=game_spec.eval_opponent_name, eval_iteration=game_spec.eval_iteration)
    agents = {agent_spec.pid: {
        "traj": PlayerTrajectory(pid=agent_spec.pid) if agent_spec.collect_data else None,
        "name": agent_spec.lora_path if agent_spec.lora_path else agent_spec.openrouter_name,
        "model": get_action_sampler_cls(agent_spec.sampler)(
            CallableActorWrapper(
                actor=actor[agent_spec.pid] if isinstance(actor, dict) else actor,
                lora_path=agent_spec.lora_path,
                obs_fmt_fn=OBSERVATION_FORMATTING[agent_spec.prompt_template],
                extract_fn=ACTION_EXTRACTION[agent_spec.action_extraction_fn],
                temperature=agent_spec.temperature,
                top_p=agent_spec.top_p,
                top_k=agent_spec.top_k,
                max_tokens=agent_spec.max_tokens,
            ) if agent_spec.openrouter_name == None else ta.agents.OpenRouterAgent(agent_spec.openrouter_name)
        )
    } for agent_spec in game_spec.agent_specs} # build agents
    env=ta.make(game_spec.env_id); env.reset(num_players=len(agents), seed=game_spec.seed); env.state.error_allowance=game_spec.error_allowance; turn=0
    while True:
        pid, obs = env.get_observation()
        action = agents[pid]["model"](obs)
        done, step_info = env.step(action.action); turn += 1 # execute the action & increment turn counter
        # general tracking
        game_information.pid.append(pid); game_information.obs.append(obs); game_information.prompts.append(action.prompt); game_information.completions.append(action.completion)
        game_information.actions.append(action.action); game_information.step_infos.append(step_info); game_information.names[pid] = agents[pid]["name"]
        # player specific trackering
        if agents[pid]["traj"] != None:
            if "logprobs" in action.sampler_info: agents[pid]["traj"].completion_logprobs.append(action.sampler_info["logprobs"])
            agents[pid]["traj"].obs.append(obs); agents[pid]["traj"].prompts.append(action.prompt); agents[pid]["traj"].prompt_ids.append(action.prompt_ids)
            agents[pid]["traj"].completions.append(action.completion); agents[pid]["traj"].completion_ids.append(action.completion_ids); agents[pid]["traj"].actions.append(action.action)
            action.format_feedback["invalid_move"] = False; agents[pid]["traj"].format_feedbacks.append(action.format_feedback); agents[pid]["traj"].step_infos.append(step_info)
            for k, v in action.sampler_info.items(): game_information.action_info.setdefault(k, []).append(v)
        if done: break
    final_rewards, game_info = env.close()
    for pid in agents.keys():
        if agents[pid]["traj"]!=None: agents[pid]["traj"].final_reward=final_rewards[pid]; agents[pid]["traj"].game_info=game_info[pid]; agents[pid]["traj"].num_turns=turn
        if game_info[pid]["invalid_move"] and agents[pid]["traj"]!=None: agents[pid]["traj"].format_feedbacks[-1]["invalid_move"]=True
    game_information.final_rewards=final_rewards; game_information.num_turns=turn; game_information.game_info=game_info
    return game_information, [agents[pid]["traj"] for pid in agents.keys() if agents[pid]["traj"]!=None]


@ray.remote
class GameScheduler:
    def __init__(self, vllm_config, tracker, buffer, model_sampler, env_sampler, action_sampler: str = "default", eval_every: Optional[int] = None, eval_runs: int = 64, max_concurrent_workers: Optional[int] = None):
        self.logger = setup_logger("game_scheduler", ray.get(tracker.get_log_dir.remote()))
        self.tracker, self.buffer = tracker, buffer
        self.model_sampler = model_sampler
        self.env_sampler = env_sampler
        self.action_sampler = action_sampler
        self.actors = [VLLMActor.options(num_gpus=1).remote(cfg=vllm_config, tracker=tracker, name=f"Actor-{i}") for i in range(int(ray.available_resources().get("GPU", 0)))]
        for actor in self.actors: ray.get(actor.ready.remote())
        self._actor_iter = itertools.cycle(self.actors)
        self.local_storage_dir = ray.get(self.tracker.get_collection_dir.remote())
        self._game_idx = 0
        self._running_jobs = {}
        self.eval_every = eval_every
        self.eval_runs = eval_runs
        self.max_concurrent_workers = max_concurrent_workers
        self._eval_iteration = -1
        self._pending_eval_specs: Optional[List] = None  # persisted across _launch_jobs calls

        # thead keeping
        self.flight: Dict[ray.ObjectRef, TaskMeta] = {}
        self._num_running = lambda typ: sum(meta.type == typ for meta in self.flight.values())
        self.logger.info("Collector initialized")

    def collect(self, num_train_workers: int, num_eval_workers: Optional[int]=None):
        self.logger.info("entered collect func")
        while ray.get(self.buffer.continue_collection.remote()):
            self.logger.info("entered colelct loop")
            self._launch_jobs(num_train_workers, num_eval_workers)
            if not self.flight: continue
            done_ref, _ = ray.wait(list(self.flight), num_returns=1)
            self._handle_finished_job(done_ref[0])
        
    def _total_running(self): return len(self.flight)
    def _under_total_cap(self): return self.max_concurrent_workers is None or self._total_running() < self.max_concurrent_workers

    def _launch_jobs(self, max_train: int, max_eval: Optional[int]):
        # Eval first (priority)
        if max_eval is not None and max_eval > 0:
            while self._num_running("eval") < max_eval and self._under_total_cap():
                try:
                    game_spec = self._next_eval_job()
                    if game_spec is None:
                        break  # not triggered yet
                    self.logger.info(f"received eval game_spec: {game_spec}")
                    ref = run_game.remote(game_spec, next(self._actor_iter))
                    self.flight[ref] = TaskMeta("eval", game_spec.env_id)
                except Exception as exc: self.logger.info(f"Exception scheduling eval game: {exc}")
        # Train fills remaining capacity
        while self._num_running("train") < max_train and self._under_total_cap():
            try:
                game_spec = self._next_train_job()
                self.logger.info(f"received train game_spec: {game_spec}")
                ref = run_game.remote(game_spec, next(self._actor_iter))
                self.flight[ref] = TaskMeta("train", game_spec.env_id)
            except Exception as exc:
                self.logger.info(f"Exception scheduling train game: {exc}")

    def _handle_finished_job(self, ref):
        meta = self.flight.pop(ref)
        try: game_information, player_trajs = ray.get(ref)
        except (RayTaskError, RayActorError) as err: self.logger.error(f"Remote episode failed for {meta.type} task: env={meta.env_id}: {err}", exc_info=True); return
        self._post_train(meta, game_information, player_trajs) if meta.type=="train" else self._post_eval(meta, game_information)

    def _next_train_job(self):
        try:
            env_spec = self.env_sampler.sample(kind="train")
            current_ckpt_uid, current_ckpt_lora_path = ray.get(self.model_sampler.get_current_ckpt.remote()) # sample the current checkpoint
            pids = list(range(env_spec.num_players))
            random.shuffle(pids); agent_specs = []
            self._running_jobs[self._game_idx] = {"env_id": env_spec.env_id, "models": []}
            for i, pid in enumerate(pids):
                if i < env_spec.num_actors:
                    self._running_jobs[self._game_idx]["models"].append({"uid": current_ckpt_uid, "pid": pid, "type": "model"})
                    agent_specs.append(AgentSpec(pid=pid, kind="checkpoint", collect_data=True, lora_path=current_ckpt_lora_path, prompt_template=env_spec.prompt_template, action_extraction_fn=env_spec.action_extraction_fn))
                else:
                    opp_uid, kind, opp_lora_path, opp_name_or_path, opp_sampling = ray.get(self.model_sampler.sample_opponent.remote())
                    if kind == "checkpoint": agent_specs.append(AgentSpec(pid=pid, kind="checkpoint", collect_data=False, lora_path=opp_name_or_path, prompt_template=env_spec.prompt_template, action_extraction_fn=env_spec.action_extraction_fn,
                                                                          temperature=opp_sampling.get("temperature"), top_p=opp_sampling.get("top_p"), top_k=opp_sampling.get("top_k"), max_tokens=opp_sampling.get("max_tokens")))
                    else: agent_specs.append(AgentSpec(pid=pid, kind=kind, lora_path=opp_lora_path, openrouter_name=opp_name_or_path)) # OpenRouter agents handle their own sampling
                    self._running_jobs[self._game_idx]["models"].append({"uid": opp_uid, "pid": pid, "type": "opponent"})
            game_spec = GameSpec(game_idx=self._game_idx, env_id=env_spec.env_id, seed=self._game_idx, agent_specs=agent_specs) # populate GameSpec
            self._game_idx += 1
            return game_spec
        except Exception as exc:
            self.logger.info(f"Exception in 'next_train_job': {exc}")
            import time 
            time.sleep(500)

    def _post_train(self, meta: TaskMeta, game_information: GameInformation, player_trajs: List[PlayerTrajectory]):
        for traj in player_trajs: 
            self.buffer.add_player_trajectory.remote(traj, env_id=meta.env_id)
            self.tracker.add_player_trajectory.remote(traj, env_id=meta.env_id)
            if len(traj.obs) > 0: 
                checkpoint_name = os.path.basename(os.path.normpath(game_information.names[traj.pid])) if game_information.names[traj.pid] else 'None'
                write_game_information_to_file(game_information, f"{self.local_storage_dir}/{checkpoint_name}.csv")
        job_info = self._running_jobs.pop(game_information.game_idx, None)
        if job_info is None: return
        actor_rs = [game_information.final_rewards[m["pid"]] for m in job_info["models"] if m["type"] == "model" if m["pid"] in game_information.final_rewards]
        opp_rs = [game_information.final_rewards[m["pid"]] for m in job_info["models"] if m["type"] == "opponent" if m["pid"] in game_information.final_rewards]
        self.env_sampler.update(avg_actor_reward=(sum(actor_rs) / len(actor_rs) if actor_rs else None), avg_opponent_reward=(sum(opp_rs) / len(opp_rs) if opp_rs else None))
        self.model_sampler.update.remote(game_info=game_information, job_info=job_info)

    def _next_eval_job(self, env_spec =None, opp_info=None):
        def _build_eval_spec(env_spec, opp_info=None):
            _, current_ckpt_lora_path = ray.get(self.model_sampler.get_current_ckpt.remote())
            pids = list(range(env_spec.num_players))
            random.shuffle(pids); agent_specs = []
            if env_spec.kind == "checkpoint":
                opp_uid, _, _, opp_path, opp_sampling = opp_info if opp_info is not None else ray.get(self.model_sampler.sample_eval_checkpoint.remote())
                eval_opponent_name = opp_uid
            else: eval_opponent_name = env_spec.fixed_opponent
            for i, pid in enumerate(pids):
                if i == 0: agent_specs.append(AgentSpec(pid=pid, kind="checkpoint", collect_data=True, lora_path=current_ckpt_lora_path, prompt_template=env_spec.prompt_template, action_extraction_fn=env_spec.action_extraction_fn, sampler=self.action_sampler))
                else:
                    if env_spec.kind == "checkpoint":
                        agent_specs.append(AgentSpec(pid=pid, kind="checkpoint", collect_data=False, lora_path=opp_path, prompt_template=env_spec.prompt_template, action_extraction_fn=env_spec.action_extraction_fn,
                                                    temperature=opp_sampling.get("temperature"), top_p=opp_sampling.get("top_p"), top_k=opp_sampling.get("top_k"), max_tokens=opp_sampling.get("max_tokens")))
                    else: agent_specs.append(AgentSpec(pid=pid, kind="openrouter", lora_path=None, openrouter_name=env_spec.fixed_opponent, sampler=self.action_sampler))
            game_spec = GameSpec(game_idx=self._game_idx, env_id=env_spec.env_id, seed=self._game_idx, agent_specs=agent_specs, eval_model_pid=pids[0], eval_opponent_name=eval_opponent_name, eval_iteration=self._eval_iteration)
            self._game_idx += 1
            return game_spec

        try:
            if self.eval_every is not None:
                # Evaluate only at fixed steps
                learner_step = ray.get(self.tracker.get_learner_step.remote())
                next_eval = (self._eval_iteration // self.eval_every + 1) * self.eval_every
                if learner_step >= next_eval:
                    self._eval_iteration = next_eval
                    eval_checkpoints = ray.get(self.model_sampler.get_eval_checkpoints.remote())
                    self._pending_eval_specs = []
                    for spec in self.env_sampler.get_eval_specs():
                        if spec.kind == "checkpoint":
                            for ckpt in eval_checkpoints: self._pending_eval_specs += [_build_eval_spec(env_spec=spec, opp_info=ckpt) for _ in range(self.eval_runs)]
                        else: self._pending_eval_specs += [_build_eval_spec(env_spec=spec) for _ in range(self.eval_runs)]
                    self.logger.info(f"Triggering eval iteration {self._eval_iteration} at learner step {learner_step} ({len(self._pending_eval_specs)} games)")
                return self._pending_eval_specs.pop() if self._pending_eval_specs else None
            else:
                # Continous evaluation
                if env_spec is None: env_spec = self.env_sampler.sample(kind="eval")
                return _build_eval_spec(env_spec=env_spec, opp_info=opp_info)
        except Exception as exc:
            self.logger.info(f"Exception in 'next_eval_job': {exc}")
            import time
            time.sleep(500)
    
    def _post_eval(self, meta: TaskMeta, game_information: GameInformation):
        self.tracker.add_eval_game_information.remote(game_information=game_information, env_id=meta.env_id, aggregate=self.eval_runs if self.eval_every else None)
