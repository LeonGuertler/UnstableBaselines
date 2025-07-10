import re, random, logging, itertools
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Tuple, Protocol

import ray
import textarena as ta
from ray.exceptions import RayActorError, RayTaskError

# local imports
from unstable.actor import VLLMActor
from unstable.core import BaseTracker, Trajectory, EpisodeResult, PlaySpec, TaskMeta, AgentSpec
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

def _iter_from_uid(uid: str) -> int: return int(m.group(1)) if (m := re.search(r"(\d+)$", uid)) else 0
def _extract_action(action: str) -> str: return (m.group(1).strip().lower() if (m := re.search(r"\[(.*?)\]", action)) else "")

@ray.remote(num_cpus=0)
def play_episode(spec: PlaySpec, actor: VLLMActor) -> EpisodeResult:
    try:
        def _build_agent(agent_spec: AgentSpec):
            match agent_spec.kind:
                case "checkpoint": return CallableActorWrapper(actor=actor, lora_path=agent_spec.model, obs_fmt_fn=OBSERVATION_FORMATTING[agent_spec.prompt_template], extract_fn=ACTION_EXTRACTION[agent_spec.action_extraction_fn])
                case "openrouter": return ta.agents.OpenRouterAgent(agent_spec.model)
                case _: raise ValueError(f"unknown kind {agent_spec.kind!r}")
        agents = {pid: _build_agent(spec.agent_specs[pid]) for pid in range(spec.num_players)}
        env=ta.make(spec.env_id); env.reset(num_players=spec.num_players, seed=spec.seed); env.state.error_allowance=0
        traj = Trajectory(); turn = 0; action_seq: List[str] = []
        while True:
            pid, obs = env.get_observation()
            if pid == spec.player_id: raw, extracted, prompt, format_feedback = agents[pid].act_full(obs)
            else: extracted = agents[pid](obs)
            done, step_info = env.step(extracted)
            if pid == spec.player_id:  # Only track the learner’s internal details.
                traj.pid.append(pid); traj.obs.append(prompt); traj.actions.append(raw); traj.extracted_actions.append(extracted)
                traj.infos.append(step_info); format_feedback["invalid_move"] = 0; traj.format_feedbacks.append(format_feedback)
            if done: break
            action_seq.append(_extract_action(extracted))
            turn += 1
        traj.final_rewards, game_info = env.close(); traj.num_turns = turn
        if spec.num_players > 1: end_by_opp_inv = game_info[1-spec.player_id]["invalid_move"]
        else: end_by_opp_inv = False # single player games have no opponent
        if game_info[spec.player_id]["invalid_move"]: traj.format_feedbacks[-1]["invalid_move"] = 1
        return EpisodeResult(traj=traj, end_by_opponent_invalid=end_by_opp_inv, action_seq=action_seq, final_rewards=traj.final_rewards)
    except Exception as e:
        print(f"EXCEPTION DURING COLLECTION {e}")

@ray.remote
class Collector:
    def __init__(
        self, num_actors: int, step_buffer, model_pool, tracker: BaseTracker, vllm_config: Dict[str, Any], training_envs: List[tuple[str, int, str|None]], evaluation_envs: List[tuple[str, int, str|None]], 
        evaluation_opponent: str="google/gemini-2.0-flash-lite-001", max_eval_games_per_ckpt: int=32, filter_opponent_invalid: bool=False, action_extraction: str="default",
    ) -> None:
        self.logger = setup_logger("collector", ray.get(tracker.get_log_dir.remote()))
        self.buffer, self.model_pool, self.tracker = step_buffer, model_pool, tracker
        self.train_envs, self.eval_envs = training_envs, evaluation_envs
        self.eval_opponent = evaluation_opponent
        self.max_eval = max_eval_games_per_ckpt
        self.filter_invalid = filter_opponent_invalid
        self.extract_key = action_extraction
        self.actors = [VLLMActor.options(num_gpus=1).remote(cfg=vllm_config, tracker=tracker, name=f"Actor-{i}") for i in range(num_actors)]
        self._actor_iter = itertools.cycle(self.actors)
        self.rng_train, self.rng_eval = random.Random(489), random.Random(977)
        self.flight: Dict[ray.ObjectRef, TaskMeta] = {}
        self.eval_counter: Dict[str, int] = {}

    def _next_actor(self):                          return next(self._actor_iter)
    def _obs_extract(self, tmpl):                   return OBSERVATION_FORMATTING[tmpl], ACTION_EXTRACTION[self.extract_key]
    def _num_running(self, typ: str) -> int:        return sum(meta.type == typ for meta in self.flight.values())
    def _make_learner_spec(self, lora_path, tmpl):  return AgentSpec(kind="checkpoint", model=lora_path, prompt_template=tmpl, action_extraction_fn=self.extract_key)
    def _make_opponent_spec(self, opp_path, opp_type, tmpl):  return AgentSpec(kind=("checkpoint" if (opp_type=="checkpoint") else "openrouter"), model=opp_path, prompt_template=tmpl, action_extraction_fn=self.extract_key)
    def _sample_env(self, rng: random.Random, envs):
        env_id, n, tmpl = rng.choice(envs)
        return env_id, n, rng.randrange(n), tmpl
 
    def collect(self, num_workers: int, num_eval_workers: int):
        while ray.get(self.buffer.continue_collection.remote()):
            self._launch_jobs(num_workers, num_eval_workers)
            if not self.flight: continue
            done_ref, _ = ray.wait(list(self.flight), num_returns=1)
            self._handle_finished(done_ref[0])

    def _launch_jobs(self, max_train: int, max_eval: int):
        while self._num_running("train") < max_train: self._submit_train()
        latest_ckpt = ray.get(self.model_pool.latest_ckpt.remote())
        if self.eval_counter.get(latest_ckpt, 0) < self.max_eval * len(self.eval_envs):
            while self._num_running("eval") < max_eval: 
                self._submit_eval(latest_ckpt)

    def _submit_train(self):
        self.logger.info(f"Submitting new train env (start of func)")
        env_id, n, pid, tmpl = self._sample_env(self.rng_train, self.train_envs)
        current_uid = ray.get(self.model_pool.latest_ckpt.remote())
        lora_path, _ = ray.get(self.model_pool.ckpt_path.remote(current_uid))
        opp_uid = ray.get(self.model_pool.sample.remote(uid_me=current_uid))
        opp_path, opp_type = ray.get(self.model_pool.ckpt_path.remote(opp_uid))
        spec = PlaySpec(env_id, n, pid, [self._make_learner_spec(lora_path, tmpl) if i == pid else self._make_opponent_spec(opp_path, opp_type, tmpl) for i in range(n)], seed=self.rng_train.getrandbits(32))
        self.logger.info(f"Submitting new train env (spec = {spec})")
        actor = self._next_actor()
        ref = play_episode.remote(spec, actor)
        self.flight[ref] = TaskMeta("train", env_id, pid, spec.seed, current_uid, opp_uid)
        self.logger.debug(f"↪ train {_iter_from_uid(current_uid)} {env_id} pid={pid}")

    def _submit_eval(self, ckpt_uid):
        env_id, n, pid, tmpl = self._sample_env(self.rng_eval, self.eval_envs)
        lora_path, _ = ray.get(self.model_pool.ckpt_path.remote(ckpt_uid))
        spec = PlaySpec(env_id, n, pid, [self._make_learner_spec(lora_path, tmpl) if i == pid else self._make_opponent_spec(self.eval_opponent, "fixed", tmpl) for i in range(n)], seed=self.rng_eval.getrandbits(32))
        actor = self._next_actor()
        ref = play_episode.remote(spec, actor)
        self.flight[ref] = TaskMeta("eval", env_id, pid, spec.seed, ckpt_uid)
        self.eval_counter[ckpt_uid] = self.eval_counter.get(ckpt_uid, 0) + 1
        self.logger.debug(f"↪ eval {_iter_from_uid(ckpt_uid)} {env_id} pid={pid}")
        
    def _handle_finished(self, ref):
        meta = self.flight.pop(ref)
        try: res: EpisodeResult = ray.get(ref)
        except (RayTaskError, RayActorError) as err:
            self.logger.error(f"Remote episode failed for {meta.type} task: env={meta.env_id}, player_id={meta.player_id}, seed={meta.seed}, ckpt_uid={meta.ckpt_uid}, opponent_uid={meta.opponent_uid or 'N/A'}: {err}", exc_info=True); return
        match meta.type:
            case "train":   self._post_train(meta, res)
            case "eval":    self._post_eval(meta, res)
            case _:         self.logger.warning(f"Unknown task type {meta.type}")

    def _post_train(self, meta: TaskMeta, res: EpisodeResult):
        print("train_done", dict(env=meta.env_id, ckpt=meta.ckpt_uid, length=len(res.traj.pid), invalid=res.end_by_opponent_invalid))
        if self.filter_invalid and res.end_by_opponent_invalid: return
        self.buffer.add_trajectory.remote(res.traj, meta.player_id, meta.env_id)
        self.tracker.add_trajectory.remote(res.traj, meta.player_id, meta.env_id)
        if meta.opponent_uid: self.model_pool.push_game_outcome.remote(uid_me=meta.ckpt_uid, uid_opp=meta.opponent_uid, final_reward=res.traj.final_rewards[meta.player_id], game_action_seq=res.action_seq, env_id=meta.env_id)

    def _post_eval(self, meta: TaskMeta, res: EpisodeResult):
        # self.tracker.add_eval_episode.remote(episode_info=None, rewards=res.final_rewards, player_id=meta.player_id, env_id=meta.env_id, iteration=meta.ckpt_uid)
        self.logger.info(res.traj)
        self.tracker.add_eval_episode.remote(rewards=res.final_rewards, player_id=meta.player_id, env_id=meta.env_id)
        self.logger.info("eval_done", extra=dict(env=meta.env_id, ckpt=meta.ckpt_uid, seed=meta.seed, reward=res.final_rewards[meta.player_id]))