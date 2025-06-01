import re, random, itertools
import textarena as ta 
from typing import List, Dict, Tuple, Optional, Any, Callable

import ray 
from ray.util.actor_pool import ActorPool

# local imports
from unstable.core import Trajectory
from unstable.tracker import WandBTracker
from unstable.actors.vllm_actor import VLLMActor
from unstable.utils.templates import OBSERVATION_FORMATTING, ACTION_EXTRACTION


class CallableActorWrapper:
    def __init__(self, actor: VLLMActor, lora_path: str, obs_formatting_fn: Callable, action_extraction_fn: Callable):
        self.actor = actor 
        self.lora_path = lora_path
        self.obs_formatting_fn = obs_formatting_fn
        self.action_extraction_fn = action_extraction_fn

    def __call__(self, observation: str):
        formatted_prompt = self.obs_formatting_fn(observation=observation)
        raw_action = ray.get(self.actor.submit_prompt.remote(prompt=formatted_prompt, lora_path=self.lora_path))
        extracted_action, format_feedback = self.action_extraction_fn(raw_action=raw_action)
        return extracted_action #raw_action, extracted_action, format_feedback, formatted_prompt

    def get_full_response(self, observation: str):
        formatted_prompt = self.obs_formatting_fn(observation=observation)
        raw_action = ray.get(self.actor.submit_prompt.remote(prompt=formatted_prompt, lora_path=self.lora_path))
        extracted_action, format_feedback = self.action_extraction_fn(raw_action=raw_action)
        return raw_action, extracted_action, format_feedback, formatted_prompt

def _iter_from_uid(uid: str) -> int:
    """ckpt-123 → 123, fallback to 0 if it doesn’t match."""
    m = re.search(r"(\d+)$", uid)
    return int(m.group(1)) if m else 0

@ray.remote 
class Collector:
    def __init__(
        self, num_actors, step_buffer, model_pool, vllm_config: Dict[str, Any], 
        training_envs: List[Tuple[str, int, Optional[str]]], evaluation_envs: List[Tuple[str, int, Optional[str]]], 
        evaluation_opponent: str = "google/gemini-2.0-flash-lite-001", num_eval_games_per_env: int = 32, eval_every_n_steps: int = 5,
        tracker: Optional[WandBTracker] = None
    ):
        """
        TODO 
        vllm_config (Dict[str, Any]): # has to include the model_name and any other inference parameters you want
        """
        self.buffer = step_buffer 
        self.model_pool = model_pool 
        self.tracker = tracker 
        self.alive = True 
        self.training_envs = training_envs
        self.evaluation_envs = evaluation_envs
        self.evaluation_opponent = evaluation_opponent
        self.num_eval_games_per_env = num_eval_games_per_env
        self.eval_every_n_steps = eval_every_n_steps
        self._last_eval_ckpt: Optional[str] = None
        self._eval_flight = []; self._pending_eval_tasks = []

        self._eval_seed_table = {env_id: [i for i in range(self.num_eval_games_per_env)] for env_id, *_ in self.evaluation_envs}

        # build the actor pool
        actors = [VLLMActor.options(num_gpus=1).remote(vllm_config=vllm_config) for _ in range(num_actors)]
        self.actor_iter = itertools.cycle(actors)


    def _sample_env(self, seed=489):
        env_id, num_players, prompt_template = random.Random(seed).choice(self.training_envs)
        player_id = random.Random(seed+1).randrange(num_players)
        return env_id, num_players, player_id, prompt_template

    def _build_game_args(self, env_id, num_players, player_id, prompt_template, seed):
        actor = next(self.actor_iter)
        current_model_uid = ray.get(self.model_pool.latest_ckpt.remote())
        lora_path = ray.get(self.model_pool.ckpt_path.remote(current_model_uid))
        opponent_uid = ray.get(self.model_pool.sample.remote(uid_me=current_model_uid))
        opponent_path_or_name = ray.get(self.model_pool.ckpt_path.remote(opponent_uid))
        return dict(
            env_id=env_id, num_players=num_players, player_id=player_id, actor=actor, lora_path=lora_path, opponent_uid=opponent_uid, 
            model_uid=current_model_uid,
            opponent_path_or_name=opponent_path_or_name, prompt_template=prompt_template, seed=seed
        )
    def _spawn_eval_sweep(self, ckpt_uid: str):
        """Populate _pending_eval_tasks; nothing is launched here."""
        lora_path = ray.get(self.model_pool.ckpt_path.remote(ckpt_uid))
        for env_id, num_players, prompt_template in self.evaluation_envs:
            seeds = self._eval_seed_table[env_id]
            for s in seeds:
                rnd = random.Random(s)
                player_id = rnd.randrange(num_players)
                actor = next(self.actor_iter)
                eval_args = dict(
                    env_id=env_id, num_players=num_players, player_id=player_id, actor=actor, lora_path=lora_path,
                    opponent_name=self.evaluation_opponent, prompt_template=prompt_template, seed=s,
                )
                # keep the metadata so we can forward it to the tracker later
                self._pending_eval_tasks.append((eval_args, env_id, player_id, ckpt_uid, s))



        # ──────────────────────────────────────────────────────────────────────────
    def collect(self, num_workers: int, num_eval_workers: int):

        # ── 1.  remote helpers ───────────────────────────────────────────────
        # @ray.remote(num_cpus=0.1)
        @ray.remote(num_cpus=0)
        def run_game(env_id, num_players, player_id, actor, lora_path, opponent_uid, model_uid, opponent_path_or_name, prompt_template, seed: int = 489):
            obs_format = OBSERVATION_FORMATTING[prompt_template]
            extract_fn = ACTION_EXTRACTION[prompt_template]
            model = CallableActorWrapper(actor, lora_path, obs_format, extract_fn)

            if opponent_uid is None: opponent = model
            elif opponent_uid.startswith("ckpt-"): opponent = CallableActorWrapper(actor, opponent_path_or_name, obs_format, extract_fn)
            else: opponent = ta.agents.OpenRouterAgent(opponent_path_or_name)

            env = ta.make(env_id)
            env.reset(num_players=num_players, seed=seed)
            env.state.error_allowance = 0

            traj, turn = Trajectory(), 0
            while True:
                pid, obs = env.get_observation()
                if pid == player_id:
                    raw, act, fb, prompt = model.get_full_response(obs)
                    done, info = env.step(act)
                    traj.pid.append(pid); traj.obs.append(prompt)
                    traj.actions.append(raw); fb["invalid_move"] = 0; traj.extracted_actions.append(act)
                    traj.infos.append(info)
                    traj.board_states.append(env.state.game_state['board'] if 'board' in env.state.game_state else None)
                    traj.format_feedbacks.append(fb)
                else:
                    done, info = env.step(opponent(obs))
                turn += 1
                if done:
                    break
            traj.final_rewards = env.close()
            traj.num_turns = turn
            if info["end_by_invalid"] and pid==player_id:  
                traj.format_feedbacks[-1]["invalid_move"] = 1 # adjust final move to invalid as necessary
            return traj, player_id, env_id

        # @ray.remote(num_cpus=0.1)
        @ray.remote(num_cpus=0)
        def run_eval_game(env_id, num_players, player_id, actor, lora_path, opponent_name, prompt_template, seed: int = 489):
            model = CallableActorWrapper( actor, lora_path, OBSERVATION_FORMATTING[prompt_template], ACTION_EXTRACTION[prompt_template])
            opponent = ta.agents.OpenRouterAgent(opponent_name)

            env = ta.make(env_id)
            env.reset(num_players=num_players, seed=seed)
            env.state.error_allowance = 0

            episode_info, turn = [], 0
            while True:
                pid, obs = env.get_observation()
                if pid == player_id:
                    full, act, fb, prompt = model.get_full_response(obs)
                    name = "current_ckpt"
                else:
                    full = act = opponent(obs)
                    name = opponent_name

                done, info = env.step(act)
                turn += 1
                episode_info.append({
                    "pid": pid, "model_name": name, "observation": obs,
                    "full_action": full, "submitted_action": act,
                    "done": done, "info": info, "step": turn
                })
                if done:
                    break

            return episode_info, player_id, env_id, env.close()

        # ── 2.  local queues / state ─────────────────────────────────────────
        train_flight: list[tuple] = []
        iter_seed = 0

        while self.alive:

            # ── A) top-up TRAINING games ────────────────────────────────────
            while len(train_flight) < num_workers:
                env_id, n, pid, tmpl = self._sample_env(seed=iter_seed)
                args = self._build_game_args(env_id, n, pid, tmpl, iter_seed)
                fut = run_game.remote(**args)
                train_flight.append((fut, args["opponent_uid"], args["model_uid"]))
                iter_seed += 1

            # ── B) maybe schedule a NEW evaluation sweep ───────────────────
            latest_ckpt = ray.get(self.model_pool.latest_ckpt.remote())
            if (not self._pending_eval_tasks and not self._eval_flight):
                if self._last_eval_ckpt is None:
                    self._spawn_eval_sweep(latest_ckpt)
                    self._last_eval_ckpt = latest_ckpt
                elif (_iter_from_uid(latest_ckpt) - _iter_from_uid(self._last_eval_ckpt)) >= self.eval_every_n_steps:
                    self._spawn_eval_sweep(latest_ckpt)
                    self._last_eval_ckpt = latest_ckpt

            # ── C) top-up EVALUATION games (respect cap) ───────────────────
            while (len(self._eval_flight) < num_eval_workers
                   and self._pending_eval_tasks):
                eval_args, env_id, pid, ckpt_uid, seed = self._pending_eval_tasks.pop(0)
                fut = run_eval_game.remote(**eval_args)
                self._eval_flight.append((fut, env_id, pid, ckpt_uid, seed))

            # ── D) wait for *any* game (train or eval) to finish ────────────
            wait_pool = ([f for f, *_ in train_flight] + [f for f, *_ in self._eval_flight])
            if not wait_pool:
                continue  # nothing running (should not happen)
            done_ref, _ = ray.wait(wait_pool, num_returns=1)
            finished = done_ref[0]

            # ── E) if it was a TRAINING game ───────────────────────────────
            idx = next((i for i, (f, _, _) in enumerate(train_flight)
                        if f == finished), None)
            if idx is not None:
                fut, opp_uid, mdl_uid = train_flight.pop(idx)
                traj, pid, env_id = ray.get(fut)
                # send to buffer / trueskill / tracker
                self.buffer.add_trajectory.remote(traj, pid, env_id)
                # self.model_pool.add_trajectory.remote(uid=mdl_uid, pid=pid, 
                #                                       env_id=env_id, traj=traj)
                if opp_uid is not None:
                    self.model_pool.update_ratings.remote(
                        uid_me=mdl_uid, uid_opp=opp_uid,
                        final_reward=traj.final_rewards[pid]
                    )
                if self.tracker:
                    self.tracker.add_trajectory.remote(traj, pid, env_id)
                continue  # back to top of loop

            # ── F) otherwise it was an EVALUATION game ─────────────────────
            idx = next(i for i, (f, *_) in enumerate(self._eval_flight) if f == finished)
            fut, env_id, pid, ckpt_uid, seed = self._eval_flight.pop(idx)
            ep_info, _, _, final_r = ray.get(fut)
            if self.tracker:
                self.tracker.add_eval_episode.remote(episode_info=ep_info, final_reward=final_r, current_ckpt_pid=pid, env_id=env_id, ckpt_iteration=ckpt_uid)
            print(f"[EVAL] ckpt={ckpt_uid} env={env_id} seed={seed} done")

    # ──────────────────────────────────────────────────────────────────────────


    def stop(self): self.alive = False