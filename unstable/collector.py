import re, ray, random, itertools
from typing import List, Dict, Tuple, Optional, Any, Callable
from ray.exceptions import RayTaskError, RayActorError

import textarena as ta 
assert ta.__version__ == "0.6.9", f"You need to use TextArena version 0.6.9 (build from source). You are using: {textarena.__version__}"

# local imports
from unstable.core import Trajectory, BaseTracker
from unstable.actor import VLLMActor
from unstable.utils.templates import OBSERVATION_FORMATTING, ACTION_EXTRACTION


class CallableActorWrapper:
    """A wrapper around VLLMActor to format observations, extract actions, and interact with remote actor."""

    def __init__(self, actor: VLLMActor, lora_path: str, obs_formatting_fn: Callable, action_extraction_fn: Callable):
        """
        Initializes the CallableActorWrapper.

        Args:
            actor (VLLMActor): Remote actor handling the prompt submission.
            lora_path (str): Path to the LoRA weights for inference.
            obs_formatting_fn (Callable): Function to format observations before prompting.
            action_extraction_fn (Callable): Function to extract actionable information from raw responses.
        """
        self.actor = actor
        self.lora_path = lora_path
        self.obs_formatting_fn = obs_formatting_fn
        self.action_extraction_fn = action_extraction_fn

    def __call__(self, observation: str) -> str:
        """ Formats observation, gets a response from actor, and extracts action """
        formatted_prompt = self.obs_formatting_fn(observation=observation)
        raw_action = ray.get(self.actor.submit_prompt.remote(prompt=formatted_prompt, lora_path=self.lora_path))
        extracted_action, _ = self.action_extraction_fn(raw_action=raw_action)
        return extracted_action

    def get_full_response(self, observation: str) -> Tuple[str, str, Dict[str, Any], str]:
        """ Returns raw response, extracted action, feedback, and formatted prompt """
        formatted_prompt = self.obs_formatting_fn(observation=observation)
        raw_action = ray.get(self.actor.submit_prompt.remote(prompt=formatted_prompt, lora_path=self.lora_path))
        extracted_action, format_feedback = self.action_extraction_fn(raw_action=raw_action)
        return raw_action, extracted_action, format_feedback, formatted_prompt


def _iter_from_uid(uid: str) -> int:
    """ Extracts numerical iteration from UID strings like 'ckpt-123'. Defaults to 0 if no number is found """
    match = re.search(r"(\d+)$", uid)
    return int(match.group(1)) if match else 0

def _extract_action(action: str, action_space=None) -> str:
    if action_space: match = action_space.search(action)
    else: match = re.search(r"\[(.*?)\]", action)
    return match.group(1).strip().lower() if match else ""


@ray.remote
class Collector:
    """Collects trajectories and evaluation episodes from environments using parallel VLLM actors."""
    def __init__(
        self,
        num_actors: int,
        step_buffer: Any,
        model_pool: Any,
        vllm_config: Dict[str, Any],
        training_envs: List[Tuple[str, int, Optional[str]]],
        evaluation_envs: List[Tuple[str, int, Optional[str]]],
        evaluation_opponent: str = "google/gemini-2.0-flash-lite-001",
        max_eval_games_per_ckpt: int = 32,
        eval_every_n_steps: int = 5,
        tracker: Optional[BaseTracker] = None,
        filter_opponent_invalid: bool = False,
        action_extraction: str = "default"
    ):
        """
        Initializes the Collector.

        Args:
            num_actors (int): Number of parallel actors.
            step_buffer (Any): Buffer to store collected trajectories.
            model_pool (Any): Pool managing model checkpoints and ratings.
            vllm_config (Dict[str, Any]): Configuration for VLLM inference.
            training_envs (List[Tuple[str, int, Optional[str]]]): List of training environments.
            evaluation_envs (List[Tuple[str, int, Optional[str]]]): List of evaluation environments.
            evaluation_opponent (str): Opponent for evaluation.
            max_eval_games_per_ckpt (int): Max number of eval games per checkpoint.
            eval_every_n_steps (int): Frequency of evaluations.
            tracker (Optional[WandBTracker]): Tracker for logging metrics.
            filter_opponent_invalid (bool): whether to remove games ending with the opponent making an invalid move
        """
        self.buffer = step_buffer
        self.model_pool = model_pool
        self.tracker = tracker
        self.alive = True
        self.training_envs = training_envs
        self.evaluation_envs = evaluation_envs
        self.evaluation_opponent = evaluation_opponent
        self.max_eval_games_per_ckpt = max_eval_games_per_ckpt
        self.eval_every_n_steps = eval_every_n_steps
        self.filter_opponent_invalid = filter_opponent_invalid
        self.action_extraction = action_extraction

        self.actors = [VLLMActor.options(num_gpus=1).remote(vllm_config=vllm_config, tracker=tracker, name=f"Actor-{i}") for i in range(num_actors)]
        self.actor_iter = itertools.cycle(self.actors)

    def get_actors(self): return self.actors
    
    def _sample_env(self, seed: int = 489, _type: str = "train") -> Tuple[str, int, int, Optional[str]]:
        """Samples a random environment from the training set."""
        env_id, num_players, prompt_template = random.Random(seed).choice(self.training_envs if _type=="train" else self.evaluation_envs)
        player_id = random.Random(seed + 1).randrange(num_players)
        return env_id, num_players, player_id, prompt_template

    def _build_game_args(self, env_id, num_players, player_id, prompt_template, seed) -> Dict[str, Any]:
        """Builds arguments for initiating a game run."""
        actor = next(self.actor_iter)
        current_model_uid = ray.get(self.model_pool.latest_ckpt.remote())
        lora_path = ray.get(self.model_pool.ckpt_path.remote(current_model_uid))
        opponent_uid = ray.get(self.model_pool.sample.remote(uid_me=current_model_uid))
        opponent_path_or_name = ray.get(self.model_pool.ckpt_path.remote(opponent_uid))
        return dict(
            env_id=env_id, num_players=num_players, player_id=player_id, actor=actor, lora_path=lora_path,
            opponent_uid=opponent_uid, model_uid=current_model_uid, opponent_path_or_name=opponent_path_or_name,
            prompt_template=prompt_template, action_extraction=self.action_extraction, seed=seed
        )

    def _build_eval_args(self, env_id, num_players, player_id, prompt_template, ckpt_uid, seed):
        lora_path = ray.get(self.model_pool.ckpt_path.remote(ckpt_uid))
        actor = next(self.actor_iter)
        return dict(
            env_id=env_id, num_players=num_players, player_id=player_id, actor=actor, lora_path=lora_path,
            opponent_name=self.evaluation_opponent, prompt_template=prompt_template, action_extraction=self.action_extraction, seed=seed
        )

    def collect(self, num_workers: int, num_eval_workers: int):
        """ Main loop for collecting trajectories and evaluations concurrently """
        @ray.remote(num_cpus=0)
        def run_game(env_id, num_players, player_id, actor, lora_path, opponent_uid, model_uid, opponent_path_or_name, prompt_template, action_extraction, seed: int = 489):
            obs_format = OBSERVATION_FORMATTING[prompt_template]
            extract_fn = ACTION_EXTRACTION[action_extraction]
            model = CallableActorWrapper(actor, lora_path, obs_format, extract_fn)
            game_action_seq = {}

            if opponent_uid is None:                opponent = model
            elif opponent_uid.startswith("ckpt-"):  opponent = CallableActorWrapper(actor, opponent_path_or_name, obs_format, extract_fn)
            else:                                   opponent = ta.agents.OpenRouterAgent(opponent_path_or_name)

            env = ta.make(env_id)
            env.reset(num_players=num_players, seed=seed)
            env.state.error_allowance = 0

            traj, turn = Trajectory(), 0
            while True:
                pid, obs = env.get_observation()
                if pid == player_id:
                    raw_act, act, fb, prompt = model.get_full_response(obs)
                    done, info = env.step(act)
                    traj.pid.append(pid)
                    traj.obs.append(prompt)
                    traj.actions.append(raw_act)
                    fb["invalid_move"] = 0
                    traj.extracted_actions.append(act)
                    traj.infos.append(info)
                    traj.format_feedbacks.append(fb)
                else:
                    act = opponent(obs)
                    done, info = env.step(act)
                turn += 1
                if done:
                    break

                if pid not in game_action_seq: game_action_seq[pid] = []
                game_action_seq[pid].append(_extract_action(act, env.action_space(pid)))
                
            traj.final_rewards = env.close()
            traj.num_turns = turn
            if info["end_by_invalid"] and pid==player_id:  
                traj.format_feedbacks[-1]["invalid_move"] = 1 # adjust final move to invalid as necessary
            return traj, player_id, env_id, (info["end_by_invalid"] and pid != player_id), game_action_seq

        @ray.remote(num_cpus=0)
        def run_eval_game(env_id, num_players, player_id, actor, lora_path, opponent_name, prompt_template, action_extraction, seed: int = 489):
            model = CallableActorWrapper( actor, lora_path, OBSERVATION_FORMATTING[prompt_template], ACTION_EXTRACTION[action_extraction])
            opponent = ta.agents.OpenRouterAgent(opponent_name)

            env = ta.make(env_id)
            env.reset(num_players=num_players, seed=seed)
            env.state.error_allowance = 0

            episode_info, turn = [], 0
            while True:
                pid, obs = env.get_observation()
                if pid == player_id:    full, act, fb, prompt = model.get_full_response(obs); name = "current_ckpt"
                else:                   full = act = opponent(obs); name = opponent_name

                done, info = env.step(act)
                turn += 1
                episode_info.append({"pid": pid, "model_name": name, "observation": obs, "full_action": full, "submitted_action": act, "done": done, "info": info, "step": turn})
                if done:
                    break

            return episode_info, player_id, env_id, env.close()

        train_flight: List[Tuple] = []
        eval_flight: List[Tuple] = []
        ckpt_eval_game_count: Dict[str, int] = {}
        iter_seed = 0
        eval_iter_seed = 0

        while self.alive:
            while len(train_flight) < num_workers:
                env_id, n, pid, tmpl = self._sample_env(seed=iter_seed)
                args = self._build_game_args(env_id, n, pid, tmpl, iter_seed)
                fut = run_game.remote(**args)
                train_flight.append((fut, args["opponent_uid"], args["model_uid"]))
                iter_seed += 1

            latest_ckpt = ray.get(self.model_pool.latest_ckpt.remote())
            while (len(eval_flight) < num_eval_workers) and (ckpt_eval_game_count.get(latest_ckpt, 0) < self.max_eval_games_per_ckpt*len(self.evaluation_envs)):
                env_id, n, pid, tmpl = self._sample_env(seed=eval_iter_seed, _type="eval")
                args = self._build_eval_args(env_id, n, pid, tmpl, latest_ckpt, eval_iter_seed)
                fut = run_eval_game.remote(**args)
                eval_flight.append((fut, env_id, pid, latest_ckpt, eval_iter_seed))
                eval_iter_seed += 1
                ckpt_eval_game_count[latest_ckpt] = ckpt_eval_game_count.get(latest_ckpt, 0) + 1 

            wait_pool = ([f for f, *_ in train_flight] + [f for f, *_ in eval_flight])
            if not wait_pool: continue  # nothing running (should not happen)
            done_ref, _ = ray.wait(wait_pool, num_returns=1)
            finished = done_ref[0]

            try: # TODO maybe move into the actual one below
                result = ray.get(finished)
            except (RayTaskError, RayActorError) as err:
                # Log & replace the failed flight entry
                logger.error("Game task failed: %s", err)
                continue  # loop back – flight slot will be refilled next iteration


            idx = next((i for i, (f, _, _) in enumerate(train_flight) if f == finished), None)
            if idx is not None:
                fut, opp_uid, mdl_uid = train_flight.pop(idx)
                traj, pid, env_id, end_by_opponent_invalid, game_action_seq = ray.get(fut)
                if self.filter_opponent_invalid and end_by_opponent_invalid:
                    continue

                self.buffer.add_trajectory.remote(traj, pid, env_id)
                if self.tracker: self.tracker.add_trajectory.remote(traj, pid, env_id)
                if opp_uid is not None:
                    self.model_pool.push_game_outcome.remote(uid_me=mdl_uid, uid_opp=opp_uid, final_reward=traj.final_rewards[pid], game_action_seq=game_action_seq, env_id=env_id)
                continue  # back to top of loop

            idx = next(i for i, (f, *_) in enumerate(eval_flight) if f == finished)
            fut, env_id, pid, ckpt_uid, seed = eval_flight.pop(idx)
            ep_info, _, _, final_r = ray.get(fut)
            if self.tracker: self.tracker.add_eval_episode.remote(episode_info=ep_info, final_rewards=final_r, player_id=pid, env_id=env_id, iteration=ckpt_uid)
            print(f"[EVAL] ckpt={ckpt_uid} env={env_id} seed={seed} done")


    def stop(self): self.alive = False