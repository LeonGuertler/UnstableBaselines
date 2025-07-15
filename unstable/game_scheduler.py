import ray, random, json
from unstable.utils import setup_logger
from unstable._types import AgentSpec, GameSpec, GameInformation

@ray.remote
class GameScheduler:
    def __init__(self, model_sampler, env_sampler, logging_dir: str):
        self.logger = setup_logger("game_scheduler", logging_dir)
        self.model_sampler = model_sampler
        self.env_sampler = env_sampler
        self._game_idx = 0
        self._running_jobs = {}

    def next_train_job(self):
        try:
            env_spec = self.env_sampler.sample(kind="train") # sample the env spec
            current_ckpt_uid, current_ckpt_lora_path = self.model_sampler.get_current_ckpt() # sample the current checkpoint
            # build the game spec and agent specs
            pids = list(range(env_spec.num_players))
            random.shuffle(pids); agent_specs = []
            self._running_jobs[self._game_idx] = {"env_id": env_spec.env_id, "models": []}
            for i, pid in enumerate(pids):
                if i < env_spec.num_actors: # add current ckpt
                    self._running_jobs[self._game_idx]["models"].append({"uid": current_ckpt_uid, "pid": pid, "type": "model"})
                    agent_specs.append(AgentSpec(pid=pid, kind="checkpoint", collect_data=True, lora_path=current_ckpt_lora_path, prompt_template=env_spec.prompt_template, action_extraction_fn=env_spec.action_extraction_fn))
                else: # sample opponent and add
                    opp_uid, kind, opp_lora_path, opp_openrouter_name = self.model_sampler.sample_opponent()
                    agent_specs.append(AgentSpec(pid=pid, kind=kind, lora_path=opp_lora_path, openrouter_name=opp_openrouter_name)) # TODO might have to adjust what is passed
                    self._running_jobs[self._game_idx]["models"].append({"uid": opp_uid, "pid": pid, "type": "opponent"})
            game_spec = GameSpec(game_idx=self._game_idx, env_id=env_spec.env_id, seed=self._game_idx, agent_specs=agent_specs) # populate GameSpec
            self._game_idx += 1
            return game_spec
        except Exception as exc:
            self.logger.info(f"Exception in 'next_train_job': {exc}")
            import time 
            time.sleep(500)

    def next_eval_job(self):
        try:
            env_spec = self.env_sampler.sample(kind="eval")
            current_ckpt_uid, current_ckpt_lora_path = self.model_sampler.get_current_ckpt() # sample the current checkpoint
            pids = list(range(env_spec.num_players))
            random.shuffle(pids); agent_specs = []
            for i, pid in enumerate(pids):
                if i == 0:  agent_specs.append(AgentSpec(pid=pid, kind="checkpoint", collect_data=True, lora_path=current_ckpt_lora_path, prompt_template=env_spec.prompt_template, action_extraction_fn=env_spec.action_extraction_fn)) # only one actor, rest fixed
                else:       agent_specs.append(AgentSpec(pid=pid, kind="openrouter", lora_path=None, openrouter_name=env_spec.fixed_opponent)) # sample opponent and add
            game_spec = GameSpec(game_idx=self._game_idx, env_id=env_spec.env_id, seed=self._game_idx, agent_specs=agent_specs, eval_model_pid=pids[0], eval_opponent_name=env_spec.fixed_opponent) # populate GameSpec
            return game_spec
        except Exception as exc:
            self.logger.info(f"Exception in 'next_eval_job': {exc}")
            import time 
            time.sleep(500)

    def update(self, game_info: GameInformation):
        self.logger.info(
            f"[GameScheduler] Game {game_info.game_idx} summary:\n"
            f"- Rewards: {json.dumps(game_info.final_rewards, indent=2)}\n"
            f"- Turns: {game_info.num_turns}\n"
            f"- Player Info: {json.dumps(game_info.game_info, indent=2)}"
        )

        job_info = self._running_jobs.pop(game_info.game_idx, None)
        if job_info is None:
            self.logger.warning(f"[GameScheduler] job_info missing for game_idx={game_info.game_idx}")
            return

        # 🔍 Log job_info["models"] for clarity
        self.logger.info(
            f"[GameScheduler] Game {game_info.game_idx} job_info:\n"
            f"{json.dumps(job_info['models'], indent=2)}"
        )

        # 🔍 Compute rewards
        actor_rs = [
            game_info.final_rewards[m["pid"]]
            for m in job_info["models"]
            if m["type"] == "model" and m["pid"] in game_info.final_rewards
        ]
        opp_rs = [
            game_info.final_rewards[m["pid"]]
            for m in job_info["models"]
            if m["type"] == "opponent" and m["pid"] in game_info.final_rewards
        ]

        # 🔍 Log reward breakdown by type
        self.logger.info(
            f"[GameScheduler] Game {game_info.game_idx} reward breakdown:\n"
            f"- actor_rs (model-controlled): {actor_rs}\n"
            f"- opp_rs (opponent-controlled): {opp_rs}"
        )

        # 🧠 Update samplers
        self.env_sampler.update(
            env_id=job_info["env_id"],
            avg_actor_reward=(sum(actor_rs) / len(actor_rs) if actor_rs else None),
            avg_opponent_reward=(sum(opp_rs) / len(opp_rs) if opp_rs else None),
        )
        self.model_sampler.update(game_info=game_info, job_info=job_info)
