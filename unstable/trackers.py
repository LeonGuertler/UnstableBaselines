import os, re, ray, time, wandb, collections, datetime, logging, numpy as np
from typing import Optional, Union, Dict
from unstable.utils import setup_logger

from unstable._types import PlayerTrajectory, GameInformation
from unstable.utils import write_game_information_to_file
Scalar = Union[int, float, bool]

class BaseTracker:
    def __init__(self, run_name: str):
        self.run_name = run_name 
        self._build_output_dir()

    def _build_output_dir(self):
        self.output_dir = os.path.join("outputs", str(datetime.datetime.now().strftime('%Y-%m-%d')), str(datetime.datetime.now().strftime('%H-%M-%S')), self.run_name)
        os.makedirs(self.output_dir)
        self.output_dirs = {}
        for folder_name in ["training_data", "eval_data", "checkpoints", "logs", "collection"]: 
            self.output_dirs[folder_name] =  os.path.join(self.output_dir, folder_name); os.makedirs(self.output_dirs[folder_name], exist_ok=True)

    def get_checkpoints_dir(self):  return self.output_dirs["checkpoints"]
    def get_train_dir(self):        return self.output_dirs["training_data"]
    def get_eval_dir(self):         return self.output_dirs["eval_data"]
    def get_collection_dir(self):   return self.output_dirs["collection"]
    def get_log_dir(self):          return self.output_dirs["logs"]
    def add_trajectory(self, trajectory: PlayerTrajectory, env_id: str): raise NotImplementedError
    def add_eval_episode(self, episode_info: Dict, final_reward: int, player_id: int, env_id: str, iteration: int): raise NotImplementedError
    def log_lerner(self, info_dict: Dict): raise NotImplementedError

    
@ray.remote
class Tracker(BaseTracker): 
    FLUSH_EVERY = 64
    def __init__(self, run_name: str, wandb_project: Optional[str]=None):
        super().__init__(run_name=run_name)
        self.logger = setup_logger("tracker", self.get_log_dir())
        self.use_wandb = False
        if wandb_project: wandb.init(project=wandb_project, name=run_name); self.use_wandb = True; wandb.define_metric("*", step_metric="learner/step")
        self._m: Dict[str, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=512))
        self._buffer: Dict[str, Scalar] = {}
        self._n = {}
        self._last_flush = time.monotonic()
        self._interface_stats = {"gpu_tok_s": {}, "TS": {}, "exploration": {}, "match_counts": {}, "format_success": None, "inv_move_rate": None, "game_len": None}

    def _put(self, k: str, v: Scalar): self._m[k].append(v)
    def _agg(self, p: str) -> dict[str, Scalar]: return {k: float(np.mean(dq)) for k, dq in self._m.items() if k.startswith(p)}
    def _flush_if_due(self):
        if time.monotonic()-self._last_flush >= self.FLUSH_EVERY:
            if self._buffer and self.use_wandb:
                try: wandb.log(self._buffer)
                except Exception as e: self.logger.warning(f"wandb.log failed: {e}")
            self._buffer.clear(); self._last_flush=time.monotonic()

    def add_player_trajectory(self, traj: PlayerTrajectory, env_id: str):
        try:
            reward = traj.final_reward; player_id = traj.pid
            self._put(f"collection-{env_id}/reward", reward)
            self._put(f"collection-{env_id}/Win Rate", int(reward>0))
            self._put(f"collection-{env_id}/Loss Rate", int(reward<0))
            self._put(f"collection-{env_id}/Draw", int(reward==0))
            self._put(f"collection-{env_id}/Reward (pid={traj.pid})", reward)
            self._put(f"collection-{env_id}/Game Length", traj.num_turns)
            for idx in range(len(traj.obs)):
                self._put(f"collection-{env_id}/Respone Length (char)", len(traj.actions[idx]))
                self._put(f"collection-{env_id}/Observation Length (char)", len(traj.obs[idx]))
                for k, v in traj.format_feedbacks[idx].items(): self._put(f"collection-{env_id}/Format Success Rate - {k}", v)
            self._n[f"collection-{env_id}"] = self._n.get(f"collection-{env_id}", 0) + 1
            self._put(f"collection-{env_id}/step", self._n[f"collection-{env_id}"])
            self._buffer.update(self._agg('collection-')); self._flush_if_due()
        except Exception as exc:
            self.logger.info(f"Exception when adding trajectory to tracker: {exc}")

    def add_eval_game_information(self, game_information: GameInformation, env_id: str):
        try:
            eval_reward = game_information.final_rewards.get(game_information.eval_model_pid, 0.0)
            _prefix = f"evaluation-{env_id}" if not game_information.eval_opponent_name else f"evaluation-{env_id} ({game_information.eval_opponent_name})"
            self._put(f"{_prefix}/Reward", eval_reward)
            self._put(f"{_prefix}/Reward (pid={game_information.eval_model_pid})", eval_reward)
            self._put(f"{_prefix}/Win Rate",  int(eval_reward>0))
            self._put(f"{_prefix}/Loss Rate", int(eval_reward<0))
            self._put(f"{_prefix}/Draw Rate", int(eval_reward==0))
            self._n[_prefix] = self._n.get(_prefix, 0) + 1
            self._put(f"{_prefix}/step", self._n[_prefix])
            self._buffer.update(self._agg('evaluation-')); self._flush_if_due()

            # try storing the eval info to file
            write_game_information_to_file(game_info=game_information, filename=os.path.join(self.get_eval_dir(), f"{env_id}-{game_information.game_idx}.csv"))

        except Exception as exc:
            self.logger.info(f"Exception when adding game_info to tracker: {exc}")

    def log_model_registry(self, ts_dict: dict[str, dict[str, float]], match_counts: dict[tuple[str, str], int]):
        self._interface_stats.update({"TS": ts_dict, "exploration": None, "match_counts": match_counts})

    def log_inference(self, actor: str, gpu_ids: list[int], stats: dict[str, float]):
        for key in stats: self._put(f"inference/{actor}/{key}", stats[key])
        for gpu_id in gpu_ids: self._interface_stats["gpu_tok_s"][gpu_id] = stats["tok_s"]
        self._buffer.update(self._agg('inference'))
    
    def log_learner(self, info: dict):
        try:
            self._m.update({f"learner/{k}": v for k, v in info.items()})
            self._buffer.update(self._agg("learner")); self._flush_if_due()
        except Exception as exc:
            self.logger.info(f"Exception in log_learner: {exc}")

    def get_interface_info(self): 
        for inf_key in ["Game Length", "Format Success Rate - correct_answer_format", "Format Success Rate - invalid_move"]: 
            self._interface_stats[inf_key] = np.mean([float(np.mean(dq)) for k,dq in self._m.items() if inf_key in k])
        return self._interface_stats
