import os, re, ray, time, wandb, collections, datetime, numpy as np
from typing import Optional, Union, Dict
from unstable.utils.logger import setup_logger

from unstable.utils._types import PlayerTrajectory, GameInformation
from unstable.utils.misc import write_game_information_to_file
Scalar = Union[int, float, bool]

class BaseTracker:
    def __init__(self, run_name: str):
        self.run_name = run_name 
        self._build_output_dir()

    def _build_output_dir(self):
        self.output_dir = os.path.join("outputs", str(datetime.datetime.now().strftime('%Y-%m-%d')), str(datetime.datetime.now().strftime('%H-%M-%S')), self.run_name)
        os.makedirs(self.output_dir)
        self.output_dirs = {}
        for folder_name in ["training_data", "checkpoints", "logs", "collection", "eval"]: 
            self.output_dirs[folder_name] =  os.path.join(self.output_dir, folder_name); os.makedirs(self.output_dirs[folder_name], exist_ok=True)

    def get_checkpoints_dir(self):  return self.output_dirs["checkpoints"]
    def get_train_dir(self):        return self.output_dirs["training_data"]
    def get_collection_dir(self):   return self.output_dirs["collection"]
    def get_eval_dir(self):         return self.output_dirs['eval']
    def get_log_dir(self):          return self.output_dirs["logs"]
    def add_trajectory(self, trajectory: PlayerTrajectory, env_id: str): raise NotImplementedError
    def add_eval_episode(self, episode_info: Dict, final_reward: int, player_id: int, env_id: str, iteration: int): raise NotImplementedError
    def log_lerner(self, info_dict: Dict): raise NotImplementedError

    
@ray.remote
class Tracker(BaseTracker): 
    FLUSH_EVERY = 64
    def __init__(self, run_name: str, wandb_project: Optional[str]=None, wandb_id: Optional[str]=None, wandb_config: Optional[Dict]=None):
        super().__init__(run_name=run_name)
        self.logger = setup_logger("tracker", self.get_log_dir())
        self.use_wandb = False
        self.learner_step = 0
        if wandb_project: wandb.init(project=wandb_project, name=run_name, config=wandb_config, id=wandb_id, resume="must" if wandb_id else None); self.use_wandb = True; wandb.define_metric("*", step_metric="learner/step")
        self._m: Dict[str, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=512))
        self._n = {}
        self._buffer: Dict[str, Scalar] = {}
        self._eval_pending: Dict[int, Dict[str, list]] = {}
        self._last_flush = time.monotonic()
        self._interface_stats = {"gpu_tok_s": {}, "TS": {}, "exploration": {}, "match_counts": {}, "format_success": None, "inv_move_rate": None, "game_len": None}

    def _put(self, k: str, v: Scalar): self._m[k].append(v)
    def _agg(self, p: str) -> dict[str, Scalar]: return {k: float(np.mean(dq)) for k, dq in self._m.items() if k.startswith(p)}
    def _num(self, k: str) -> int: return len(self._m[k]) if k in self._m else 0
    def _clear(self, p: str) -> None:
        for k in list(self._m.keys()):
            if k.startswith(p): del self._m[k]
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
                self._put(f"collection-{env_id}/Respone Length (char)", len(traj.completions[idx]))
                self._put(f"collection-{env_id}/Observation Length (char)", len(traj.obs[idx]))
                for k, v in traj.format_feedbacks[idx].items(): self._put(f"collection-{env_id}/Format Success Rate - {k}", v)
            self._n[f"collection-{env_id}"] = self._n.get(f"collection-{env_id}", 0) + 1
            self._put(f"collection-{env_id}/step", self._n[f"collection-{env_id}"])
            self._buffer.update(self._agg('collection-')); self._flush_if_due()
        except Exception as exc:
            self.logger.info(f"Exception when adding trajectory to tracker: {exc}")

    def add_eval_game_information(self, game_information: GameInformation, env_id: str, aggregate: int = None):
        try:
            eval_reward = game_information.final_rewards.get(game_information.eval_model_pid, 0.0)
            _prefix = f"evaluation-{env_id}"
            _lbl = f" ({game_information.eval_opponent_name})" if game_information.eval_opponent_name else ""
            it = game_information.eval_iteration if aggregate is not None else None
            if it not in self._eval_pending: self._eval_pending[it] = collections.defaultdict(list)
            buf = self._eval_pending[it]
            buf[f"{_prefix}/Reward{_lbl}"].append(eval_reward)
            buf[f"{_prefix}/Reward (pid={game_information.eval_model_pid}){_lbl}"].append(eval_reward)
            buf[f"{_prefix}/Win Rate{_lbl}"].append(int(eval_reward>0))
            buf[f"{_prefix}/Loss Rate{_lbl}"].append(int(eval_reward<0))
            buf[f"{_prefix}/Draw Rate{_lbl}"].append(int(eval_reward==0))
            for pid, info in game_information.game_info.items():
                if pid == game_information.eval_model_pid: buf[f"{_prefix}/Invalid Move Loss Rate{_lbl}"].append(int(info.get("invalid_move")))
                if info.get("invalid_move") and pid != game_information.eval_model_pid: buf[f"{_prefix}/Invalid Move Win Rate{_lbl}"].append(int(eval_reward>0))
            buf[f"{_prefix}/Game Length{_lbl}"].append(game_information.num_turns)
            if aggregate is not None:
                buf[f"{_prefix}/Iteration"].append(it)
                if len(buf[f"{_prefix}/Iteration"]) >= aggregate:
                    self._buffer.update({k: float(np.mean(v)) for k, v in buf.items()}); self._flush_if_due()
                    del self._eval_pending[it]
            else: self._buffer.update({k: float(np.mean(v)) for k, v in buf.items()}); self._flush_if_due()
            write_game_information_to_file(game_info=game_information, filename=os.path.join(self.get_eval_dir(), f"{env_id}-{game_information.game_idx}.csv"))
        except Exception as exc:
            self.logger.info(f"Exception when adding game_info to tracker: {exc}")

    def log_model_sampler(self, ts_dict: dict[str, dict[str, float]], match_counts: dict[tuple[str, str], int]):
        self._interface_stats.update({"TS": ts_dict, "exploration": None, "match_counts": match_counts})

    def log_inference(self, actor: str, gpu_ids: list[int], stats: dict[str, float]):
        for key in stats: self._put(f"inference/{actor}/{key}", stats[key])
        for gpu_id in gpu_ids: self._interface_stats["gpu_tok_s"][gpu_id] = stats["tok_s"]
        self._buffer.update(self._agg('inference'))
    
    def log_learner(self, info: dict):
        try:
            self.learner_step = info["step"]
            self._m.update({f"learner/{k}": v for k, v in info.items()})
            self._buffer.update(self._agg("learner")); self._flush_if_due()
        except Exception as exc:
            self.logger.info(f"Exception in log_learner: {exc}")

    def get_learner_step(self): return self.learner_step

    def get_interface_info(self):
        print("Computing interface stats...")
        for inf_key in ["Game Length", "Format Success Rate - correct_answer_format", "Format Success Rate - invalid_move"]: 
            self._interface_stats[inf_key] = np.mean([float(np.mean(dq)) for k,dq in self._m.items() if inf_key in k])
            print(self._interface_stats[inf_key])
        return self._interface_stats
