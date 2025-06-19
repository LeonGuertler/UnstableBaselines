import os, re, ray, time, wandb, collections
from typing import Optional, Union
import numpy as np
from unstable.core import BaseTracker, Trajectory


@ray.remote
class Tracker(BaseTracker):
    COLLECT_FLUSH_EVERY: int = 64 # How often to push collection stats (in *trajectories*)
    TOPK: int = 25 # Only keep the K most‑played match‑ups in the dashboard table

    def __init__(self, run_name: str, accum_strategy: str="ma", output_dir: Optional[str]=None, wandb_project: Optional[str]=None) -> None:
        super().__init__(run_name=run_name, output_dir=output_dir)
        self.accum_strategy = accum_strategy
        self.wandb_project = wandb_project
        self._ma_range = 512; self._tau = 0.001

        # metric storage
        self._metrics: dict[str, Union[collections.deque, float]] = {}
        self._inference: dict[str, dict[str, float]] = {}

        self._buffer: dict[str, Union[int, float, bool, "wandb.Table"]] = {}
        self._collect_counter: int = 0  # trajectories since last flush

        self._collection_snapshot: dict[str, float] = {}
        self._eval_snapshot: dict[str, float] = {}
        self._ts_snapshot: dict[str, dict]  = {}
        self._match_counts: dict[tuple[str, str], int] = {}

        self._gpu_tok_rate: dict[int, float] = collections.defaultdict(float)
        self._gpu_last_seen: dict[int, float] = collections.defaultdict(lambda: 0.0)
        self._gpu_timeout = 10.0   # seconds without update ⇒ show 0 toks/s

        if self.wandb_project is not None:
            wandb.init(project=self.wandb_project, name=run_name)

    def _aggregate(self, prefix: str) -> dict[str, Union[int, float]]:
        """Collapse internal metric deques/emas into scalars for prefix """
        out: dict[str, Union[int, float]] = {}
        for name, val in self._metrics.items():
            if not name.startswith(prefix):  continue
            out[name] = (float(np.mean(val)) if val else 0.0) if isinstance(val, collections.deque) else val
        return out

    def _flush(self, force: bool = False):
        wandb.log(self._buffer)
        self._buffer.clear()

    # metric bookkeeping
    def _update_metric(self, name: str, value: float | int | bool, prefix: str, env_id: str):
        for key in (f"{prefix}-{env_id}/{name}", f"{prefix}-all/{name}"):
            match self.accum_strategy:
                case "ma":  self._metrics.setdefault(key, collections.deque(maxlen=self._ma_range)).append(value)
                case "ema": self._metrics[key] = self._metrics.get(key, 0.0) * (1 - self._tau) + self._tau * value
                case _:     raise NotImplementedError

    def _batch_update_metrics(self, pairs: list[tuple[str, float|int|bool]], prefix: str, env_id: str) -> None:
        for n, v in pairs:  
            self._update_metric(n, v, prefix, env_id)

    # trajectory collection
    def _add_trajectory_to_metrics(self, trajectory: Trajectory, player_id: int, env_id: str, prefix: str="collection") -> None:
        m_reward, o_reward = trajectory.final_rewards[player_id], trajectory.final_rewards[1 - player_id]
        pairs = [("Game Length", trajectory.num_turns)]
        match len(trajectory.final_rewards):
            case 1: pairs.append(("Reward", m_reward))
            case 2: pairs.extend([("Win Rate", int(m_reward > o_reward)), ("Loss Rate", int(m_reward < o_reward)), ("Draw Rate", int(m_reward == o_reward)), ("Reward", m_reward), (f"Reward (pid={player_id})", m_reward)])
            case _: raise NotImplementedError

        for i, pid in enumerate(trajectory.pid):
            if pid != player_id: continue
            for k, v in trajectory.format_feedbacks[i].items():
                pairs.append((f"Format Success Rate - {k}", v))
            pairs.extend([("Response Length (avg. char)", len(trajectory.actions[i])), ("Observation Length (avg. char)", len(trajectory.obs[i]))])
        self._batch_update_metrics(pairs, prefix, env_id)

    def add_trajectory(self, trajectory: Trajectory, player_id: int, env_id: str, prefix: str = "collection"):
        self._add_trajectory_to_metrics(trajectory, player_id, env_id, prefix)
        self._buffer.update(self._aggregate(prefix))
        self._collect_counter += 1
        if self._collect_counter % self.COLLECT_FLUSH_EVERY == 0: self._flush()
        self._collection_snapshot = self._aggregate("collection")

    def log_learner(self, info: dict):
        self._metrics.update({f"learner/{k}": v for k, v in info.items()})
        self._buffer.update(self._aggregate("learner"))

    def _add_eval_episode_to_metrics(self, episode_info: dict, final_rewards: dict, player_id: int, env_id: str, prefix: str="evaluation") -> None:
        pairs = [("Game Length", len(episode_info)), ("Reward", final_rewards[player_id])]
        if len(final_rewards) == 2:
            m_reward, o_reward = final_rewards[player_id], final_rewards[1 - player_id]
            pairs.extend([("Win Rate", int(m_reward > o_reward)), ("Loss Rate", int(m_reward < o_reward)), ("Draw Rate", int(m_reward == o_reward)), ("Reward", m_reward), (f"Reward (pid={player_id})", m_reward)])
        elif len(final_rewards) > 2: raise NotImplementedError
        self._batch_update_metrics(pairs, prefix, env_id)

    def add_eval_episode(self, episode_info: dict, final_rewards: dict, player_id: int, env_id: str, iteration: int, prefix: str="evaluation") -> None:
        self._add_eval_episode_to_metrics(episode_info, final_rewards, player_id, env_id, prefix)
        self._buffer.update(self._aggregate(prefix))
        self._flush()
        self._eval_snapshot = self._aggregate("evaluation")

    def log_model_pool(self, step: int, match_counts: dict, ts_dict: dict, action_distribution: dict) -> None:
        ckpt_ids = [u for u in ts_dict if re.fullmatch(r"ckpt-\d+", u)]
        cur = max(ckpt_ids, key=lambda u: int(u.split("-")[1]), default=None)
        if cur is not None:
            self._buffer["trueskill/mu"] = ts_dict[cur]["mu"]
            self._buffer["trueskill/sigma"] = ts_dict[cur]["sigma"]

        # top‑K match‑ups table
        top = sorted(match_counts.items(), key=lambda kv: kv[1], reverse=True)[: self.TOPK]
        table = wandb.Table(columns=["uid_a", "uid_b", "games"], data=[[*pair, cnt] for pair, cnt in top])
        self._buffer["pool/top_matchups"] = table

        # action distribution
        for n, c in action_distribution.items():
            if '/' in n: self._buffer[f"actions-{n.split('/')[0]}/{n.split('/')[1]}"] = c
            else: self._buffer[f"actions/{n}"] = c
        self._flush(force=True)
        self._ts_snapshot = ts_dict
        self._match_counts = match_counts
        
    def log_inference(self, actor: str, gpu_ids: list[int], stats: dict[str, dict[str, float]]):
        self._inference[actor] = stats
        share = sum(meta.get("tok_s", 0.0) for meta in stats.values()) / max(1, len(gpu_ids))
        now = time.monotonic()
        for gid in gpu_ids:
            self._gpu_tok_rate[gid]  = share
            self._gpu_last_seen[gid] = now

        if self.wandb_project:
            flat_stats = {f"inference/{actor}/queued": sum(s["queued"] for s in stats.values()), f"inference/{actor}/running": sum(s["running"] for s in stats.values()), f"inference/{actor}/tokens_sec": sum(s["tok_s"] for s in stats.values())}
            self._buffer.update(flat_stats)
            self._flush()

    def get_latest_inference_metrics(self) -> dict[str, dict[str, float]]: return self._inference
    def get_gpu_tok_rates(self) -> dict[int, float]: return {gid: (rate if time.monotonic() - self._gpu_last_seen[gid] <= self._gpu_timeout else 0.0) for gid, rate in self._gpu_tok_rate.items()}
    def get_latest_learner_metrics(self): return {k.split('/',1)[1]: v for k,v in self._buffer.items() if k.startswith('learner/')}
    
    def get_collection_metrics(self):   return self._collection_snapshot
    def get_eval_metrics(self):         return self._eval_snapshot
    def get_ts_snapshot(self):          return self._ts_snapshot
    def get_match_counts(self):         return self._match_counts
    def get_wandb_project(self):        return self.wandb_project
    def get_run_name(self):             return self.run_name