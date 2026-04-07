import ray, random, copy, trueskill
from dataclasses import asdict
from collections import defaultdict, deque
from typing import Dict, Any, List

from unstable.utils._types import GameInformation, ModelMeta
from unstable.utils.logger import setup_logger

class BaseModelSampler:
    def __init__(self, tracker, beta: float = 4.0,
                 opponent_temperature: float = None, opponent_top_p: float = None,
                 opponent_top_k: int = None, opponent_max_tokens: int = None):
        self.TS = trueskill.TrueSkill(beta=beta)
        self._db: dict[str, ModelMeta] = {}
        self._match_counts = defaultdict(int)
        self._exploration = defaultdict(lambda: defaultdict(dict))
        self._current_ckpt_uid : str | None = None; self.active_ckpt = deque()
        self._tracker = tracker; self._update_step: int = 1; 
        self._opp_temperature = opponent_temperature
        self._opp_top_p = opponent_top_p
        self._opp_top_k = opponent_top_k
        self._opp_max_tokens = opponent_max_tokens
        self.logger = setup_logger("model_sampler", ray.get(self._tracker.get_log_dir.remote()))

    @staticmethod
    def _scores_to_ranks(scores: List[float]) -> List[int]:
        order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        ranks = [0]*len(scores); rank = 0
        for i, idx in enumerate(order):
            if i and scores[idx] != scores[order[i-1]]: rank = i  # next rank starts here
            ranks[idx] = rank
        return ranks
    
    def add_checkpoint(self, uid: str, path: str, iteration: int, inherit: bool=True, eval: bool=False):
        self.logger.info(f"tryin to add ckpt: {uid}, path {path}, iteration {iteration}, inherit: {inherit}")
        if uid in self._db: return
        rating = self.TS.Rating(mu=self._db[self._current_ckpt_uid].rating.mu, sigma=self._db[self._current_ckpt_uid].rating.sigma*2) if (inherit and self._current_ckpt_uid in self._db) else self.TS.create_rating()
        self._db[uid] = ModelMeta(uid=uid, kind="checkpoint", path_or_name=path, rating=rating, iteration=iteration, eval=eval)
        self._current_ckpt_uid = uid # make it current
        self.logger.info(f"added ckpt: {uid}, path {path}, iteration {iteration}, inherit: {inherit}")
    
    def get_all_models(self): return copy.deepcopy(self._db)
    def get_current_ckpt(self) -> str|None: return self._current_ckpt_uid
    def get_name_or_lora_path(self, uid: str) -> str: return self._db[uid].path_or_name
    def add_eval_checkpoint(self, uid: str, path: str):
        if uid in self._db: return
        self._db[uid] = ModelMeta(uid=uid, kind="checkpoint", path_or_name=path, rating=self.TS.create_rating(), iteration=0, eval=True)
        self.logger.info(f"added eval-only checkpoint: {uid}, path {path}")

    def add_fixed_checkpoint(self, uid: str, path: str):
        if uid in self._db: return
        self._db[uid] = ModelMeta(uid=uid, kind="fixed_checkpoint", path_or_name=path, rating=self.TS.create_rating(), iteration=0)
        self.logger.info(f"added fixed checkpoint opponent: {uid}, path {path}")

    def add_fixed(self, name: str, prior_mu: float = 25.):
        if f"fixed-{name}" not in self._db: self._db[f"fixed-{name}"] = ModelMeta(f"fixed-{name}", "fixed", name, self.TS.create_rating(mu=prior_mu))

    def get_current_ckpt(self):         
        current_ckpt_lora_path = self.get_name_or_lora_path(uid=self._current_ckpt_uid)
        return self._current_ckpt_uid, current_ckpt_lora_path

    def get_eval_checkpoints(self):
        candidates = [m for m in self._db.values() if m.kind == "checkpoint" and m.eval]
        if not candidates: candidates = [self._db[self._current_ckpt_uid]]
        return [(m.uid, m.kind, None, m.path_or_name, self.get_opponent_sampling_params()) for m in candidates]
    
    def update(self, game_info: GameInformation, job_info: Dict[str, Any],  dummy_uid: str="fixed-env"):
        uids = [m["uid"] for m in job_info["models"] if m["pid"] in game_info.final_rewards]
        scores = [game_info.final_rewards[m["pid"]] for m in job_info["models"] if m["pid"] in game_info.final_rewards]
        if len(uids) == 1:
            if dummy_uid not in self._db: self.add_fixed(name=dummy_uid.replace("fixed-", ""), prior_mu=25.0)
            uids = [uids[0], dummy_uid]
            scores = [scores[0], 0.0]
        rating_groups = [[self._db[uid].rating] for uid in uids]
        ranks = self._scores_to_ranks(scores)
        new_groups = self.TS.rate(rating_groups, ranks=ranks)
        for uid, (new_rating,) in zip(uids, new_groups):
            self._db[uid].rating = new_rating
            self._db[uid].games += 1
            if ranks[uids.index(uid)] == 0:               self._db[uid].wins  += 1
            elif ranks.count(ranks[uids.index(uid)]) > 1: self._db[uid].draws += 1
        for i, uid_i in enumerate(uids):
            for uid_j in uids[i+1:]:
                self._match_counts[tuple(sorted((uid_i, uid_j)))] += 1
        self._update_step += 1
        if not self._update_step%10: self._tracker.log_model_sampler.remote(ts_dict={uid: asdict(meta) for uid, meta in self._db.items()}, match_counts=copy.deepcopy(self._match_counts))

    def get_opponent_sampling_params(self) -> dict:
        return {"temperature": self._opp_temperature, "top_p": self._opp_top_p,
                "top_k": self._opp_top_k, "max_tokens": self._opp_max_tokens}

    def sample_eval_checkpoint(self):
        eval_ckpt = random.choice(self.get_eval_checkpoints())
        self.logger.info(f"sampling eval checkpoint opponent: {eval_ckpt[0]}")
        return eval_ckpt

    def sample_opponent(self): raise NotImplementedError


@ray.remote
class MirrorModelSampler(BaseModelSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def sample_opponent(self): 
        current_uid = self.get_current_ckpt()[0]
        opponent_meta = self._db[current_uid]
        self.logger.info(f"sampling mirror opponent: {opponent_meta.uid}")
        return opponent_meta.uid, opponent_meta.kind, None, opponent_meta.path_or_name, self.get_opponent_sampling_params()
    
@ray.remote
class FixedOpponentModelSampler(BaseModelSampler):
    def __init__(self, include_current_ckpt: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.include_current_ckpt = include_current_ckpt

    def sample_opponent(self):
        available_models = [model_meta for uid, model_meta in self.get_all_models().items() if (model_meta.active and model_meta.kind in ("fixed", "fixed_checkpoint")) or (model_meta.uid==self.get_current_ckpt()[0] and self.include_current_ckpt)]
        opponent_meta = random.choice(available_models)
        self.logger.info(f"sampling fixed opponent: {opponent_meta.uid}")
        return opponent_meta.uid, opponent_meta.kind, None, opponent_meta.path_or_name, self.get_opponent_sampling_params()


@ray.remote
class AsynchronousModelSampler(BaseModelSampler):
    def __init__(self, k: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.k = k
    
    def add_checkpoint(self, uid, path, iteration, inherit = True):
        super().add_checkpoint(uid, path, iteration, inherit)
        if self.k is not None: 
            self.active_ckpt.append(uid)
            if len(self.active_ckpt) > self.k: self._db[self.active_ckpt.popleft()].active = False
    
    def sample_opponent(self): 
        opponent_meta = random.choice([model_meta for uid, model_meta in self.get_all_models().items() if (model_meta.active and model_meta.kind=="checkpoint")])
        self.logger.info(f"sampling opponent: {opponent_meta.uid}")
        return opponent_meta.uid, opponent_meta.kind, None, opponent_meta.path_or_name, self.get_opponent_sampling_params()
    

@ray.remote
class WinRateModelSampler(BaseModelSampler):
    def __init__(self, threshold: float = 0.7, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self._recent_outcomes = deque()
        self._current_opponent_meta = None

    def add_checkpoint(self, uid: str, path: str, iteration: int, inherit: bool = True):
        promote_opponent = self._should_rotate()
        super().add_checkpoint(uid, path, iteration, inherit)
        if self._current_opponent_meta is None or promote_opponent: self._current_opponent_meta = self._db[uid]
        self.logger.info(f"New opponent promoted! {self._current_opponent_meta.uid}. Win rate: {self._current_win_rate()}" if promote_opponent else f"No opponent change. Win rate: {self._current_win_rate()}")
        self._recent_outcomes.clear()

    def update(self, game_info: GameInformation, job_info: Dict[str, Any]):
        super().update(game_info, job_info)
        actor = next((m for m in job_info["models"] if m["type"] == "model"), None)
        opp   = next((m for m in job_info["models"] if m["type"] == "opponent"), None)
        if actor and opp and actor["pid"] in game_info.final_rewards and opp["pid"] in game_info.final_rewards:
            actor_r, opp_r = game_info.final_rewards[actor["pid"]], game_info.final_rewards[opp["pid"]]
            self._recent_outcomes.append(1 if actor_r > opp_r else 0)

    def sample_opponent(self):
        if self._current_opponent_meta is None: self._current_opponent_meta = self._db[self._current_ckpt_uid]
        self.logger.info(f"sampling opponent: {self._current_opponent_meta.uid}, current win rate: {sum(self._recent_outcomes)}/{len(self._recent_outcomes)}")
        return self._current_opponent_meta.uid, self._current_opponent_meta.kind, None, self._current_opponent_meta.path_or_name, self.get_opponent_sampling_params()

    def _current_win_rate(self) -> float:  return sum(self._recent_outcomes) / len(self._recent_outcomes) if self._recent_outcomes else 0.0
    def _should_rotate(self): return self._current_win_rate() > self.threshold
