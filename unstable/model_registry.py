import ray, copy, trueskill
from dataclasses import dataclass, asdict
from collections import defaultdict
from typing import Dict, List

from unstable._types import ModelMeta
from unstable.utils import setup_logger


@ray.remote
class ModelRegistry:
    def __init__(self, tracker, beta: float = 4.0):
        self.TS = trueskill.TrueSkill(beta=beta)
        self._db: dict[str, ModelMeta] = {}
        self._match_counts = defaultdict(int) # (uid_a, uid_b) -> n
        self._exploration = defaultdict(lambda: defaultdict(dict))
        self._current_ckpt_uid : str | None = None 
        self._tracker = tracker; self._update_step: int = 1
        self.logger = setup_logger("model_registry", ray.get(self._tracker.get_log_dir.remote()))

    @staticmethod
    def _scores_to_ranks(scores: List[float]) -> List[int]:
        order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        ranks = [0]*len(scores); rank = 0
        for i, idx in enumerate(order):
            if i and scores[idx] != scores[order[i-1]]: rank = i  # next rank starts here
            ranks[idx] = rank
        return ranks

    def add_checkpoint(self, uid: str, path: str, iteration: int, inherit: bool=True):
        self.logger.info(f"tryin to add ckpt: {uid}, path {path}, iteration {iteration}, inherit: {inherit}")
        if uid in self._db: return
        rating = self.TS.Rating(mu=self._db[self._current_ckpt_uid].rating.mu, sigma=self._db[self._current_ckpt_uid].rating.sigma*2) if (inherit and self._current_ckpt_uid in self._db) else self.TS.create_rating()
        self._db[uid] = ModelMeta(uid=uid, kind="checkpoint", path_or_name=path, rating=rating, iteration=iteration)
        self._current_ckpt_uid = uid # make it current
        self.logger.info(f"added ckpt: {uid}, path {path}, iteration {iteration}, inherit: {inherit}")

    def get_all_models(self): return copy.deepcopy(self._db)
    def get_current_ckpt(self) -> str|None: return self._current_ckpt_uid
    def get_name_or_lora_path(self, uid: str) -> str: return self._db[uid].path_or_name
    def add_fixed(self, name: str, prior_mu: float = 25.): 
        if f"fixed-{name}" not in self._db: self._db[f"fixed-{name}"] = ModelMeta(f"fixed-{name}", "fixed", name, self.TS.create_rating(mu=prior_mu))

    def update_ratings(self, uids: List[str], scores: List[float], env_id: str, dummy_uid: str="fixed-env") -> None:
        if len(uids) == 1:
            if dummy_uid not in self._db: self.add_fixed(name=dummy_uid.replace("fixed-", ""), prior_mu=25.0)
            uids = [uids[0], dummy_uid]
            scores = [scores[0], 0.0] # any baseline score works
        rating_groups = [[self._db[uid].rating] for uid in uids]
        ranks = self._scores_to_ranks(scores)
        new_groups = self.TS.rate(rating_groups, ranks=ranks)

        # flatten, then write back
        for uid, (new_rating,) in zip(uids, new_groups):
            self._db[uid].rating = new_rating
            self._db[uid].games += 1
            if ranks[uids.index(uid)] == 0:               self._db[uid].wins  += 1
            elif ranks.count(ranks[uids.index(uid)]) > 1: self._db[uid].draws += 1

        # update pair-wise match matrix for analysis/debugging
        for i, uid_i in enumerate(uids):
            for uid_j in uids[i+1:]:
                self._match_counts[tuple(sorted((uid_i, uid_j)))] += 1
        self._update_step += 1

        # push to tracker every n update steps
        if not self._update_step%10: self._tracker.log_model_registry.remote(ts_dict={uid: asdict(meta) for uid, meta in self._db.items()}, match_counts=copy.deepcopy(self._match_counts))

