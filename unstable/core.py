from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Dict
import trueskill, math, ray, random, time

# local imports
# from unstable.tracker import WandBTracker

@dataclass
class Trajectory:
    pid: List[int] = field(default_factory=list)
    obs: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    extracted_actions: List[str] = field(default_factory=list)
    infos: List[Dict] = field(default_factory=list)
    final_rewards: Dict[int, float] = field(default_factory=dict)
    num_turns: int = field(default_factory=int)
    format_feedbacks: List[Dict] = field(default_factory=list)
    board_states: List[str] = field(default_factory=list)


@dataclass
class Step: # TODO change to only contain necessary + info-dict for flexibility
    pid: int
    obs: str
    act: str
    reward: float
    env_id: str
    raw_reward: float
    transformed_end_of_game_reward: float
    step_reward: float

@dataclass
class Opponent:
    uid: str # “ckpt-1234” or “gemini-flash”
    kind: str # {"checkpoint","fixed"}
    path_or_name: str # LoRA dir or OpenRouter model id
    rating: trueskill.Rating # trueskill.Rating(mu, sigma)
    active: bool = True




@ray.remote
class ModelPool:
    def __init__(self, sample_mode, max_active_lora, tracker=None, lag_range=(1,7), beta=4.0):
        self.TS = trueskill.TrueSkill(beta=beta)
        self._models   = {}         # uid -> Opponent dataclass
        self._ckpt_log = []         # ordered list of checkpoint uids
        self.lag_lo, self.lag_hi = lag_range
        self.sample_mode = sample_mode #"self-play", "lagged", "fixed", "adaptive-trueskill"
        self.max_active_lora = max_active_lora
        self._latest_uid = None
        self._last_ckpt = None

        # for tracking
        self._match_counts = defaultdict(int) #  (uid_a, uid_b) -> games played
        self._rating_log = [] #  [(time, uid, μ, σ), …]
        self._step_counter = 0 #  learner step snapshot id
        self._tracker = tracker

    def current_uid(self):
        return self._latest_uid

    def add_checkpoint(self, path: str, iteration: int):
        uid = f"ckpt-{iteration}"

        # inherit μ/σ if a previous checkpoint exists
        if self._latest_uid and self._latest_uid in self._models:
            prev_rating = self._models[self._latest_uid].rating
            init_rating = self.TS.Rating(mu=prev_rating.mu, sigma=prev_rating.sigma * 2) #self.TS.Rating(mu=prev_rating.mu, prev_rating.sigma*2)
        else:
            init_rating = self.TS.create_rating()   # default prior

        self._models[uid] = Opponent(uid, "checkpoint", path, rating=init_rating)
        self._ckpt_log.append(uid)
        self._latest_uid = uid                     # promote to “current”
        self._maintain_active_pool() # update current ckpt pool

    def add_fixed(self, name, prior_mu=25.0):
        uid = f"fixed-{name}"
        if uid not in self._models:
            self._models[uid] = Opponent(uid, "fixed", name, rating=self.TS.create_rating(prior_mu))

    def latest_ckpt(self):
        return self._ckpt_log[-1] if self._ckpt_log else None

    def ckpt_path(self, uid):
        if uid is None: return None
        return self._models[uid].path_or_name

    def sample(self, uid_me):
        if self.sample_mode == "fixed": # randomly sample one of the fixed opponents provided
            return self._sample_fixed_opponent()

        elif self.sample_mode == "mirror": # literally play against yourself
            return self.latest_ckpt()

        elif self.sample_mode == "lagged": # sample an opponent randomly from the available checkpoints (will be lagged by default)
            return self._sample_lagged_opponent()
        
        elif self.sample_mode == "random": # randomly select an opponent from prev checkpoints and fixed opponents
            return self._sample_random_opponent()
        
        elif self.sample_mode == "match-quality": # sample an opponent (fixed or prev) based on the TrueSkill match quality
            return self._sample_match_quality_opponent(uid_me=uid_me)
        
        elif self.sample_mode == "ts-dist": # sample an opponent (fixed or prev) based on the absolute difference in TrueSkill scores
            return self._sample_ts_dist_opponent(uid_me=uid_me)
        
        elif self.sample_mode == "ts-dist-biased": # sample an opponent (fixed or prev) based on the exp(mu_i-mu_b) distance (i.e. biased towards stronger opponents)
            return self._sample_ts_dist_biased_opponent(uid_me=uid_me)        
        
        elif self.sample_mode == "exploration": # sample an opponent based on the expected number of unique board states when playing against that opponent
            return self._sample_exploration_opponent(uid_me=uid_me)
        
        
        else:
            raise ValueError(self.sample_mode)


    def _sample_fixed_opponent(self):
        fixed = [u for u,m in self._models.items() if m.kind=="fixed"]
        return random.choice(fixed)

    def _sample_lagged_opponent(self):
        available = [u for u,m in self._models.items() if (m.kind=="checkpoint" and m.active==True)]
        return random.choice(available)

    def _sample_random_opponent(self):
        fixed = [u for u,m in self._models.items() if m.kind=="fixed"]
        checkpoints = [u for u,m in self._models.items() if (m.kind=="checkpoint" and m.active==True)]
        return random.choice(fixed+checkpoints)

    def _sample_match_quality_opponent(self, uid_me: str) -> str:
        """ Pick an opponent with probability ∝ TrueSkill match-quality """
        cand, weights = [], []
        for uid, opp in self._models.items():
            if uid == uid_me or not opp.active:
                continue
            q = self.TS.quality([self._models[uid_me].rating, opp.rating]) # ∈ (0,1]
            cand.append(uid)
            weights.append(q) # already scaled

        # softmax the weights
        weights_sum = sum(weights)
        for i in range(len(weights)): 
            weights[i] = weights[i] / weights_sum
        # # fallback – nothing active except myself
        if not cand:
            return None
        return random.choices(cand, weights=weights, k=1)[0]

    def _sample_ts_dist_opponent(self, uid_me: str) -> str:
        """Sample by *absolute* TrueSkill μ-distance (closer ⇒ higher prob)."""
        cand, weights = [], []
        for uid, opp in self._models.items():
            if uid == uid_me or not opp.active:
                continue
            d = abs(self._models[uid_me].rating.mu - opp.rating.mu)
            cand.append(uid)
            weights.append(d)

        weights_sum = sum(weights)
        for i in range(len(weights)):
            weights[i] = 1 - (weights[i] / weights_sum) # smaller dist greater match prob
        if not cand:
            return None
        return random.choices(cand, weights=weights, k=1)[0]

    def _sample_ts_dist_biased_opponent(self, uid_me: str):
        cand, weights = [], []
        for uid, opp in self._models.items():
            if uid == uid_me or not opp.active:
                continue
            d = math.exp((opp.rating.mu-self._models[uid_me].rating.mu)/25)
            cand.append(uid)
            weights.append(d)

        weights_sum = sum(weights)
        for i in range(len(weights)):
            weights[i] = weights[i]/weights_sum
        if not cand:
            return None
        return random.choices(cand, weights=weights, k=1)[0]

    def _sample_exploration_opponent(self, uid_me: str):
        # TODO 
        return 



    def update_ratings(self, uid_me, uid_opp, final_reward):
        if uid_me not in self._models or uid_opp not in self._models:
            return  # skip if either side is unknown

        a = self._models[uid_me].rating
        b = self._models[uid_opp].rating

        if final_reward == 1: # uid_me wins → order is (a, b)
            new_a, new_b = self.TS.rate_1vs1(a, b)
        elif final_reward == -1: # uid_opp wins → order is (b, a)
            new_b, new_a = self.TS.rate_1vs1(b, a)
        elif final_reward == 0: # draw
            new_a, new_b = self.TS.rate_1vs1(a, b, drawn=True)
        else: # unexpected reward value
            return  

        self._models[uid_me].rating = new_a
        self._models[uid_opp].rating = new_b
        print(f"\nUPDATED TRUESKILL:\n\t{uid_me} -> {self._models[uid_me].rating}\n\t{uid_opp} -> {self._models[uid_opp].rating}")

        # register the game for tracking
        self._register_game(uid_me=uid_me, uid_opp=uid_opp)

    def _exp_win(self, A, B): return self.TS.cdf((A.mu - B.mu) / ((2*self.TS.beta**2 + A.sigma**2 + B.sigma**2) ** 0.5))
    def _activate(self, uid): self._models[uid].active = True
    def _retire(self, uid): self._models[uid].active = False

    def _maintain_active_pool(self):
        current = self.latest_ckpt()
        if current is None: return # nothing to do yet

        # collect candidate ckpts (exclude current, fixed models)
        cands = [uid for uid, m in self._models.items() if m.kind == "checkpoint" and uid != current]
        if not cands: return # only the current ckpt exists

        cur_rating = self._models[current].rating
        β = self.TS.beta

        # score candidates according to sampling mode
        scores = {}
        if self.sample_mode in {"random", "lagged"}: # more recent → smaller index in ckpt log
            # scores = {uid: -self._ckpt_log.index(uid) for uid in cands}
            scores = {uid: self._ckpt_log.index(uid) for uid in cands}

        elif self.sample_mode == "match-quality":
            for uid in cands:
                scores[uid] = self.TS.quality([cur_rating, self._models[uid].rating]) # higher is better

        elif self.sample_mode == "ts-dist":
            for uid in cands:
                scores[uid] = -abs(cur_rating.mu - self._models[uid].rating.mu) # larger (i.e. −distance) is better

        elif self.sample_mode == "ts-dist-biased":
            pos, neg = {}, {}
            for uid in cands:
                delta = self._models[uid].rating.mu - cur_rating.mu
                (pos if delta >= 0 else neg)[uid] = delta
            # keep strongest positives first; if not enough, use closest negatives
            ordered = (sorted(pos, key=pos.__getitem__, reverse=True) + sorted(neg, key=lambda u: abs(neg[u])))
            scores = {uid: len(ordered) - i for i, uid in enumerate(ordered)}# bigger rank ⇒ bigger score

        else:
            # default fallback: recent
            scores = {uid: -self._ckpt_log.index(uid) for uid in cands}

        # pick the N best, plus the current ckpt
        N = self.max_active_lora - 1 # slot for current already reserved
        keep = {current} | set(sorted(scores, key=scores.__getitem__, reverse=True)[:max(0, N)])

        # flip active flags
        for uid, opp in self._models.items():
            if opp.kind != "checkpoint":
                continue
            opp.active = (uid in keep)



    def _register_game(self, uid_me, uid_opp):
        if uid_opp is None: # self-play
            uid_opp = uid_me
        pair = tuple(sorted((uid_me, uid_opp)))
        self._match_counts[pair] += 1


    def snapshot(self, iteration: int):
        self._step_counter = iteration
        for uid, opp in self._models.items():
            self._rating_log.append((time.time(), uid, opp.rating.mu, opp.rating.sigma))
            if self._tracker: self._tracker.log_trueskill.remote(step=iteration, uid=uid, mu=opp.rating.mu, sigma=opp.rating.sigma)
        # push matchup matrix (sparse dict is fine)
        if self._tracker: self._tracker.log_matchup_counts.remote(step=iteration, counts=dict(self._match_counts))









# To implement
# -2) "fixed" - randomly sample from the fixed opponents provided
# -1) "lagged" - sample from the past n checkpoints randomly
# 0) "mirror" - mirrored self-play
# 1) "random", always keep the n latest checkpoins and sample randomly from those and the fixed opponents
# 2) "match-quality", keep the n checkpoints with the highest match quality for the current checkpoint and sample based on the match quality
# 3) "trueskill-dist", sample based on the trueskill dist (and keep those with the lowest TS distance)
# 4) "trueskill-dist-biased", as the name suggests
# 5) "exploration" somehow keep track of the action or board-state exploration / diversity during matchups and sample based on that  



