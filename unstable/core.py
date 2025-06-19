import os, ray, torch, datetime, trueskill, math
from dataclasses import dataclass, field
from collections import Counter, deque
from typing import List, Dict, Optional


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


@dataclass
class Step:
    pid: int
    obs: str
    act: str
    reward: float
    env_id: str
    step_info: Dict


@dataclass
class Opponent:
    uid: str # “ckpt-1234” or “gemini-flash”
    kind: str # {"checkpoint","fixed"}
    path_or_name: str # LoRA dir or OpenRouter model id
    rating: trueskill.Rating # trueskill.Rating(mu, sigma)
    active: bool = True


class BaseAlgo:
    def initialize(self, model, tokenizer, device, max_train_len: Optional[int]= None, accelerator=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_train_len = max_train_len
        self.accel = accelerator

    def prepare_batch(self, steps):
        """ Turn a list[Step] into tensors on self.dev. Return whatever update() needs """
        raise NotImplementedError

    def update(self, batch):
        """ One gradient update on *this worker only*. Must call .backward() but NOT .step(). Return latest loss as float (for logging) """
        raise NotImplementedError


class BaseTracker:
    def __init__(self, run_name: str, output_dir: Optional[str] = None):
        self.run_name = run_name 
        self._build_output_dir(output_dir=output_dir)

    def _build_output_dir(self, output_dir: Optional[str]):
        self.output_dir = os.path.join("outputs", str(datetime.datetime.now().strftime('%Y-%m-%d')), str(datetime.datetime.now().strftime('%H-%M-%S')), self.run_name) if not output_dir else output_dir
        os.makedirs(self.output_dir)
        self.output_dirs = {}
        for folder_name in ["training_data", "eval_data", "checkpoints"]: 
            self.output_dirs[folder_name] =  os.path.join(self.output_dir, folder_name); os.makedirs(self.output_dirs[folder_name], exist_ok=True)

    def get_checkpoints_dir(self):  return self.output_dirs["checkpoints"]
    def get_train_dir(self):        return self.output_dirs["training_data"]
    def get_eval_dir(self):         return self.output_dirs["eval_data"]
    
    def add_trajectory(self, trajectory: Trajectory, player_id: int, env_id: str): raise NotImplementedError
    def add_eval_episode(self, episode_info: Dict, final_reward: int, player_id: int, env_id: str, iteration: int): raise NotImplementedError
    def log_lerner(self, info_dict: Dict): raise NotImplementedError


class ActionSeqTracker:
    def __init__(self, sliding_window_size: int = None):
        assert sliding_window_size is None or sliding_window_size > 0, "sliding_window_size must be None or > 0"

        self.sliding_window_size = sliding_window_size
        self.unigrams, self.bigrams, self.trigrams, self.fourgrams, self.fivegrams = {}, {}, {}, {}, {} # Env -> Opponent -> Counter of n-grams
        if self.sliding_window_size:
            self.unigrams_deque, self.bigrams_deque, self.trigrams_deque, self.fourgrams_deque, self.fivegrams_deque = {}, {}, {}, {}, {}

    def add(self, action_seq: List[str], env_id: str, opp_uid: str):
        for n, name in [(1,"unigrams"), (2, "bigrams"), (3, "trigrams"), (4, "fourgrams"), (5, "fivegrams")]:
            attr = getattr(self, name)
            if env_id not in attr: attr[env_id] = {}
            if opp_uid not in attr[env_id]: attr[env_id][opp_uid] = Counter()
            n_grams = [tuple(action_seq[i:i+n]) for i in range(len(action_seq) - n + 1)]
            attr[env_id][opp_uid].update(Counter(n_grams))

            if self.sliding_window_size:
                queues = getattr(self, f"{name}_deque")
                if env_id not in queues: queues[env_id] = deque()
                queue = queues[env_id]
                queue.append((opp_uid,n_grams))
                if len(queue) > self.sliding_window_size:
                    removed_step = queue.popleft()
                    attr[env_id][removed_step[0]].subtract(removed_step[1]); attr[env_id][removed_step[0]] += Counter()

    def unique(self, n: str, env_id: str):
        return set(sum(getattr(self, n)[env_id].values(), Counter()).keys())
    
    def count(self, n: str, env_id: str):
        return sum(sum(getattr(self, n)[env_id].values(), Counter()).values())
    
    def unique_count(self, n: str, env_id: str):
        return len(self.unique(n, env_id))
    
    def entropy(self, n: str, env_id: str):
        return -sum(p * math.log2(p) for p in self._normalize_ngrams(sum(getattr(self, n)[env_id].values(), Counter())).values() if p > 0)
    
    def opponents(self, env_id: str):
        return set([opp_uid for n in ["unigrams", "bigrams", "trigrams", "fourgrams", "fivegrams"] for opp_uid in getattr(self, n)[env_id].keys()])
    
    @staticmethod
    def _normalize_ngrams(ngrams): total = sum(ngrams.values()); return {k: v / total for k, v in ngrams.items()}
