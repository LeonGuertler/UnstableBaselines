import trueskill
from dataclasses import dataclass, field
from typing import List, Any, Dict, Optional


@dataclass
class Step:
    pid: int
    obs: str 
    act: str
    reward: float
    env_id: str
    step_info: Optional[Dict]

@dataclass
class PlayerTrajectory:
    pid:                int = field(default_factory=int)
    final_reward:       float = field(default_factory=float)
    obs:                List[str] = field(default_factory=list)
    actions:            List[str] = field(default_factory=list)
    extracted_actions:  List[str] = field(default_factory=list)
    format_feedbacks:   List[Dict] = field(default_factory=list)
    step_infos:         List[Dict] = field(default_factory=list)
    game_info:          Dict = field(default_factory=dict)
    num_turns:          int = field(default_factory=int)


@dataclass
class GameInformation:
    game_idx:           int = field(default_factory=int)
    env_id:             str = field(default_factory=str)
    pid:                List[int] = field(default_factory=list)
    obs:                List[str] = field(default_factory=list)
    full_actions:       List[str] = field(default_factory=list)
    extracted_actions:  List[str] = field(default_factory=list)
    step_infos:         List[Dict] = field(default_factory=list)
    game_info:          Dict = field(default_factory=dict)
    final_rewards:      Dict[int, float] = field(default_factory=dict)
    num_turns:          int = field(default_factory=int)
    names:              Dict[int, str] = field(default_factory=dict)
    eval_model_pid:     Optional[int] = None
    eval_opponent_name: Optional[str] = None

@dataclass 
class AgentSpec:
    pid: int
    kind: str # "checkpoint" | "openrouter"
    collect_data: bool = False
    openrouter_name: str|None = None
    lora_path: str|None = None
    prompt_template: str = "default" # prompt template key
    action_extraction_fn: str = "default"

@dataclass
class GameSpec:
    game_idx: int
    env_id: str
    seed: int
    agent_specs: List[AgentSpec]
    eval_model_pid: Optional[int] = None
    eval_opponent_name: Optional[str] = None


@dataclass
class TaskMeta:
    type: str  # "train" | "eval"
    env_id: str


@dataclass
class TrainEnvSpec:
    env_id: str
    num_players: int
    num_actors: int
    prompt_template: str 
    action_extraction_fn: str = "default"

@dataclass 
class EvalEnvSpec:
    env_id: str 
    num_players: int 
    prompt_template: str
    action_extraction_fn: str = "default"
    fixed_opponent: str = "google/gemini-2.0-flash-lite-001"
    # forced_pid: Optional[List] = None # whether to force a specific pid for the collection models


@dataclass
class ModelMeta:
    uid: str
    kind: str # "checkpoint" | "fixed"
    path_or_name: str # local path or OpenRouter id
    rating: trueskill.Rating # μ / σ
    games: int = 0
    wins: int = 0
    draws: int = 0
    active: bool = True
    iteration: int|None = None


