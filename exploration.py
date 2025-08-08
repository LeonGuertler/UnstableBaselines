import os, sys
import re
import csv
import time
import math
import uuid
import json
import random
import torch
import copy
import argparse
from tqdm import tqdm
from tqdm.asyncio import tqdm
from collections import Counter
from vllm.lora.request import LoRARequest
from vllm import EngineArgs, LLMEngine, SamplingParams

from unstable.utils.templates import OBSERVATION_FORMATTING, extract_action_and_format_feedback

import wandb
import textarena as ta


class Engine:
    def __init__(self, base_engine_args):
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus == 0: raise RuntimeError("No CUDA devices visible – cannot initialise Engine.")
        self.engines = [];  self._pending = []
        self._last_pending_print = None
        for gpu_id in range(self.num_gpus):
            cfg = copy.deepcopy(base_engine_args)
            cfg.tensor_parallel_size = 1
            prev_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            try: engine = LLMEngine.from_engine_args(cfg)
            finally: os.environ["CUDA_VISIBLE_DEVICES"] = prev_visible
            self.engines.append(engine); self._pending.append(0)
        print(f"Initialised {self.num_gpus} independent vLLM engines (one per GPU).")

    def add_request(self, *args, **kwargs):
        target_idx = self._pending.index(min(self._pending))
        self.engines[target_idx].add_request(*args, **kwargs)
        self._pending[target_idx] += 1

    def step(self):
        all_responses = []
        if self._last_pending_print != self._pending:
            self._last_pending_print = copy.deepcopy(self._pending)
            status = " ".join(f"GPU {idx}: {p}" for idx, p in enumerate(self._pending))
            sys.stdout.write(f"\rPending per engine: {status}")
            sys.stdout.flush()
        for idx, eng in enumerate(self.engines):
            if not self._pending[idx] >= 1: continue
            res = eng.step()
            for r in res:
                if r.outputs[-1].finish_reason is not None: 
                    self._pending[idx] -= 1
                    all_responses.append(r)
        return all_responses

    def __getattr__(self, item): return getattr(self.engines[0], item)


def play_episodes(env_id, engine, sampling_params, lora_req, template, n_episodes: int = 5, k_actions: int = 30, max_turns: int = 40):
    logged_responses = {}; metrics = {'entropy': {}, 'num_actions': {}}; turn = 0
    envs = [ta.make(env_id) for _ in range(n_episodes)]
    for env in envs:
        env.reset(num_players=_num_players[env_id])
        env.state.error_allowance = 0
    if args.fixed_opponent: agents = ['self', ta.agents.OpenRouterAgent(model_name=args.fixed_opponent)]; random.shuffle(agents)
    else: agents = ['self', 'self']

    with tqdm(total=max_turns, desc="Running episodes...") as pbar:
        done_flags = [False] * len(envs)
        while not all(done_flags) or turn < max_turns:

            # Generate requests over environments
            print('Generating requests...')
            reqs = {}; actions = {}
            for i, env in enumerate(envs):
                if done_flags[i]: continue
                pid, obs = env.get_observation()
                if i == 0: 
                    print('###### OBSERVATION ######')
                    print(obs)
                agent = agents[pid]

                if agent == 'self':
                    for _ in range(k_actions): 
                        req_id = str(uuid.uuid4())
                        reqs[req_id] = i
                        engine.add_request(req_id, template(obs), sampling_params, lora_request=lora_req)
                        logged_responses[req_id] = {"turn": env.state.turn, "pid": pid, "env_id": env_id, "observation": obs, "response": None, "action": None}
                else:
                    actions.setdefault(i, []).append(agent(obs))
            
            # Process responses
            print('Running...')
            while reqs: 
                responses = engine.step()
                for r in responses:
                    r_id = r.request_id; r = r.outputs[-1]
                    action = _extract_action(extract_action_and_format_feedback(r.text)[0], action_space=_action_spaces[env_id])
                    if action is not None: actions.setdefault(reqs[r_id], []).append(action)   
                    logged_responses[r_id]["response"], logged_responses[r_id]["action"] = r.text, action
                    del reqs[r_id]
            
            # Update metrics and step environments
            print('Results...')
            for i, act in actions.items():
                action_dist = Counter(act)
                print(f'###### ENV {i} ######')
                print('ACTION DISTRIBUTION', action_dist)
                metrics['entropy'].setdefault(turn, []).append(_entropy(action_dist))
                metrics['num_actions'].setdefault(turn, []).append(len(action_dist))
                action = f'[{random.choices(list(action_dist.keys()), weights=list(action_dist.values()))[0]}]'
                d, _ = envs[i].step(action)
                print('FINAL ACTION:', action, 'DONE:', d)
                if d: done_flags[i] = True

            pbar.update(1); turn += 1

    for env in envs: env.close()
    metrics['entropy'] = {k: sum(v) / len(v) for k, v in metrics['entropy'].items()}
    metrics['num_actions'] = {k: sum(v) / len(v) for k, v in metrics['num_actions'].items()}
    return logged_responses, metrics


_action_spaces = {
    'SimpleTak-v0-train': rf"\[\s*({'|'.join(str(i) for i in range(4**2))})\s*\]",
    'TicTacToe-v0-train': rf"\[\s*({'|'.join(str(i) for i in range(3**2))})\s*\]",
    'KuhnPoker-v0-train': r"\[(Check|Bet|Fold|Call)\]",
    'ConnectFour-v0-train': r".*\[(?:col\s*)?(\d+)\].*",
    'Wordle-v0-train': r"\[(\w+)\]",
    "Minesweeper-v0-train": r"\[(\d+\s\d+)\]",
    "Battleship-v0-train": r"\[([A-Za-z]\d+)\]",
    "Bandit-v0-summarize-train": r"\[(red|blue|green|yellow|purple)\]"
}
_num_players = {
    'SimpleTak-v0-train': 2,
    'TicTacToe-v0-train': 2,
    'KuhnPoker-v0-train': 2,
    'ConnectFour-v0-train': 2,
    "Battleship-v0-train": 2,
    'Wordle-v0-train': 1,
    "Minesweeper-v0-train": 1,
    "Bandit-v0-summarize-train": 1
}
def _extract_action(action: str, action_space=None) -> str: 
    if (m := re.search(r".*" if action_space is None else action_space, action)):
        return m.group(1).strip().lower() 
    else: return None
def _entropy(actions: Counter) -> float: return -sum((c / sum(actions.values())) * math.log2(c / sum(actions.values())) for c in actions.values() if c > 0)
def _write_data(responses, path: str, filename: str):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, filename), mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['prompt', 'response'])  # header
        for step in responses.values(): writer.writerow([step["observation"], step["response"]])

if __name__ == "__main__":
    # eval args
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--env_id", type=str, default="Battleship-v0-train")
    parser.add_argument("--template", type=str, default="qwen3-zs")
    parser.add_argument("--checkpoint_dir", type=str, default="/work/tgrams/UnstableBaselines/outputs/2025-07-20/07-22-41/exploration-Qwen3-4B-Base-ConnectFour-v0-train-1752988954")
    parser.add_argument("--output_dir", type=str, default="/work/tgrams/UnstableBaselines/outputs/exploration")
    parser.add_argument("--eval_every", type=int, default=25)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--max_turns", type=int, default=40)
    parser.add_argument("--fixed_opponent", type=str, default=None) # "google/gemini-2.0-flash-lite-001"

    # vLLM
    parser.add_argument("--max_parallel_seq", type=int, default=180)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_loras", type=int, default=8)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=4096)
    args = parser.parse_args()

    # checkpoint data
    checkpoints = sorted(list(os.listdir(os.path.join(args.checkpoint_dir, 'checkpoints'))), 
                         key=lambda x: int(x.split("-")[-1])) if args.checkpoint_dir else ["None.csv"]
    if len(checkpoints) > 1: checkpoints = [cp for cp in checkpoints if (int(cp.split("-")[-1])-1) % args.eval_every == 0]
    with open(os.path.join(args.checkpoint_dir, 'checkpoints', checkpoints[0], 'adapter_config.json'), 'r') as f: config = json.load(f)
    args.output_dir = os.path.join(args.output_dir, config['base_model_name_or_path'], args.env_id)
    
    # wandb
    wandb.init(project="UnstableBaselines-exploration-eval", config=vars(args), name=args.name if args.name else f"{args.env_id}-{config['base_model_name_or_path']}-{time.time()}")
    
    engine_args = EngineArgs(
        model=config['base_model_name_or_path'],
        enable_lora=True,
        max_loras=args.max_loras,
        max_lora_rank=config['r'],
        max_cpu_loras=args.max_loras,
        max_num_seqs=args.max_parallel_seq,
        task="generate",
        max_model_len=args.max_model_len,
        tensor_parallel_size=1,          # one GPU per engine
        disable_custom_all_reduce=True,
        enforce_eager=False,
        disable_log_stats=True,
    )

    try: engine = Engine(engine_args)
    except Exception as e: print(f"vLLM engine initialisation failed: {e}"); raise

    # run
    for checkpoint in checkpoints:
        print(f"Evaluating {checkpoint}...")
        
        lora_path = os.path.join(args.checkpoint_dir, 'checkpoints', checkpoint)
        lora_req = LoRARequest(lora_path, int(checkpoint.split("-")[-1]), lora_path) if lora_path else None; print(lora_req)
        sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)

        metrics = {'entropy': {}, 'num_actions': {}}
        reponses, metrics = play_episodes(args.env_id, engine, sampling_params, lora_req, OBSERVATION_FORMATTING[args.template], args.episodes, args.max_turns)
        _write_data(reponses, os.path.join(args.output_dir, checkpoint), f'{checkpoint}.csv')
        wandb.log({
            **{f"{args.env_id}-Turn-{k}/entropy": v for k, v in metrics['entropy'].items()},
            **{f"{args.env_id}-Turn-{k}/actions": v for k, v in metrics['num_actions'].items()},
            f"{args.env_id}/entropy": sum(metrics['entropy'].values()) / len(metrics['entropy']),
            f"{args.env_id}/actions": sum(metrics['num_actions'].values()) / len(metrics['num_actions']),
        })
