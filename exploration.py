# First, we sample observations via self-play or against a fixed opponent. 
# Then, we revisit each observation and sample k actions to assess uncertainty. 
# Generating observations is necessary to make sure that the observations actually stem from the agent. 
# Otherwise, if observations are too far from the policy, our entropy estimates will be biased.  
import os
import re
import csv
import time
import math
import uuid
import json
import random
import argparse
import pandas as pd
from tqdm import tqdm
from tqdm.asyncio import tqdm
from collections import Counter
from vllm.lora.request import LoRARequest
from vllm import EngineArgs, LLMEngine, SamplingParams

from unstable.utils.templates import OBSERVATION_FORMATTING, extract_action_and_format_feedback

import wandb
import textarena as ta



def measure_exploration(checkpoint: str, engine: LLMEngine, args):
    lora_path = os.path.join(args.checkpoint_dir, 'checkpoints', checkpoint)
    lora_req = LoRARequest(lora_path, int(checkpoint.split("-")[-1]), lora_path) if lora_path else None; print(lora_req)
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    if args.generate_data: dataset = _generate_data(args.env_id, engine, sampling_params, lora_req, OBSERVATION_FORMATTING[args.template], args.generate_data)
    else: dataset = _read_data(os.path.join(args.checkpoint_dir, 'collection', f'{checkpoint}.csv'), args.observations_per_turn)

    metrics = {"all": {"entropy": [], "unique_actions": []}}
    for example in tqdm(dataset, desc="Evaluating examples"):
        actions = _sample_actions(
            example,
            engine,
            sampling_params,
            lora_req,
            OBSERVATION_FORMATTING[args.template],
            args.actions
        )
        entropy = _entropy(actions)
        if example["turn"] not in metrics: metrics[example["turn"]] = {"entropy": [], "unique_actions": []}
        metrics[example["turn"]]["entropy"].append(entropy); metrics[example["turn"]]["unique_actions"].append(len(actions))
        metrics["all"]["entropy"].append(entropy); metrics["all"]["unique_actions"].append(len(actions))
    return {k: sum(v["entropy"]) / len(v["entropy"]) for k, v in metrics.items()}, \
           {k: sum(v["unique_actions"]) / len(v["unique_actions"]) for k, v in metrics.items()}


_action_spaces = {
    'SimpleTak-v0-train': rf"\[\s*({'|'.join(str(i) for i in range(4**2))})\s*\]",
    'TicTacToe-v0-train': rf"\[\s*({'|'.join(str(i) for i in range(3**2))})\s*\]",
    'ConnectFour-v0-train': r".*\[(?:col\s*)?(\d+)\].*",
    'Wordle-v0-train': r"\[(\w+)\]",
    "Minesweeper-v0-train": r"\[(\d+)\s(\d+)\]",
    "Bandit-v0-train": r"\[(red|blue|green|yellow|purple)\]"
}
_num_players = {
    'SimpleTak-v0-train': 2,
    'TicTacToe-v0-train': 2,
    'ConnectFour-v0-train': 2,
    'Wordle-v0-train': 1,
    "Minesweeper-v0-train": 1,
    "Bandit-v0-train": 1
}
def _sample_actions(example, engine, sampling_params, lora_req, template, k_actions: int = 25):
    sampled_actions = Counter(); request_ids = [str(uuid.uuid4()) for _ in range(k_actions)]; llm_responses = []; _running = 0
    prompt = template(example["observation"])
    for request_id in request_ids: engine.add_request(request_id, prompt, sampling_params, lora_request=lora_req); _running += 1
    while _running:
        responses = engine.step()
        for r in responses:
            r = r.outputs[-1]
            if r.finish_reason is not None:
                a = _extract_action(extract_action_and_format_feedback(r.text)[0], action_space=_action_spaces[args.env_id])
                llm_responses.append({"observation": example["observation"], "response": r.text, "action": a})
                if a is not None: sampled_actions[a] += 1
                _running -= 1
    _write_data(llm_responses, os.path.join(args.output_dir, str(example["turn"])), f'{checkpoint}.csv')
    return sampled_actions
def _extract_action(action: str, action_space=None) -> str: 
    if (m := re.search(r".*" if action_space is None else action_space, action)):
        print(m)
        return m.group(1).strip().lower() 
    else: return None
def _entropy(actions: Counter) -> float: return -sum((c / sum(actions.values())) * math.log2(c / sum(actions.values())) for c in actions.values() if c > 0)
def _read_data(path: str, sample_k: int = 5): 
    df = pd.read_csv(path); observations = []
    df = df[df['actions'] != 'actions']
    for turn in df['turn'].unique():
        seen = set()
        unique = df[df['turn'] == turn].drop_duplicates(subset=['obs'])
        for _, d in unique.sample(min(sample_k, len(unique))).iterrows(): 
            if d['obs'] not in seen:
                observations.append({'turn': int(d['turn']), 'observation': d['obs'], 'id':len(observations), 'env_id': args.env_id})
                seen.add(d['obs'])
    return observations
def _write_data(responses, path: str, filename: str):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, filename), mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['prompt', 'response'])  # header
        for step in responses: writer.writerow([step["observation"], step["response"]])
def _generate_data(env_id, engine, sampling_params, lora_req, template, turns: int = 20):
    env=ta.make(env_id); observations = []
    if args.fixed_opponent: agents = ['self', ta.agents.OpenRouterAgent(model_name=args.fixed_opponent)]; random.shuffle(agents)
    else: agents = ['self', 'self']
    with tqdm(total=turns, desc="Generating observations") as pbar:
        while len(observations) < turns:
            turn = 0
            env.reset(num_players=_num_players[env_id]); env.state.error_allowance=0
            while True:
                pid, obs = env.get_observation(); observations.append({"turn": turn, "id": len(observations), "pid": pid, "env_id": env_id, "observation": obs})
                pbar.update(1)
                action = None
                if agents[pid] == 'self':
                    engine.add_request(str(uuid.uuid4()), template(obs), sampling_params, lora_request=lora_req)
                    while action is None:
                        responses = engine.step()
                        for r in responses:
                            r = r.outputs[-1]
                            if r.finish_reason is not None: action = extract_action_and_format_feedback(r.text)[0]
                else: action = agents[pid](obs)
                done, _ = env.step(action)
                if done or len(observations) >= turns: break
                turn += 1
            env.close()
    return observations

if __name__ == "__main__":
    # eval args
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="ConnectFour-v0-train")
    parser.add_argument("--template", type=str, default="qwen3-zs")
    parser.add_argument("--checkpoint_dir", type=str, default="/work/tgrams/UnstableBaselines/outputs/2025-07-20/07-22-41/exploration-Qwen3-4B-Base-ConnectFour-v0-train-1752988954")
    parser.add_argument("--output_dir", type=str, default="/work/tgrams/UnstableBaselines/outputs/exploration")
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--actions", type=int, default=100)
    parser.add_argument("--observations_per_turn", type=int, default=3)
    parser.add_argument("--generate_data", type=int, default=40)
    parser.add_argument("--fixed_opponent", type=str, default=None) # "google/gemini-2.0-flash-lite-001"

    # vLLM
    parser.add_argument("--max_parallel_seq", type=int, default=120)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
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
    wandb.init(project="UnstableBaselines-exploration-eval", config=vars(args), name=f"{args.env_id}-{config['base_model_name_or_path']}-{time.time()}")
    
    # vLLM engine
    engine_args = EngineArgs(
        model=config['base_model_name_or_path'], enable_lora=True, max_loras=args.max_loras, max_lora_rank=config['r'],
        max_cpu_loras=args.max_loras, max_num_seqs=args.max_parallel_seq, task="generate", max_model_len=args.max_model_len, 
        tensor_parallel_size=args.tensor_parallel_size, disable_custom_all_reduce=True, enforce_eager=False, disable_log_stats=True,  # Reduce logging overhead
    )
    try: engine = LLMEngine.from_engine_args(engine_args); print("VLLM engine initialized successfully")
    except Exception as e: print(f"vLLM engine initialization failed: {e}"); raise

    # run
    for checkpoint in checkpoints:
        print(f"Evaluating {checkpoint}...")
        avg_entropy, avg_actions = measure_exploration(checkpoint, engine, args)
        wandb.log({**{f"Turn {k}/avg entropy": v for k, v in avg_entropy.items()}, 
                   **{f"Turn {k}/avg actions": v for k, v in avg_actions.items()}})
