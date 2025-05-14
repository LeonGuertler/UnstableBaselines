#!/usr/bin/env python
# collect_sft_dataset.py
#
# ⚠️  Requirements
#   pip install openai ray textarena tqdm
#   export OPENROUTER_API_KEY="sk-..."
#
# Example:
#   python collect_sft_dataset.py \
#       --model deepseek-chat \
#       --env TicTacToe-v0 \
#       --outfile data/sft_train.jsonl \
#       --num_actors 4 \
#       --num_workers 256 \
#       --max_samples 10000

import os, re, json, argparse, random, pathlib, concurrent.futures
from typing import List, Dict
from tqdm import tqdm
import textarena as ta
from utils.templates import apply_default_template, OBSERVATION_FORMATTING


STANDARD_GAME_PROMPT = "You are Assistant, a large language model playing two-player zero-sum games. Think inside <think></think> tags and output your move in [square brackets]."

class OpenRouterAgent:
    """ Calls OpenRouter and returns **full** reply: <think>…</think>\\n[move] """
    def __init__(self, model_name: str, system_prompt: str=STANDARD_GAME_PROMPT, verbose: bool=False, **kwargs):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.kwargs = kwargs

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("Set OPENROUTER_API_KEY first.")
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    # ---------- helpers ---------- #
    def _single_call(self, prompt: str) -> str:
        msgs = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt},]
        response = self.client.chat.completions.create(model=self.model_name, messages=msgs, n=1, **self.kwargs)
        reasoning = response.choices[0].message.reasoning
        answer = response.choices[0].message.content
        return reasoning, answer

    def __call__(self, observation: str) -> str:
        prompt = apply_default_template(observation)
        reasoning, answer = self._single_call(observation)
        return reasoning, answer


def make_env(env_id: str):
    env = ta.make(env_id)
    env = ta.wrappers.FirstLastObservationWrapper(env)
    env.reset(num_players=2)
    env.state.error_allowance = 0
    return env

def play_episode(args, agent: OpenRouterAgent) -> List[Dict]:
    env = make_env(args.env_id)
    records = []
    done, steps = False, 0
    while not done and steps < args.max_env_steps:
        pid, obs = env.get_observation()
        formatted = OBSERVATION_FORMATTING[args.observation_template](observation=obs)
        try:
            reasoning, answer = agent(obs) 
        except Exception as e:
            # Abort this game but keep the thread alive
            print(f"⚠️  Episode aborted: {e}")
            return records
        print(reasoning, answer)
        done, _ = env.step(action=answer)
        records.append({"observation": obs, "formatted_observation": formatted, "reasoning": reasoning, "answer": answer})
        steps += 1
    return records

def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="deepseek/deepseek-r1")
    p.add_argument("--env_id", default="TicTacToe-v0")
    p.add_argument("--outfile", default="data/sft_dataset.jsonl")
    p.add_argument("--episodes", type=int, default=128)      # number of games
    p.add_argument("--threads", type=int, default=128)        # thread pool size
    p.add_argument("--max_env_steps", type=int, default=32)
    p.add_argument("--observation_template", default="default")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    return p.parse_args()


def main():
    args = cli()
    pathlib.Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)

    # Build one agent per thread so connections are reused
    agents = [OpenRouterAgent(model_name=args.model, temperature=args.temperature, top_p=args.top_p) for _ in range(args.threads)]
    samples: List[Dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = [pool.submit(play_episode, args, agents[i % args.threads]) for i in range(args.episodes)]
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), unit="episode"):
            samples.extend(fut.result())

    # Write JSONL once all threads are done
    with open(args.outfile, "w", encoding="utf-8") as f:
        for rec in samples:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✓ Saved {len(samples)} samples from {args.episodes} episodes → {args.outfile}")


if __name__ == "__main__":
    main()