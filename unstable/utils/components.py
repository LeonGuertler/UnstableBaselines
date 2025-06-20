import time
import ray
import math
import json
import textarena as ta
from collections import Counter
from typing import List, Tuple, Dict, Any
import itertools

from unstable.actor import VLLMActor
from unstable.core import BaseTracker
from unstable.collector import CallableActorWrapper
from unstable.model_pool import ModelPool
from unstable.utils.templates import OBSERVATION_FORMATTING, ACTION_EXTRACTION, extract_action


@ray.remote
class StateActionExploration:
    def __init__(self, model_pool: ModelPool, tracker: BaseTracker, vllm_config: Dict[str, Any], num_actors: int, prompt_template: str, action_extraction: str, runs: int = 25):
        self.model_pool = model_pool
        self.tracker = tracker
        self.alive = True
        self.vllm_config = vllm_config
        self.prompt_template = prompt_template
        self.action_extraction = action_extraction
        self.data = [json.loads(line) for line in open("unstable/utils/example_data.jsonl", "r") if line.strip()]
        self.dataset_len = len(self.data)
        self.runs = runs

        self.actors = [VLLMActor.options(num_gpus=1).remote(vllm_config=vllm_config, tracker=tracker, name=f"EntropyEstimator-{i}") for i in range(num_actors)]
        self.actor_iter = itertools.cycle(self.actors)

    def run(self):
        @ray.remote(num_cpus=0)
        def run_state_action_entropy(item, actor, lora_path, prompt_template, action_extraction):
            model = CallableActorWrapper(actor, lora_path, OBSERVATION_FORMATTING[prompt_template], ACTION_EXTRACTION[action_extraction])

            action_distribution = Counter()
            while sum(action_distribution.values()) < self.runs:
                print(f"Running {sum(action_distribution.values())}/{self.runs}")
                full, act, fb, prompt = model.get_full_response(item["observation"])
                valid_action = extract_action(act, ta.make(item["env_id"]).action_space(item["pid"]))
                if valid_action: action_distribution[valid_action] = action_distribution.get(valid_action, 0) + 1

            total = sum(action_distribution.values())
            entropy = -sum((count / total) * math.log2(count / total) for count in action_distribution.values())

            return {"entropy": entropy, "distribution": action_distribution}
        
        dataset = iter(self.data)
        entropy_results = {}
        ckpt_state_action_distribution: Dict[str, int] = {}
        state_action_entropy_estimation_flight: List[Tuple] = []
        while True:
            latest_ckpt = ray.get(self.model_pool.latest_ckpt.remote())
            if latest_ckpt not in ckpt_state_action_distribution:
                while sum(len(lst) for lst in entropy_results.values()) < self.dataset_len:
                    while (len(state_action_entropy_estimation_flight) < 20) and (ckpt_state_action_distribution.get(latest_ckpt, 0) < self.dataset_len):
                        item = next(dataset)
                        lora_path = ray.get(self.model_pool.ckpt_path.remote(latest_ckpt))
                        actor = next(self.actor_iter)
                        fut = run_state_action_entropy.remote(item, actor, lora_path, self.prompt_template, self.action_extraction)
                        state_action_entropy_estimation_flight.append((fut, item["env_id"], item["pid"], latest_ckpt, item["id"]))
                        ckpt_state_action_distribution[latest_ckpt] = ckpt_state_action_distribution.get(latest_ckpt, 0) + 1

                    wait_pool = [f for f, *_ in state_action_entropy_estimation_flight]
                    if not wait_pool: continue  # nothing running (should not happen)
                    done_ref, _ = ray.wait(wait_pool, num_returns=1)
                    finished = done_ref[0]

                    idx = next(i for i, (f, *_) in enumerate(state_action_entropy_estimation_flight) if f == finished)
                    fut, env_id, pid, ckpt_uid, seed = state_action_entropy_estimation_flight.pop(idx)
                    res = ray.get(fut)
                    print(f"ITEM ID: {item['id']}, ENV ID: {env_id}, ENTROPY: {res['entropy']}, DISTRIBUTION: {res['distribution']}")
                    entropy_results[f"exploration/{env_id}-entropy"] = entropy_results.get(f"exploration/{env_id}-entropy", []) + [res['entropy']]
                    time.sleep(2.0)
                
                entropy_results = {k: sum(v) / len(v) for k, v in entropy_results.items()}
                print(f"[STATE ACTION ENTROPY] Final Results", f'{entropy_results}')
                self.tracker.log.remote(entropy_results)
                dataset = iter(self.data)
                entropy_results = {}
            else:
                time.sleep(5.0)
