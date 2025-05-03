import argparse, os, re, time, random, gc, wandb
import asyncio, threading

import numpy as np 
from tqdm import tqdm
from collections import deque
from typing import List, Dict, Tuple, Optional

import ray, torch, vllm 
from transformers import AutoModelForCausalLM, AutoTokenizer

import textarena as ta

# local imports
from learners import PPOLearner
from trajectory_buffer import Trajectory, Step, StepBuffer
from utils import REWARD_TRANSFORMATIONS, validate_requested_gpus, average_weights, reserve_resources_for_learners

@ray.remote(num_gpus=1, num_cpus=1)
class RayLearner(PPOLearner):
    def __init__(self, args):
        super().__init__(args)

@ray.remote(num_gpus=1)
class VLLMActor:
    def __init__(self, args):
        gpu_ids = ray.get_gpu_ids()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        torch.cuda.set_device(0)

        self.llm = vllm.LLM(model=args.model_name, trust_remote_code=True, dtype="bfloat16", task="generate")
        self.sampling_params = vllm.SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)

        self.queue = deque()
        self.loop = asyncio.get_event_loop()
        self.loop.create_task(self._batch_loop())

        self.lock = threading.Lock()

    async def submit_prompt(self, prompt: str):
        fut = asyncio.Future()
        self.queue.append((prompt, fut))
        return await fut

    async def _batch_loop(self):
        while True:
            await asyncio.sleep(0.02)
            if not self.queue:
                continue
            batch = []
            while self.queue:
                batch.append(self.queue.popleft())
            prompts, futures = zip(*batch)
            try:
                outputs = await asyncio.to_thread(self.llm.generate, prompts, self.sampling_params, use_tqdm=True)
                for fut, out in zip(futures, outputs):
                    fut.set_result(out.outputs[0].text)
            except Exception as e:
                for fut in futures:
                    fut.set_exception(e)

    def update_weights(self, weights: dict):
        print("\n\nUPDATING ACTOR WEIGHTS")
        t0 = time.time()
        with self.lock:
            with torch.no_grad():
                executor = self.llm.llm_engine.model_executor
                model = executor.driver_worker.worker.get_model()
                device = next(model.parameters()).device
                state_dict = model.state_dict()
                for k in weights:
                    if k in state_dict and state_dict[k].shape == weights[k].shape:
                        tensor = torch.from_numpy(weights[k].copy()).to(device)
                        state_dict[k].copy_(tensor)
        print(f"Finished updating weights in {time.time()-t0} seconds.\n\n")


class Collector:
    def __init__(self, args): 
        self.args = args
        self.group_0: List[ray.actor.ActorHandle] = []
        self.group_1: List[ray.actor.ActorHandle] = []
        self.current_group_id = 0

    def initialize(self, num_actors: int):
        assert num_actors % 2 == 0, f"expected an even number of actors for play against prev checkpoint"
        self.group_0 = [VLLMActor.remote(self.args) for _ in range(num_actors // 2)]
        self.group_1 = [VLLMActor.remote(self.args) for _ in range(num_actors // 2)]

    def get_current_and_prev_client(self):
        # return two actors. One with the previous checkoint and one with the current one
        current_group = self.group_0 if self.current_group_id == 0 else self.group_1
        prev_group = self.group_1 if self.current_group_id == 0 else self.group_0
        return random.choice(current_group), random.choice(prev_group)

    def update_all_weights(self, weights: dict):
        current_group = self.group_0 if self.current_group_id == 0 else self.group_1
        ray.get([client.update_weights.remote(weights) for client in current_group])
        self.current_group_id = 1 - self.current_group_id  # flip roles



def apply_r1_template(observation: str) -> str:
    """R1 template with thinking/answer tags."""
    return (
        f"A conversation between User and Assistant. You are playing a two-player zero-sum game. Make valid moves to win. "
        f"Make sure you enclose the final action you are submitting in squared brackets. "
        f"The Assistant first thinks about the reasoning process in the mind and then provides the move. "
        f"The reasoning process is enclosed within <think> </think>. Everything outside the think tags will be submitted to the environment.\n"
        f"User: {observation}\nAssistant: <think>"
    )


def extract_action_and_format_feedback(raw_action: str) -> Tuple[str, Dict[str, bool]]:
    """
    Extracts the action as everything after </think> and provides feedback on tag formatting.
    Returns:
        action (str): Extracted action string.
        format_feedback (dict): Dictionary with format correctness flags.
    """
    think_match = re.search(r"<think>(.*?)</think>", raw_action, re.DOTALL)
    
    action = ""
    if think_match:
        end_idx = think_match.end()
        action = raw_action[end_idx:].strip()
    
    format_feedback = {"has_think": think_match is not None, "has_answer": False, "order_correct": False}

    return action, format_feedback




@ray.remote(num_cpus=0.1)
def collect_episode_once(env_id: str, buffer, actor1, actor2, use_all_data: bool, max_env_steps: int = 64):
    def _make_env():
        env = ta.make(env_id)
        env = ta.wrappers.FirstLastObservationWrapper(env)
        env.reset(num_players=2)
        env.state.error_allowance = 0
        return env

    env = _make_env()
    traj = Trajectory()
    done, steps = False, 0

    a1_pid = int(np.random.uniform() < 0.5)
    actors = {a1_pid: actor1, 1 - a1_pid: actor2}

    while not done and steps < max_env_steps:
        pid, obs = env.get_observation()
        formatted_prompt = apply_r1_template(observation=obs)
        action = ray.get(actors[pid].submit_prompt.remote(formatted_prompt))

        # extract environment action
        extracted_action, format_feedback = extract_action_and_format_feedback(raw_action=action)
        done, _ = env.step(action=extracted_action)
        if use_all_data or pid == a1_pid:
            traj.pid.append(pid)
            traj.obs.append(formatted_prompt)
            traj.actions.append(action)
            traj.format_feedbacks.append(format_feedback)
            steps += 1

    traj.final_rewards = env.close() if done else {0: 0, 1: 0}
    traj.num_turns = steps

    ray.get(buffer.add_trajectory.remote(traj, current_checkpoint_pid=a1_pid))

def start_collection_loop(args, collector, buffer, max_outstanding):
    def loop():
        outstanding = []
        while True:
            # Clean up finished futures
            ready, not_ready = ray.wait(outstanding, timeout=0, num_returns=len(outstanding))
            outstanding = list(not_ready)

            if len(outstanding) < max_outstanding*5:
                a1, a2 = collector.get_current_and_prev_client()
                fut = collect_episode_once.remote(
                    env_id=args.train_env_id, buffer=buffer, actor1=a1, actor2=a2,
                    use_all_data=args.use_all_data, max_env_steps=args.max_env_steps,
                )
                outstanding.append(fut)
            time.sleep(0.05)
    thread = threading.Thread(target=loop, daemon=True)
    thread.start()


@ray.remote
def train_loop(learners, buffer, collector, args):
    num_learners = len(learners)
    while True:
        # print(ray.get(buffer.size.remote()))
        if ray.get(buffer.size.remote()) >= args.batch_size:
            trajs = ray.get(buffer.get_batch.remote(args.batch_size))

            split_size = len(trajs) // num_learners

            # split batch among learners
            batches = [trajs[i*split_size : (i+1)*split_size] for i in range(num_learners - 1)]
            batches.append(trajs[(num_learners - 1)*split_size:])  # remainder goes to last
            # === Run updates ===
            update_futures = [learners[i].update.remote(batches[i]) for i in range(num_learners)]
            weights_list = ray.get(update_futures)

            # === Average and sync ===
            avg_weights = average_weights(weights_list)

            collector.update_all_weights(avg_weights)  # no ray.get!


            log_future = buffer.log_training_info_to_wandb.remote()
            if num_learners != 1:
                weight_futures = [learner.update_weights.remote(avg_weights) for learner in learners]
                ray.get([log_future] + weight_futures) # Wait for all
            else:
                ray.get([log_future])

            print("✅ All updates + syncing done:", time.time())
            avg_weights = None
            print("Weights updated at", time.time())
        else:
            time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_env_id", default="TicTacToe-v0")
    ap.add_argument("--model_name", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--total_iters", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--base_port", type=int, default=8000)
    ap.add_argument("--num_actors", type=int, default=3)
    ap.add_argument("--num_learners", type=int, default=1)
    
    ap.add_argument("--wandb", action="store_true") 
    ap.add_argument("--wandb_project_name", type=str, default="UnstableBaselines")

    ap.add_argument("--max_buffer_size", type=int, default=4096)
    ap.add_argument("--use_all_data", action="store_true") # i.e. use prev checkpoint perspective as well
    ap.add_argument("--num_collection_workers", type=int, default=128)
    ap.add_argument("--max_env_steps", type=int, default=32)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=64)
    ap.add_argument("--normalize_role_advantage", action="store_true")

    ap.add_argument("--format_reward_think", type=float, default=0.1)
    ap.add_argument("--format_reward_action", type=float, default=0.1)
    ap.add_argument("--format_reward_order", type=float, default=0.1)

    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--gradient_checkpointing", action="store_true") 
    ap.add_argument("--bf16_training", action="store_true") 
    ap.add_argument("--ppo_epochs", type=int, default=1)

    ap.add_argument("--reward_scale", type=float, default=1.0)
    ap.add_argument("--ppo_clip_lower", type=float, default=0.2)
    ap.add_argument("--ppo_clip_upper", type=float, default=0.2)
    ap.add_argument("--ppo_value_clip", type=float, default=0.2)
    ap.add_argument("--vf_coef", type=float, default=1.0)
    ap.add_argument("--kl_penalty_coef", type=float, default=0.0)
    ap.add_argument("--gradient_clip", type=float, default=1.0)

    ap.add_argument("--reward_strategy", type=str, default="win-loss", choices=["win-loss", "raw"])

    args = ap.parse_args() 


    # check whether the gpu counts are correct
    total_gpus, total_cpus = validate_requested_gpus(args=args)
    ray.init(num_gpus=total_gpus)
    pg = reserve_resources_for_learners(args.num_learners)
    learners = [RayLearner.options(placement_group=pg, placement_group_bundle_index=i, num_cpus=2, num_gpus=1).remote(args=args) for i in range(args.num_learners)]

    buffer = StepBuffer.remote(args=args)
    collector = Collector(args=args)
    collector.initialize(num_actors=args.num_actors)
    start_collection_loop(args, collector, buffer, max_outstanding=total_cpus)
    train_loop.remote(learners, buffer, collector, args)

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Shutting down.")

if __name__ == "__main__":
    main()