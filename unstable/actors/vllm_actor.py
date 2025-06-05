import os, time
import asyncio
from collections import deque
from typing import Optional, Dict, Any

import ray, torch, vllm
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.lora.request import LoRARequest


@ray.remote
class VLLMActor:
    def __init__(self, vllm_config: Dict[str, Any]):
        try:
            gpu_ids = ray.get_gpu_ids() 
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
            torch.cuda.set_device(0)

            engine_args = EngineArgs(
                model=vllm_config["model_name"], enable_lora=True, max_loras=vllm_config.get("max_loras", 5), max_lora_rank=vllm_config["lora_config"]["lora_rank"], 
                max_cpu_loras=vllm_config.get("max_loras", 5), max_num_seqs=vllm_config["max_parallel_seq"], task="generate"
            )
            self.engine = LLMEngine.from_engine_args(engine_args)
            self.sampling_params = SamplingParams(temperature=vllm_config.get("temperature", 0.7),  top_p=vllm_config.get("top_p", 0.95), max_tokens=vllm_config.get("max_tokens", 4096))

            self._queue = deque()
            self._futures = {}
            self._next_id = 0

            loop = asyncio.get_event_loop()
            loop.create_task(self._batch_loop())
        
        except Exception as e:
            import traceback
            print("Error initializing VLLMActor:", e)
            traceback.print_exc()
            raise

    async def submit_prompt(self, prompt: str, lora_path: Optional[str] = None) -> str:
        fut = asyncio.Future()
        self._queue.append((prompt, lora_path, fut))
        return await fut

    async def _batch_loop(self):
        while True:
            await asyncio.sleep(0.02)
            while self._queue:
                prompt, path, fut = self._queue.popleft()
                req_id = str(self._next_id)
                self._next_id += 1
                self._futures[req_id] = fut
                lora_req = LoRARequest(lora_name=path, lora_path=path, lora_int_id=abs(hash(path))) if path is not None else None
                print(f"[Actor {os.getpid()}] submitting req {req_id} with {lora_req}")
                self.engine.add_request(req_id, prompt, self.sampling_params, lora_request=lora_req)
            # Step the engine and only resolve once finish_reason is non-None
            for out in self.engine.step():
                token = out.outputs[-1] # take the last token in this partial output
                if token.finish_reason is None: continue # skip interim/newline events
                fut = self._futures.pop(out.request_id, None) # now it’s done—fulfil the future
                if fut: fut.set_result(token.text)

