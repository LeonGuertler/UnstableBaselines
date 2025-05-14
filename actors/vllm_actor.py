import os, time
import asyncio
from collections import deque
from typing import Optional

import ray, torch, vllm
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.lora.request import LoRARequest


class VLLMActor:
    def __init__(self, args):
        gpu_ids = ray.get_gpu_ids() 
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        torch.cuda.set_device(0)

        self.args = args
        engine_args = EngineArgs(
            model=args.model_name,
            tokenizer="outputs/sft_lora_4b/checkpoint-3", 
            enable_lora=True, 
            max_loras=2, 
            max_lora_rank=args.lora_rank, 
            max_cpu_loras=2, 
            max_num_seqs=args.max_vllm_seq, 
            tokenizer_mode="slow"
        )
        self.engine = LLMEngine.from_engine_args(engine_args)
        self.sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)

        self._queue = deque()
        self._futures = {}
        self._next_id = 0

        loop = asyncio.get_event_loop()
        loop.create_task(self._batch_loop())

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

                if path is not None:
                    lora_req = LoRARequest(lora_name=path, lora_path=path, lora_int_id=abs(hash(path)))
                else:
                    lora_req = None

                print(f"[Actor {os.getpid()}] submitting req {req_id} with {lora_req}")
                self.engine.add_request(req_id, prompt, self.sampling_params, lora_request=lora_req)

            # 2) Step the engine and only resolve once finish_reason is non-None
            for out in self.engine.step():
                # take the last token in this partial output
                token = out.outputs[-1]

                # skip interim/newline events
                if token.finish_reason is None:
                    continue

                # now it’s done—fulfil the future
                fut = self._futures.pop(out.request_id, None)
                if fut:
                    fut.set_result(token.text)

