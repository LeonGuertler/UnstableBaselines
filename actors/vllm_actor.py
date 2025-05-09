# # import os, time, asyncio, threading
# # import ray, vllm, torch
# # import numpy as np
# # from collections import deque

# # from actors.lora_actor_mixin import LoRAHotSwapMixin

# # class VLLMActor(LoRAHotSwapMixin):
# #     def __init__(self, args, lora_path=None):
# #         gpu_ids = ray.get_gpu_ids()
# #         os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
# #         torch.cuda.set_device(0)

# #         self.lora_path = lora_path
# #         self.llm = vllm.LLM(
# #             model=args.model_name,
# #             trust_remote_code=True,
# #             dtype="bfloat16",
# #             task="generate",
# #             max_num_seqs=256,
# #             enable_lora=True
# #         )

# #         self.sampling_params = vllm.SamplingParams(
# #             temperature=args.temperature,
# #             top_p=args.top_p,
# #             max_tokens=args.max_tokens
# #         )

# #         self.lora_request = (
# #             LoRARequest("lora_adapter", abs(hash(lora_path)) % (2**31), lora_path)
# #             if lora_path else None
# #         )

# #         self.queue = deque()
# #         self.loop = asyncio.get_event_loop()
# #         self.loop.create_task(self._batch_loop())
# #         self.lock = threading.Lock()
# import os, time, asyncio, threading
# import numpy as np
# import ray, vllm, torch
# from vllm.lora.request import LoRARequest
# from collections import deque



# class VLLMActor:
#     def __init__(self, args):
#         gpu_ids = ray.get_gpu_ids()
#         os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
#         torch.cuda.set_device(0)

#         # build vllm engine
#         engine_args = vllm.EngineArgs(
#             model=args.model_name,
#             enable_lora=True,
#             max_loras=2,
#             max_lora_rank=8,
#             max_cpu_loras=2,
#             max_num_seqs=256
#         )
#         self.llm = vllm.LLMEngine.from_engine_args(engine_args)

#         self.sampling_params = vllm.SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)

#         self.queue = deque()
#         self.loop = asyncio.get_event_loop()
#         self.loop.create_task(self._batch_loop())



#         async def submit_prompt(self, prompt: str, lora_path: Optional[str]):
#             fut = asyncio.Future()
#             self.queue.append((prompt, lora_path, fut))
#             return await fut


#         async def _batch_loop(self):
#             while True:
#                 await asyncio.sleep(0.02)
#                 if not self.queue:
#                     continue
#                 batch = []
#                 while self.queue:
#                     batch.append(self.queue.popleft())
#                 prompts, lora_path, futures = zip(*batch)
#                 # create submission
#                 if lora_path is not None:

#                 try:
#                     outputs = await asyncio.to_thread(self.llm.generate, prompts, self.sampling_params, use_tqdm=True)
#                     for fut, out in zip(futures, outputs):
#                         fut.set_result(out.outputs[0].text)
#                 except Exception as e:
#                     for fut in futures:
#                         fut.set_exception(e)




import os, time
import asyncio
from collections import deque
from typing import Optional

import ray, torch, vllm
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.lora.request import LoRARequest


class VLLMActor:
    def __init__(self, args):
        gpu_ids = ray.get_gpu_ids() # 1) Pin to the GPUs Ray gave us
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        torch.cuda.set_device(0)

        # 2) Build the engine with LoRA support
        self.args = args
        engine_args = EngineArgs(model=args.model_name, enable_lora=True, max_loras=2, max_lora_rank=args.lora_rank, max_cpu_loras=2, max_num_seqs=args.max_vllm_seq)
        self.engine = LLMEngine.from_engine_args(engine_args)
        # self.llm = vllm.LLM(
        #     model=args.model_name, enable_lora=True, max_loras=5, max_lora_rank=args.lora_rank, max_cpu_loras=5, max_num_seqs=args.max_vllm_seq
        # )
        self.sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)

        # 3) Our own request queue + bookkeeping
        self._queue = deque()       # items: (prompt, Optional[LoRARequest], future)
        self._futures = {}          # request_id → asyncio.Future
        self._next_id = 0

        # 4) Kick off the background task
        loop = asyncio.get_event_loop()
        loop.create_task(self._batch_loop())

    async def submit_prompt(self, prompt: str, lora_path: Optional[str] = None) -> str:
        fut = asyncio.Future()
        self._queue.append((prompt, lora_path, fut))
        return await fut


    async def _batch_loop(self):
        while True:
            # small pause so we batch up incoming prompts
            await asyncio.sleep(0.02)

            # 1) Drain our own queue into the vLLM engine
            while self._queue:
                prompt, path, fut = self._queue.popleft()
                req_id = str(self._next_id)
                self._next_id += 1
                self._futures[req_id] = fut

                # build LoRARequest if needed
                if path is not None:
                    lora_req = LoRARequest(
                        lora_name=path,
                        lora_path=path,
                        lora_int_id=abs(hash(path))
                    )
                else:
                    lora_req = None

                print(f"[Actor {os.getpid()}] submitting req {req_id} with {lora_req}")
                self.engine.add_request(
                    req_id,
                    prompt,
                    self.sampling_params,
                    lora_request=lora_req
                )

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


# TODO add lora_rank to args
