import os, time, asyncio, threading
import ray, vllm, torch
import numpy as np
from collections import deque



class VLLMActor:
    def __init__(self, args):
        gpu_ids = ray.get_gpu_ids()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        torch.cuda.set_device(0)

        self.llm = vllm.LLM(model=args.model_name, trust_remote_code=True, dtype="bfloat16", task="generate", max_num_seqs=256)
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
