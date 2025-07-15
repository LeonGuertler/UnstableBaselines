import os, time, asyncio
from collections import defaultdict, deque
from typing import Optional, Dict, Any

import ray
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.lora.request import LoRARequest

from unstable.utils.logging import setup_logger


@ray.remote
class VLLMActor:
    def __init__(self, cfg: Dict[str, Any], tracker, name: str):
        self.logger = setup_logger(f"actor-{name}", ray.get(tracker.get_log_dir.remote())) # set up logging
        self.gpu_ids = ray.get_gpu_ids()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))
        
        engine_args = EngineArgs(
            model=cfg["model_name"], enable_lora=True, max_loras=cfg["max_loras"], max_lora_rank=cfg["lora_config"]["lora_rank"], 
            max_cpu_loras=cfg["max_loras"], max_num_seqs=cfg["max_parallel_seq"], task="generate", max_model_len=cfg["max_model_len"],
            disable_custom_all_reduce=True, enforce_eager=False, disable_log_stats=True,  # Reduce logging overhead
        )
        try: self.engine = LLMEngine.from_engine_args(engine_args); self.logger.info("VLLM engine initialized successfully")
        except Exception as e: self.logger.error(f"VLLM engine initialization failed: {e}"); raise
        self.logger.info(f"vLLM model path or name: {engine_args.model}")
        self.logger.info(f"Model architecture: {self.engine.model_config.__dict__}")
            
        self.sampling_params = SamplingParams(temperature=cfg.get("temperature", 0.7), top_p=cfg.get("top_p", 0.95), max_tokens=cfg.get("max_tokens", 4096))

        self._queue = deque()
        self._futures = {}
        self._next_id = 0
        self._req2lora = {}
        self._prev_tok_cnt = defaultdict(int)

        self.tracker = tracker
        self.name = name

        self._queued = 0
        self._running = 0
        self._tok_hist = deque()
        self._batch_task = asyncio.create_task(self._batch_loop())
        self._report_task = asyncio.create_task(self._report_loop())
        self._lora_ids: Dict[str, int] = {"base": 0}
        self._next_lora_id = 1
        self._last_step_time = time.monotonic()  # Add health check flag

    async def submit_prompt(self, prompt: str, lora_path: Optional[str] = None) -> str:
        if lora_path is not None and not isinstance(lora_path, str): lora_path = str(lora_path)
        fut = asyncio.Future()
        self._queued += 1
        self._queue.append((prompt, lora_path, fut))
        return await fut

    async def _batch_loop(self):
        while True:
            try:
                await asyncio.sleep(0.02)
                if time.monotonic() - self._last_step_time > 30: 
                    self.logger.error(f"Potential deadlock detected - no engine steps for {time.monotonic() - self._last_step_time:.1f} seconds\nRunning requests: {dict(self._running)}\nQueue size: {len(self._queue)}") # 30 second deadlock detection
                while self._queue:
                    prompt, path, fut = self._queue.popleft()
                    lora = path or "base"
                    req_id = str(self._next_id); self._next_id += 1
                    self._futures[req_id] = fut
                    self._req2lora[req_id] = lora
                    self._queued -= 1
                    self._running += 1

                    if path:
                        if path not in self._lora_ids:
                            self._lora_ids[path] = self._next_lora_id
                            self._next_lora_id += 1
                        lora_req = LoRARequest(path, self._lora_ids[path], path)
                    else:
                        lora_req = None
                    try: self.engine.add_request(req_id, prompt, self.sampling_params, lora_request=lora_req)
                    except Exception as e:
                        self.logger.error(f"Failed to add request {req_id}: {e}")
                        self._running -= 1
                        self._req2lora.pop(req_id, None)
                        fut.set_exception(e)
                        continue
                try:
                    step_start = time.monotonic()
                    outs = self.engine.step()
                    step_duration = time.monotonic() - step_start
                    self._last_step_time = time.monotonic()
                    if step_duration > 5.0: self.logger.warning(f"Slow engine step: {step_duration:.1f}s") # Log slow steps
                except Exception as exc:   
                    self.logger.exception(f"engine.step() failed - running: {dict(self._running)}"); await asyncio.sleep(1.0)  # Brief pause before retry
                    continue

                for out in outs:
                    req_id = out.request_id
                    lora = self._req2lora.get(req_id, "base")
                    segment = out.outputs[-1]

                    tok_ids = getattr(segment, "token_ids", None) or []
                    prev = self._prev_tok_cnt[req_id]
                    new_tok = max(0, len(tok_ids) - prev)
                    self._prev_tok_cnt[req_id] = len(tok_ids)

                    now = time.monotonic()
                    for _ in range(new_tok): 
                        self._tok_hist.append(now)
                    if segment.finish_reason is not None:
                        fut = self._futures.pop(req_id, None)
                        if fut and not fut.done():
                            fut.set_result(segment.text)
                        self._running -= 1
                        self._req2lora.pop(req_id, None)
                        self._prev_tok_cnt.pop(req_id, None)
            except Exception as e: self.logger.exception(f"Critical error in batch loop: {e}"); await asyncio.sleep(1.0)  # Prevent tight error loop

    async def _report_loop(self):
        self.logger.info("Starting _report_loop")
        while True:
            await asyncio.sleep(5.0) # only send every 5 sec
            stats = {"queued": self._queued, "running": self._running, "tok_s": self._tok_rate()}
            self.logger.info(f"inside while loop _report_loop stats: {stats}")
            try: ray.get(self.tracker.log_inference.remote(actor=self.name, gpu_ids=self.gpu_ids, stats=stats))
            except Exception as e: self.logger.warning(f"tracker logging failed: {e}")

    def _tok_rate(self, window: float = 2.0) -> float:
        now  = time.monotonic()
        while self._tok_hist and now - self._tok_hist[0] > window:
            self._tok_hist.popleft()
        return len(self._tok_hist) / window