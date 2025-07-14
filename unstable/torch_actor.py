import os, time, asyncio
from collections import defaultdict, deque
from typing import Optional, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

from unstable.utils.logging import setup_logger

class PyTorchActor:
    def __init__(self, cfg: Dict[str, Any], tracker, name: str):
        self.logger = setup_logger(f"actor-{name}", tracker.get_log_dir())  # Adjusted for local call, assuming tracker has local methods
        self.gpu_ids = [0]  # Assume single GPU or adjust for multi-GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))
        
        # Load base model and tokenizer
        self.base_model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            max_position_embeddings=cfg.get("max_model_len", 8192)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # For generation
        
        # PEFT for LoRA support
        self.model = self.base_model  # Start with base
        self.lora_adapters = {}  # Dict: path -> adapter_name
        self.next_adapter_id = 1
        self.max_loras = cfg.get("max_loras", 8)
        
        self.temperature = cfg.get("temperature", 0.7)
        self.top_p = cfg.get("top_p", 0.95)
        self.max_tokens = cfg.get("max_tokens", 4096)
        self.max_parallel_seq = cfg.get("max_parallel_seq", 128)

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
        self._last_step_time = time.monotonic()

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
                    self.logger.error(f"Potential deadlock detected - no engine steps for {time.monotonic() - self._last_step_time:.1f} seconds\nRunning requests: {self._running}\nQueue size: {len(self._queue)}")
                if not self._queue:
                    continue
                
                # Collect current requests (up to max_parallel_seq)
                requests = []
                while self._queue and len(requests) < self.max_parallel_seq:
                    requests.append(self._queue.popleft())
                
                # Group by LoRA for efficient processing
                groups = defaultdict(list)
                for prompt, path, fut in requests:
                    lora = path or "base"
                    req_id = str(self._next_id); self._next_id += 1
                    self._futures[req_id] = fut
                    self._req2lora[req_id] = lora
                    self._queued -= 1
                    self._running += 1
                    groups[lora].append((req_id, prompt))
                
                for lora, group in groups.items():
                    if lora != "base":
                        if lora not in self._lora_ids:
                            # Load LoRA adapter if not present
                            self._lora_ids[lora] = self._next_lora_id
                            self._next_lora_id += 1
                            # Assuming LoRA path is a PEFT-compatible directory
                            self.model = PeftModel.from_pretrained(self.base_model, lora, adapter_name=lora)
                        self.model.set_active_adapters(lora)
                    else:
                        self.model.disable_adapters() if hasattr(self.model, 'disable_adapters') else None
                    
                    # Prepare batch
                    req_ids, prompts = zip(*group)
                    inputs = self.tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=True).to(self.model.device)
                    
                    # Autoregressive generation loop (custom for chunking/dynamic feel)
                    generated_ids = inputs.input_ids.clone()
                    step_start = time.monotonic()
                    done = [False] * len(prompts)
                    for step in range(self.max_tokens):
                        if all(done):
                            break
                        
                        # Forward pass
                        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            outputs = self.model(input_ids=generated_ids, attention_mask=(generated_ids != self.tokenizer.pad_token_id))
                            logits = outputs.logits[:, -1, :]
                        
                        # Sampling
                        logits = logits / self.temperature
                        if self.top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                            sorted_indices_to_remove = cumulative_probs > self.top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                            logits = logits.masked_fill(indices_to_remove, -float('inf'))
                        
                        probs = torch.softmax(logits, dim=-1)
                        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                        
                        # Append next tokens
                        generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(1)], dim=1)
                        
                        # Check for EOS and mark done
                        for i in range(len(next_tokens)):
                            if done[i]:
                                continue
                            if next_tokens[i] == self.tokenizer.eos_token_id:
                                done[i] = True
                        
                        # Token counting
                        new_tok = sum(1 for d in done if not d)  # Approximate new tokens
                        now = time.monotonic()
                        for _ in range(new_tok):
                            self._tok_hist.append(now)
                        
                    step_duration = time.monotonic() - step_start
                    self._last_step_time = time.monotonic()
                    if step_duration > 5.0:
                        self.logger.warning(f"Slow generation step: {step_duration:.1f}s")
                    
                    # Decode and set results
                    for i, req_id in enumerate(req_ids):
                        generated_text = self.tokenizer.decode(generated_ids[i][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                        fut = self._futures.pop(req_id, None)
                        if fut and not fut.done():
                            fut.set_result(generated_text)
                        self._running -= 1
                        self._req2lora.pop(req_id, None)
                        self._prev_tok_cnt.pop(req_id, None)  # Adjusted, as no token_ids
                    
            except Exception as e:
                self.logger.exception(f"Critical error in batch loop: {e}")
                await asyncio.sleep(1.0)

    async def _report_loop(self):
        self.logger.info("Starting _report_loop")
        while True:
            await asyncio.sleep(5.0)
            stats = {"queued": self._queued, "running": self._running, "tok_s": self._tok_rate()}
            self.logger.info(f"inside while loop _report_loop stats: {stats}")
            try:
                self.tracker.log_inference(actor=self.name, gpu_ids=self.gpu_ids, stats=stats)  # Adjusted for local
            except Exception as e:
                self.logger.warning(f"tracker logging failed: {e}")

    def _tok_rate(self, window: float = 2.0) -> float:
        now = time.monotonic()
        while self._tok_hist and now - self._tok_hist[0] > window:
            self._tok_hist.popleft()
        return len(self._tok_hist) / window if window > 0 else 0