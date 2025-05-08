import os, time, asyncio, threading
import ray, vllm, torch
import numpy as np
from collections import deque

import hashlib

def hash_tensor(t: torch.Tensor) -> str:
    return hashlib.md5(t.detach().cpu().to(torch.float32).numpy().tobytes()).hexdigest()

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

    def init_process_group(
        self, master_addr, master_port, rank, world, group_name, backend
    ):
        os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = master_addr, master_port
        dist.init_process_group(backend, rank=rank, world_size=world)
        self.model_update_pg = dist.new_group(range(world), backend=backend)

    # ---------- weight update RPCs ----------
    def prepare_weight(self, name, dtype, shape):
        """Create an empty tensor buffer inside the live model."""
        with torch.no_grad():
            param = dict(self.model.named_parameters())[name]
            assert param.shape == tuple(shape)
            # Allocate an empty tensor sharing param’s storage
            self._tmp_dst = param.data            # keep reference
            return True                           # ACK

    def finish_weight_update(self):
        """Called once after the last param — just to sync CUDA stream."""
        torch.cuda.empty_cache()



    # def update_weights(self, weights):
    #     with self.lock, torch.no_grad():
    #         sd = {k: t.to(dtype=self.model.dtype, device=self.device)
    #               for k, t in weights.items()}
    #         self.model.load_state_dict(sd, strict=True)

    # def update_weights(self, weights_np: dict):
    #     """
    #     Parameters
    #     ----------
    #     weights_np : Dict[str, np.ndarray]
    #         State-dict coming from the learner. Key pattern:
    #         "module.model.layers.{i}.self_attn.[q|k|v]_proj.weight", etc.
    #     """
    #     print("\nUPDATING ACTOR WEIGHTS")
    #     t0 = time.time()

    #     with self.lock, torch.no_grad():
    #         # ── live model used by vLLM ────────────────────────────────────────────
    #         model  = self.llm.llm_engine.model_executor.driver_worker.worker.get_model()
    #         n_layer = model.config.num_hidden_layers

    #         # ── 1. strip "module." prefix ────────────────────────────────────────
    #         stripped = {k[7:] if k.startswith("module.") else k: v
    #                     for k, v in weights_np.items()}

    #         # ── 2. build clean dict & collect Q/K/V per layer ────────────────────
    #         clean_sd = {}
    #         qkv_buckets = {i: {} for i in range(n_layer)}

    #         for k, arr in stripped.items():
    #             # attention splits
    #             if ".self_attn.q_proj.weight" in k:
    #                 i = int(k.split(".")[3])              # layer index
    #                 qkv_buckets[i]["q"] = arr
    #             elif ".self_attn.k_proj.weight" in k:
    #                 i = int(k.split(".")[3])
    #                 qkv_buckets[i]["k"] = arr
    #             elif ".self_attn.v_proj.weight" in k:
    #                 i = int(k.split(".")[3])
    #                 qkv_buckets[i]["v"] = arr
    #             else:
    #                 # everything else copies as-is
    #                 clean_sd[k] = torch.from_numpy(arr.copy())

    #         # ── 3. fuse QKV tensors layer-by-layer ──────────────────────────────
    #         for i in range(n_layer):
    #             bucket = qkv_buckets[i]
    #             with suppress(KeyError):
    #                 q = torch.from_numpy(bucket["q"].copy())
    #                 k = torch.from_numpy(bucket["k"].copy())
    #                 v = torch.from_numpy(bucket["v"].copy())
    #                 fused = torch.cat([q, k, v], dim=0)
    #                 fused_key = f"model.layers.{i}.self_attn.qkv_proj.weight"
    #                 clean_sd[fused_key] = fused

    #         # ── 4. sanity check & load ──────────────────────────────────────────
    #         missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    #         if missing:
    #             print("[WARNING] Missing keys after fusion:", len(missing))
    #         if unexpected:
    #             print("[WARNING] Unexpected keys after fusion:", len(unexpected))

    #         # optional: verify a hash difference
    #         import hashlib
    #         def _h(t): return hashlib.md5(
    #             t.detach().cpu().to(torch.float32).numpy().tobytes()).hexdigest()
    #         print("   └── first-param hash now:", _h(next(model.parameters())))

    #     print(f"Finished updating weights in {time.time() - t0:.2f}s")







    # def update_weights(self, weights): 
    #     print("\nUPDATING ACTOR WEIGHTS")
    #     print("type(weights):", type(weights))

    #     t0 = time.time()

    #     with self.lock, torch.no_grad():
    #         model = self.llm.llm_engine.model_executor.driver_worker.worker.get_model()
    #         clean = {k[7:] if k.startswith("module.") else k: v for k, v in weights.items()}

    #         for layer in range(n_layers):
    #             q = clean.pop(f"model.layers.{layer}.self_attn.q_proj.weight")
    #             k = clean.pop(f"model.layers.{layer}.self_attn.k_proj.weight")
    #             v = clean.pop(f"model.layers.{layer}.self_attn.v_proj.weight")
    #             clean[f"model.layers.{layer}.self_attn.qkv_proj.weight"] = torch.cat([q, k, v], dim=0)


    #         print("[BEFORE]", hash_tensor(next(model.parameters())))
    #         # Convert weights but keep on CPU
    #         torch_weights = {k: torch.from_numpy(v.copy()) for k, v in weights.items()}  # copy avoids non-writable warning
    #         model_keys = set(model.state_dict().keys())
    #         weight_keys = set(torch_weights.keys())
    #         print("[DEBUG] Missing keys:", model_keys - weight_keys)
    #         print("[DEBUG] Extra keys:", weight_keys - model_keys)

    #         model.load_state_dict(torch_weights, strict=False)  # load_state_dict will handle the device placement

    #         print("[AFTER]", hash_tensor(next(model.parameters())))
    #     print(f"Finished updating weights in {time.time() - t0:.2f}s\n")


    # def update_weights(self, weights): 
    #     print("\nUPDATING ACTOR WEIGHTS")
    #     t0 = time.time()

    #     with self.lock, torch.no_grad():
    #         model = self.llm.llm_engine.model_executor.driver_worker.worker.get_model()
    #         device = next(model.parameters()).device

    #         torch_weights = {k: torch.from_numpy(v).to(device) for k, v in weights.items()}
    #         model.load_state_dict(torch_weights, strict=False)  # strict=False to avoid shape/key mismatches

    #     print(f"Finished updating weights in {time.time() - t0:.2f}s\n")


    # def update_weights(self, weights): 
    #     print("\nUPDATING ACTOR WEIGHTS")
    #     t0 = time.time()

    #     with self.lock, torch.no_grad():
    #         model = (self.llm.llm_engine.model_executor.driver_worker.worker.get_model())
    #         device = next(model.parameters()).device
    #         state_dict = model.state_dict()

    #         for k, w in weights.items():
    #             if k in state_dict and state_dict[k].shape == w.shape:
    #                 tensor = torch.from_numpy(w).to(device) #, non_blocking=True)
    #                 state_dict[k].copy_(tensor)

    #     print(f"Finished updating weights in {time.time() - t0:.2f}s\n")