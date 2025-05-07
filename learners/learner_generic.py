import time, ray, torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from ray.train import get_context
# from algorithms import Reinforce
# from algorithms import build_algo   # Reinforce, PPO, etc.
from learners.reinforce import Reinforce

def train_loop_per_worker(cfg):
    # ---- Ray Train context & DDP ranks ---------------------------------
    ctx   = get_context()
    rank  = ctx.get_world_rank()
    world = ctx.get_world_size()

    local_gpu = rank % torch.cuda.device_count()
    device    = torch.device(f"cuda:{local_gpu}")

    # dist.init_process_group("nccl", rank=rank, world_size=world)

    import torch.distributed as dist
    assert dist.is_initialized() # sanity-check


    # ---- model, tokenizer, algorithm -----------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device)

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_gpu], output_device=local_gpu
    )

    tok   = AutoTokenizer.from_pretrained(cfg["model_name"],
                                          trust_remote_code=True)
    # algo  = build_algo(cfg["algo"], cfg, model, tok, device)
    algo  = Reinforce(cfg, model, tok, device)
    opt   = algo.opt
    buf   = cfg["step_buffer"]

    B, G, CLIP, T, CKPT = (
        cfg["batch_size"], cfg["grad_accum"],
        cfg["grad_clip"],  cfg["total_iters"],
        cfg["checkpoint_freq"],
    )

    for it in range(T):
        # ------ wait until buffer non-empty -----------------------------
        while ray.get(buf.size.remote()) < B*world:
            time.sleep(0.2)

        # ------ each worker independently pulls *its own* mini-batch ----
        steps   = ray.get(buf.get_batch.remote(B))
        batch   = algo.prepare_batch(steps)
        # loss    = algo.update(batch)

        mini_batch_size = len(batch)//cfg["grad_accum"]
        for i in range(cfg["grad_accum"]):
            start, end = i*mini_batch_size, (i+1)*mini_batch_size
            # mini_batch = batch[start:end] 
            mini_batch = tuple(lst[start:end] for lst in batch)
            loss = algo.update(mini_batch)
        
        # step
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        opt.zero_grad()

        if rank==0 and (it+1)%CKPT==0:
                ctx.save_checkpoint({
                    "model": model.module.state_dict(),
                    "iteration": it + 1,
                })


        # if (it + 1) % G == 0:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        #     opt.step(); opt.zero_grad()

        #     # optional checkpoint from rank 0
        #     if rank == 0 and (it + 1) % CKPT == 0:
        #         ctx.save_checkpoint({
        #             "model": model.module.state_dict(),
        #             "iteration": it + 1,
        #         })

        # if rank == 0 and (it + 1) % 10 == 0:
        ctx.report({"iter": it + 1, "loss": loss})
