import time, ray, torch
from ray.train import get_context
from transformers import AutoModelForCausalLM, AutoTokenizer

# local imports
from algorithms import Reinforce


def train_loop_per_worker(cfg):
    args = cfg["args"]; buffer = cfg["buffer"]

    # ---- Ray Train context & DDP ranks ---------------------------------
    ctx = get_context()
    rank = ctx.get_world_rank()
    world_size = ctx.get_world_size()

    local_gpu = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{local_gpu}")

    import torch.distributed as dist
    assert dist.is_initialized() # sanity-check

    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_gpu], output_device=local_gpu)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    algo = Reinforce(args, model, tokenizer, device)
    optimizer = algo.optimizer

    gpu_batch_size = args.batch_size // world_size
    iteration = 0
    while True:
        while ray.get(buffer.size.remote()) < args.batch_size: # wait until buffer has enough
            time.sleep(0.2)

        # ------ each worker independently pulls *its own* mini-batch ----
        batch = ray.get(buffer.get_batch.remote(gpu_batch_size))
        optimizer.zero_grad(set_to_none=True)

        mini_batch_size = len(batch)//args.gradient_accumulation_steps
        # print(f"{len(batch)=}")
        # print(f"{args.gradient_accumulation_steps=}")
        # print(f"{mini_batch_size=}")
        for i in range(args.gradient_accumulation_steps):
            start, end = i*mini_batch_size, (i+1)*mini_batch_size
            # print(f"{start=}")
            # print(f"{end=}")
            mini_batch = batch[start:end] 
            # print(mini_batch)
            loss = algo.update(mini_batch)
        
        # step
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
        # opt.zero_grad()

        if rank==0 and (iteration+1)%args.save_every_n_update_steps==0:
                ctx.save_checkpoint({"model": model.module.state_dict(), "iteration": iteration + 1})

        # ctx.report({"iter": iteration + 1, "loss": loss})

        iteration += 1
