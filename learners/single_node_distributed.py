import time, ray, torch
from ray.train import get_context
from ray.air import session #, Checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

# local imports
from algorithms import Reinforce, PPO


def train_loop_per_worker(cfg):
    args = cfg["args"]; buffer = cfg["buffer"]

    # Ray Train context & DDP ranks
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

        batch = ray.get(buffer.get_batch.remote(gpu_batch_size)) # each worker independently pulls *its own* mini-batch
        optimizer.zero_grad(set_to_none=True)

        mini_batch_size = len(batch)//args.gradient_accumulation_steps
        for i in range(args.gradient_accumulation_steps):
            start, end = i*mini_batch_size, (i+1)*mini_batch_size
            mini_batch = batch[start:end] 
            loss = algo.update(mini_batch)
        
        # step
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()

        # if rank == 0 and iteration % args.save_every_n_update_steps == 0:
        # ckpt = Checkpoint.from_dict({"model": model.module.state_dict(), "iteration": iteration})
        # if rank==0:
        #     ckpt = {"model": model.module.state_dict(), "iteration": iteration}
        #     session.report({"iteration": iteration}, checkpoint=ckpt)
        if rank == 0:
            session.report({
                "iteration": iteration,
                "state_dict_keys": list(model.module.state_dict().keys())  # just as sanity check
            })
        # if rank==0 and (iteration+1)%args.save_every_n_update_steps==0:
        #         ctx.save_checkpoint({"model": model.module.state_dict(), "iteration": iteration + 1})

        iteration += 1
