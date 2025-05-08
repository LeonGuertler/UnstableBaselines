import ray, torch, time, wandb
from ray.train import get_context
from ray.air import session
from ray.train import Checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

# local imports
from algorithms import Reinforce, PPO



# def broadcast_to_vllm(model_ddp, actors):
#     """Run on rank-0 learner *only*."""
#     model = model_ddp.module                  # unwrap DDP
#     num, t0 = 0, time.time()

#     for name, param in model.named_parameters():
#         num += 1
#         shape, dtype = tuple(param.shape), torch_type_codec(param.dtype)

#         # Ask every actor to allocate its destination tensor (non-blocking)
#         futs = [a.prepare_weight.remote(name, dtype, shape)
#                 for a in actors]

#         # If you use ZeRO-3, gather full param on rank-0 only here
#         with deepspeed.zero.GatheredParameters([param],
#                                                enabled=ds_config.zero_stage==3):
#             dist.broadcast(param.data, 0, group=model_update_pg)

#         # sync actors, free temp cuda memory on last param
#         if num == len(list(model.named_parameters())):
#             [a.finish_weight_update.remote() for a in actors]

#         # ensure RPCs complete before next param (optional but safe)
#         ray.get(futs)

#     dist.barrier(group=model_update_pg)
#     print(f"[BROADCAST] {num} tensors in {time.time()-t0:.2f}s")



def broadcast_one_version(model_ddp, collector):
    model = model_ddp.module
    actors = ray.get(collector.get_current_group.remote())   # handles
    pg     = ray.get(collector.get_current_pg.remote())      # torch PG handle

    for name, p in model.named_parameters():
        # 1. ask only current actors to allocate their dst tensor
        futs = [a.prepare_weight.remote(name, str(p.dtype), p.shape) for a in actors]
        # 2. broadcast to that PG
        dist.broadcast(p.data, 0, group=pg)
        ray.get(futs)                 # wait so we don’t overrun actors

    [a.finish_weight_update.remote() for a in actors]
    dist.barrier(group=pg)

    # current group becomes "previous" opponents
    collector.flip.remote()




def train_loop_per_worker(cfg):
    args = cfg["args"]; buffer = cfg["buffer"]; collector = cfg["collector"]

    # init wandb
    wandb.init(project=args.wandb_project_name, name=f"{args.wandb_name}-learner", config=args)

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

        metrics = {}

        mini_batch_size = len(batch)//args.gradient_accumulation_steps
        for i in range(args.gradient_accumulation_steps):
            start, end = i*mini_batch_size, (i+1)*mini_batch_size
            mini_batch = batch[start:end] 
            update_info = algo.update(mini_batch)

            for k in update_info:
                metrics[k] = metrics.get(k, 0.0) + update_info[k]
        
        # step
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()


        # log to WandB
        if rank == 0: # TODO add the learner name here since each learner might track custom stuff
            avg_metrics = {f"learner/{k}": v / args.gradient_accumulation_steps for k, v in metrics.items()}
            avg_metrics["learner/iteration"] = iteration
            avg_metrics["learner/grad_norm"] = sum(p.grad.data.norm(2).item()**2 for p in model.parameters() if p.grad is not None) ** 0.5
            avg_metrics["learner/lr"] = optimizer.param_groups[0]["lr"]
            wandb.log(avg_metrics)
            # log straight to wandb

        if rank == 0:
            broadcast_to_vllm(model, collector)
            # raw_sd   = model.module.state_dict() if world_size > 1 else model.state_dict()
            # clean_sd = prepare_vllm_state_dict(raw_sd)       # ← your fusion helper
            # cpu_state = {k: v.detach().cpu().contiguous()    # ⇐ KEEP AS TORCH TENSORS
            #             for k, v in clean_sd.items()}
            # weights_ref = ray.put(cpu_state)
            # collector.update_all_weights.remote(weights_ref)



        # if rank == 0:
        #     # state_dict = model.module.state_dict() if world_size > 1 else model.state_dict()
        #     # cpu_state = {k: v.detach().to(dtype=torch.float32).cpu().numpy() for k, v in state_dict.items()}
        #     # weights_ref = ray.put(cpu_state)  # ✅ put ONCE — do NOT ray.get() this again
        #     # print("[learner] type(weights_ref):", type(weights_ref))
        #     # collector.update_all_weights.remote(weights_ref)  # ✅ passes ObjectRef to remote actor
        #     raw_sd   = model.module.state_dict() if world_size > 1 else model.state_dict()
        #     clean_sd = prepare_vllm_state_dict(raw_sd)        # ➟ keys match vLLM
        #     weights_ref = ray.put({k: v.cpu().numpy() for k, v in clean_sd.items()})
        #     collector.update_all_weights.remote(weights_ref)


            # weights_ref = ray.put(cpu_state)
            # ray.get(collector.update_all_weights.remote(weights_ref))
            # collector.update_all_weights(weights_ref)
            # collector.update_all_weights.remote(weights_ref)


            # session.report({"iteration": iteration, "weights": cpu_state})

            # checkpoint = Checkpoint.from_dict({"iteration": iteration, "model": cpu_weights})
            # session.report({"iteration": iteration}, checkpoint=checkpoint)

        iteration += 1


        # if rank == 0:
        #     cpu_state = {k: v.detach().cpu() for k, v in model.module.state_dict().items()}
        #     session.report({
        #         "iteration": iteration,
        #         "model": cpu_state
        #     })

        # iteration += 1


    
