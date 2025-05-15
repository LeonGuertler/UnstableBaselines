import ray, torch, os, time, wandb
from ray.train import get_context
from ray.air import session
from ray.train import Checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from learners.lora_utils import build_lora_model, load_lora_state
from peft import get_peft_model_state_dict, set_peft_model_state_dict

# local imports
from algorithms import Reinforce, PPO



def train_loop_per_worker(cfg):
    args = cfg["args"]; buffer = cfg["buffer"]; collector = cfg["collector"]
    wandb.init(project=args.wandb_project_name, name=f"{args.wandb_name} (learner)", config=args) # init wandb

    root_dir = os.getcwd()
    root_checkpoint_dir = os.path.join(root_dir, args.output_dir_checkpoints)
    print(f'WILL STORE TO: {root_checkpoint_dir}')

    # Ray Train context & DDP ranks
    ctx = get_context()
    rank = ctx.get_world_rank()
    world_size = ctx.get_world_size()

    local_gpu = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{local_gpu}")

    import torch.distributed as dist
    assert dist.is_initialized() # sanity-check

    # load base + LoRA
    base = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
    peft_model = build_lora_model(model=base, r=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout).to(device)

    # load initial weights if provided
    if args.initial_lora_path and args.initial_lora_path.lower() != "none":
        load_lora_state(peft_model, args.initial_lora_path)
        torch.cuda.empty_cache()

    model = torch.nn.parallel.DistributedDataParallel(peft_model, device_ids=[local_gpu], output_device=local_gpu, find_unused_parameters=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr) # optimizer over only the adapters
    algo = Reinforce(args, model, tokenizer, device)

    gpu_batch_size = args.batch_size // world_size
    iteration = 0
    while True:
        while ray.get(buffer.size.remote()) < args.batch_size*2: # wait until buffer has enough + stability buffer
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

        if rank == 0:
            checkpoint_folder_path = os.path.join(root_checkpoint_dir, f"iteration-{iteration}")
            os.makedirs(checkpoint_folder_path, exist_ok=True)
            peft_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            peft_model.save_pretrained(checkpoint_folder_path)
            ray.get(collector.add_new_lora_paths.remote(checkpoint_folder_path))
            session.report({"iteration": iteration})
        iteration += 1



    
