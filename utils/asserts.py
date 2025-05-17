import os, wandb, torch, ray
from transformers import AutoConfig

def _validate_requested_gpus(args):
    total_gpus = args.num_actors + args.num_learners
    available_gpus = torch.cuda.device_count()
    cpu_count = os.cpu_count()
    print("\n" + "="*55)
    print("System Resource Allocation Summary")
    print("="*55)
    def line(label, used, available, symbol=""): print(f"{label:<40} {used:>5} / {available:<5} {symbol}")
    line("GPUs for Actors", args.num_actors, available_gpus)
    line("GPUs for Learners", args.num_learners, available_gpus)
    line("Total GPUs Requested", total_gpus, available_gpus, "✅" if total_gpus == available_gpus else ("⚠️" if total_gpus < available_gpus else "❌"))
    print("-" * 55)
    line("Total CPUs", cpu_count, cpu_count)
    line("Threads for Data Collection", args.num_collection_workers, "N/A")
    line("Threads for Evaluation", args.num_evaluation_workers, "N/A")
    assert total_gpus >= (args.num_actors+args.num_learners), f"You can not have more learners + actors than gpus."


def _validate_batch_sizes(args):
    assert args.batch_size % args.gradient_accumulation_steps == 0, "Batch size must be divisible by gradient accumulation steps."
    per_step_batch_size = args.batch_size // args.gradient_accumulation_steps
    world_size = args.num_learners  # Learners are typically used for gradient syncing
    assert per_step_batch_size % world_size == 0, f"Per-step batch size ({per_step_batch_size}) must be divisible by number of learners ({world_size})."
    per_device_batch = per_step_batch_size // world_size
    print("\n" + "=" * 55)
    print("Batch Size Configuration Summary")
    print("=" * 55)
    def line(label, value): print(f"{label:<40} {value:>10}")
    line("Total Batch Size", args.batch_size)
    line("Gradient Accumulation Steps", args.gradient_accumulation_steps)
    line("Effective Batch per Step", per_step_batch_size)
    line("Number of Learners (world size)", world_size)
    line("Per-Device Batch Size", per_device_batch)
    print("-" * 55)
    if per_device_batch > 512: print("⚠️  Warning: Per-device batch size is large. May cause OOM issues or slow convergence.")

def _misc_asserts(args):
    try:
        wandb.ensure_configured()
    except wandb.errors.UsageError as e:
        raise AssertionError("Weights & Biases is enabled but not logged in.\nRun `wandb login` in your terminal or set the WANDB_API_KEY environment variable.")
    if "OPENROUTER_API_KEY" not in os.environ:
        raise AssertionError("Missing OPENROUTER_API_KEY in environment variables.\nExport it using `export OPENROUTER_API_KEY=your_key_here` before running the script.")
    if not (args.self_play_opponent_lag_upper > args.self_play_opponent_lag_lower): 
        raise AssertionError(f"Invalid self-play lag range: upper={args.self_play_opponent_lag_upper} must be strictly greater than lower={args.self_play_opponent_lag_lower}.\nUpdate the args so that `--self_play_opponent_lag_upper` > `--self_play_opponent_lag_lower`.")

def _model_and_lora_asserts(args):
    assert args.lora_rank > 0, "LoRA rank must be greater than 0."
    assert args.lora_alpha > 0, "LoRA alpha must be greater than 0."
    assert 0.0 <= args.lora_dropout <= 1.0, "LoRA dropout must be between 0 and 1."
    try:
        AutoConfig.from_pretrained(args.model_name)
    except Exception as e:
        raise AssertionError(f"Unable to fetch model config for '{args.model_name}'. Make sure the model name is correct and that you're online or it's available locally.\nDetails: {e}")


def assert_args(args):
    _validate_requested_gpus(args=args)
    _validate_batch_sizes(args=args)
    _misc_asserts(args=args)
    _model_and_lora_asserts(args)
