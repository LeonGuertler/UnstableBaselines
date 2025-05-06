import os, torch 
from typing import List, Dict

import ray 
from ray.util.placement_group import placement_group


def validate_requested_gpus(args):
    total_gpus = args.num_actors + args.num_learners
    available_gpus = torch.cuda.device_count()
    cpu_count = os.cpu_count()
    cpu_reserved = args.num_learners
    cpu_remaining = cpu_count - cpu_reserved
    safe_worker_count = int(cpu_count * 0.7)

    print("\n" + "="*48)
    print("System Resource Allocation Summary")
    print("="*48)

    def line(label, used, available, symbol=""):
        print(f"{label:<35} {used:>5} / {available:<5} {symbol}")

    line("GPUs for Actors", args.num_actors, available_gpus)
    line("GPUs for Learners", args.num_learners, available_gpus)
    line("Total GPUs Requested", total_gpus, available_gpus, "✅" if total_gpus == available_gpus else ("⚠️" if total_gpus < available_gpus else "❌"))

    print("-" * 48)
    line("Total CPUs", cpu_count, cpu_count)
    line("CPUs Reserved for Learners", cpu_reserved, cpu_count)
    line("CPUs Remaining for Workers", cpu_remaining, cpu_count)
    line("Safe Collector Limit (70%)", safe_worker_count, cpu_count)
    return total_gpus, safe_worker_count

def reserve_resources_for_learners(num_learners):
    # Reserve each learner 1 GPU and 1 CPU
    bundles = [{"CPU": 2, "GPU": 1} for _ in range(num_learners)]
    pg = placement_group(bundles, strategy="STRICT_PACK")
    ray.get(pg.ready())
    return pg
