import argparse

from unstable.train import train
from unstable.utils.templates import get_algorithm_config


def parse_envs(envs_arg: str): return [e.strip() for e in envs_arg.split(",") if e.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="grpo-connect-four", help="The run name.")
    parser.add_argument("--algorithm", type=str, default="grpo", help="The algorithm to train on.") 
    parser.add_argument("--envs", type=str, default="ConnectFour-v0-train", help=(
        "Comma-separated list of environments to train on. "
        "Example: 'ConnectFour-v0-train,AnotherEnv-v0'"
        ),
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B-Base", help="The model to train on.")
    parser.add_argument("--template", type=str, default="qwen3-zs", help="The prompt template to use.")
    args = parser.parse_args()
    env_ids = parse_envs(args.envs)
    config = get_algorithm_config(args.algorithm)
    config["run"] = args.name
    config["model_name"] = args.model
    config["learner"]["model_name"] = args.model
    config["vllm_config"]["model_name"] = args.model
    config["env_sampler"]["train"] = [
        {
            "id": env_id,
            "num_players": 2,
            "num_actors": 1,
            "prompt_template": args.template,
        }
        for env_id in env_ids
    ]
    train(config)