import argparse

from unstable.train import train
from unstable.utils.templates import get_algorithm_config


def parse_envs(envs_arg: str): return [e.strip() for e in envs_arg.split(",") if e.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="grpo-connect-four", help="The run name.")
    parser.add_argument("--algorithm", type=str, default="grpo", help="The algorithm to train on.") 
    parser.add_argument("--players", type=int, default=2, help="The number of players in the environment.")
    parser.add_argument("--envs", type=str, default="ConnectFour-v0-train", help=(
        "Comma-separated list of environments to train on. "
        "Example: 'ConnectFour-v0-train,AnotherEnv-v0'"
        ),
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B-Base", help="The model to train on.")
    parser.add_argument("--template", type=str, default="qwen3-zs", help="The prompt template to use.")
    parser.add_argument("--group_size", type=int, default=1, help="Number of rollouts per prompt for GRPO grouping.")
    parser.add_argument("--action_extraction_fn", type=str, default="default", help="Action extraction function to use.")
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
            "num_players": args.players,
            "num_actors": 1,
            "prompt_template": args.template,
            "group_size": args.group_size,
            "action_extraction_fn": args.action_extraction_fn,
        }
        for env_id in env_ids
    ]
    train(config)