import argparse

from unstable.train import train
from unstable.utils.templates import get_algorithm_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="grpo-connect-four", help="The run name. Default is grpo-connect-four.")
    parser.add_argument("--env", type=str, default="ConnectFour-v0-train", help="The environment to train on. Default is ConnectFour.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B-Base", help="The model to train on. Default is Qwen3-1.7B-Base.")
    args = parser.parse_args()

    config = get_algorithm_config("grpo")
    config['model_name'] = args.model; config['learner']['model_name'] = args.model; config['vllm_config']['model_name'] = args.model
    config['env_sampler']['train'] = [{'id': args.env, 'num_players': 2, 'num_actors': 2, 'prompt_template': "qwen3-zs"}]
    train(config)