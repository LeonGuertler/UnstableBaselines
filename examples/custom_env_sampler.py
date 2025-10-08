from unstable.train import train
from unstable.utils.templates import get_algorithm_config
from unstable.collection.env_samplers import UniformRandomEnvSampler


if __name__ == "__main__":
    config = get_algorithm_config("ppo")
    config['learner']['learning_rate'] = 1e-5
    config['learner']['grad_clip'] = 0.2
    config['env_sampler']['type'] = UniformRandomEnvSampler
    train(config)