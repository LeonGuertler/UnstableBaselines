from unstable.train import train
from unstable.common.utils.templates import get_algorithm_config
from unstable.common.env_samplers import UniformRandomEnvSampler


config = get_algorithm_config("ppo")
config['learner']['learning_rate'] = 1e-5
config['learner']['grad_clip'] = 0.2
config['env_sampler']['type'] = UniformRandomEnvSampler
train(config)