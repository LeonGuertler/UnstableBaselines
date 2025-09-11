from unstable.train import train
from unstable.common.utils.templates import get_algorithm_config


config = get_algorithm_config("reinforce")
config['learner']['learning_rate'] = 1e-5
config['learner']['grad_clip'] = 0.2
config['replay_buffer']['max_buffer_size'] = 800
train(config)