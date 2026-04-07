import os
import yaml
from unstable.train import train

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    train(config)
