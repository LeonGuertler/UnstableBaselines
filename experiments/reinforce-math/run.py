import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from textarena.envs.registration import register_with_versions
from textarena.wrappers import SingleTurnObservationWrapper
from unstable.train import train

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# SingleTurnEnv (1 Player)
register_with_versions(id="math-12k-v0",        entry_point="env:SingleTurnDatasetEnv", wrappers={"default": [SingleTurnObservationWrapper]}, dataset=os.path.join(DATA_DIR, "math_12k.jsonl"),        type="math",            verifier="judge")
register_with_versions(id="math-lvl3to5-8k-v0", entry_point="env:SingleTurnDatasetEnv", wrappers={"default": [SingleTurnObservationWrapper]}, dataset=os.path.join(DATA_DIR, "math_lvl3to5_8k.jsonl"), type="math",            verifier="judge")
register_with_versions(id="math500-v0",          entry_point="env:SingleTurnDatasetEnv", wrappers={"default": [SingleTurnObservationWrapper]}, dataset=os.path.join(DATA_DIR, "math500.jsonl"),         type="math",            verifier="judge")
register_with_versions(id="minerva-v0",          entry_point="env:SingleTurnDatasetEnv", wrappers={"default": [SingleTurnObservationWrapper]}, dataset=os.path.join(DATA_DIR, "minerva.jsonl"),         type="math",            verifier="judge")
register_with_versions(id="aime-2024-v0",        entry_point="env:SingleTurnDatasetEnv", wrappers={"default": [SingleTurnObservationWrapper]}, dataset=os.path.join(DATA_DIR, "aime2024.jsonl"),        type="math",            verifier="judge")
register_with_versions(id="aime-2025-v0",        entry_point="env:SingleTurnDatasetEnv", wrappers={"default": [SingleTurnObservationWrapper]}, dataset=os.path.join(DATA_DIR, "aime2025.jsonl"),        type="math",            verifier="judge")
register_with_versions(id="gpqa-v0",             entry_point="env:SingleTurnDatasetEnv", wrappers={"default": [SingleTurnObservationWrapper]}, dataset=os.path.join(DATA_DIR, "gpqa.jsonl"),            type="multiple-choice", verifier="judge")
register_with_versions(id="mmlu-pro-v0",         entry_point="env:SingleTurnDatasetEnv", wrappers={"default": [SingleTurnObservationWrapper]}, dataset=os.path.join(DATA_DIR, "mmlu_pro.jsonl"),        type="multiple-choice", verifier="judge")

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    train(config)
