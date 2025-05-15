import os, csv
import time, datetime
from typing import List


def initialize_local_folder_structure(args):
    args.wandb_name = f"{args.model_name}-{args.train_env_id}-run-{int(time.time())}"

    # confirm outputs folder exists
    os.makedirs(args.output_dir, exist_ok=True)

    # create date folder
    date_folder = datetime.datetime.now().strftime('%Y-%m-%d')
    os.makedirs(args.output_dir, exist_ok=True)

    # create run folder
    run_folder_name = f"{datetime.datetime.now().strftime('%H-%M-%S')}-{args.model_name.replace('/', '-')}-{args.train_env_id}"
    args.run_folder = os.path.join(args.output_dir, date_folder, run_folder_name)
    os.makedirs(args.run_folder, exist_ok=True)

    # create train/eval/checkpoint folders
    args.output_dir_train = os.path.join(args.run_folder, "training_data")
    args.output_dir_eval = os.path.join(args.run_folder, "eval_data")
    args.output_dir_checkpoints = os.path.join(args.run_folder, "checkpoints")

    # create necessary folders
    os.makedirs(args.output_dir_train, exist_ok=True)
    os.makedirs(args.output_dir_eval, exist_ok=True)
    os.makedirs(args.output_dir_checkpoints, exist_ok=True)

    # set absolute paths where necessary
    args.initial_lora_path = os.path.abspath(args.initial_lora_path)

    return args

def write_eval_data_to_file(episode_info, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(episode_info[0].keys()))
        writer.writeheader()
        writer.writerows(episode_info)

def write_training_data_to_file(batch, filename: str):
    with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['pid', 'obs', 'act', 'reward'])  # header
        for step in batch:
            writer.writerow([step.pid, step.obs, step.act, step.reward])