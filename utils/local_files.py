import csv
from typing import List

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