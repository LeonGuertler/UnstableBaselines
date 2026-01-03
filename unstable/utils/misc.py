import csv, json, os
from unstable.utils._types import GameInformation


def write_training_data_to_file(batch, filename: str):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists: writer.writerow(['pid', 'prompt', 'completion', 'reward', "env_id", "step_info"])  # header
        for step in batch: writer.writerow([step.pid, step.prompt, step.completion, step.reward, step.env_id, step.step_info])

def write_game_information_to_file(game_info: GameInformation, filename: str) -> None:
    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["game_idx", "turn_idx", "pid", "name", "prompt", "full_action", "extracted_action", "step_info", "final_reward", "eval_model_pid", "eval_opponent_name"])
        if not file_exists: writer.writeheader()
        for t in range(game_info.num_turns or len(game_info.prompts)):
            pid = game_info.pid[t] if t < len(game_info.pid) else None
            row = {"game_idx": game_info.game_idx, "turn_idx": t, "pid": pid, "name": game_info.names.get(pid, ""), "prompt": game_info.prompts[t] if t < len(game_info.prompts) else "", "full_action": game_info.completions[t] if t < len(game_info.completions) else "", "extracted_action": game_info.actions[t] if t < len(game_info.actions) else "", "step_info": json.dumps(game_info.step_infos[t] if t < len(game_info.step_infos) else {}, ensure_ascii=False), "final_reward": game_info.final_rewards.get(pid, ""), "eval_model_pid": game_info.eval_model_pid, "eval_opponent_name": game_info.eval_opponent_name}
            writer.writerow(row)