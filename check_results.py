import pandas as pd 
import numpy as np 


results_path = "eval_results/Qwen/Qwen3-4B-Base/eval_results.csv"

df = pd.read_csv(results_path)

for env_id in df["env_id"].unique():
    sub_df = df[df["env_id"]==env_id]
    sub_df["win"] = (sub_df["eval_model_reward"] > sub_df["avg_opponent_reward"]).astype(bool)
    sub_df["loss"] = (sub_df["eval_model_reward"] < sub_df["avg_opponent_reward"]).astype(bool)
    sub_df["draw"] = (sub_df["eval_model_reward"] == sub_df["avg_opponent_reward"]).astype(bool)
    
    win_rate = sub_df["win"].sum() / len(sub_df) 
    loss_rate = sub_df["loss"].sum() / len(sub_df) 
    draw_rate = sub_df["draw"].sum() / len(sub_df) 
    avg_reward = sub_df["eval_model_reward"].mean()
    # input(sub_df)
    print(f"Env-id: {env_id}")
    print(f"{win_rate=}")
    print(f"{loss_rate=}")
    print(f"{draw_rate=}")
    print(f"{avg_reward=}")
    input()
# input(df)