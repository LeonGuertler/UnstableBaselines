import argparse



def get_args():
    # base args
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--debugging", type=bool, default=False)

    # general configs
    ap.add_argument("--gradient_accumulation_steps", type=int, default=64)
    ap.add_argument("--gradient_checkpointing", action="store_true") 
    ap.add_argument("--bf16_training", action="store_true") 
    ap.add_argument("--gradient_clip", type=float, default=1.0)

    # reward design
    ap.add_argument("--format_reward_think", type=float, default=0.25)
    ap.add_argument("--format_reward_valid_move", type=float, default=1.0)
    ap.add_argument("--format_penalty_invalid_move", type=float, default=-1.0)

    # faster running vars
    ap.add_argument("--num_actors", type=int, default=3)
    ap.add_argument("--num_learners", type=int, default=1)
    ap.add_argument("--num_collection_workers", type=int, default=384)
    ap.add_argument("--num_evaluation_workers", type=int, default=4)
    ap.add_argument("--max_vllm_seq", type=int, default=384)

    # collection params
    ap.add_argument("--train_env_id", type=lambda arg: arg.split(',') if ',' in arg else arg, default="TicTacToe-v0", help="Single env as string or multiple envs as comma-separated string")
    ap.add_argument("--max_env_steps", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--observation_format_template", type=str, default="default")
    ap.add_argument("--action_extraction_template", type=str, default="default")
    ap.add_argument("--self_play_opponent_lag_lower", type=int, default=7)
    ap.add_argument("--self_play_opponent_lag_upper", type=int, default=11)
    ap.add_argument("--use_all_data", type=bool, default=False, help="Whether to use traces from both players or only the current player")

    # eval params
    ap.add_argument("--eval_env_id", type=lambda arg: arg.split(',') if ',' in arg else arg, default="TicTacToe-v0", help="Single env as string or multiple envs as comma-separated string")
    # ap.add_argument("--eval_env_id", default="TicTacToe-v0")
    ap.add_argument("--max_env_steps_eval", type=int, default=64)
    ap.add_argument("--eval_model_name", type=str, default="google/gemini-2.0-flash-lite-001")

    # directory and local logging args 
    ap.add_argument("--output_dir", type=str, default="outputs/")
    ap.add_argument("--save_strategy", type=str, default="best", choices=["steps"])
    ap.add_argument("--save_every_n_update_steps", type=int, default=3) #25)
    ap.add_argument("--log_training_data", type=bool, default=True)

    # wandb & tracking params
    ap.add_argument("--wandb", action="store_true") 
    ap.add_argument("--wandb_project_name", type=str, default="UnstableBaselines")
    ap.add_argument("--ma_range", type=int, default=100)

    # lora
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=int, default=0.0)
    ap.add_argument("--initial_lora_path", type=str, default=None)
    ap.add_argument("--vllm_max_loras", type=int, default=4)

    args = ap.parse_args() 
    args.max_buffer_size = args.batch_size*3 # default TODO maybe move at some point
    return args