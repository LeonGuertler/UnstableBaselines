python3 unstable.py \
    --model_name "Qwen/Qwen3-4B-base" \
    --train_env_id "SimpleTak-v0,Snake-v0,ConnectFour-v0" \
    --eval_env_id "SimpleTak-v0,TicTacToe-v0,ConnectFour-v0,Snake-v0"\
    --wandb \
    --debugging True \
    --num_actors 7 \
    --num_learners 1 \
    --lr 5e-5 \
    --batch_size 384 \
    --gradient_accumulation_steps 384 \
    --max_tokens 4096 \
    --gradient_checkpointing \
    --bf16_training \
    --num_collection_workers 384 \
    --num_evaluation_workers 64 \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0 \
    --initial_lora_path None \
    --self_play_opponent_lag_lower 5 \
    --self_play_opponent_lag_upper 11 \
    --format_reward_think 0.0 \
    --format_reward_valid_move 0.0 \
    --format_penalty_invalid_move 0.0


# --batch_size 512 \
# --gradient_accumulation_steps 512 \
# --num_collection_workers 512 \
# --num_evaluation_workers 32 \

# google/gemini-2.0-flash-001
    # --initial_lora_path "checkpoint-3" \