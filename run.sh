python3 unstable.py \
    --model_name "Qwen/Qwen3-4B-base" \
    --train_env_id "SimpleTak-v0" \
    --eval_env_id "SimpleTak-v0,TicTacToe-v0"\
    --wandb \
    --num_actors 3 \
    --num_learners 1 \
    --lr 2e-5 \
    --batch_size 32 \
    --gradient_accumulation_steps 32 \
    --max_tokens 4096 \
    --gradient_checkpointing \
    --bf16_training \
    --num_collection_workers 128 \
    --num_evaluation_workers 32 \
    --lora_rank 32 \
    --lora_alpha 32 \
    --lora_dropout 0 \
    --initial_lora_path "checkpoint-3" \
    --self_play_opponent_lag 7


    # --batch_size 512 \
    # --gradient_accumulation_steps 512 \
    # --num_collection_workers 512 \
    # --num_evaluation_workers 32 \