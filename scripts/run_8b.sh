python3 unstable.py \
    --model_name "Qwen/Qwen3-8B-base" \
    # --train_env_id "SimpleTak-v0" \
    --wandb \
    --num_actors 2 \
    --num_learners 1 \
    --lr 1e-4 \
    --batch_size 384 \
    --gradient_accumulation_steps 384 \
    --max_tokens 1024 \
    --gradient_checkpointing \
    --bf16_training \
    --num_collection_workers 384 \
    --num_evaluation_workers 8 \
    --lora_rank 32 \
    --lora_alpha 32 \
    --lora_dropout 0