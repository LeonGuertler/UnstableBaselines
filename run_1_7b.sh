python3 unstable.py \
    --model_name "Qwen/Qwen3-1.7B" \
    --wandb \
    --num_actors 2 \
    --num_learners 1 \
    --num_collection_workers 512 \
    --ppo_epochs 1 \
    --normalize_role_advantage \
    --batch_size 128 \
    --gradient_accumulation_steps 128 \
    --max_tokens 2048 \
    --gradient_checkpointing \
    --bf16_training \
    --use_all_data

