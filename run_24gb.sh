python3 unstable.py \
    --model_name "Qwen/Qwen3-0.6B" \
    --wandb \
    --num_actors 2 \
    --num_learners 1 \
    --num_collection_workers 512 \
    --batch_size 64 \
    --gradient_accumulation_steps 64 \
    --max_tokens 1024 \
    --gradient_checkpointing \
    --bf16_training \
    --use_all_data

