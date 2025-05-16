python3 unstable.py \
    --model_name "Qwen/Qwen3-0.6B" \
    --wandb \
    --num_actors 3 \
    --num_learners 1 \
    --lr 1e-4 \
    --batch_size 384 \
    --gradient_accumulation_steps 384 \
    --max_tokens 4096 \
    --gradient_checkpointing \
    --bf16_training \
    --num_collection_workers 120 \
    --num_evaluation_workers 6 \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --temperature 0.5

