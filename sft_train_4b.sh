python3 sft.py \
    --model_name "Qwen/Qwen3-4B-base" \
    --train_file "data/sft_dataset.jsonl" \
    --output_dir "outputs/sft_lora_4b" \
    --batch_size 32 \
    --epochs 3 \
    --lora_rank 32 \
    --lora_alpha 32 \
    --wandb_project "UnstableBaselines" \
    --wandb_name "tictactoe-sft"