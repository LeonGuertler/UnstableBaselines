# python3 sft.py \
#     --model_name "Qwen/Qwen3-4B-base" \
#     --train_file "data/sft_dataset.jsonl" \
#     --output_dir "outputs/sft_lora_4b" \
#     --batch_size 32 \
#     --epochs 1 \
#     --lora_rank 128 \
#     --lora_alpha 256 \
#     --wandb_project "UnstableBaselines" \
#     --wandb_name "tictactoe-sft"


python3 sft.py \
    --model_name "Qwen/Qwen3-4B-base" \
    --train_files data/Snake.jsonl data/SimpleTak.jsonl data/SimpleNegotiation.jsonl \
    --output_dir "sft_lora_4b" \
    --batch_size 32 \
    --epochs 1 \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.03 \
    --use_rslora True\
    --wandb_project "UnstableBaselines" \
    --wandb_name "combined-sft"
