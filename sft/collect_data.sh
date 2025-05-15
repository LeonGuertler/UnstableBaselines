python3 sft_generate_data.py \
    --model "deepseek/deepseek-r1" \
    --env_id "TicTacToe-v0" \
    --outfile "data/sft_dataset.jsonl" \
    --episodes 128 \
    --threads 128
