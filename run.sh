# python3 unstable.py --wandb --num_actors 2 --num_learners 2 --num_collection_workers 200 --ppo_epochs 1 --normalize_role_advantage --batch_size 64 --gradient_accumulation_steps 32 --max_tokens 2048 --gradient_checkpointing --bf16_training --use_all_data


# python3 unstable.py --wandb --num_actors 2 --num_learners 2 --num_collection_workers 1 --ppo_epochs 1 --normalize_role_advantage --batch_size 512 --gradient_accumulation_steps 256 --max_tokens 1500 --gradient_checkpointing --bf16_training --use_all_data



# python3 unstable.py --wandb --num_actors 2 --num_learners 2 --num_collection_workers 200 --ppo_epochs 1 --normalize_role_advantage --batch_size 512 --gradient_accumulation_steps 256 --max_tokens 1500 --gradient_checkpointing --bf16_training --use_all_data
python3 unstable.py --wandb --num_actors 2 --num_learners 2 --num_collection_workers 200 --ppo_epochs 1 --normalize_role_advantage --batch_size 64 --gradient_accumulation_steps 32 --max_tokens 500 --gradient_checkpointing --bf16_training --use_all_data