# unstable-baselines (WIP)
(it's calles `unstable-baselines` becuase the maintained OpenAI baselines package is called `stable-baselines`)


# Getting Started
It makes sense to start from the game-tuned checkpoint (three epochs of sft on 750 observation-actions pairs generated via R1 self-play on TicTacToe-v0). 
The checkpoint is zipped `lora_ckpt.zip`. You have to unzip it. The `run.sh` script will use this as the initial set of lora weights by default. 
Afterward you can just run `bash run.sh`. Depending on how many GPUs you have, you can set the `--num_actors` and `--num_learners`. Keep in mind that collection is much much much more expensive than lora-training (so 7-1 is prob a good ratio).



# TODOs
- track action diversity and log it to wandb (only really possible in fixed move games like TicTacToe, etc. but probably super interesting and valuable)
- multi-gpu TorchTrainer
- seperate the logs for everything (and actually log to files) for easier debugging
- Organize .sh scripts

- store specific checkpoints




KIV:
- track time per complete episode
- track time per turn
- track time per generated char
- maybe dynamically adjust the number of gpus used for training/collection


## General TODOs:
- sharding for both training and collection
- single-gpu training
- multi-node training


# Ideas:
- increase sample weight based on SD of reward 
- somehow account for stochacisity in environment rewards (maybe somehow include reward certainty)
- dynamic GRPO rollouts (either by window or by least return confidence)
