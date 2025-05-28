# unstable-baselines (WIP)
(It's called `unstable-baselines` because the maintained OpenAI baselines package is called `stable-baselines`, duh)


# Getting Started
It makes sense to start from the game-tuned checkpoint (three epochs of sft on 750 observation-actions pairs generated via R1 self-play on TicTacToe-v0). 
The checkpoint is zipped `lora_ckpt.zip`. You have to unzip it. The `run.sh` script will use this as the initial set of lora weights by default. 
Afterward, you can just run `bash run.sh`. Depending on how many GPUs you have, you can set the `--num_actors` and `--num_learners`. Keep in mind that collection is much much much more expensive than lora-training (so 7-1 is prob a good ratio).

(`bash run_24gb.sh` should work well for machine with 4x24gb (vRam) gpus)

Also, this is currently using the new version of textarena (wich is on the `v0.6.9` branch). To run this code, you will have to build textaren locally:
1. `git clone https://github.com/LeonGuertler/TextArena.git`
2. `cd TextArena`
3. `git checkout v0.6.9`
4. `pip uninstall textarena` (just to make sure)
5. `python3 setup.py sdist bdist_wheel`
6. `pip install -e .`
7. all done, you are now running textarena v0.6.9 


### TODOs for Leon now:
- normalize sample rewards by environment (done)
- single player envs 
- fixed opponent (sampling) multi player envs (done - randomly sampling now, but might be worth chaning in the future)
- make single-player game collection easy


### TODOs for Tim:
- track action/state diversity and log it to wandb (only really possible in fixed move games like TicTacToe, etc. but probably super interesting and valuable)
- store specific checkpoints
- fixed number of eval games after each update step


### TODOs for Bobby:
- 


# TODOs
- multi-gpu TorchTrainer
- seperate the logs for everything (and actually log to files) for easier debugging
- Organize .sh scripts

- make wandb logging optional (Bobby?)
- simplify wandb naming (maybe remove company name (i.e. 'Qwen') and add the username of the person running it)
- make it easy to specify a list of eval (and training) opponents (TODO - are multiple eval opponents actually necessary)


- simplify the algorithm vs leaner split to make it more modular (i.e. move the optimizer step into the algorithm to allow for PPO etc. training; (also rename learner to something else))
- It probably makes sense to empower the TrajectoryBuffer to actually manage the env for each step. That way it'll be easier to add GRPO with dynamic branching later on.



# General TODOs:
- sharding for both training and collection
- single-gpu training
- multi-node training


## KIV:
- track time per complete episode
- track time per turn
- track time per generated char
- maybe dynamically adjust the number of gpus used for training/collection



# Ideas:
- Increase sample weight based on SD of reward 
- Account for stochacisity in environment rewards (maybe somehow include reward certainty)
- Dynamic GRPO rollouts (either by window or by least return confidence)

