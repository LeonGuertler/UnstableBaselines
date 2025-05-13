# unstable-baselines
(it's calles `unstable-baselines` becuase the maintained OpenAI baselines package is called `stable-baselines`)



`bash run_24gb.sh` should work on 4090s (default is 2gpus for collection 1 for training) 

# TODOs

<!-- - make num train gpus more flexible (i.e. 1-n) -->
<!-- - create warning if not all gpus are used -->
<!-- - track invalid move rate -->
<!-- - better default name for wandb run -->
<!-- - keep track of win-rate by pid -->
<!-- - make sure to only submit the final action (i.e. add action extraction logic) -->
<!-- - add format reward -->
<!-- - add standard formatting options -->
<!-- - add eval metrics to wandb -->
<!-- - store sample CoTs -->
<!-- - add a moving-average tracker and add tau/ma for both the wandb tracking -->
<!-- - dynamically collect eval episodes in parallel -->
<!-- - randomly subsample when  hitting max buffer size -->
<!-- - add training metrics to wandb (the actual training metrics) 
        -> pass the tracker into the training loop?
        -> maybe get the algo to return a dict of stuff worth tracking -->
<!-- - maybe allow for uneven number of actor gpus -->
<!-- - training and vllm inf with hot-swapped LoRA weights (https://docs.vllm.ai/en/stable/features/lora.html -> doesn't seem hard and given the recent papers on this super worth doing) -->


- maybe dynamically adjust the number of gpus used for training/collection
- store specific checkpoints
- track reward sd (echo trap)
- fix multi-gpu training
- split output dir by date, then time
- fix where the lora weights end up


KIV:
- track time per complete episode
- track time per turn
- track time per generated char
- optionally allocate an evaluate GPU


## General TODOs:
- sharding for both training and collection
- single-gpu training
- multi-node training



# Ideas:
- increase sample weight based on SD of reward 
- somehow account for stochacisity in environment rewards (maybe somehow include reward certainty)
- allow for uneven numbers of actors (maybe have the extra one focus on evals)
- dynamic GRPO rollouts (either by window or by least return confidence)