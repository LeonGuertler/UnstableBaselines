# unstable-baselines


# TODOs

<!-- - make num train gpus more flexible (i.e. 1-n) -->
<!-- - create warning if not all gpus are used -->
<!-- - track invalid move rate -->
<!-- - better default name for wandb run -->
<!-- - keep track of win-rate by pid -->
<!-- - make sure to only submit the final action (i.e. add action extraction logic) -->
<!-- - add format reward -->
<!-- - add standard formatting options -->


- maybe dynamically adjust the number of gpus used for training/collection
- add training metrics to wandb (the actual training metrics)
- add eval metrics to wandb
- store specific checkpoints
- store sample CoTs

- optionally allocate an evaluate GPU


- track reward sd (echo trap)

KIV:
- track time per complete episode
- track time per turn
- track time per generated char




# Ideas:
- increase sample weight based on SD of reward 
- somehow account for stochacisity in environment rewards (maybe somehow include reward certainty)