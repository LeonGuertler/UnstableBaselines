
<div align="center">
<picture>
  <source media="(prefers-color-scheme: light)" srcset="docs/_static/sloganlogo.png">
  <img alt="UnstableBaselines logo" src="docs/_static/sloganlogo.png" width="80%" height="80%">
</picture>


An Async, Online, Multi-Turn, Multi-Agent RL library for training reasoning models on TextArena games.

<h3>

[Documentation](https://ub.readthedocs.io/en/latest/)

</h3>

[![GitHub Repo stars](https://img.shields.io/github/stars/LeonGuertler/UnstableBaselines)](https://github.com/LeonGuertler/UnstableBaselines/stargazers)
[![Discord](https://img.shields.io/discord/1257951838322561075?color=7289DA&label=Discord)](https://discord.gg/KPacHzK23e)
[![TextArena](https://img.shields.io/badge/TextArena-181717)](https://github.com/LeonGuertler/TextArena)
<!-- [![TextArena](https://img.shields.io/badge/TextArena-v0.6.9-181717)](https://github.com/LeonGuertler/TextArena) -->
</div>

---


### Why Unstable Baselines?
UnstableBaselines is an Async-, Online-, Multi-Agent RL library focused on simplicity and hackability. Since multiple recent papers showed the sufficiency of LoRA for reasoning tuning, and the fact that opponent sampling for self-play strategies beyond mirror self-play work best when using LoRA weights (since vLLM allows for hot-swapping), we built UnstableBaselines as a LoRA first RL library. We tried to keep the code as straight forward as possible. It is currently around **1.2K** lines long, semi-readable and all our learners run on a **single GPU**. The main focus of unstable-baselines is to enable fast prototyping/research. For something a bit more production ready we recommend to use [oat](https://github.com/sail-sg/oat) or [verifiers](https://github.com/willccbb/verifiers).


### Key Features
* **Asynchronous collection & learning** – actors generate data while learners train.
* **Multi‑agent, multi‑turn** focus with self‑play or fixed opponents.
* **LoRA‑first** fine‑tuning workflow for fast, lightweight updates.
* **Accessability** All our learners can train on a single GPU!
* **Composable reward transforms** at step, game, and sampling stages.


### Quickstart
You can use our PyPi-package:
```bash
pip3 install unstable-rl
```
and run Unstable Baselines from the command-line interface:
```bash
python -m unstable.train --config ppo
```


### Citation [![DOI](https://zenodo.org/badge/975887163.svg)](https://doi.org/10.5281/zenodo.15719270)

If you use **UnstableBaselines** in your research, please cite:

```bibtex
@software{guertler_leon_2025_15719271,
  author={Guertler, Leon and Grams, Tim and Liu, Zichen and Cheng, Bobby},
  title={{UnstableBaselines}},
  month=jun,
  year=2025,
  publisher={Zenodo},
  version={0.1.0},
  doi={10.5281/zenodo.15719271},
  url={https://doi.org/10.5281/zenodo.15719271}
}

```

Developed in partnership with [PlasticLabs](https://plasticlabs.ai/).
