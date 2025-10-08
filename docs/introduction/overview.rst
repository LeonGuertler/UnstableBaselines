Overview
~~~~~~~~

Why Unstable Baselines?
"""""""""""""""""""""""
UnstableBaselines is an Async-, Online-, Multi-Agent RL library focused on simplicity and hackability. 
It is designed for fast research iteration on reasoning models for TextArena games with a LoRA-first workflow. 
We do not harness complex GPU-scaling strategies and, therefore, offer lightweight and easy to use RL for small models. 
We believe reinforcement learning should not only belong to large research groups with massive compute resources.
Unstable Baselines is, therefore, designed to run learning algorithms on a single GPU, making reinforcement learning accessible for (almost) everyone.

Why Reinforcement Learning for Text-Based Games?
""""""""""""""""""""""""""""""""""""""""""""""""""""
Text-based games provide an ideal setting for developing reasoning-capable language models through interaction rather than imitation. 
Self-play introduces an emergent curriculum: as agents improve, they automatically encounter stronger opponents and more complex situations, ensuring a continuously evolving training signal. 
This interplay of interaction, adaptation, and self-generated challenge makes RL in text-based games a powerful approach to building adaptive, self-improving language agents capable of grounded reasoning and communication.

Architecture
""""""""""""
.. raw:: html

    <div align="center">
        <img style="width: 800px;" src="../_static/architecture.png" />
    </div>

**Action Sampler.** Text

**Environment Sampler.** Text

**Model Sampler.** Text

**Game Scheduler.** Text

**Learner.** Text

**Replay Buffer.** Text

Configuration
"""""""""""""
Unstable Baselines follows a modular architecture, which allows for easy extension and customization.

