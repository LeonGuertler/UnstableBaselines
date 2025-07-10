import time, pathlib, ray, torch
from typing import Any, Dict, Optional, List
import random

import tree
from dataclasses import replace
from unstable.buffer import EpisodeBuffer
from unstable.core import BaseAlgo
from unstable.model_pool import ModelPool
from unstable.learners.utils import build_peft_model, enable_full_activation_ckpt
from unstable.utils.logging import setup_logger
from unstable.reward_transformations.transformation_sampling import (
    NormalizeRewardsByEnv,
)


def compute_gae(rewards, values, gamma=1.0, gae_lambda=1.0):
    # Compute gae (for policy learning) and return (for critic learning)
    assert len(rewards) == len(values)
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            nextnonterminal = 0  # Assume a complete episode
            nextvalues = 0  # Does not matter
        else:
            nextnonterminal = 1.0
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = (
            delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        )
    return advantages


@ray.remote
class A2CLearner:
    def __init__(
        self,
        model_name: str,
        episode_buffer: EpisodeBuffer,
        model_pool: ModelPool,
        algorithm: BaseAlgo,
        batch_size: int,
        mini_batch_size: int,
        infer_mini_batch_size: int,
        max_generation_len: int,
        normalize_adv: bool = False,
        learning_rate: float = 1e-5,
        critic_learning_rate: float = 1e-5,
        grad_clip: float = 1.0,
        batch_delay_buffer: float = 1.5,
        lora_cfg: Dict[str, Any] = {},
        initial_lora_path: Optional[str] = None,
        ckpt_root: str = "checkpoints",
        save_every: int = 1,
        tracker=None,
        activation_checkpointing: bool = True,
        gradient_checkpointing: bool = True,
        use_trainer_cache: bool = False,
        max_train_len: Optional[int] = None,
    ):
        self.logger = setup_logger(
            "learner", ray.get(tracker.get_log_dir.remote())
        )  # set up logging
        self.episode_buffer, self.model_pool, self.tracker = (
            episode_buffer,
            model_pool,
            tracker,
        )
        (
            self.batch_size,
            self.mini_batch_size,
            self.infer_mini_batch_size,
            self.grad_clip,
        ) = (batch_size, mini_batch_size, infer_mini_batch_size, grad_clip)
        self.algorithm = algorithm
        self.batch_delay_buffer = batch_delay_buffer
        self.save_every = save_every
        self.normalize_adv = normalize_adv
        self.ckpt_root = pathlib.Path(
            ray.get(self.tracker.get_checkpoints_dir.remote())
            if self.tracker
            else ckpt_root
        )
        self.ckpt_root.mkdir(parents=True, exist_ok=True)

        torch.set_float32_matmul_precision("high")
        torch.set_default_dtype(torch.bfloat16)

        gpu_ids = ray.get_gpu_ids()
        self.device = (
            torch.device(f"cuda:{gpu_ids[0]}") if gpu_ids else torch.device("cpu")
        )
        self.model, self.tokenizer = build_peft_model(
            model_name, self.device, lora_cfg, initial_lora_path
        )
        self.model.to(torch.bfloat16)
        self.critic, _ = build_peft_model(
            model_name, self.device, lora_cfg, initial_lora_path, critic_model=True
        )

        if not use_trainer_cache:
            self.model.config.use_cache = False
            self.critic.config.use_cache = False
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()  # gradient checkpointing
            self.critic.gradient_checkpointing_enable()  # gradient checkpointing
        if activation_checkpointing:
            enable_full_activation_ckpt(self.model)
            enable_full_activation_ckpt(self.critic)

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate
        )
        self.critic_optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.critic.parameters()),
            lr=critic_learning_rate,
        )

        self.algorithm.initialize(
            model=(self.model, self.critic),
            tokenizer=self.tokenizer,
            device=self.device,
            max_train_len=max_train_len,
            max_generation_len=max_generation_len,
        )
        self._step = 0
        self._samples_seen = 0  # training counters

    def train(self, iterations: int):
        self.logger.info("Learner starting training loop …")
        while self._step < iterations:
            while (
                ray.get(self.episode_buffer.size.remote())
                < self.batch_size * self.batch_delay_buffer
            ):
                time.sleep(0.2)
            self.logger.info("Starting learning step")

            try:
                batch_episodes: List = ray.get(
                    self.episode_buffer.get_batch.remote(self.batch_size)
                )
            except Exception as exc:
                self.logger.exception(
                    f"could not fetch batch (step={self._step}) -\n\n{exc}\n\n"
                )
                raise

            all_samples = tree.flatten(batch_episodes)
            num_samples = len(all_samples)

            # construct value targets
            all_values = []
            for i in range(0, len(all_samples), self.infer_mini_batch_size):
                batch_steps = all_samples[i : i + self.infer_mini_batch_size]
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    _, state_enc, _, _, _, _, _ = self.algorithm.prepare_batch(
                        batch_steps
                    )
                    with torch.no_grad():
                        # take the first token's output
                        target_values = self.critic(**state_enc)[:, 0]
                        all_values.append(target_values)
            all_values = torch.cat(all_values).float().cpu()
            ep_values = torch.split(all_values, [len(ep) for ep in batch_episodes])
            ep_rewards = [
                torch.tensor([step.reward for step in ep]) for ep in batch_episodes
            ]
            ep_advantages = []
            ep_returns = []
            for rewards, values in zip(ep_rewards, ep_values):
                adv = compute_gae(rewards, values)
                ep_advantages.append(adv)
                ep_returns.append(adv + values)

            batch = []
            for i, ep in enumerate(batch_episodes):
                for j, step in enumerate(ep):
                    step = replace(step, reward=ep_advantages[i][j].item())
                    step = replace(
                        step,
                        step_info={
                            **step.step_info,
                            "return": ep_returns[i][j].item(),
                            "advantage": ep_advantages[i][j].item(),
                        },
                    )
                    batch.append(step)
            assert len(batch) >= self.batch_size

            self.optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)

            metrics_acc: Dict[str, float] = {}

            batch = random.sample(batch, self.batch_size)
            num_steps = self.batch_size // self.mini_batch_size
            print(
                f"Got {num_samples} many samples. Running for {num_steps} steps (i.e. mini batch size: {self.mini_batch_size})"
            )

            if self.normalize_adv:
                batch = NormalizeRewardsByEnv(True)(batch)

            for i in range(num_steps):
                sub = batch[i * self.mini_batch_size : (i + 1) * self.mini_batch_size]
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    update_metrics = self.algorithm.update(sub, scaling=num_steps)
                for k, v in update_metrics.items():
                    metrics_acc[k] = metrics_acc.get(k, 0.0) + v
                self.logger.info(f"Mini-step metrics: {update_metrics}")
            self.logger.info(f"Step metrics: {metrics_acc}")

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)

            try:
                self.optimizer.step()
                self.critic_optimizer.step()
            except Exception as exc:
                self.logger.exception(
                    f"optimizer.step crashed on step {self._step} -\n\n{exc}\n\n"
                )
                raise
            self._step += 1

            log = {f"{k}": v / num_steps for k, v in metrics_acc.items()}
            grad_norm = (
                sum(
                    p.grad.data.norm(2).item() ** 2
                    for p in self.model.parameters()
                    if p.grad is not None
                )
                ** 0.5
            )
            critic_grad_norm = (
                sum(
                    p.grad.data.norm(2).item() ** 2
                    for p in self.critic.parameters()
                    if p.grad is not None
                )
                ** 0.5
            )
            log.update(
                {
                    "step": self._step,
                    "samples_seen": self._samples_seen,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "grad_norm": grad_norm,
                    "critic_lr": self.critic_optimizer.param_groups[0]["lr"],
                    "grad_norm": critic_grad_norm,
                }
            )
            self.tracker.log_learner.remote(log)
            self._samples_seen += self.batch_size

            # save & register checkpoint every step
            if self._step % self.save_every == 0:
                try:
                    self._save_checkpoint()
                except Exception as exc:
                    self.logger.exception(
                        f"failed to save checkpoint {ckpt_dir} -\n\n{exc}\n\n"
                    )
                    raise
                if self.model_pool and self._last_ckpt:
                    self.model_pool.add_checkpoint.remote(
                        str(self._last_ckpt), self._step
                    )
                    self.logger.info(f"[Learner] +registered -> {self._last_ckpt}")
                    self.model_pool.snapshot.remote(self._step)

        self.logger.info("[Learner] training finished.")
        self.episode_buffer.stop.remote()

    def _save_checkpoint(self):
        ckpt_dir = self.ckpt_root / f"iteration-{self._step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model = self.model.module if hasattr(self.model, "module") else self.model
        model.save_pretrained(ckpt_dir, save_adapter=True)
        self._last_ckpt = ckpt_dir
        self.logger.info(f"[Learner] saved -> {ckpt_dir}")
