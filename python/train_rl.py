#!/usr/bin/env python3
"""Train DQN/DDPG-style power allocation baselines without teacher labels.

The paper compares GNN against DQN and DDPG as learning baselines.  In this
project they are implemented as one-step RL/contextual-bandit policies: the
state is the current large-scale channel/access snapshot and the reward is a
differentiable large-scale sum-rate proxy. No supervised power labels are loaded.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from dataclasses import dataclass
from typing import Iterable

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from dataset import _as_lkn
from config import MODEL_DIR, TRAINING_DATA_GLOB
from rl_models import (
    MLP,
    DDPGActor,
    DDPGCritic,
    build_state,
    fpcp_shares,
    masked_row_softmax,
    proxy_sum_rate,
)


DEFAULT_ACTION_ALPHAS = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5)


class RLDataset(Dataset):
    """Feature-only dataset for RL baselines.

    This loader intentionally ignores ``labels`` in the training file, so DQN
    and DDPG remain independent of supervised power-allocation labels.
    """

    def __init__(self, mat_path: str, L: int = 100, K: int = 20):
        self.mat_path = mat_path
        self.L = L
        self.K = K
        paths = sorted(glob.glob(mat_path)) if any(ch in mat_path for ch in "*?[]") else [mat_path]
        if not paths:
            raise FileNotFoundError(f"No training data matched: {mat_path}")

        sqrt_gain_parts = []
        d_mask_parts = []
        sigma_parts = []
        snr_parts = []
        for path in paths:
            sqrt_gain, d_mask, sigma_e, snrs = self._load_one(path, L, K)
            sqrt_gain_parts.append(sqrt_gain)
            d_mask_parts.append(d_mask)
            sigma_parts.append(sigma_e)
            snr_parts.append(snrs)

        self.sqrt_gain = np.concatenate(sqrt_gain_parts, axis=2)
        self.d_mask = np.concatenate(d_mask_parts, axis=2)
        self.sigma_e = np.concatenate(sigma_parts, axis=0)
        self.snrs = np.concatenate(snr_parts, axis=0)
        self.n_snaps = self.sqrt_gain.shape[2]

        print(f"Loaded RL feature dataset: {self.n_snaps} snapshots from {len(paths)} file(s)")
        print(f"  Shape: sqrtGain={self.sqrt_gain.shape}, D={self.d_mask.shape}")
        print(f"  SNR range: {np.unique(self.snrs)}")
        print("  Labels: not loaded; DQN/DDPG train from reward only")

    @staticmethod
    def _load_one(mat_path: str, L: int, K: int):
        with h5py.File(mat_path, "r") as f:
            features = f["features"]
            sqrt_gain = np.array(features["sqrtGain"])
            d_mask = np.array(features["D"])
            sigma_e = np.array(features["sigma_e"])

            snrs = []
            meta = f["meta"]
            meta_raw = np.array(meta)
            for i in range(meta_raw.shape[1]):
                struct = f[meta_raw[0, i]]
                snrs.append(float(np.array(struct["SNR_dB"])[0, 0]))

        sqrt_gain = _as_lkn(sqrt_gain, L, K, "features.sqrtGain").astype(np.float32)
        d_mask = (_as_lkn(d_mask, L, K, "features.D") > 0.5).astype(np.float32)
        sigma_flat = np.array(sigma_e, dtype=np.float32).flatten()
        n_snaps = sqrt_gain.shape[2]
        if sigma_flat.size == n_snaps:
            sigma_e = sigma_flat
        elif sigma_flat.size == 1:
            sigma_e = np.full(n_snaps, float(sigma_flat[0]), dtype=np.float32)
        else:
            sigma_e = np.full(n_snaps, float(sigma_flat[0]), dtype=np.float32)
        return sqrt_gain, d_mask, sigma_e, np.array(snrs, dtype=np.float32)

    def __len__(self) -> int:
        return self.n_snaps

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sqrt_gain = self.sqrt_gain[:, :, idx]
        d_mask = self.d_mask[:, :, idx]
        return {
            "sqrt_gain": torch.from_numpy(sqrt_gain),
            "D": torch.from_numpy(d_mask),
            "snr_db": torch.tensor(self.snrs[idx], dtype=torch.float32),
            "sigma_e": torch.tensor(self.sigma_e[idx], dtype=torch.float32),
        }


@dataclass
class TrainStats:
    best_val_reward: float = -math.inf
    best_epoch: int = 0


def make_loaders(dataset: Dataset, batch_size: int, val_split: float, seed: int):
    n_val = max(1, int(len(dataset) * val_split)) if len(dataset) > 1 else 0
    n_train = len(dataset) - n_val
    if n_val > 0:
        train_set, val_set = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(seed),
        )
    else:
        train_set, val_set = dataset, dataset
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def train_dqn(args, dataset: RLDataset, device: torch.device, state_dim: int):
    action_alphas = [float(x) for x in args.action_alphas]
    model = MLP(state_dim, len(action_alphas), args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    train_loader, val_loader = make_loaders(dataset, args.batch_size, args.val_split, args.seed)
    stats = TrainStats()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            state = build_state(batch["sqrt_gain"], batch["D"], batch["snr_db"], batch["sigma_e"])
            pt_batch = torch.pow(10.0, batch["snr_db"].view(-1, 1) / 10.0)
            target_rewards = []
            for alpha in action_alphas:
                shares = fpcp_shares(batch["sqrt_gain"], batch["D"], alpha)
                target_rewards.append(proxy_sum_rate(shares, batch["sqrt_gain"], batch["D"], pt_batch))
            target = torch.stack(target_rewards, dim=1)
            target = (target - target.mean(dim=1, keepdim=True)) / target.std(dim=1, keepdim=True).clamp_min(1.0e-4)
            q_pred = model(state)
            loss = nn.functional.mse_loss(q_pred, target)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_loss += float(loss.detach()) * state.size(0)

        val_reward = evaluate_dqn(model, val_loader, action_alphas, device)
        train_loss /= max(1, len(train_loader.dataset))
        print(f"[DQN] epoch {epoch:03d} loss={train_loss:.5f} val_reward={val_reward:.5f}")

        if val_reward > stats.best_val_reward:
            stats.best_val_reward = val_reward
            stats.best_epoch = epoch
            save_checkpoint(
                os.path.join(args.output_dir, "best_dqn_power.pt"),
                method="dqn",
                L=dataset.L,
                K=dataset.K,
                state_dim=state_dim,
                hidden_dim=args.hidden_dim,
                state_dict=model.state_dict(),
                action_alphas=action_alphas,
                reward_proxy="large_scale_sum_rate",
            )

    save_checkpoint(
        os.path.join(args.output_dir, "final_dqn_power.pt"),
        method="dqn",
        L=dataset.L,
        K=dataset.K,
        state_dim=state_dim,
        hidden_dim=args.hidden_dim,
        state_dict=model.state_dict(),
        action_alphas=action_alphas,
        reward_proxy="large_scale_sum_rate",
    )
    print(f"[DQN] best epoch={stats.best_epoch} val_reward={stats.best_val_reward:.5f}")


@torch.no_grad()
def evaluate_dqn(model: nn.Module, loader: Iterable, action_alphas: list[float],
                 device: torch.device) -> float:
    model.eval()
    rewards = []
    for batch in loader:
        batch = batch_to_device(batch, device)
        state = build_state(batch["sqrt_gain"], batch["D"], batch["snr_db"], batch["sigma_e"])
        pt_batch = torch.pow(10.0, batch["snr_db"].view(-1, 1) / 10.0)
        q = model(state)
        actions = torch.argmax(q, dim=1)
        batch_rewards = []
        for i, alpha in enumerate(action_alphas):
            shares = fpcp_shares(batch["sqrt_gain"], batch["D"], alpha)
            reward = proxy_sum_rate(shares, batch["sqrt_gain"], batch["D"], pt_batch)
            batch_rewards.append(torch.where(actions == i, reward, torch.zeros_like(reward)))
        rewards.append(torch.stack(batch_rewards, dim=0).sum(dim=0).detach().cpu())
    return float(torch.cat(rewards).mean()) if rewards else 0.0


def train_ddpg(args, dataset: RLDataset, device: torch.device, state_dim: int):
    actor = DDPGActor(
        state_dim,
        dataset.L,
        dataset.K,
        args.hidden_dim,
        base_alpha=args.ddpg_base_alpha,
        residual_scale=args.ddpg_residual_scale,
    ).to(device)
    critic = DDPGCritic(state_dim, dataset.L * dataset.K, args.hidden_dim).to(device)
    actor_opt = torch.optim.AdamW(actor.parameters(), lr=args.lr, weight_decay=1.0e-4)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=args.lr, weight_decay=1.0e-4)
    train_loader, val_loader = make_loaders(dataset, args.batch_size, args.val_split, args.seed)
    stats = TrainStats()

    for epoch in range(1, args.epochs + 1):
        actor.train()
        critic.train()
        critic_loss_total = 0.0
        actor_loss_total = 0.0
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            state = build_state(batch["sqrt_gain"], batch["D"], batch["snr_db"], batch["sigma_e"])
            pt_batch = torch.pow(10.0, batch["snr_db"].view(-1, 1) / 10.0)

            with torch.no_grad():
                policy_shares = actor(state, batch["D"], batch["sqrt_gain"])
                reward = proxy_sum_rate(policy_shares, batch["sqrt_gain"], batch["D"], pt_batch)
            q_pred = critic(state, policy_shares.detach())
            critic_loss = nn.functional.mse_loss(q_pred, reward)
            critic_opt.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
            critic_opt.step()

            policy_shares = actor(state, batch["D"], batch["sqrt_gain"])
            q_policy = critic(state, policy_shares)
            reward_policy = proxy_sum_rate(policy_shares, batch["sqrt_gain"], batch["D"], pt_batch)
            entropy = masked_entropy(policy_shares, batch["D"])
            anchor = fpcp_shares(batch["sqrt_gain"], batch["D"], args.ddpg_anchor_alpha)
            anchor_loss = nn.functional.mse_loss(policy_shares, anchor)
            actor_loss = (-q_policy.mean()
                          - args.reward_mix * reward_policy.mean()
                          - args.ddpg_entropy_coef * entropy
                          + args.ddpg_anchor_coef * anchor_loss)
            actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
            actor_opt.step()

            critic_loss_total += float(critic_loss.detach()) * state.size(0)
            actor_loss_total += float(actor_loss.detach()) * state.size(0)

        val_reward = evaluate_actor(actor, val_loader, device)
        n_train = max(1, len(train_loader.dataset))
        print(
            f"[DDPG] epoch {epoch:03d} critic_loss={critic_loss_total/n_train:.5f} "
            f"actor_loss={actor_loss_total/n_train:.5f} val_reward={val_reward:.5f}"
        )

        if val_reward > stats.best_val_reward:
            stats.best_val_reward = val_reward
            stats.best_epoch = epoch
            save_checkpoint(
                os.path.join(args.output_dir, "best_ddpg_power.pt"),
                method="ddpg",
                L=dataset.L,
                K=dataset.K,
                state_dim=state_dim,
                hidden_dim=args.hidden_dim,
                actor_state_dict=actor.state_dict(),
                critic_state_dict=critic.state_dict(),
                reward_proxy="large_scale_sum_rate",
                actor_type="fpcp_residual",
                ddpg_base_alpha=args.ddpg_base_alpha,
                ddpg_residual_scale=args.ddpg_residual_scale,
            )

    save_checkpoint(
        os.path.join(args.output_dir, "final_ddpg_power.pt"),
        method="ddpg",
        L=dataset.L,
        K=dataset.K,
        state_dim=state_dim,
        hidden_dim=args.hidden_dim,
        actor_state_dict=actor.state_dict(),
        critic_state_dict=critic.state_dict(),
        reward_proxy="large_scale_sum_rate",
        actor_type="fpcp_residual",
        ddpg_base_alpha=args.ddpg_base_alpha,
        ddpg_residual_scale=args.ddpg_residual_scale,
    )
    print(f"[DDPG] best epoch={stats.best_epoch} val_reward={stats.best_val_reward:.5f}")


@torch.no_grad()
def evaluate_actor(actor: DDPGActor, loader: Iterable, device: torch.device) -> float:
    actor.eval()
    rewards = []
    for batch in loader:
        batch = batch_to_device(batch, device)
        state = build_state(batch["sqrt_gain"], batch["D"], batch["snr_db"], batch["sigma_e"])
        pt_batch = torch.pow(10.0, batch["snr_db"].view(-1, 1) / 10.0)
        shares = actor(state, batch["D"], batch["sqrt_gain"])
        rewards.append(proxy_sum_rate(shares, batch["sqrt_gain"], batch["D"], pt_batch).detach().cpu())
    return float(torch.cat(rewards).mean()) if rewards else 0.0


def masked_entropy(shares: torch.Tensor, d_mask: torch.Tensor) -> torch.Tensor:
    probs = shares.clamp_min(1.0e-12)
    entropy = -(probs * torch.log(probs) * d_mask).sum(dim=2)
    active_rows = (d_mask.sum(dim=2) > 0).float()
    return (entropy * active_rows).sum() / active_rows.sum().clamp_min(1.0)


def save_checkpoint(path: str, **payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN/DDPG RL power allocation baselines")
    parser.add_argument("--data", default=TRAINING_DATA_GLOB, help="MATLAB v7.3 training dataset")
    parser.add_argument("--method", choices=["dqn", "ddpg", "all"], default="all")
    parser.add_argument("--output_dir", default=str(MODEL_DIR))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--pt", type=float, default=10.0)
    parser.add_argument("--reward_mix", type=float, default=0.2,
                        help="Direct reward term mixed into DDPG actor loss")
    parser.add_argument("--ddpg_entropy_coef", type=float, default=0.02,
                        help="Entropy regularization to avoid DDPG one-UE collapse")
    parser.add_argument("--ddpg_base_alpha", type=float, default=-1.0,
                        help="FPCP alpha used as the residual DDPG base policy")
    parser.add_argument("--ddpg_residual_scale", type=float, default=0.75,
                        help="Bound on DDPG residual logits around the FPCP base policy")
    parser.add_argument("--ddpg_anchor_alpha", type=float, default=-1.0,
                        help="FPCP alpha used as a weak DDPG stabilizing anchor")
    parser.add_argument("--ddpg_anchor_coef", type=float, default=0.05,
                        help="Strength of the weak DDPG FPCP-anchor regularizer")
    parser.add_argument("--action_alphas", type=float, nargs="+", default=list(DEFAULT_ACTION_ALPHAS),
                        help="DQN discrete FPCP alpha actions")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RLDataset(args.data, L=args.L, K=args.K)

    sample = dataset[0]
    state_dim = build_state(
        sample["sqrt_gain"].unsqueeze(0),
        sample["D"].unsqueeze(0),
        sample["snr_db"].unsqueeze(0),
        sample["sigma_e"].unsqueeze(0),
    ).shape[1]
    print(f"State dim: {state_dim}; device: {device}")

    if args.method in ("dqn", "all"):
        train_dqn(args, dataset, device, state_dim)
    if args.method in ("ddpg", "all"):
        train_ddpg(args, dataset, device, state_dim)


if __name__ == "__main__":
    main()
