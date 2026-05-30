#!/usr/bin/env python3
"""
Unsupervised training script for the paper-aligned DCGNN.

Reference:
  Zhao et al., "A Dynamic Power Allocation Approach for Downlink Cell-Free
  Massive MIMO With Graph Neural Network", IEEE TVT, Vol. 74, No. 4, Apr. 2025.

Key training parameters from the paper (Section V.A):
  - L = 100 APs, K = 40/60/80 UEs, N = 4 antennas per AP
  - 500 m × 500 m area, AP height 3 m
  - pmax = 100 mW, pilot power = 100 mW, noise = −114 dBm
  - 2-hidden-layer GNN (32 → 16), z = 15 neighbours
  - Initial learning rate 10⁻³, Adam with exponential decay
  - 5 000 training time slots, 500 testing time slots
  - Unsupervised: loss = −(1/K) Σ SE_k
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

# Ensure the local python/ package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dcgnn_paper import (
    PaperDCGNN,
    generate_training_beta,
    save_paper_dcgnn,
    unsupervised_loss,
)
from config import MODEL_DIR


# ═══════════════════════════════════════════════════════════════════════════
# Training utilities
# ═══════════════════════════════════════════════════════════════════════════


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def train_epoch(
    model: PaperDCGNN,
    optimizer: optim.Optimizer,
    scheduler,
    rng: np.random.Generator,
    batch_size: int = 1,
) -> dict[str, float]:
    """Single unsupervised training epoch.

    Each batch generates a fresh random scenario, so the model never
    sees the same data twice — matching the paper's time-slot approach.
    """
    model.train()

    beta = generate_training_beta(
        L=model.L,
        K=model.K,
        batch_size=batch_size,
        rng=rng,
    )
    beta_t = torch.from_numpy(beta).to(next(model.parameters()).device)

    optimizer.zero_grad()
    rho, se = model.forward(beta_t, return_se=True)
    loss = -se.mean()

    if torch.isnan(loss) or torch.isinf(loss):
        return {"loss": float("nan"), "mean_SE": float("nan")}

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    with torch.no_grad():
        return {
            "loss": float(loss),
            "mean_SE": float(se.mean()),
            "max_SE": float(se.max()),
            "min_SE": float(se.min()),
            "mean_rho": float(rho.mean()),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }


@torch.no_grad()
def evaluate(
    model: PaperDCGNN,
    rng: np.random.Generator,
    num_slots: int = 500,
    batch_size: int = 1,
) -> dict[str, float]:
    """Evaluate over `num_slots` independent scenarios."""
    model.eval()
    se_accum = 0.0
    se_list = []

    for _ in range(num_slots):
        beta = generate_training_beta(
            L=model.L,
            K=model.K,
            batch_size=batch_size,
            rng=rng,
        )
        beta_t = torch.from_numpy(beta).to(next(model.parameters()).device)
        rho, se = model.forward(beta_t, return_se=True)
        se_accum += float(se.mean())
        se_list.append(float(se.mean()))

    se_arr = np.array(se_list)
    return {
        "mean_SE": float(np.mean(se_arr)),
        "median_SE": float(np.median(se_arr)),
        "std_SE": float(np.std(se_arr)),
        "max_SE": float(np.max(se_arr)),
        "min_SE": float(np.min(se_arr)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train paper-aligned DCGNN (unsupervised, IEEE TVT 2025)"
    )
    p.add_argument("--L", type=int, default=100, help="Number of APs")
    p.add_argument("--K", type=int, default=40, help="Number of UEs")
    p.add_argument("--z", type=int, default=15, help="Neighbours per node")
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[32, 16],
                   help="Hidden GNN layer dimensions")
    p.add_argument("--pmax", type=float, default=100.0, help="Per-AP power limit [mW]")
    p.add_argument("--noise-dbm", type=float, default=-114.0,
                   help="Noise power [dBm]")
    p.add_argument("--epochs", type=int, default=5000, help="Training time slots")
    p.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    p.add_argument("--lr-decay", type=float, default=0.9995,
                   help="Exponential LR decay per slot")
    p.add_argument("--eval-interval", type=int, default=250,
                   help="Evaluate every N slots")
    p.add_argument("--eval-slots", type=int, default=500,
                   help="Evaluation scenarios")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=str(MODEL_DIR))
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--square-length", type=float, default=500.0,
                   help="Coverage area side length [m]")
    p.add_argument("--ap-height", type=float, default=3.0,
                   help="AP height [m]")
    p.add_argument("--sigma-sf", type=float, default=8.0,
                   help="Shadow fading std [dB]")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # ── Build model ──
    model = PaperDCGNN(
        L=args.L,
        K=args.K,
        z=args.z,
        hidden_dims=list(args.hidden_dims),
        pmax=args.pmax,
        tau_c=200,
        tau_p=10,
        sigma2=1.0,  # beta is already "over noise"
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"  Paper DCGNN -- Unsupervised Training")
    print(f"{'='*60}")
    print(f"  L={args.L}  K={args.K}  N=4 (paper)  z={args.z}")
    print(f"  GNN layers: {args.hidden_dims}")
    print(f"  Parameters: {n_params:,}")
    print(f"  pmax={args.pmax} mW  noise={args.noise_dbm} dBm")
    print(f"  Area: {args.square_length}×{args.square_length} m  "
          f"AP height: {args.ap_height} m")
    print(f"  Shadow σ: {args.sigma_sf} dB")
    print(f"  Training slots: {args.epochs}  lr0={args.lr}  "
          f"decay={args.lr_decay}")
    print(f"{'='*60}\n")

    # ── Optimiser (Adam with exponential decay, as in paper) ──
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    best_se = -float("inf")
    best_epoch = 0
    eval_se_history: list[float] = []

    # ── Training loop ──
    t_start = time.perf_counter()
    for slot in range(1, args.epochs + 1):
        stats = train_epoch(model, optimizer, scheduler, rng)

        if np.isnan(stats["loss"]):
            print(f"  [slot {slot:5d}]  LOSS=NaN — stopping")
            break

        if slot % 100 == 0 or slot == 1:
            print(
                f"  [slot {slot:5d}/{args.epochs}]  "
                f"loss={_fmt(stats['loss'])}  "
                f"mean_SE={_fmt(stats['mean_SE'])}  "
                f"rho_mean={_fmt(stats['mean_rho'])}  "
                f"lr={stats['lr']:.2e}"
            )

        # ── Evaluation ──
        if slot % args.eval_interval == 0 or slot == args.epochs:
            eval_stats = evaluate(model, rng, num_slots=args.eval_slots)
            eval_se_history.append(eval_stats["mean_SE"])
            print(
                f"\n  ── Eval @ slot {slot} ──\n"
                f"    mean_SE={_fmt(eval_stats['mean_SE'])}  "
                f"median={_fmt(eval_stats['median_SE'])}  "
                f"std={_fmt(eval_stats['std_SE'])}  "
                f"[{_fmt(eval_stats['min_SE'])}, {_fmt(eval_stats['max_SE'])}]\n"
            )

            if eval_stats["mean_SE"] > best_se:
                best_se = eval_stats["mean_SE"]
                best_epoch = slot
                output_path = Path(args.output_dir) / "best_paper_dcgnn.pt"
                save_paper_dcgnn(
                    model,
                    output_path,
                    epoch=slot,
                    best_val_se=best_se,
                    args=vars(args),
                )
                print(f"    → best model saved to {output_path}")

    elapsed = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"  Training complete in {elapsed:.0f}s ({args.epochs} slots)")
    print(f"  Best val SE = {_fmt(best_se)} @ slot {best_epoch}")
    print(f"{'='*60}")

    # ── Final save ──
    final_path = Path(args.output_dir) / "final_paper_dcgnn.pt"
    save_paper_dcgnn(model, final_path, epoch=args.epochs, args=vars(args))
    print(f"Final model saved to {final_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
