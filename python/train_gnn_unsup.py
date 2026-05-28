#!/usr/bin/env python3
"""Train an unsupervised GNN power allocator without WMMSE labels.

U-GNN uses the existing AP-UE graph features, but its loss is a differentiable
large-scale sum-rate proxy. The saved checkpoint is compatible with
``gnn_runtime.py`` because the model still emits per-AP share logits.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split

from dataset import GNNDataset
from project_paths import MODEL_DIR, TRAINING_DATA_GLOB
from train_gnn import PowerGNN_GAT, custom_collate


EPS = 1.0e-8
MAX_LOGIT = 30.0
MAX_SQRT_GAIN = 1.0e6
MAX_PT = 1.0e6


def _graph_sqrt_gain(batch, L: int, K: int) -> torch.Tensor:
    """Recover masked sqrt-gain AP features as (B, L, K)."""

    nodes_per_graph = L + K
    node_pos = torch.arange(batch.x.size(0), device=batch.x.device) % nodes_per_graph
    is_ap = node_pos < L
    sqrt_gain = batch.x[is_ap, :K].view(batch.num_graphs, L, K)
    return torch.nan_to_num(sqrt_gain, nan=0.0, posinf=MAX_SQRT_GAIN, neginf=0.0).clamp(0.0, MAX_SQRT_GAIN)


def shares_from_logits(logits: torch.Tensor, d_mask: torch.Tensor, temperature: float) -> torch.Tensor:
    valid_rows = d_mask.sum(dim=2, keepdim=True) > 0
    logits = torch.nan_to_num(logits, nan=0.0, posinf=MAX_LOGIT, neginf=-MAX_LOGIT).clamp(-MAX_LOGIT, MAX_LOGIT)
    scaled_logits = logits / max(float(temperature), 1.0e-6)
    masked_logits = scaled_logits.masked_fill(d_mask <= 0.5, -1.0e9)
    share = torch.softmax(masked_logits, dim=2) * d_mask.float()
    row_sum = share.sum(dim=2, keepdim=True)
    return torch.where(valid_rows, share / row_sum.clamp_min(1.0e-12), share)


def fpcp_shares(sqrt_gain: torch.Tensor, d_mask: torch.Tensor, alpha: float) -> torch.Tensor:
    """Fractional power-control shares with the MATLAB FPCP convention."""

    sqrt_gain = torch.nan_to_num(sqrt_gain, nan=0.0, posinf=MAX_SQRT_GAIN, neginf=0.0).clamp(0.0, MAX_SQRT_GAIN)
    gain = sqrt_gain.square().clamp_min(1.0e-12)
    weights = torch.pow(gain, -float(alpha)) * d_mask
    row_sum = weights.sum(dim=2, keepdim=True)
    served = d_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
    epa = d_mask / served
    return torch.where(row_sum > 0, weights / row_sum.clamp_min(1.0e-12), epa)


def _renormalize_shares(shares: torch.Tensor, d_mask: torch.Tensor) -> torch.Tensor:
    valid_rows = d_mask.sum(dim=2, keepdim=True) > 0
    shares = torch.nan_to_num(shares, nan=0.0, posinf=1.0, neginf=0.0).clamp_min(0.0) * d_mask.float()
    row_sum = shares.sum(dim=2, keepdim=True)
    served = d_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
    epa = d_mask / served
    return torch.where(valid_rows & (row_sum > 0), shares / row_sum.clamp_min(1.0e-12), epa)


def policy_from_logits(
    logits: torch.Tensor,
    d_mask: torch.Tensor,
    sqrt_gain: torch.Tensor,
    args,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Convert model output to per-AP shares.

    ``share_logits`` is the legacy pure neural policy. ``share_logits_mix`` adds
    one AP-level gate and learns a residual policy around an analytic FPCP prior:
    share = lambda * share_gnn + (1 - lambda) * share_fpcp. The prior is a
    differentiable training scaffold, not an inference-time hard fallback.
    """

    K = d_mask.size(2)
    raw_logits = logits[:, :, :K]
    nn_shares = shares_from_logits(raw_logits, d_mask, args.share_temperature)
    aux = {"nn_entropy": share_entropy(nn_shares, d_mask).detach()}

    if logits.size(2) <= K:
        return nn_shares, aux

    prior = fpcp_shares(sqrt_gain, d_mask, getattr(args, "mixture_alpha", -1.0))
    gate = torch.sigmoid(logits[:, :, K:K + 1])
    gate_min = float(getattr(args, "mix_lambda_min", 0.05))
    gate_max = float(getattr(args, "mix_lambda_max", 0.95))
    gate = gate_min + (gate_max - gate_min) * gate
    shares = _renormalize_shares(gate * nn_shares + (1.0 - gate) * prior, d_mask)

    active = (d_mask.sum(dim=2, keepdim=True) > 0).float()
    aux["mix_lambda_tensor"] = gate
    aux["mix_lambda"] = ((gate * active).sum() / active.sum().clamp_min(1.0)).detach()
    return shares, aux


def policy_from_batch(logits: torch.Tensor, batch, args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    B, L = logits.size(0), logits.size(1)
    d_mask = batch.D_mask.view(B, L, -1).float()
    sqrt_gain = _graph_sqrt_gain(batch, L, d_mask.size(2))
    shares, aux = policy_from_logits(logits, d_mask, sqrt_gain, args)
    return shares, d_mask, sqrt_gain, aux


def equivalent_sum_rate(shares: torch.Tensor, sqrt_gain: torch.Tensor,
                        d_mask: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
    """Differentiable equivalent-channel sum-rate proxy.

    The exact MATLAB simulation remains the final evaluator. This proxy gives
    U-GNN a teacher-free training signal while matching the deterministic
    equivalent-rate form used by ``computeRhoWMMSE.m``:
    ``SINR_k = |diag(H_eff V)|^2 / (sum_i |H_eff V_i|^2 - desired + 1)``.
    """

    snr_db = torch.nan_to_num(snr_db.float(), nan=0.0, posinf=60.0, neginf=-60.0).clamp(-60.0, 60.0)
    pt = torch.pow(torch.tensor(10.0, device=shares.device), snr_db.view(-1, 1, 1) / 10.0).clamp_max(MAX_PT)
    rho = torch.nan_to_num(pt * shares * d_mask, nan=0.0, posinf=MAX_PT, neginf=0.0).clamp_min(0.0)
    v = torch.sqrt(rho + EPS) * d_mask
    h_eff = sqrt_gain.transpose(1, 2).clamp_min(0.0)
    hv = torch.nan_to_num(torch.bmm(h_eff, v), nan=0.0, posinf=1.0e12, neginf=-1.0e12).clamp(-1.0e12, 1.0e12)
    desired = torch.diagonal(hv, dim1=1, dim2=2).square().clamp_max(1.0e24)
    total_rx = hv.square().sum(dim=2).clamp_max(1.0e24)
    interference = (total_rx - desired).clamp_min(0.0)
    sinr = desired / (interference + 1.0)
    rate = torch.log2(1.0 + sinr.clamp_min(0.0)).sum(dim=1)
    return torch.nan_to_num(rate, nan=0.0, posinf=1.0e6, neginf=0.0)


def share_entropy(shares: torch.Tensor, d_mask: torch.Tensor) -> torch.Tensor:
    entropy = -(shares.clamp_min(1.0e-12) * torch.log(shares.clamp_min(1.0e-12))).sum(dim=2)
    served = d_mask.sum(dim=2).clamp_min(1.0)
    norm = torch.log(served.clamp_min(2.0))
    normalized = torch.where(served > 1.0, entropy / norm.clamp_min(1.0e-12), torch.ones_like(entropy))
    active = (served > 1.0).float()
    return (normalized * active).sum(dim=1) / active.sum(dim=1).clamp_min(1.0)


def best_baseline(sqrt_gain: torch.Tensor, d_mask: torch.Tensor, snr_db: torch.Tensor,
                  alphas: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
    baseline_shares = []
    baseline_rates = []
    for alpha in alphas:
        shares = fpcp_shares(sqrt_gain, d_mask, alpha)
        baseline_shares.append(shares)
        baseline_rates.append(equivalent_sum_rate(shares, sqrt_gain, d_mask, snr_db))

    rate_stack = torch.stack(baseline_rates, dim=1)
    share_stack = torch.stack(baseline_shares, dim=1)
    best_rate, best_idx = rate_stack.max(dim=1)
    gather_idx = best_idx.view(-1, 1, 1, 1).expand(-1, 1, d_mask.size(1), d_mask.size(2))
    best_share = share_stack.gather(1, gather_idx).squeeze(1)
    return best_rate.detach(), best_share.detach()


def unsup_loss(logits: torch.Tensor, batch, args) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    shares, d_mask, sqrt_gain, policy_aux = policy_from_batch(logits, batch, args)
    snr_db = batch.snr.view(-1).float()

    rate = equivalent_sum_rate(shares, sqrt_gain, d_mask, snr_db)
    loss = -rate.mean()
    diagnostics = {"rate": rate.detach()}
    diagnostics.update({k: v for k, v in policy_aux.items() if k != "mix_lambda_tensor"})

    baseline_rate, baseline_share = best_baseline(sqrt_gain, d_mask, snr_db, args.baseline_alphas)
    advantage = rate - baseline_rate
    diagnostics["baseline_rate"] = baseline_rate
    diagnostics["advantage"] = advantage.detach()

    if args.baseline_guard_weight > 0:
        guard = F.relu(args.baseline_margin - advantage).mean()
        loss = loss + args.baseline_guard_weight * guard
        diagnostics["guard"] = guard.detach()

    if args.anchor_weight > 0:
        underperform = F.relu(args.baseline_margin - advantage).detach().view(-1, 1, 1)
        share_mask = d_mask > 0.5
        anchor_per_entry = (shares - baseline_share).square() * underperform
        anchor = anchor_per_entry[share_mask].mean() if torch.any(share_mask) else logits.sum() * 0.0
        loss = loss + args.anchor_weight * anchor
        diagnostics["anchor"] = anchor.detach()

    if "mix_lambda_tensor" in policy_aux and args.mix_lambda_floor_weight > 0:
        active = d_mask.sum(dim=2, keepdim=True) > 0
        mix_floor = F.relu(args.mix_lambda_floor - policy_aux["mix_lambda_tensor"][active]).square().mean()
        loss = loss + args.mix_lambda_floor_weight * mix_floor
        diagnostics["mix_floor"] = mix_floor.detach()

    if args.entropy_weight > 0:
        entropy = share_entropy(shares, d_mask)
        entropy_floor_loss = F.relu(args.entropy_floor - entropy).square().mean()
        loss = loss + args.entropy_weight * entropy_floor_loss
        diagnostics["entropy"] = entropy.detach()
        diagnostics["entropy_floor"] = entropy_floor_loss.detach()

    if args.fairness_weight > 0:
        pt = torch.pow(torch.tensor(10.0, device=shares.device), snr_db.view(-1, 1, 1) / 10.0)
        pt = pt.clamp_max(MAX_PT)
        rho = torch.nan_to_num(pt * shares * d_mask, nan=0.0, posinf=MAX_PT, neginf=0.0).clamp_min(0.0)
        v = torch.sqrt(rho + EPS) * d_mask
        h_eff = sqrt_gain.transpose(1, 2).clamp_min(0.0)
        hv = torch.nan_to_num(torch.bmm(h_eff, v), nan=0.0, posinf=1.0e12, neginf=-1.0e12).clamp(-1.0e12, 1.0e12)
        desired = torch.diagonal(hv, dim1=1, dim2=2).square().clamp_max(1.0e24)
        total_rx = hv.square().sum(dim=2).clamp_max(1.0e24)
        sinr = desired / ((total_rx - desired).clamp_min(0.0) + 1.0)
        se = torch.nan_to_num(torch.log2(1.0 + sinr.clamp_min(0.0)), nan=0.0, posinf=1.0e6, neginf=0.0)
        fairness = se.var(dim=1, unbiased=False).mean()
        loss = loss + args.fairness_weight * fairness
        diagnostics["fairness"] = fairness.detach()

    diagnostics["loss"] = loss.detach()
    return loss, diagnostics


def distillation_loss(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    d_mask: torch.Tensor,
    student_temperature: float,
    teacher_temperature: float,
) -> torch.Tensor:
    K = d_mask.size(2)
    student_share = shares_from_logits(logits[:, :, :K], d_mask, student_temperature)
    teacher_share = shares_from_logits(teacher_logits.detach()[:, :, :K], d_mask, teacher_temperature)
    share_mask = d_mask > 0.5
    if not torch.any(share_mask):
        return logits.sum() * 0.0
    mse = F.mse_loss(student_share[share_mask], teacher_share[share_mask], reduction="mean")
    ce = -(teacher_share[share_mask] * torch.log(student_share[share_mask].clamp_min(1.0e-12))).mean()
    return mse + 0.05 * ce


def perturb_batch_features(batch, noise_std: float):
    noisy = batch.clone()
    if noise_std <= 0:
        return noisy

    L = int(noisy.num_ap_nodes[0] if hasattr(noisy.num_ap_nodes, "__len__") else noisy.num_ap_nodes)
    K = int(noisy.num_ue_nodes[0] if hasattr(noisy.num_ue_nodes, "__len__") else noisy.num_ue_nodes)
    nodes_per_graph = L + K
    node_pos = torch.arange(noisy.x.size(0), device=noisy.x.device) % nodes_per_graph
    is_ap = node_pos < L

    ap_noise = torch.exp(torch.randn_like(noisy.x[is_ap, :K]) * noise_std).clamp(0.5, 2.0)
    ue_noise = torch.exp(torch.randn_like(noisy.x[~is_ap, :L]) * noise_std).clamp(0.5, 2.0)
    noisy.x[is_ap, :K] = (noisy.x[is_ap, :K] * ap_noise).clamp(0.0, MAX_SQRT_GAIN)
    noisy.x[~is_ap, :L] = (noisy.x[~is_ap, :L] * ue_noise).clamp(0.0, MAX_SQRT_GAIN)
    return noisy


def consistency_loss(model, logits: torch.Tensor, batch, args) -> torch.Tensor:
    if args.consistency_weight <= 0:
        return logits.sum() * 0.0

    base_shares, d_mask, _, _ = policy_from_batch(logits, batch, args)
    noisy_batch = perturb_batch_features(batch, args.consistency_noise_std)
    noisy_logits = model(noisy_batch)
    noisy_shares, _, _, _ = policy_from_batch(noisy_logits, noisy_batch, args)
    share_mask = d_mask > 0.5
    if not torch.any(share_mask):
        return logits.sum() * 0.0
    return F.mse_loss(noisy_shares[share_mask], base_shares.detach()[share_mask], reduction="mean")


def _empty_metrics() -> dict[str, float]:
    return {"loss": 0.0, "rate": 0.0, "baseline_rate": 0.0, "advantage": 0.0}


def _accumulate_metrics(total: dict[str, float], diagnostics: dict[str, torch.Tensor], bs: int):
    for key, value in diagnostics.items():
        if key not in total:
            total[key] = 0.0
        total[key] += float(value.mean().item()) * bs


def _finalize_metrics(total: dict[str, float], n_samples: int) -> dict[str, float]:
    return {key: value / max(n_samples, 1) for key, value in total.items()}


def gradients_are_finite(model: torch.nn.Module) -> bool:
    for param in model.parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            return False
    return True


def train_epoch(model, teacher_model, loader, optimizer, scheduler, device, args):
    model.train()
    totals = _empty_metrics()
    n_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss, diagnostics = unsup_loss(logits, batch, args)
        if args.consistency_weight > 0:
            consistency = consistency_loss(model, logits, batch, args)
            loss = loss + args.consistency_weight * consistency
            diagnostics["consistency"] = consistency.detach()
            diagnostics["loss"] = loss.detach()
        if teacher_model is not None and args.distill_weight > 0:
            with torch.no_grad():
                teacher_logits = teacher_model(batch)
            d_mask = batch.D_mask.view(logits.size(0), logits.size(1), -1).float()
            distill = distillation_loss(
                logits,
                teacher_logits,
                d_mask,
                args.share_temperature,
                args.teacher_temperature,
            )
            loss = loss + args.distill_weight * distill
            diagnostics["distill"] = distill.detach()
            diagnostics["loss"] = loss.detach()
        if not torch.isfinite(loss):
            print("Warning: non-finite U-GNN loss, skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not gradients_are_finite(model):
            print("Warning: non-finite U-GNN gradient, skipping optimizer step")
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.step()
        scheduler.step()

        bs = logits.size(0)
        _accumulate_metrics(totals, diagnostics, bs)
        n_samples += bs

    return _finalize_metrics(totals, n_samples)


@torch.no_grad()
def evaluate(model, teacher_model, loader, device, args):
    model.eval()
    totals = _empty_metrics()
    n_samples = 0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        loss, diagnostics = unsup_loss(logits, batch, args)
        if teacher_model is not None and args.distill_weight > 0:
            teacher_logits = teacher_model(batch)
            d_mask = batch.D_mask.view(logits.size(0), logits.size(1), -1).float()
            distill = distillation_loss(
                logits,
                teacher_logits,
                d_mask,
                args.share_temperature,
                args.teacher_temperature,
            )
            loss = loss + args.distill_weight * distill
            diagnostics["distill"] = distill.detach()
            diagnostics["loss"] = loss.detach()
        bs = logits.size(0)
        _accumulate_metrics(totals, diagnostics, bs)
        n_samples += bs

    return _finalize_metrics(totals, n_samples)


def _checkpoint_arg(checkpoint, name, default):
    if not isinstance(checkpoint, dict):
        return default
    if name in checkpoint:
        return checkpoint[name]
    args = checkpoint.get("args")
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


def _load_state_dict(path: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    return checkpoint, state


def try_load_gnn_checkpoint(model: PowerGNN_GAT, path: str, device: torch.device, role: str,
                            allow_partial: bool = False) -> bool:
    if not path:
        return False
    if not os.path.isfile(path):
        print(f"{role} checkpoint not found: {path}")
        return False
    try:
        checkpoint, state = _load_state_dict(path)
        if allow_partial:
            current = model.state_dict()
            compatible = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
            skipped = sorted(set(state.keys()) - set(compatible.keys()))
            model.load_state_dict(compatible, strict=False)
            if skipped:
                print(f"  partial load skipped {len(skipped)} incompatible tensor(s)")
        else:
            model.load_state_dict(state, strict=True)
        model.to(device)
        print(f"Loaded {role} checkpoint: {path}")
        if isinstance(checkpoint, dict):
            print(
                f"  checkpoint model_type={checkpoint.get('model_type', 'unknown')}, "
                f"output_kind={checkpoint.get('output_kind', 'unknown')}"
            )
        return True
    except Exception as exc:
        print(f"WARNING: failed to load {role} checkpoint {path}: {exc}")
        return False


def resolve_optional_path(path: str) -> str:
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath(path),
        os.path.abspath(os.path.join(script_dir, path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[-1]


def load_dataset(pattern: str):
    data_files = sorted(glob.glob(pattern))
    if not data_files:
        print(f"ERROR: No files matched pattern: {pattern}")
        sys.exit(1)
    print(f"Found {len(data_files)} data file(s):")
    for path in data_files:
        print(f"  - {path}")
    if len(data_files) == 1:
        return GNNDataset(data_files[0])
    parts = [GNNDataset(path) for path in data_files]
    return ConcatDataset(parts), parts[0].L, parts[0].K


def main():
    parser = argparse.ArgumentParser(description="Train U-GNN without WMMSE labels")
    parser.add_argument("--data", type=str, default=TRAINING_DATA_GLOB)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_max", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--fairness_weight", type=float, default=0.01)
    parser.add_argument("--baseline_guard_weight", type=float, default=1.0)
    parser.add_argument("--baseline_margin", type=float, default=0.02)
    parser.add_argument("--anchor_weight", type=float, default=0.2)
    parser.add_argument("--entropy_weight", type=float, default=0.30)
    parser.add_argument("--entropy_floor", type=float, default=0.45)
    parser.add_argument("--share_temperature", type=float, default=1.25)
    parser.add_argument("--output_kind", type=str, default="share_logits_mix",
                        choices=["share_logits", "share_logits_mix"])
    parser.add_argument("--mixture_alpha", type=float, default=-1.0)
    parser.add_argument("--mix_lambda_min", type=float, default=0.05)
    parser.add_argument("--mix_lambda_max", type=float, default=0.95)
    parser.add_argument("--mix_lambda_floor", type=float, default=0.20)
    parser.add_argument("--mix_lambda_floor_weight", type=float, default=0.05)
    parser.add_argument("--consistency_weight", type=float, default=0.02)
    parser.add_argument("--consistency_noise_std", type=float, default=0.03)
    parser.add_argument("--baseline_alphas", type=str, default="-1.0,-0.5,0.0,0.5,1.0")
    parser.add_argument("--init_model", type=str, default="")
    parser.add_argument("--teacher_model", type=str, default="")
    parser.add_argument("--disable_gnn_init", action="store_true")
    parser.add_argument("--pure_unsup", action="store_true")
    parser.add_argument("--distill_weight", type=float, default=0.0)
    parser.add_argument("--teacher_temperature", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default=str(MODEL_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Optional cap for smoke tests; 0 uses the full dataset")
    args = parser.parse_args()
    if args.pure_unsup:
        args.disable_gnn_init = True
        args.init_model = ""
        args.teacher_model = ""
        args.distill_weight = 0.0
    args.baseline_alphas = [float(x.strip()) for x in args.baseline_alphas.split(",") if x.strip()]
    if not args.baseline_alphas:
        args.baseline_alphas = [-1.0, 0.0, 1.0]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Training objective: U-GNN-v2 equivalent-rate proxy with differentiable FPCP safety prior")
    print(f"Baseline alphas: {args.baseline_alphas}")
    print(f"Output kind: {args.output_kind}, distill_weight={args.distill_weight}, init_model='{args.init_model}'")

    loaded = load_dataset(args.data)
    if isinstance(loaded, tuple):
        dataset, L, K = loaded
    else:
        dataset = loaded
        L, K = dataset.L, dataset.K
    if args.max_samples and args.max_samples > 0 and args.max_samples < len(dataset):
        dataset = Subset(dataset, list(range(args.max_samples)))
    print(f"Total dataset size: {len(dataset)}")

    val_size = max(1, int(len(dataset) * args.val_split)) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    if val_size > 0:
        train_set, val_set = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        train_set, val_set = dataset, dataset

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=0,
    )

    model = PowerGNN_GAT(
        L=L,
        K=K,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        output_scale=1.0,
        output_kind=args.output_kind,
    ).to(device)
    print(f"Creating U-GNN: L={L}, K={K}, params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    init_model_path = "" if args.disable_gnn_init else resolve_optional_path(args.init_model)
    if init_model_path:
        try_load_gnn_checkpoint(model, init_model_path, device, "initial GNN", allow_partial=True)

    teacher_model = None
    teacher_model_path = resolve_optional_path(args.teacher_model) if args.teacher_model else init_model_path
    if args.distill_weight > 0 and teacher_model_path:
        teacher_model = PowerGNN_GAT(
            L=L,
            K=K,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=0.0,
            output_scale=1.0,
            output_kind="share_logits",
        ).to(device)
        if try_load_gnn_checkpoint(teacher_model, teacher_model_path, device, "teacher GNN"):
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad_(False)
        else:
            teacher_model = None

    steps_per_epoch = max(1, math.ceil(len(train_set) / args.batch_size))
    total_steps = max(1, args.epochs * steps_per_epoch)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr_max / 10.0, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr_max,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10,
        final_div_factor=100,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_state = None
    best_epoch = 0
    best_val_reward = -float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, teacher_model, train_loader, optimizer, scheduler, device, args)
        val_metrics = evaluate(model, teacher_model, val_loader, device, args)
        val_reward = val_metrics["rate"]

        if epoch == 0 or (epoch + 1) % 5 == 0:
            distill_msg = ""
            if "distill" in val_metrics:
                distill_msg = f", distill {val_metrics['distill']:.5f}"
            print(
                f"Epoch {epoch + 1}/{args.epochs} - "
                f"Train loss {train_metrics['loss']:.4f}, rate {train_metrics['rate']:.4f}, "
                f"adv {train_metrics['advantage']:.4f} - "
                f"Val loss {val_metrics['loss']:.4f}, rate {val_metrics['rate']:.4f}, "
                f"baseline {val_metrics['baseline_rate']:.4f}, adv {val_metrics['advantage']:.4f}"
                f"{distill_msg}"
            )

        if val_reward > best_val_reward and np.isfinite(val_reward):
            best_val_reward = val_reward
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    best_path = os.path.join(args.output_dir, "best_ugnn_power.pt")
    torch.save({
        "epoch": best_epoch,
        "model_state_dict": best_state,
        "model_type": "ugnn",
        "feature_schema": "masked_gain_snr_csi_degree_sumgain_v2",
        "training_objective": "unsupervised_large_scale_sum_rate",
        "proxy": "equivalent_channel_sum_rate_v2",
        "fairness_weight": args.fairness_weight,
        "baseline_guard_weight": args.baseline_guard_weight,
        "baseline_margin": args.baseline_margin,
        "anchor_weight": args.anchor_weight,
        "entropy_weight": args.entropy_weight,
        "entropy_floor": args.entropy_floor,
        "share_temperature": args.share_temperature,
        "mixture_alpha": args.mixture_alpha,
        "mix_lambda_min": args.mix_lambda_min,
        "mix_lambda_max": args.mix_lambda_max,
        "mix_lambda_floor": args.mix_lambda_floor,
        "mix_lambda_floor_weight": args.mix_lambda_floor_weight,
        "consistency_weight": args.consistency_weight,
        "consistency_noise_std": args.consistency_noise_std,
        "baseline_alphas": args.baseline_alphas,
        "init_model": init_model_path,
        "teacher_model": teacher_model_path if teacher_model is not None else "",
        "distill_weight": args.distill_weight,
        "teacher_temperature": args.teacher_temperature,
        "val_reward": best_val_reward,
        "norm_method": "teacher_free_share_logits_v2",
        "output_kind": args.output_kind,
        "output_scale": model.output_scale,
        "args": vars(args),
    }, best_path)
    print(f"Best U-GNN model saved to {best_path} (val reward={best_val_reward:.4f})")

    final_path = os.path.join(args.output_dir, "final_ugnn_power.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_type": "ugnn",
        "feature_schema": "masked_gain_snr_csi_degree_sumgain_v2",
        "training_objective": "unsupervised_large_scale_sum_rate",
        "proxy": "equivalent_channel_sum_rate_v2",
        "fairness_weight": args.fairness_weight,
        "baseline_guard_weight": args.baseline_guard_weight,
        "baseline_margin": args.baseline_margin,
        "anchor_weight": args.anchor_weight,
        "entropy_weight": args.entropy_weight,
        "entropy_floor": args.entropy_floor,
        "share_temperature": args.share_temperature,
        "mixture_alpha": args.mixture_alpha,
        "mix_lambda_min": args.mix_lambda_min,
        "mix_lambda_max": args.mix_lambda_max,
        "mix_lambda_floor": args.mix_lambda_floor,
        "mix_lambda_floor_weight": args.mix_lambda_floor_weight,
        "consistency_weight": args.consistency_weight,
        "consistency_noise_std": args.consistency_noise_std,
        "baseline_alphas": args.baseline_alphas,
        "init_model": init_model_path,
        "teacher_model": teacher_model_path if teacher_model is not None else "",
        "distill_weight": args.distill_weight,
        "teacher_temperature": args.teacher_temperature,
        "norm_method": "teacher_free_share_logits_v2",
        "output_kind": args.output_kind,
        "output_scale": model.output_scale,
        "args": vars(args),
    }, final_path)
    print(f"Final U-GNN model saved to {final_path}")


if __name__ == "__main__":
    main()
