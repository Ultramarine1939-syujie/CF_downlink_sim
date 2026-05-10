#!/usr/bin/env python3
"""
Federated Averaging (FedAvg) training for GNN power allocation.

Phase 3: 联邦学习集成
"""

import argparse
import copy
import glob
import os
import random
import time
import csv
import json
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split, Subset

from dataset import GNNDataset
from train_gnn import PowerGNN_GAT, custom_collate, compute_loss, evaluate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _shard_indices(idxs, num_clients: int):
    splits = np.array_split(np.array(idxs, dtype=np.int64), num_clients)
    return [s.tolist() for s in splits]


def split_dataset_to_clients(train_dataset, num_clients: int, seed: int, split_by: str):
    n = len(train_dataset)
    idxs = list(range(n))

    if split_by == "random":
        rng = random.Random(seed)
        rng.shuffle(idxs)
        shards = _shard_indices(idxs, num_clients)
        return [Subset(train_dataset, indices=s) for s in shards]

    meta_pairs = []
    for i in range(n):
        sample = train_dataset[i]
        snr = int(float(sample["snr"].view(-1)[0].item()))
        mode = str(sample.get("mode", "All"))
        if split_by == "snr":
            key = snr
        elif split_by == "mode":
            key = mode
        elif split_by == "snr_mode":
            key = (snr, mode)
        else:
            raise ValueError(f"Unknown split_by: {split_by}")
        meta_pairs.append((key, i))

    meta_pairs.sort(key=lambda x: x[0])
    sorted_idxs = [i for _, i in meta_pairs]
    shards = _shard_indices(sorted_idxs, num_clients)
    return [Subset(train_dataset, indices=s) for s in shards]


def local_train(model, dataloader, device, lr: float, local_epochs: int, weight_decay: float,
                scheduler: str, use_amp: bool):
    model = model.to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, len(dataloader))
    total_steps = max(1, int(local_epochs) * steps_per_epoch)

    if scheduler == "onecycle":
        lr_sched = OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=100.0
        )
    elif scheduler == "none":
        lr_sched = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler}")

    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    total_loss = 0.0
    total_samples = 0

    for _ in range(local_epochs):
        for batch in dataloader:
            batch = batch.to(device)

            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                rho_pred = model(batch)
                y_batched = batch.y.view(-1, model.L, model.K)
                D_batched = batch.D_mask.view(-1, model.L, model.K)
                nz_batched = batch.rho_is_nonzero.view(-1, model.L, model.K)
                loss = compute_loss(rho_pred, y_batched, D_batched, nz_batched)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            if lr_sched is not None:
                lr_sched.step()

            bs = int(rho_pred.shape[0])
            total_loss += float(loss.item()) * bs
            total_samples += bs

    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss, total_samples


def fedavg_aggregate(state_dicts, weights):
    if len(state_dicts) == 0:
        raise ValueError("Empty state_dicts for aggregation")

    total_weight = float(sum(weights))
    if total_weight <= 0:
        raise ValueError("Non-positive total aggregation weight")

    agg = OrderedDict()
    for k in state_dicts[0].keys():
        agg[k] = torch.zeros_like(state_dicts[0][k], device="cpu")

    for sd, w in zip(state_dicts, weights):
        w = float(w) / total_weight
        for k, v in sd.items():
            agg[k] += v.detach().to("cpu") * w

    return agg


def save_checkpoint(model, path, extra: dict):
    payload = {
        "model_state_dict": {k: v.detach().to("cpu") for k, v in model.state_dict().items()},
        "norm_method": "per_sample_signedlog_minmax",
        "output_scale": float(getattr(model, "output_scale", 1.0)),
    }
    payload.update(extra)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def main():
    parser = argparse.ArgumentParser(description="FedAvg training for GNN power allocation (Phase 3)")
    parser.add_argument("--data", type=str,
                        default="../data/gnn_training/*.mat",
                        help="Path/pattern to .mat training data (supports glob)")
    parser.add_argument("--rounds", type=int, default=50, help="Number of federated rounds")
    parser.add_argument("--frac", type=float, default=0.2, help="Client sampling fraction C in (0,1]")
    parser.add_argument("--local_epochs", type=int, default=1, help="Local epochs per selected client")
    parser.add_argument("--batch_size", type=int, default=32, help="Local batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Local learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="AdamW weight decay")
    parser.add_argument("--scheduler", type=str, default="onecycle", choices=["onecycle", "none"],
                        help="Local LR scheduler")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="../models", help="Output directory")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N rounds")
    parser.add_argument("--num_clients", type=int, default=0, help="Number of clients (default: L from dataset)")
    parser.add_argument("--split_by", type=str, default="random", choices=["random", "snr", "mode", "snr_mode"],
                        help="Client data split strategy (to simulate non-IID)")
    parser.add_argument("--amp", action="store_true", help="Enable AMP on CUDA for faster local updates")
    parser.add_argument("--init_ckpt", type=str, default="", help="Optional init checkpoint (.pt) for global model")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Model hidden dim")
    parser.add_argument("--num_heads", type=int, default=4, help="GAT heads")
    parser.add_argument("--num_layers", type=int, default=3, help="GAT layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_files = sorted(glob.glob(args.data))
    if len(data_files) == 0:
        raise SystemExit(f"ERROR: No files matched pattern: {args.data}")

    if len(data_files) == 1:
        dataset = GNNDataset(data_files[0])
        L, K = dataset.L, dataset.K
    else:
        all_datasets = [GNNDataset(f) for f in data_files]
        from torch.utils.data import ConcatDataset
        dataset = ConcatDataset(all_datasets)
        L, K = all_datasets[0].L, all_datasets[0].K

    num_clients = int(args.num_clients) if int(args.num_clients) > 0 else int(L)
    num_clients = max(1, num_clients)

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(args.seed))

    client_subsets = split_dataset_to_clients(train_dataset, num_clients=num_clients, seed=args.seed, split_by=args.split_by)
    client_sizes = [len(s) for s in client_subsets]

    print(f"Loaded {len(dataset)} snapshots | Train={train_size}, Val={val_size}")
    print(f"Clients={num_clients} | avg client size={np.mean(client_sizes):.1f}, "
          f"min={np.min(client_sizes)}, max={np.max(client_sizes)}")
    print(f"Split by: {args.split_by}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=0
    )

    global_model = PowerGNN_GAT(
        L=L, K=K,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        output_scale=1.0
    ).to(device)

    if args.init_ckpt:
        ckpt_path = args.init_ckpt
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                global_model.load_state_dict(ckpt["model_state_dict"], strict=True)
            elif isinstance(ckpt, dict):
                global_model.load_state_dict(ckpt, strict=True)
            print(f"Loaded init checkpoint: {ckpt_path}")
        else:
            print(f"WARNING: init_ckpt not found: {ckpt_path}")

    out_dir = args.output_dir
    final_path = os.path.join(out_dir, "gnn_power_fedavg.pt")
    log_path = os.path.join(out_dir, "fedavg_log.csv")
    summary_path = os.path.join(out_dir, "fedavg_summary.json")
    log_rows = []

    best_val = float("inf")
    best_round = 0
    best_state = None

    for r in range(args.rounds):
        round_t0 = time.time()
        m = max(1, int(round(args.frac * num_clients)))
        selected = random.sample(range(num_clients), m)

        local_state_dicts = []
        local_weights = []
        local_losses = []
        clients_used = 0
        total_samples_used = 0

        for cid in selected:
            subset = client_subsets[cid]
            if len(subset) == 0:
                continue

            local_loader = DataLoader(
                subset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=custom_collate,
                num_workers=0
            )

            local_model = copy.deepcopy(global_model)
            loss, n_samples = local_train(
                local_model,
                local_loader,
                device=device,
                lr=args.lr,
                local_epochs=args.local_epochs,
                weight_decay=args.weight_decay,
                scheduler=args.scheduler,
                use_amp=args.amp
            )
            local_losses.append(loss)
            local_state_dicts.append({k: v.detach().to("cpu") for k, v in local_model.state_dict().items()})
            local_weights.append(n_samples)
            clients_used += 1
            total_samples_used += int(n_samples)

        if len(local_state_dicts) == 0:
            print(f"[Round {r+1}/{args.rounds}] No client updates, skipping aggregation")
            continue

        agg_state = fedavg_aggregate(local_state_dicts, local_weights)
        global_model.load_state_dict(agg_state)

        val_mse, val_corr, val_nz_mse = evaluate(global_model, val_loader, device=device)

        mean_local_loss = float(np.mean(local_losses)) if len(local_losses) else float("nan")
        print(f"[Round {r+1}/{args.rounds}] clients={len(local_state_dicts)}/{m} "
              f"local_loss={mean_local_loss:.6f} | val_mse={val_mse:.6f} "
              f"val_nz_mse={val_nz_mse:.6f} val_corr={val_corr:.4f}")

        model_bytes = int(sum(v.numel() * v.element_size() for v in global_model.state_dict().values()))
        comm_up_bytes = int(clients_used * model_bytes)
        comm_down_bytes = int(clients_used * model_bytes)
        comm_total_bytes = int(comm_up_bytes + comm_down_bytes)
        round_sec = float(time.time() - round_t0)

        log_rows.append({
            "round": int(r + 1),
            "clients_total": int(num_clients),
            "clients_selected": int(m),
            "clients_used": int(clients_used),
            "samples_used": int(total_samples_used),
            "local_loss_mean": float(mean_local_loss),
            "val_mse": float(val_mse),
            "val_nz_mse": float(val_nz_mse),
            "val_corr": float(val_corr),
            "round_sec": float(round_sec),
            "model_bytes": int(model_bytes),
            "comm_up_bytes": int(comm_up_bytes),
            "comm_down_bytes": int(comm_down_bytes),
            "comm_total_bytes": int(comm_total_bytes),
        })

        if np.isfinite(val_mse) and val_mse < best_val:
            best_val = float(val_mse)
            best_round = int(r)
            best_state = {k: v.detach().to("cpu").clone() for k, v in global_model.state_dict().items()}

        if args.save_every > 0 and ((r + 1) % args.save_every == 0):
            round_path = os.path.join(out_dir, f"gnn_power_fedavg_round{r+1:04d}.pt")
            save_checkpoint(
                global_model,
                round_path,
                extra={"round": int(r + 1), "clients": int(num_clients), "frac": float(args.frac)}
            )

    if best_state is not None:
        global_model.load_state_dict(best_state)
        save_checkpoint(
            global_model,
            final_path,
            extra={"best_round": int(best_round + 1), "best_val_mse": float(best_val),
                   "clients": int(num_clients), "frac": float(args.frac)}
        )
        print(f"Best global model saved: {final_path} (round={best_round+1}, val_mse={best_val:.6f})")
    else:
        save_checkpoint(
            global_model,
            final_path,
            extra={"clients": int(num_clients), "frac": float(args.frac)}
        )
        print(f"Final global model saved: {final_path}")

    if len(log_rows) > 0:
        os.makedirs(out_dir, exist_ok=True)
        fieldnames = list(log_rows[0].keys())
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(log_rows)

        corrs = [row["val_corr"] for row in log_rows if np.isfinite(row["val_corr"])]
        max_corr = float(max(corrs)) if len(corrs) else float("nan")
        target_corr = float(0.98 * max_corr) if np.isfinite(max_corr) else float("nan")
        round_98 = None
        if np.isfinite(target_corr):
            for row in log_rows:
                if float(row["val_corr"]) >= target_corr:
                    round_98 = int(row["round"])
                    break

        summary = {
            "metric": "val_corr_proxy",
            "max_val_corr": max_corr,
            "target_98pct": target_corr,
            "round_to_98pct": round_98,
            "log_path": os.path.basename(log_path),
            "final_model_path": os.path.basename(final_path),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"FedAvg log saved: {log_path}")
        print(f"FedAvg summary saved: {summary_path}")


if __name__ == "__main__":
    main()

