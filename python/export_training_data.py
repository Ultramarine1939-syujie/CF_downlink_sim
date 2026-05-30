#!/usr/bin/env python3
"""Generate GNN training data without MATLAB."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from simulator import (
    channel_estimates,
    compute_rho_distributed_wmmse,
    compute_rho_dist,
    compute_rho_epa,
    compute_se,
    db2pow,
    default_params,
    generate_setup,
    precoding_mr,
    write_training_dataset_h5,
)
from config import TRAINING_DATA_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Python-generated GNN training data")
    parser.add_argument("--output-dir", type=str, default=str(TRAINING_DATA_DIR))
    parser.add_argument("--snapshots-per-snr", type=int, default=None)
    parser.add_argument("--snr-db", type=float, nargs="+", default=None)
    parser.add_argument("--realizations", type=int, default=None)
    parser.add_argument("--setups", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true")
    return parser.parse_args()


def generate_single_snapshot(
    params: dict[str, Any],
    snr_db: float,
    mode: str,
    si: int,
    mode_idx: int,
    snap: int,
    seed_offset: int,
    augment: bool,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, float, float, float, dict[str, Any]]:
    L = int(params["system"]["L"])
    K = int(params["system"]["K"])
    N = int(params["system"]["N"])
    tau_p = int(params["system"]["tau_p"])
    tau_c = int(params["system"]["tau_c"])
    p = float(params["power"]["p"])
    sigma_e = float(params["csi"]["sigma_e"])
    nbr = int(params["training"]["nbrOfRealizations"])
    nbr_setups = int(params["training"]["nbrOfSetups"])
    Pt = float(db2pow(snr_db))

    seed = seed_offset + si * 100000 + mode_idx * 10000 + snap
    rng = np.random.default_rng(seed)
    setup = generate_setup(
        L,
        K,
        N,
        tau_p,
        nbr_setups,
        seed,
        float(params["channel"]["ASD_varphi"]),
        float(params["channel"]["ASD_theta"]),
        params["channel"],
    )
    gain = setup["gainOverNoise"][:, :, 0]
    R = setup["R"][:, :, :, :, 0]
    pilot_index = setup["pilotIndex"][:, 0]
    D = setup["D_small"][:, :, 0]

    Hhat, _, _, _ = channel_estimates(R, nbr, L, K, N, tau_p, pilot_index, p, rng)
    H = Hhat + np.sqrt(sigma_e**2 / 2.0) * (rng.standard_normal(Hhat.shape) + 1j * rng.standard_normal(Hhat.shape))
    V_mr, sc_mr = precoding_mr(Hhat, nbr, N, K, L)

    rho_dwmmse, _ = compute_rho_distributed_wmmse(D, gain, Pt, params["dwmmse"]["rounds"], params["dwmmse"]["damping"])
    rho_dist = compute_rho_dist(D, gain, Pt)
    rho_epa = compute_rho_epa(D, Pt)

    esr_dw = float(np.sum(compute_se(H, V_mr, sc_mr, D, tau_c, tau_p, nbr, N, K, L, rho_dwmmse)))
    esr_d = float(np.sum(compute_se(H, V_mr, sc_mr, D, tau_c, tau_p, nbr, N, K, L, rho_dist)))
    esr_e = float(np.sum(compute_se(H, V_mr, sc_mr, D, tau_c, tau_p, nbr, N, K, L, rho_epa)))

    D_aug = D.copy()
    rho_dw_aug = rho_dwmmse.copy()
    rho_d_aug = rho_dist.copy()
    rho_e_aug = rho_epa.copy()
    sigma_e_aug = sigma_e

    if augment:
        drop_min = float(params["training"]["dataAug_dropRate_min"])
        drop_max = float(params["training"]["dataAug_dropRate_max"])
        drop_rate = drop_min + (drop_max - drop_min) * rng.random()
        dropped = (rng.random((L, K)) < drop_rate) & (D_aug > 0.5)
        D_aug[dropped] = 0.0
        rho_dw_aug[dropped] = 0.0
        rho_d_aug[dropped] = 0.0
        rho_e_aug[dropped] = 0.0
        sigma_var = float(params["training"]["dataAug_sigma_e_var"])
        sigma_e_aug = sigma_e * (1.0 + sigma_var * (rng.random() - 0.5) * 2.0)

    meta = {"SNR_dB": float(snr_db), "mode": mode, "seed": int(seed), "snapIdx": int(snap)}
    return np.sqrt(gain), D_aug, float(sigma_e_aug), rho_dw_aug, rho_d_aug, rho_e_aug, esr_dw, esr_d, esr_e, meta


def export_dataset(args: argparse.Namespace) -> Path:
    params = default_params()
    if args.snapshots_per_snr is not None:
        params["training"]["nSnapshotsPerSNR"] = args.snapshots_per_snr
    if args.snr_db is not None:
        params["training"]["SNR_dB"] = list(args.snr_db)
    if args.realizations is not None:
        params["training"]["nbrOfRealizations"] = args.realizations
    if args.setups is not None:
        params["training"]["nbrOfSetups"] = args.setups

    L = int(params["system"]["L"])
    K = int(params["system"]["K"])
    N = int(params["system"]["N"])
    tau_c = int(params["system"]["tau_c"])
    tau_p = int(params["system"]["tau_p"])
    sigma_e = float(params["csi"]["sigma_e"])
    nbr = int(params["training"]["nbrOfRealizations"])
    nbr_setups = int(params["training"]["nbrOfSetups"])
    snr_range = list(params["training"]["SNR_dB"])
    modes = list(params["training"]["accessModes"])
    snapshots_per_snr = int(params["training"]["nSnapshotsPerSNR"])
    n_snaps = len(snr_range) * len(modes) * snapshots_per_snr

    print("=== Python GNN training dataset export ===")
    print(f"Output dir: {args.output_dir}")
    print(f"SNR range: {snr_range} dB")
    print(f"Snapshots per SNR/mode: {snapshots_per_snr}")
    print(f"Modes: {modes}")
    print(f"Realizations/snapshot: {nbr}")
    print(f"Total snapshots: {n_snaps}")

    sqrt_gain_all = np.zeros((L, K, n_snaps), dtype=np.float32)
    D_all = np.zeros((L, K, n_snaps), dtype=np.float32)
    sigma_all = np.zeros((1, 1, n_snaps), dtype=np.float32)
    rho_dw_all = np.zeros((L, K, n_snaps), dtype=np.float32)
    rho_d_all = np.zeros((L, K, n_snaps), dtype=np.float32)
    rho_e_all = np.zeros((L, K, n_snaps), dtype=np.float32)
    esr_dw_all = np.zeros(n_snaps, dtype=np.float32)
    esr_d_all = np.zeros(n_snaps, dtype=np.float32)
    esr_e_all = np.zeros(n_snaps, dtype=np.float32)
    meta_rows: list[dict[str, Any]] = []

    t0 = time.perf_counter()
    idx = 0
    for si, snr_db in enumerate(snr_range, start=1):
        print(f"--- SNR = {snr_db:g} dB ---")
        for mode_idx, mode in enumerate(modes, start=1):
            mode_t0 = time.perf_counter()
            for snap in range(1, snapshots_per_snr + 1):
                values = generate_single_snapshot(
                    params,
                    float(snr_db),
                    mode,
                    si,
                    mode_idx,
                    snap,
                    args.seed,
                    augment=not args.no_augment,
                )
                sqrt_g, D, sigma_aug, rho_dw, rho_d, rho_e, esr_dw, esr_d, esr_e, meta = values
                sqrt_gain_all[:, :, idx] = sqrt_g.astype(np.float32)
                D_all[:, :, idx] = D.astype(np.float32)
                sigma_all[0, 0, idx] = np.float32(sigma_aug)
                rho_dw_all[:, :, idx] = rho_dw.astype(np.float32)
                rho_d_all[:, :, idx] = rho_d.astype(np.float32)
                rho_e_all[:, :, idx] = rho_e.astype(np.float32)
                esr_dw_all[idx] = np.float32(esr_dw)
                esr_d_all[idx] = np.float32(esr_d)
                esr_e_all[idx] = np.float32(esr_e)
                meta_rows.append(meta)
                idx += 1
            done = idx
            print(f"  [{mode}] completed {done}/{n_snaps} snapshots ({done / n_snaps * 100:.1f}%) in {time.perf_counter() - mode_t0:.1f}s")

    features = {"sqrtGain": sqrt_gain_all, "D": D_all, "sigma_e": sigma_all}
    labels = {
        "rho_DWMMSE": rho_dw_all,
        "rho_Dist": rho_d_all,
        "rho_EPA": rho_e_all,
        "ESR_DWMMSE": esr_dw_all.reshape(-1, 1),
        "ESR_Dist": esr_d_all.reshape(-1, 1),
        "ESR_EPA": esr_e_all.reshape(-1, 1),
    }
    sys_config = {
        "L": L,
        "K": K,
        "N": N,
        "tau_c": tau_c,
        "tau_p": tau_p,
        "ASD_varphi": float(params["channel"]["ASD_varphi"]),
        "ASD_theta": float(params["channel"]["ASD_theta"]),
        "p": float(params["power"]["p"]),
        "sigma_e": sigma_e,
        "nbrOfRealizations": nbr,
        "nbrOfSetups": nbr_setups,
        "SNR_dB_range": np.asarray(snr_range, dtype=np.float64),
        "nSnapshotsPerSNR": snapshots_per_snr,
        "accessModes": modes,
        "useParallel": False,
    }

    out_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"gnn_training_data_python_{timestamp}.mat"
    print(f"\nSaving to: {out_file}")
    write_training_dataset_h5(out_file, features, labels, meta_rows, sys_config)

    elapsed = time.perf_counter() - t0
    print("\n========================================")
    print("  Python dataset export complete")
    print("========================================")
    print(f"Total snapshots: {n_snaps}")
    print(f"Elapsed: {elapsed:.1f}s ({elapsed / 60.0:.1f} min)")
    print(f"sqrtGain: {features['sqrtGain'].shape}")
    print(f"rho_DWMMSE: {labels['rho_DWMMSE'].shape}")
    print(f"ESR_DWMMSE: min={esr_dw_all.min():.2f}, max={esr_dw_all.max():.2f}, mean={esr_dw_all.mean():.2f}")
    return out_file


def main() -> int:
    args = parse_args()
    export_dataset(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
