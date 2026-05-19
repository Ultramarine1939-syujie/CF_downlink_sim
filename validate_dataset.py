#!/usr/bin/env python3
"""Validate GNN training dataset quality - Using h5py for MATLAB v7.3 files"""
import h5py
import numpy as np
import sys

def as_lkn(arr, L, K):
    """Normalize MATLAB v7.3/HDF5 arrays to (L, K, N_snap)."""
    arr = np.asarray(arr)
    if arr.ndim != 3:
        return arr
    if arr.shape[0] == L and arr.shape[1] == K:
        return arr
    if arr.shape[1] == K and arr.shape[2] == L:
        return np.transpose(arr, (2, 1, 0))
    if arr.shape[0] == K and arr.shape[1] == L:
        return np.transpose(arr, (1, 0, 2))
    return arr

def validate_dataset(mat_path):
    print(f"Loading: {mat_path}")

    with h5py.File(mat_path, 'r') as f:
        print("\n" + "="*60)
        print("  Dataset Validation Report")
        print("="*60)

        # Check top-level fields
        print("\n[1] Top-level structure:")
        for key in f.keys():
            print(f"  - {key} (type: {type(f[key]).__name__})")

        # Read features
        features = f['features']
        print(f"\n[2] Features subfields:")
        if hasattr(features, 'keys'):
            for key in features.keys():
                print(f"  - {key}")
        else:
            print(f"  (Dataset, shape: {features.shape})")

        # Read labels
        labels = f['labels']
        print(f"\n[3] Labels subfields:")
        if hasattr(labels, 'keys'):
            for key in labels.keys():
                print(f"  - {key}")
        else:
            print(f"  (Dataset, shape: {labels.shape})")

        # Read meta and sysConfig
        print(f"\n[4] Meta information:")
        meta = f['meta']
        if hasattr(meta, 'keys'):
            for key in meta.keys():
                print(f"  - {key}")
        else:
            print(f"  (Dataset, shape: {meta.shape})")

        sysConfig = f['sysConfig']
        if hasattr(sysConfig, 'keys'):
            print(f"\n[5] SysConfig subfields:")
            for key in sysConfig.keys():
                print(f"  - {key}")

        # Read system config
        L = int(np.array(sysConfig['L'])[0,0])
        K = int(np.array(sysConfig['K'])[0,0])
        N = int(np.array(sysConfig['N'])[0,0])
        tau_c = int(np.array(sysConfig['tau_c'])[0,0])
        tau_p = int(np.array(sysConfig['tau_p'])[0,0])
        SNR_dB_range = np.array(sysConfig['SNR_dB_range']).flatten()
        nSnapshotsPerSNR = int(np.array(sysConfig['nSnapshotsPerSNR'])[0,0])

        print(f"\n[6] System config:")
        print(f"  L={L}, K={K}, N={N}")
        print(f"  tau_c={tau_c}, tau_p={tau_p}")
        print(f"  SNR_dB_range: {SNR_dB_range}")
        print(f"  nSnapshotsPerSNR: {nSnapshotsPerSNR}")
        N_snaps_expected = len(SNR_dB_range) * nSnapshotsPerSNR  # DCC only

        # Get dimensions
        sqrtGain = as_lkn(np.array(features['sqrtGain']), L, K)
        D = as_lkn(np.array(features['D']), L, K)
        sigma_e = np.array(features['sigma_e'])
        rho_WMMSE = as_lkn(np.array(labels['rho_WMMSE']), L, K)
        rho_Dist = as_lkn(np.array(labels['rho_Dist']), L, K)
        rho_EPA = as_lkn(np.array(labels['rho_EPA']), L, K)
        ESR_WMMSE = np.array(labels['ESR_WMMSE']).flatten()
        ESR_Dist = np.array(labels['ESR_Dist']).flatten()
        ESR_EPA = np.array(labels['ESR_EPA']).flatten()

        N_snaps = sqrtGain.shape[2]

        print(f"\n[7] Features dimension check:")
        print(f"  sqrtGain: {sqrtGain.shape} (expected LxKxN = {L}x{K}x{N_snaps})")
        print(f"  D:        {D.shape}")
        print(f"  sigma_e:  {sigma_e.shape}")

        print(f"\n[8] Labels dimension check:")
        print(f"  rho_WMMSE: {rho_WMMSE.shape}")
        print(f"  rho_Dist:  {rho_Dist.shape}")
        print(f"  rho_EPA:   {rho_EPA.shape}")
        print(f"  ESR_WMMSE: {ESR_WMMSE.shape}")
        print(f"  ESR_Dist:  {ESR_Dist.shape}")
        print(f"  ESR_EPA:   {ESR_EPA.shape}")

        # Dimension consistency check
        print(f"\n[9] Dimension consistency check:")
        errors = []
        warnings = []

        if sqrtGain.shape != (L, K, N_snaps):
            errors.append(f"sqrtGain dimension error: {sqrtGain.shape} vs expected ({L}, {K}, {N_snaps})")
        if D.shape != (L, K, N_snaps):
            errors.append(f"D dimension error: {D.shape} vs expected ({L}, {K}, {N_snaps})")
        if sigma_e.size != N_snaps:
            errors.append(f"sigma_e dimension error: {sigma_e.shape} has {sigma_e.size} values vs expected {N_snaps}")
        if rho_WMMSE.shape != (L, K, N_snaps):
            errors.append(f"rho_WMMSE dimension error: {rho_WMMSE.shape} vs expected ({L}, {K}, {N_snaps})")
        if ESR_WMMSE.shape != (N_snaps,):
            errors.append(f"ESR_WMMSE dimension error: {ESR_WMMSE.shape} vs expected ({N_snaps},)")
        if rho_Dist.shape != (L, K, N_snaps):
            errors.append(f"rho_Dist dimension error: {rho_Dist.shape} vs expected ({L}, {K}, {N_snaps})")
        if rho_EPA.shape != (L, K, N_snaps):
            errors.append(f"rho_EPA dimension error: {rho_EPA.shape} vs expected ({L}, {K}, {N_snaps})")

        if N_snaps != N_snaps_expected:
            warnings.append(f"Snapshot count mismatch: actual {N_snaps} vs expected {N_snaps_expected}")

        if errors:
            for e in errors:
                print(f"  [FAIL] {e}")
        else:
            print(f"  [PASS] All dimensions correct")

        if warnings:
            for w in warnings:
                print(f"  [WARN] {w}")

        # Physical validity check
        print(f"\n[10] Physical validity check:")

        # D matrix: should be 0 or 1
        D_unique = np.unique(D)
        if set(D_unique.flatten()) <= {0, 1}:
            print(f"  [PASS] D matrix values correct (only 0, 1)")
        else:
            print(f"  [FAIL] D matrix contains invalid values: {D_unique}")

        # rho should be non-negative
        rho_W_nan = np.isnan(rho_WMMSE).sum()
        rho_D_nan = np.isnan(rho_Dist).sum()
        rho_E_nan = np.isnan(rho_EPA).sum()
        if rho_W_nan == 0 and rho_D_nan == 0 and rho_E_nan == 0:
            print(f"  [PASS] rho has no NaN values")
        else:
            print(f"  [FAIL] rho has NaN: WMMSE={rho_W_nan}, Dist={rho_D_nan}, EPA={rho_E_nan}")

        # Check rho non-negative
        if np.all(rho_WMMSE >= 0) and np.all(rho_Dist >= 0) and np.all(rho_EPA >= 0):
            print(f"  [PASS] All rho values are non-negative")
        else:
            print(f"  [FAIL] rho contains negative values!")

        # sigma_e should be positive
        sigma_e_flat = sigma_e.flatten()
        if np.all(sigma_e_flat > 0):
            print(f"  [PASS] sigma_e all positive: range=[{sigma_e_flat.min():.3f}, {sigma_e_flat.max():.3f}]")
        else:
            print(f"  [FAIL] sigma_e contains non-positive values!")

        # ESR should be positive
        if np.all(ESR_WMMSE > 0) and np.all(ESR_Dist > 0) and np.all(ESR_EPA > 0):
            print(f"  [PASS] All ESR values are valid")
            print(f"     ESR_WMMSE: mean={ESR_WMMSE.mean():.2f}, range=[{ESR_WMMSE.min():.2f}, {ESR_WMMSE.max():.2f}]")
            print(f"     ESR_Dist:  mean={ESR_Dist.mean():.2f}, range=[{ESR_Dist.min():.2f}, {ESR_Dist.max():.2f}]")
            print(f"     ESR_EPA:   mean={ESR_EPA.mean():.2f}, range=[{ESR_EPA.min():.2f}, {ESR_EPA.max():.2f}]")
        else:
            print(f"  [FAIL] ESR contains invalid values!")

        # Infinity check
        print(f"\n[11] Infinity check:")
        sqrtGain_inf = np.isinf(sqrtGain).sum()
        rho_W_inf = np.isinf(rho_WMMSE).sum()
        ESR_W_inf = np.isinf(ESR_WMMSE).sum()
        print(f"  sqrtGain Inf: {sqrtGain_inf}")
        print(f"  rho_WMMSE Inf: {rho_W_inf}")
        print(f"  ESR_WMMSE Inf: {ESR_W_inf}")

        if sqrtGain_inf == 0 and rho_W_inf == 0 and ESR_W_inf == 0:
            print(f"  [PASS] No infinity values")

        # WMMSE gain check
        print(f"\n[12] WMMSE vs Dist gain statistics:")
        gain_vs_D = (ESR_WMMSE - ESR_Dist) / np.maximum(ESR_Dist, 1e-10) * 100
        print(f"  mean={gain_vs_D.mean():.2f}%, range=[{gain_vs_D.min():.2f}%, {gain_vs_D.max():.2f}%]")

        # Data distribution check
        print(f"\n[13] sqrtGain distribution:")
        sg = sqrtGain.flatten()
        print(f"  min={sg.min():.4f}, max={sg.max():.4f}, mean={sg.mean():.4f}, std={sg.std():.4f}")

        # rho power sum check
        print(f"\n[14] rho power sum check:")
        rho_W_sums = rho_WMMSE.sum(axis=(0, 1))
        rho_D_sums = rho_Dist.sum(axis=(0, 1))
        rho_E_sums = rho_EPA.sum(axis=(0, 1))
        print(f"  rho_WMMSE: mean={rho_W_sums.mean():.4f}, std={rho_W_sums.std():.4f}")
        print(f"  rho_Dist:  mean={rho_D_sums.mean():.4f}, std={rho_D_sums.std():.4f}")
        print(f"  rho_EPA:   mean={rho_E_sums.mean():.4f}, std={rho_E_sums.std():.4f}")

        # meta information check
        print(f"\n[15] Meta information:")
        if hasattr(meta, 'shape'):
            n_meta = meta.shape[1] if len(meta.shape) > 1 else meta.shape[0]
            print(f"  Snapshot count: {n_meta}")
        else:
            print(f"  (Unable to get meta info)")

        print("\n" + "="*60)
        print("  Validation Complete")
        print("="*60)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        mat_path = sys.argv[1]
    else:
        mat_path = "gnn_training_data_20260426_230559.mat"
    validate_dataset(mat_path)
