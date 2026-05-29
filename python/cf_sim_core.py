"""Pure-Python Cell-Free downlink simulation primitives.

The functions in this module mirror the MATLAB implementation under
``matlab/`` closely enough to run the full pipeline without crossing the
MATLAB/Python bridge.  They intentionally keep array shapes aligned with the
MATLAB code:

* channels: ``(L*N, realizations, K)``
* precoders: ``(L*N, K, realizations)``
* AP-UE matrices: ``(L, K)``
"""

from __future__ import annotations

import hashlib
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

EPS = np.finfo(float).eps


def db2pow(db: np.ndarray | float) -> np.ndarray | float:
    return np.power(10.0, np.asarray(db) / 10.0)


def pow2db(power: np.ndarray | float) -> np.ndarray | float:
    return 10.0 * np.log10(np.maximum(power, EPS))


def default_params() -> dict[str, Any]:
    return {
        "system": {"L": 100, "K": 20, "N": 1, "tau_c": 200, "tau_p": 10},
        "channel": {
            "ASD_varphi": math.radians(15),
            "ASD_theta": math.radians(15),
            "squareLength": 1000.0,
            "B": 20e6,
            "noiseFigure": 7.0,
            "alpha": 36.7,
            "constantTerm": -30.5,
            "sigma_sf": 4.0,
            "decorr": 9.0,
            "distanceVertical": 10.0,
            "antennaSpacing": 0.5,
        },
        "power": {"p": 100.0, "SNR_dB": [5, 10, 15, 20, 25, 30]},
        "csi": {"sigma_e": 0.3, "nIter": 5},
        "simulation": {
            "numScenarios": 10,
            "nbrOfRealizations": 200,
            "seed": 42,
            "accessModes": ["DCC"],
        },
        "runtime": {"runStage": 3, "useCache": True, "verbose": True, "verboseAlgo": False},
        "output": {
            "isSaveFig": True,
            "isSaveData": True,
            "savePath": "Imgs",
            "dataPath": "SimulationData",
            "cleanOldFigures": True,
        },
        "wmmse": {"maxIter": 30, "simMaxIter": 20, "tol": 1e-4},
        "dwmmse": {"rounds": 5, "damping": 0.6},
        "fpcp": {"alpha": -1.0},
        "syncAblation": {
            "enable": True,
            "fronthaulMbps": 1000.0,
            "syncRttMs": 0.05,
            "dccPayloadRatio": 0.35,
            "includeComputeTime": True,
            "includeModelInferenceEstimate": True,
            "inferenceGops": 80.0,
            "controllerGops": 200.0,
            "edgeFeatureBytes": 8.0,
            "modelParamBytes": 4.0,
            "gnnLayers": 3,
            "gnnHiddenDim": 128,
            "gnnHeads": 4,
            "dcgnnTopZ": 15,
            "localHiddenDim": 96,
            "localLayers": 3,
            "rlHiddenDim": 256,
            "rlLayers": 3,
        },
        "gnn": {
            "fullModelFile": Path("models") / "best_gat_gnn_power.pt",
            "localModelFile": Path("models") / "best_local_gnn_power.pt",
            "dcgnnModelFile": Path("models") / "best_dcgnn_power.pt",
            "ugnnModelFile": Path("models") / "best_ugnn_power.pt",
        },
        "rl": {
            "dqnModelFile": Path("models") / "best_dqn_power.pt",
            "ddpgModelFile": Path("models") / "best_ddpg_power.pt",
        },
        "training": {
            "nSnapshotsPerSNR": 500,
            "nbrOfRealizations": 200,
            "nbrOfSetups": 2,
            "SNR_dB": [5, 10, 15, 20, 25, 30],
            "accessModes": ["DCC"],
            "dataAug_dropRate_min": 0.1,
            "dataAug_dropRate_max": 0.3,
            "dataAug_sigma_e_var": 0.2,
        },
    }


def fingerprint(*values: Any) -> str:
    payload = json.dumps(values, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def complex_normal(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def hermitian_sqrtm(mat: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh((mat + mat.conj().T) / 2.0)
    vals = np.maximum(vals.real, 0.0)
    return (vecs * np.sqrt(vals)) @ vecs.conj().T


def hermitian_toeplitz(first_col: np.ndarray) -> np.ndarray:
    n = first_col.size
    out = np.empty((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            if i >= j:
                out[i, j] = first_col[i - j]
            else:
                out[i, j] = np.conj(first_col[j - i])
    return out


def local_scattering_matrix(
    n_antennas: int,
    varphi: float,
    theta: float,
    asd_varphi: float,
    asd_theta: float,
    antenna_spacing: float = 0.5,
    quadrature_order: int = 24,
) -> np.ndarray:
    if n_antennas == 1:
        return np.ones((1, 1), dtype=np.complex128)

    first_col = np.zeros(n_antennas, dtype=np.complex128)
    first_col[0] = 1.0

    gh_x, gh_w = np.polynomial.hermite.hermgauss(quadrature_order)
    gh_w = gh_w / np.sqrt(np.pi)

    for col in range(1, n_antennas):
        distance = antenna_spacing * col
        if asd_varphi > 0 and asd_theta > 0:
            delta = np.sqrt(2.0) * asd_varphi * gh_x[:, None]
            eps = np.sqrt(2.0) * asd_theta * gh_x[None, :]
            phase = 2j * np.pi * distance * np.sin(varphi + delta) * np.cos(theta + eps)
            first_col[col] = np.sum((gh_w[:, None] * gh_w[None, :]) * np.exp(phase))
        elif asd_varphi > 0:
            delta = np.sqrt(2.0) * asd_varphi * gh_x
            phase = 2j * np.pi * distance * np.sin(varphi + delta) * np.cos(theta)
            first_col[col] = np.sum(gh_w * np.exp(phase))
        elif asd_theta > 0:
            eps = np.sqrt(2.0) * asd_theta * gh_x
            phase = 2j * np.pi * distance * np.sin(varphi) * np.cos(theta + eps)
            first_col[col] = np.sum(gh_w * np.exp(phase))
        else:
            first_col[col] = np.exp(2j * np.pi * distance * np.sin(varphi) * np.cos(theta))

    corr = hermitian_toeplitz(first_col)
    tr = np.trace(corr).real
    if tr > 0:
        corr = corr * (n_antennas / tr)
    return corr


def generate_setup(
    L: int,
    K: int,
    N: int,
    tau_p: int,
    nbr_of_setups: int,
    seed: int | None,
    asd_varphi: float,
    asd_theta: float,
    channel_cfg: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    cfg = default_params()["channel"]
    if channel_cfg:
        cfg.update(channel_cfg)

    rng = np.random.default_rng(seed if seed and seed > 0 else None)
    square_length = float(cfg["squareLength"])
    noise_variance_dbm = -174.0 + 10.0 * np.log10(float(cfg["B"])) + float(cfg["noiseFigure"])
    alpha = float(cfg["alpha"])
    constant_term = float(cfg["constantTerm"])
    sigma_sf = float(cfg["sigma_sf"])
    decorr = float(cfg["decorr"])
    distance_vertical = float(cfg["distanceVertical"])
    antenna_spacing = float(cfg["antennaSpacing"])

    gain_db = np.zeros((L, K, nbr_of_setups), dtype=np.float64)
    R = np.zeros((N, N, L, K, nbr_of_setups), dtype=np.complex128)
    distances = np.zeros((L, K, nbr_of_setups), dtype=np.float64)
    pilot_index = np.zeros((K, nbr_of_setups), dtype=np.int64)
    D = np.zeros((L, K, nbr_of_setups), dtype=np.float64)
    D_small = np.zeros((L, K, nbr_of_setups), dtype=np.float64)
    ap_positions = np.zeros(L, dtype=np.complex128)
    ue_positions = np.zeros(K, dtype=np.complex128)

    wrap_horizontal = np.tile(np.array([-square_length, 0.0, square_length]), (3, 1))
    wrap_vertical = wrap_horizontal.T
    wrap_locations = wrap_horizontal.ravel(order="F") + 1j * wrap_vertical.ravel(order="F")

    for setup_idx in range(nbr_of_setups):
        ap_positions = (rng.random(L) + 1j * rng.random(L)) * square_length
        ue_positions = np.zeros(K, dtype=np.complex128)
        ap_wrapped = ap_positions[:, None] + wrap_locations[None, :]
        shadow_corr = sigma_sf**2 * np.ones((K, K), dtype=np.float64)
        shadow_ap_realizations = np.zeros((K, L), dtype=np.float64)

        for k in range(K):
            ue_pos = (rng.random() + 1j * rng.random()) * square_length
            abs_dist = np.abs(ap_wrapped - ue_pos)
            which_pos = np.argmin(abs_dist, axis=1)
            distance_ap_ue = abs_dist[np.arange(L), which_pos]
            distances[:, k, setup_idx] = np.sqrt(distance_vertical**2 + distance_ap_ue**2)

            if k > 0:
                shortest = np.array([
                    np.min(np.abs(ue_pos - ue_positions[i] + wrap_locations))
                    for i in range(k)
                ])
                new_col = sigma_sf**2 * np.power(2.0, -shortest / decorr)
                term1 = np.linalg.solve(shadow_corr[:k, :k].T, new_col).T
                mean_values = term1 @ shadow_ap_realizations[:k, :]
                variance = sigma_sf**2 - term1 @ new_col
                std_value = np.sqrt(max(float(variance), 0.0))
            else:
                new_col = np.array([], dtype=np.float64)
                mean_values = 0.0
                std_value = sigma_sf

            shadowing = mean_values + std_value * rng.standard_normal(L)
            gain_db[:, k, setup_idx] = (
                constant_term
                - alpha * np.log10(distances[:, k, setup_idx])
                + shadowing
                - noise_variance_dbm
            )

            if k > 0:
                shadow_corr[:k, k] = new_col
                shadow_corr[k, :k] = new_col
            shadow_ap_realizations[k, :] = shadowing
            ue_positions[k] = ue_pos

            master = int(np.argmax(gain_db[:, k, setup_idx]))
            D[master, k, setup_idx] = 1.0
            if k < tau_p:
                pilot_index[k, setup_idx] = k
            else:
                pilot_interference = np.zeros(tau_p)
                for t in range(tau_p):
                    used = pilot_index[:k, setup_idx] == t
                    pilot_interference[t] = np.sum(db2pow(gain_db[master, :k, setup_idx][used]))
                pilot_index[k, setup_idx] = int(np.argmin(pilot_interference))

            for l in range(L):
                wrapped_ap = ap_wrapped[l, which_pos[l]]
                angle_varphi = np.angle(ue_positions[k] - wrapped_ap)
                angle_theta = np.arcsin(distance_vertical / distances[l, k, setup_idx])
                R[:, :, l, k, setup_idx] = db2pow(gain_db[l, k, setup_idx]) * local_scattering_matrix(
                    N, angle_varphi, angle_theta, asd_varphi, asd_theta, antenna_spacing
                )

        for l in range(L):
            for t in range(tau_p):
                pilot_ues = np.where(pilot_index[:, setup_idx] == t)[0]
                if pilot_ues.size == 0:
                    continue
                strongest = pilot_ues[np.argmax(gain_db[l, pilot_ues, setup_idx])]
                D[l, strongest, setup_idx] = 1.0

        for k in range(K):
            temp = np.full(L, -np.inf)
            served = D[:, k, setup_idx] > 0.5
            temp[served] = gain_db[served, k, setup_idx]
            D_small[int(np.argmax(temp)), k, setup_idx] = 1.0

    return {
        "gainOverNoisedB": gain_db,
        "gainOverNoise": db2pow(gain_db),
        "R": R,
        "pilotIndex": pilot_index,
        "D": D,
        "D_small": D_small,
        "APpositions": ap_positions,
        "UEpositions": ue_positions,
        "distances": distances,
    }


def channel_estimates(
    R: np.ndarray,
    nbr_of_realizations: int,
    L: int,
    K: int,
    N: int,
    tau_p: int,
    pilot_index: np.ndarray,
    p: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    H = complex_normal((L * N, nbr_of_realizations, K), rng)
    for l in range(L):
        rows = slice(l * N, (l + 1) * N)
        for k in range(K):
            H[rows, :, k] = np.sqrt(0.5) * hermitian_sqrtm(R[:, :, l, k]) @ H[rows, :, k]

    Np = np.sqrt(0.5) * complex_normal((N, nbr_of_realizations, L, tau_p), rng)
    Hhat = np.zeros((L * N, nbr_of_realizations, K), dtype=np.complex128)
    B = np.zeros_like(R, dtype=np.complex128)
    C = np.zeros_like(R, dtype=np.complex128)
    eye_n = np.eye(N, dtype=np.complex128)

    for l in range(L):
        rows = slice(l * N, (l + 1) * N)
        for t in range(tau_p):
            mask = pilot_index == t
            if not np.any(mask):
                continue
            yp = np.sqrt(p) * tau_p * np.sum(H[rows, :, :][:, :, mask], axis=2) + np.sqrt(tau_p) * Np[:, :, l, t]
            psi = p * tau_p * np.sum(R[:, :, l, mask], axis=2) + eye_n
            for k in np.where(mask)[0]:
                rpsi = np.linalg.solve(psi.T, R[:, :, l, k].T).T
                Hhat[rows, :, k] = np.sqrt(p) * rpsi @ yp
                B[:, :, l, k] = p * tau_p * rpsi @ R[:, :, l, k]
                C[:, :, l, k] = R[:, :, l, k] - B[:, :, l, k]
    return Hhat, H, B, C


def precoding_mr(Hhat: np.ndarray, nbr: int, N: int, K: int, L: int) -> tuple[np.ndarray, np.ndarray]:
    V = np.transpose(Hhat, (0, 2, 1)).copy()
    # MATLAB reshape(Hhat, [N, L, nbr, K]) uses column-major order, yielding
    #   H4_matlab(a, l, n, k) = Hhat((l-1)*N+a, n, k).
    # Python C-order reshape(L, N, nbr, K) gives H4(l, a, n, k) = Hhat(l*N+a, n, k).
    # Transpose swaps axes 0<->1 to match MATLAB's (N, L, nbr, K) layout.
    H4 = Hhat.reshape(L, N, nbr, K).transpose(1, 0, 2, 3)  # (N, L, nbr, K)
    scaling = np.mean(np.sum(np.abs(H4) ** 2, axis=0), axis=1)  # sum over N, mean over nbr -> (L, K)
    return V, scaling


def precoding_lmmse(
    Hhat: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    nbr: int,
    N: int,
    K: int,
    L: int,
    p: float,
) -> tuple[np.ndarray, np.ndarray]:
    scaling = np.zeros((L, K), dtype=np.float64)
    V = np.zeros((L * N, K, nbr), dtype=np.complex128)

    if N == 1:
        hhat_sq = np.sum(np.abs(Hhat) ** 2, axis=0, keepdims=True)
        c_sum = np.sum(C[0, 0, :, :], axis=0).reshape(1, 1, K)
        denom = p * hhat_sq + p * c_sum + 1.0
        V_all = (p / denom) * Hhat
        V_all = V_all * D.reshape(L, 1, K)
        V = np.transpose(V_all, (0, 2, 1)).copy()
        scaling = np.mean(np.abs(V_all) ** 2, axis=1).reshape(L, K)
        return V, scaling

    eye_n = np.eye(N, dtype=np.complex128)
    for n in range(nbr):
        for l in range(L):
            served = D[l, :] > 0.5
            if not np.any(served):
                continue
            rows = slice(l * N, (l + 1) * N)
            h_all = Hhat[rows, n, :].reshape(N, K)
            c_sum_l = np.sum(C[:, :, l, :], axis=2)
            A = p * (h_all @ h_all.conj().T) + p * c_sum_l + eye_n
            v_full = p * np.linalg.solve(A, h_all)
            v_full[:, ~served] = 0.0
            V[rows, :, n] = v_full
            scaling[l, :] += np.sum(np.abs(v_full) ** 2, axis=0) / nbr
    return V, scaling


def precoding_lmmse_global(
    Hhat: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    nbr: int,
    N: int,
    K: int,
    L: int,
    p: float,
) -> tuple[np.ndarray, np.ndarray]:
    scaling = np.zeros((L, K), dtype=np.float64)
    V = np.zeros((L * N, K, nbr), dtype=np.complex128)

    if N == 1:
        c_sum = np.sum(C[0, 0, :, :], axis=1)
        for l in range(L):
            h_ln = Hhat[l, :, :]
            denom = p * np.sum(np.abs(h_ln) ** 2, axis=1) + p * c_sum[l] + 1.0
            v_ln = (p / denom[:, None]) * h_ln
            v_ln[:, D[l, :] <= 0.5] = 0.0
            V[l, :, :] = v_ln.T
            scaling[l, :] = np.mean(np.abs(v_ln) ** 2, axis=0)
        return V, scaling

    eye_n = np.eye(N, dtype=np.complex128)
    # Match MATLAB: sum(C, 4) = sum over users → per-AP (N, N, L) covariance
    c_sum_all = np.sum(C, axis=3)  # (N, N, L)
    for n in range(nbr):
        for l in range(L):
            served = D[l, :] > 0.5
            if not np.any(served):
                continue
            rows = slice(l * N, (l + 1) * N)
            h_all = Hhat[rows, n, :].reshape(N, K)
            c_sum_l = c_sum_all[:, :, l]  # (N, N), per-AP
            A = p * (h_all @ h_all.conj().T) + p * c_sum_l + eye_n
            v_full = p * np.linalg.solve(A, h_all)
            v_full[:, ~served] = 0.0
            V[rows, :, n] = v_full
            scaling[l, :] += np.sum(np.abs(v_full) ** 2, axis=0) / nbr
    return V, scaling


def precoding_robust_mmse(
    Hhat: np.ndarray,
    D: np.ndarray,
    nbr: int,
    N: int,
    K: int,
    L: int,
    Pt: float,
    sigma_e: float,
    n_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    if N == 1:
        return precoding_robust_mmse_scalar(Hhat, D, nbr, K, L, Pt, sigma_e, n_iter)

    eye_n = np.eye(N, dtype=np.complex128)
    scaling = np.zeros((L, K), dtype=np.float64)
    V = np.zeros((L * N, K, nbr), dtype=np.complex128)
    tau = np.sqrt(1.0 + sigma_e**2)

    for n in range(nbr):
        for l in range(L):
            served = D[l, :] > 0.5
            if not np.any(served):
                continue
            rows = slice(l * N, (l + 1) * N)
            h_local = Hhat[rows, n, :][:, served].reshape(N, int(np.sum(served)))
            p_bar = np.linalg.solve(h_local @ h_local.conj().T + (K / Pt) * eye_n, h_local)
            for _ in range(n_iter):
                denom = _positive_trace_power(p_bar @ p_bar.conj().T)
                f = np.sqrt(Pt / denom)
                theta = sigma_e**2 * (h_local @ h_local.conj().T / K)
                P = f * tau * p_bar
                lam = np.trace(eye_n).real / (f**2 * Pt) - np.trace(P @ P.conj().T @ theta).real / Pt
                lam = max(float(lam), 1e-6)
                A = h_local @ h_local.conj().T + (1.0 + f**2) * theta + lam * f**2 * tau * eye_n
                p_bar = np.linalg.solve(A, h_local)
            denom = _positive_trace_power(p_bar @ p_bar.conj().T)
            f = np.sqrt(Pt / denom)
            v_tmp = f * tau * p_bar
            v_full = np.zeros((N, K), dtype=np.complex128)
            v_full[:, served] = v_tmp
            V[rows, :, n] = v_full
            scaling[l, :] += np.sum(np.abs(v_full) ** 2, axis=0) / nbr
    return V, scaling


def precoding_robust_mmse_scalar(
    Hhat: np.ndarray,
    D: np.ndarray,
    nbr: int,
    K: int,
    L: int,
    Pt: float,
    sigma_e: float,
    n_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized R-MMSE precoder for the common single-antenna AP case."""

    H_lnk = Hhat.reshape(L, nbr, K)
    D_l1k = (D > 0.5).astype(float)[:, None, :]
    H_local = H_lnk * D_l1k
    sum_h2 = np.sum(np.abs(H_local) ** 2, axis=2, keepdims=True)

    p_bar = np.divide(
        H_local,
        sum_h2 + (K / Pt),
        out=np.zeros_like(H_local, dtype=np.complex128),
        where=(D_l1k > 0),
    )
    tau = np.sqrt(1.0 + sigma_e**2)
    theta = sigma_e**2 * sum_h2 / K

    for _ in range(n_iter):
        p_power = np.sum(np.abs(p_bar) ** 2, axis=2, keepdims=True)
        valid = p_power > 0
        f2 = np.divide(Pt, p_power, out=np.zeros_like(p_power, dtype=np.float64), where=valid)
        p_tx_power = f2 * tau**2 * p_power
        lam_raw = np.divide(1.0, f2 * Pt, out=np.full_like(f2, np.inf), where=(f2 > 0)) - p_tx_power * theta / Pt
        lam = np.maximum(lam_raw, 1e-6)
        denom = sum_h2 + (1.0 + f2) * theta + lam * f2 * tau
        p_bar = np.divide(H_local, denom, out=np.zeros_like(H_local, dtype=np.complex128), where=(D_l1k > 0) & (denom > 0))

    p_power = np.sum(np.abs(p_bar) ** 2, axis=2, keepdims=True)
    f = np.sqrt(np.divide(Pt, p_power, out=np.zeros_like(p_power, dtype=np.float64), where=(p_power > 0)))
    V_lnk = f * tau * p_bar
    V = np.transpose(V_lnk, (0, 2, 1)).reshape(L, K, nbr)
    scaling = np.mean(np.abs(V_lnk) ** 2, axis=1)
    return V, scaling


def _positive_trace_power(mat: np.ndarray) -> float:
    """Return trace power without clipping small positive MATLAB values.

    MATLAB's R-MMSE iteration can intentionally produce very small, but still
    positive, ``trace(P_bar*P_bar')`` values.  Clipping those values to machine
    epsilon changes the later normalization factor by many orders of magnitude,
    so only invalid or non-positive values are replaced.
    """

    value = float(np.real(np.trace(mat)))
    if not np.isfinite(value) or value <= 0.0:
        return np.finfo(float).tiny
    return value


def compute_se(
    H: np.ndarray,
    V: np.ndarray,
    scaling: np.ndarray,
    D: np.ndarray,
    tau_c: int,
    tau_p: int,
    nbr: int,
    N: int,
    K: int,
    L: int,
    rho: np.ndarray,
) -> np.ndarray:
    prelog = 1.0 - tau_p / tau_c
    power_scale = np.zeros((L, K), dtype=np.float64)
    valid = (D > 0.5) & (scaling > 0) & (rho > 0)
    power_scale[valid] = np.sqrt(rho[valid] / scaling[valid])

    if N == 1:
        W = V * power_scale[:, :, None]
    else:
        row_scale = np.repeat(power_scale, N, axis=0)
        W = V * row_scale[:, :, None]

    H_all = np.transpose(H, (0, 2, 1))
    G = np.einsum("akn,ajn->kjn", np.conj(H_all), W, optimize=True)
    diag = np.stack([G[k, k, :] for k in range(K)], axis=0)
    signal = np.mean(diag, axis=1)
    interf = np.mean(np.sum(np.abs(G) ** 2, axis=1), axis=1)
    sinr = np.abs(signal) ** 2 / np.maximum(interf - np.abs(signal) ** 2 + 1.0, EPS)
    return prelog * np.real(np.log2(1.0 + sinr))


def compute_rho_epa(D: np.ndarray, Pt: float) -> np.ndarray:
    served = np.sum(D > 0.5, axis=1, keepdims=True)
    return np.where(served > 0, Pt * (D > 0.5) / np.maximum(served, 1), 0.0).astype(float)


def compute_rho_dist(D: np.ndarray, gain_over_noise: np.ndarray, Pt: float) -> np.ndarray:
    weights = np.sqrt(np.maximum(gain_over_noise, 0.0)) * (D > 0.5)
    row_sum = np.sum(weights, axis=1, keepdims=True)
    return np.where(row_sum > 0, Pt * weights / np.maximum(row_sum, EPS), 0.0)


def compute_rho_fpcp(D: np.ndarray, gain_over_noise: np.ndarray, Pt: float, alpha: float = -1.0) -> np.ndarray:
    safe_gain = np.maximum(gain_over_noise, EPS)
    weights = np.power(safe_gain, -alpha) * (D > 0.5)
    row_sum = np.sum(weights, axis=1, keepdims=True)
    fallback = compute_rho_epa(D, Pt)
    return np.where(row_sum > 0, Pt * weights / np.maximum(row_sum, EPS), fallback)


def compute_rho_random(scaling: np.ndarray, D: np.ndarray, Pt: float, rng: np.random.Generator) -> np.ndarray:
    L, K = D.shape
    d_random = rng.random(K)
    d_random = d_random / max(float(np.sqrt(np.sum(d_random**2))), EPS)
    d2 = d_random**2
    ap_power = (scaling * D) @ d2
    eta = np.zeros(L)
    nonzero = ap_power > 0
    eta[nonzero] = np.sqrt((Pt / L) / ap_power[nonzero])
    return (eta[:, None] ** 2) * (scaling * D) * d2[None, :]


def compute_rho_distributed_wmmse(
    D: np.ndarray,
    gain_over_noise: np.ndarray,
    Pt: float,
    max_rounds: int = 5,
    damping: float = 0.6,
) -> tuple[np.ndarray, dict[str, float]]:
    D_bin = (D > 0.5).astype(float)
    gain = np.maximum(gain_over_noise, 0.0) * D_bin
    rho = compute_rho_epa(D_bin, Pt)
    for _ in range(max_rounds):
        received_power = np.sum(gain * rho, axis=0, keepdims=True)
        self_power = gain * rho
        interference_proxy = np.maximum(received_power - self_power, 0.0)
        score = gain / (1.0 + interference_proxy)
        score *= D_bin
        empty = np.sum(score, axis=1, keepdims=True) <= EPS
        score = np.where(empty, D_bin, score)
        rho_next = Pt * score / np.maximum(np.sum(score, axis=1, keepdims=True), EPS)
        rho = damping * rho_next + (1.0 - damping) * rho
        rho = enforce_per_ap_power(rho, D_bin, Pt)
    return rho, {"rounds": float(max_rounds), "messageBytes": float(max_rounds * D.shape[1] * 8)}


def enforce_per_ap_power(rho: np.ndarray, D: np.ndarray, Pt: float) -> np.ndarray:
    rho = np.maximum(rho, 0.0) * D
    row_power = np.sum(rho, axis=1, keepdims=True)
    over = row_power[:, 0] > Pt
    if np.any(over):
        rho[over, :] *= Pt / np.maximum(row_power[over], EPS)
    return rho


def build_effective_channel(Hhat: np.ndarray, N: int, K: int, L: int, nbr: int) -> np.ndarray:
    H_eff = np.zeros((K, L), dtype=np.float64)
    for l in range(L):
        rows = slice(l * N, (l + 1) * N)
        for k in range(K):
            H_lk = Hhat[rows, :nbr, k]
            H_eff[k, l] = np.sqrt(max(float(np.mean(np.sum(np.abs(H_lk) ** 2, axis=0))), 0.0))
    return H_eff


def compute_equiv_sinr(H_eff: np.ndarray, V: np.ndarray) -> np.ndarray:
    HV = H_eff @ V
    desired = np.abs(np.diag(HV)) ** 2
    interf = np.maximum(np.sum(np.abs(HV) ** 2, axis=1) - desired, 0.0)
    return desired / np.maximum(interf + 1.0, EPS)


def compute_equiv_rate(H_eff: np.ndarray, V: np.ndarray) -> float:
    return float(np.sum(np.log2(1.0 + compute_equiv_sinr(H_eff, V))))


def project_per_ap(V: np.ndarray, D: np.ndarray, Pt: float) -> np.ndarray:
    out = V * D
    row_power = np.sum(np.abs(out) ** 2, axis=1, keepdims=True)
    over = row_power[:, 0] > Pt
    if np.any(over):
        out[over, :] *= np.sqrt(Pt / np.maximum(row_power[over], EPS))
    return out


def solve_for_lambda(A: np.ndarray, B: np.ndarray, D: np.ndarray, lam: np.ndarray, reg: float) -> np.ndarray:
    L, K = D.shape
    A_lam = A + np.diag(lam) + reg * np.eye(L)
    V = np.zeros((L, K), dtype=np.complex128)
    if np.all(D > 0.5):
        return np.linalg.solve(A_lam, B)
    for k in range(K):
        served = np.where(D[:, k] > 0.5)[0]
        if served.size:
            V[served, k] = np.linalg.solve(A_lam[np.ix_(served, served)], B[served, k])
    return V


def solve_transmit_update(
    A: np.ndarray,
    B: np.ndarray,
    D: np.ndarray,
    Pt: float,
    max_dual_iter: int,
    reg: float,
) -> np.ndarray:
    L, _ = D.shape
    lam = np.zeros(L)
    diag_scale = max(float(np.real(np.trace(A))) / max(L, 1), 1e-6)
    step0 = diag_scale / max(Pt, 1.0)
    V = np.zeros_like(B, dtype=np.complex128)
    for dual_iter in range(1, max_dual_iter + 1):
        V = solve_for_lambda(A, B, D, lam, reg)
        row_power = np.sum(np.abs(V) ** 2, axis=1)
        violation = row_power - Pt
        if np.max(violation) <= max(1e-6, 1e-5 * Pt):
            break
        lam = np.maximum(0.0, lam + (step0 / np.sqrt(dual_iter)) * violation)
    return project_per_ap(V, D, Pt)


def accept_monotone_update(
    H_eff: np.ndarray,
    V_old: np.ndarray,
    V_candidate: np.ndarray,
    D: np.ndarray,
    Pt: float,
    prev_rate: float,
) -> tuple[np.ndarray, float]:
    alpha = 1.0
    while alpha >= 1e-3:
        trial = project_per_ap((1.0 - alpha) * V_old + alpha * V_candidate, D, Pt)
        rate = compute_equiv_rate(H_eff, trial)
        if rate >= prev_rate - 1e-9:
            return trial, rate
        alpha *= 0.5
    return V_old, prev_rate


def compute_rho_wmmse(
    Hhat: np.ndarray,
    D: np.ndarray,
    Pt: float,
    N: int,
    K: int,
    L: int,
    nbr: int,
    max_iter: int = 30,
    tol: float = 1e-4,
    verbose: bool = False,
) -> tuple[np.ndarray, int]:
    D_bin = (D > 0.5).astype(float)
    if Hhat.size == 0 or np.all(D_bin == 0):
        return compute_rho_epa(D_bin, Pt), 0

    H_eff = build_effective_channel(Hhat, N, K, L, nbr)
    rho0 = compute_rho_dist(D_bin, H_eff.T, Pt)
    rho0[~np.isfinite(rho0)] = 0.0
    V = project_per_ap(np.sqrt(np.maximum(rho0, 0.0)) * D_bin, D_bin, Pt)
    prev_rate = compute_equiv_rate(H_eff, V)
    best_rate = prev_rate
    best_V = V.copy()
    reg = 1e-6 * max(float(np.real(np.trace(H_eff.conj().T @ H_eff))) / max(L, 1), 1.0)

    for it in range(1, max_iter + 1):
        V_old = V.copy()
        HV = H_eff @ V
        total_rx = np.sum(np.abs(HV) ** 2, axis=1) + 1.0
        desired = np.diag(HV)
        u = desired / np.maximum(total_rx, EPS)
        mse = 1.0 - 2.0 * np.real(np.conj(u) * desired) + np.abs(u) ** 2 * total_rx
        mse = np.maximum(np.real(mse), 1e-12)
        w = 1.0 / mse
        A = H_eff.conj().T @ (np.diag(w * np.abs(u) ** 2) @ H_eff)
        A = (A + A.conj().T) / 2.0 + reg * np.eye(L)
        B = H_eff.conj().T @ np.diag(w * np.conj(u))
        candidate = solve_transmit_update(A, B, D_bin, Pt, 25, reg)
        V, rate = accept_monotone_update(H_eff, V_old, candidate, D_bin, Pt, prev_rate)

        power_change = np.linalg.norm(V.ravel() - V_old.ravel()) / max(np.linalg.norm(V_old.ravel()), EPS)
        rate_change = abs(rate - prev_rate) / max(abs(prev_rate), 1.0)
        if rate > best_rate:
            best_rate = rate
            best_V = V.copy()
        if verbose and (it == 1 or it % 5 == 0):
            sinr = compute_equiv_sinr(H_eff, V)
            preview = ", ".join(f"{x:.2f}" for x in sinr[:3])
            print(f"    [WMMSE] Iter {it:2d}: WSR={rate:.4f}, PowerChange={power_change:.6f}, SINR=[{preview}]")
        if it > 1 and (power_change < tol or rate_change < tol):
            return finalize_rho(best_V, D_bin, Pt), it
        prev_rate = rate
    return finalize_rho(best_V, D_bin, Pt), max_iter


def finalize_rho(V: np.ndarray, D: np.ndarray, Pt: float) -> np.ndarray:
    rho = np.abs(V) ** 2 * D
    rho = enforce_per_ap_power(rho, D, Pt)
    rho[~np.isfinite(rho)] = 0.0
    return rho


def empty_timing() -> dict[str, float]:
    return {
        "total_sec": 0.0,
        "bridge_sec": 0.0,
        "load_sec": 0.0,
        "feature_sec": 0.0,
        "collate_sec": 0.0,
        "forward_sec": 0.0,
        "post_sec": 0.0,
        "python_total_sec": 0.0,
    }


_MODEL_WARNED: set[str] = set()


def _warn_model_once(key: str, message: str) -> None:
    if key not in _MODEL_WARNED:
        print(f"  [WARN] {message}")
        _MODEL_WARNED.add(key)


def compute_rho_gnn(
    D: np.ndarray,
    gain_over_noise: np.ndarray,
    Pt: float,
    model_path: Path | str,
    sigma_e: float,
) -> tuple[np.ndarray, dict[str, float]]:
    t0 = time.perf_counter()
    model_path = Path(model_path)
    if not model_path.is_file():
        timing = empty_timing()
        timing["total_sec"] = time.perf_counter() - t0
        timing["forward_sec"] = timing["total_sec"]
        return compute_rho_epa(D, Pt), timing
    try:
        import gnn_runtime

        out = gnn_runtime.infer(str(model_path), np.sqrt(np.maximum(gain_over_noise, 0.0)), D, Pt, sigma_e)
        timing = {**empty_timing(), **{k: float(v) for k, v in out.items() if k.endswith("_sec")}}
        timing["mix_lambda_mean"] = float(out.get("mix_lambda_mean", 1.0))
        timing["total_sec"] = float(out.get("python_total_sec", time.perf_counter() - t0))
        return np.asarray(out["rho"], dtype=np.float64), timing
    except Exception as exc:  # pragma: no cover - defensive fallback for missing PyG/checkpoint mismatch
        _warn_model_once(str(model_path), f"GNN inference failed for {model_path.name}; falling back to EPA. {exc}")
        timing = empty_timing()
        timing["total_sec"] = time.perf_counter() - t0
        timing["forward_sec"] = timing["total_sec"]
        return compute_rho_epa(D, Pt), timing


def compute_rho_local_gnn(
    D: np.ndarray,
    gain_over_noise: np.ndarray,
    Pt: float,
    model_path: Path | str,
    sigma_e: float,
) -> tuple[np.ndarray, dict[str, float]]:
    t0 = time.perf_counter()
    model_path = Path(model_path)
    if not model_path.is_file():
        timing = empty_timing()
        timing["total_sec"] = time.perf_counter() - t0
        timing["forward_sec"] = timing["total_sec"]
        return compute_rho_epa(D, Pt), timing
    try:
        import gnn_runtime_local

        out = gnn_runtime_local.infer(str(model_path), np.sqrt(np.maximum(gain_over_noise, 0.0)), D, Pt, sigma_e)
        timing = {**empty_timing(), **{k: float(v) for k, v in out.items() if k.endswith("_sec")}}
        timing["total_sec"] = float(out.get("python_total_sec", time.perf_counter() - t0))
        return np.asarray(out["rho"], dtype=np.float64), timing
    except Exception as exc:  # pragma: no cover
        _warn_model_once(str(model_path), f"Local-GNN inference failed for {model_path.name}; falling back to EPA. {exc}")
        timing = empty_timing()
        timing["total_sec"] = time.perf_counter() - t0
        timing["forward_sec"] = timing["total_sec"]
        return compute_rho_epa(D, Pt), timing


def normalized_power_entropy(rho: np.ndarray, D: np.ndarray) -> float:
    D_bin = (D > 0.5).astype(float)
    rho = np.maximum(rho, 0.0) * D_bin
    row_sum = np.sum(rho, axis=1, keepdims=True)
    served = np.sum(D_bin, axis=1, keepdims=True)
    active = (row_sum[:, 0] > 0) & (served[:, 0] > 1)
    if not np.any(active):
        return 1.0
    p = rho[active, :] / np.maximum(row_sum[active, :], EPS)
    h = -np.sum(p * np.log(np.maximum(p, EPS)), axis=1)
    return float(np.mean(h / np.log(np.maximum(served[active, 0], 2.0))))


def compute_rho_ugnn(
    D: np.ndarray,
    gain_over_noise: np.ndarray,
    Pt: float,
    model_path: Path | str,
    sigma_e: float,
    guard_enabled: bool = False,
    min_entropy: float = 0.35,
    guard_alpha: float = -1.0,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    t0 = time.perf_counter()
    rho_nn, timing = compute_rho_gnn(D, gain_over_noise, Pt, model_path, sigma_e)
    entropy = normalized_power_entropy(rho_nn, D)
    row_power = np.sum(np.maximum(rho_nn, 0.0), axis=1)
    active_rows = np.sum(D > 0.5, axis=1) > 0
    invalid = (not np.isfinite(rho_nn).all()) or np.any(row_power[active_rows] <= 0)
    if invalid or (guard_enabled and entropy < min_entropy):
        rho = compute_rho_fpcp(D, gain_over_noise, Pt, guard_alpha)
        timing["guard_triggered"] = 1.0
    else:
        rho = rho_nn
        timing["guard_triggered"] = 0.0
    timing["guard_entropy"] = entropy
    timing["guard_invalid"] = float(invalid)
    timing["guard_sec"] = time.perf_counter() - t0 - timing.get("total_sec", 0.0)
    timing["total_sec"] = time.perf_counter() - t0
    return rho, timing, rho_nn


def compute_rho_rl(
    D: np.ndarray,
    gain_over_noise: np.ndarray,
    Pt: float,
    model_path: Path | str,
    sigma_e: float,
) -> tuple[np.ndarray, dict[str, float]]:
    t0 = time.perf_counter()
    model_path = Path(model_path)
    if not model_path.is_file():
        timing = empty_timing()
        timing["total_sec"] = time.perf_counter() - t0
        timing["forward_sec"] = timing["total_sec"]
        return compute_rho_epa(D, Pt), timing
    try:
        import rl_runtime

        out = rl_runtime.infer(str(model_path), np.sqrt(np.maximum(gain_over_noise, 0.0)), D, Pt, sigma_e)
        timing = {**empty_timing(), **{k: float(v) for k, v in out.items() if k.endswith("_sec")}}
        timing["total_sec"] = float(out.get("python_total_sec", time.perf_counter() - t0))
        return np.asarray(out["rho"], dtype=np.float64), timing
    except Exception as exc:  # pragma: no cover
        _warn_model_once(str(model_path), f"RL inference failed for {model_path.name}; falling back to EPA. {exc}")
        timing = empty_timing()
        timing["total_sec"] = time.perf_counter() - t0
        timing["forward_sec"] = timing["total_sec"]
        return compute_rho_epa(D, Pt), timing


def build_algo_table(pa_keys: list[str] | None = None, pc_keys: list[str] | None = None, modes: list[str] | None = None) -> list[dict[str, Any]]:
    pa = {
        "baseline": ("Baseline", "distributed"),
        "random": ("Random", "distributed"),
        "EPA": ("EPA", "distributed"),
        "FPCP": ("FPCP", "distributed"),
        "DWMMSE": ("D-WMMSE", "distributed"),
        "WMMSE": ("WMMSE", "centralized_reference"),
        "GNN": ("GNN", "low_latency_centralized"),
        "LocalGNN": ("Local-GNN", "distributed"),
        "DCGNN": ("DCGNN", "low_latency_centralized"),
        "UGNN": ("U-GNN", "low_latency_centralized"),
        "DQN": ("DQN", "low_latency_centralized"),
        "DDPG": ("DDPG", "low_latency_centralized"),
    }
    pc = {
        "MR": ("MR", "distributed"),
        "LMMSE": ("L-MMSE", "distributed"),
        "RMMSE": ("R-MMSE", "distributed"),
        "LMMSE_G": ("L-MMSE-G", "centralized_reference"),
    }
    if pa_keys is None:
        pa_keys = list(pa)
    if pc_keys is None:
        pc_keys = list(pc)
    if modes is None:
        modes = ["DCC"]

    out: list[dict[str, Any]] = []
    for pa_key in pa_keys:
        for pc_key in pc_keys:
            for mode in modes:
                pa_name, pa_arch = pa[pa_key]
                pc_name, pc_arch = pc[pc_key]
                out.append({
                    "id": len(out) + 1,
                    "pa": pa_key,
                    "pc": pc_key,
                    "mode": mode,
                    "name": f"{pa_name}+{pc_name} ({mode})",
                    "pcArch": pc_arch,
                    "paArch": pa_arch,
                    "isDistributed": pc_arch == "distributed" and pa_arch == "distributed",
                })
    return out


def method_names() -> list[str]:
    return ["Baseline", "FPCP", "D-WMMSE", "WMMSE", "GNN", "Local-GNN", "DCGNN", "U-GNN", "DQN", "DDPG"]


def pa_to_method(pa_key: str) -> str | None:
    return {
        "baseline": "Baseline",
        "FPCP": "FPCP",
        "DWMMSE": "D-WMMSE",
        "WMMSE": "WMMSE",
        "GNN": "GNN",
        "LocalGNN": "Local-GNN",
        "DCGNN": "DCGNN",
        "UGNN": "U-GNN",
        "DQN": "DQN",
        "DDPG": "DDPG",
    }.get(pa_key)


def print_final_results(
    ESR_mean: np.ndarray,
    algo_table: list[dict[str, Any]],
    snr_db: np.ndarray,
    num_scenarios: int,
    nbr: int,
) -> None:
    avg = np.mean(ESR_mean, axis=1)
    order = np.argsort(avg)[::-1]
    names = [a["name"] for a in algo_table]
    distributed = np.array([bool(a["isDistributed"]) for a in algo_table])
    baseline_idx = next((i for i, n in enumerate(names) if "Baseline" in n and "MR" in n and "DCC" in n), 0)
    wmmse_idx = next((i for i, n in enumerate(names) if "WMMSE" in n and "D-WMMSE" not in n and "L-MMSE-G" not in n), baseline_idx)
    baseline_avg = avg[baseline_idx]
    wmmse_avg = avg[wmmse_idx]

    print("\n=====================================================================")
    print("  DISTRIBUTED DOWNLINK CANDIDATES (Python)")
    print("=====================================================================")
    print(f"  {'Rank':<4}  {'Algorithm':<30}  {'Avg ESR':>10}  {'vs Baseline':>12}  {'vs WMMSE':>12}")
    print("---------------------------------------------------------------------")
    rank = 0
    for idx in order:
        if not distributed[idx]:
            continue
        rank += 1
        pct_base = (avg[idx] - baseline_avg) / max(abs(baseline_avg), 1.0) * 100.0
        print(f"  {rank:<4d}  {names[idx]:<30}  {avg[idx]:10.2f}  {pct_base:+11.2f}%  {avg[idx] - wmmse_avg:+12.2f}")

    print("\nBest distributed algorithms per SNR:")
    for si, snr in enumerate(snr_db):
        sorted_idx = np.argsort(ESR_mean[:, si])[::-1]
        printed = 0
        for idx in sorted_idx:
            if distributed[idx]:
                printed += 1
                print(f"  SNR={snr:5.1f} dB  #{printed}  {names[idx]:<30}  ESR={ESR_mean[idx, si]:8.2f}")
                if printed >= 3:
                    break
    print(f"\nSummary: {num_scenarios} scenarios x {nbr} realizations x {len(snr_db)} SNR")
    best_high = int(np.argmax(ESR_mean[:, -1]))
    print(f"Overall best at highest SNR: {names[best_high]}  ESR={ESR_mean[best_high, -1]:.2f} @ {snr_db[-1]:.1f} dB")
    print("=====================================================================\n")


def write_training_dataset_h5(
    output_file: Path,
    features: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    meta_rows: list[dict[str, Any]],
    sys_config: dict[str, Any],
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_file, "w") as f:
        g_feat = f.create_group("features")
        for key, value in features.items():
            g_feat.create_dataset(key, data=value)

        g_lab = f.create_group("labels")
        for key, value in labels.items():
            g_lab.create_dataset(key, data=value)

        g_sys = f.create_group("sysConfig")
        for key, value in sys_config.items():
            if isinstance(value, (str, bytes)):
                g_sys.create_dataset(key, data=np.bytes_(value))
            elif isinstance(value, (list, tuple)) and value and isinstance(value[0], str):
                dt = h5py.string_dtype(encoding="utf-8")
                g_sys.create_dataset(key, data=np.asarray(value, dtype=object), dtype=dt)
            else:
                arr = np.asarray(value)
                if arr.ndim == 0:
                    arr = arr.reshape(1, 1)
                g_sys.create_dataset(key, data=arr)

        meta = f.create_dataset("meta", shape=(1, len(meta_rows)), dtype=h5py.ref_dtype)
        for idx, row in enumerate(meta_rows):
            grp = f.create_group(f"meta_{idx:06d}")
            grp.create_dataset("SNR_dB", data=np.asarray([[float(row["SNR_dB"])]], dtype=np.float64))
            mode_chars = np.asarray([ord(ch) for ch in str(row.get("mode", "DCC"))], dtype=np.uint16).reshape(-1, 1)
            grp.create_dataset("mode", data=mode_chars)
            grp.create_dataset("seed", data=np.asarray([[int(row.get("seed", 0))]], dtype=np.int64))
            grp.create_dataset("snapIdx", data=np.asarray([[int(row.get("snapIdx", idx + 1))]], dtype=np.int64))
            meta[0, idx] = grp.ref


def write_simulation_results_csv(
    output_file: Path,
    ESR_mean: np.ndarray,
    ESR_best: np.ndarray,
    ESR_best_algo: list[str],
    algo_table: list[dict[str, Any]],
    snr_db: np.ndarray,
    perf: dict[str, Any],
) -> None:
    """Write all Python simulation results as one tidy CSV table.

    The file intentionally stores one observation per row instead of embedding
    JSON/HDF5 arrays, so pandas/R/Excel-style readers can ingest it directly.
    """

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["table", "id", "algorithm", "pa", "pc", "mode", "method", "snr_db", "metric", "value", "avg_esr", "best_algorithm", "is_best"])
        for idx, algo in enumerate(algo_table):
            avg_esr = float(np.mean(ESR_mean[idx, :]))
            for si, snr in enumerate(snr_db):
                is_best = int(algo["name"] == ESR_best_algo[si])
                writer.writerow(["esr", algo["id"], algo["name"], algo["pa"], algo["pc"], algo["mode"], "", f"{snr:.8g}", "ESR_mean", f"{ESR_mean[idx, si]:.8g}", f"{avg_esr:.8g}", ESR_best_algo[si], is_best])
            writer.writerow(["esr_summary", algo["id"], algo["name"], algo["pa"], algo["pc"], algo["mode"], "", "", "avg_ESR", f"{avg_esr:.8g}", f"{avg_esr:.8g}", "", ""])

        method_names = list(perf.get("methodNames", []))
        mode_names = list(perf.get("modeNames", []))
        for metric in ("time_pa_sec", "time_core_sec", "comm_bytes"):
            values = perf.get(metric)
            if values is None:
                continue
            arr = np.asarray(values)
            for mi, method in enumerate(method_names):
                for si, snr in enumerate(snr_db):
                    for mode_idx, mode in enumerate(mode_names):
                        writer.writerow(["perf", "", "", "", "", mode, method, f"{snr:.8g}", metric, f"{arr[mi, si, mode_idx]:.8g}", "", "", ""])

        pc_names = ["MR", "LMMSE", "RMMSE", "LMMSE_G"]
        pc_values = perf.get("time_pc_sec")
        if pc_values is not None:
            arr = np.asarray(pc_values)
            for pi, pc_name in enumerate(pc_names[: arr.shape[0]]):
                for si, snr in enumerate(snr_db):
                    for mode_idx, mode in enumerate(mode_names):
                        writer.writerow(["perf_pc", "", "", "", pc_name, mode, "", f"{snr:.8g}", "time_pc_sec", f"{arr[pi, si, mode_idx]:.8g}", "", "", ""])

        for method, value in perf.get("model_param_count", {}).items():
            writer.writerow(["model_param_count", "", "", "", "", "", method, "", "model_param_count", f"{float(value):.8g}", "", "", ""])


def write_esr_csv(output_file: Path, ESR_mean: np.ndarray, algo_table: list[dict[str, Any]], snr_db: np.ndarray) -> None:
    """Backward-compatible compact ESR table writer."""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        f.write("id,algorithm,pa,pc,mode,avg_esr," + ",".join(f"snr_{snr:g}dB" for snr in snr_db) + "\n")
        for idx, algo in enumerate(algo_table):
            values = ",".join(f"{v:.8g}" for v in ESR_mean[idx, :])
            f.write(
                f"{algo['id']},{algo['name']},{algo['pa']},{algo['pc']},{algo['mode']},"
                f"{np.mean(ESR_mean[idx, :]):.8g},{values}\n"
            )


def maybe_plot_esr(
    save_dir: Path,
    ESR_mean: np.ndarray,
    algo_table: list[dict[str, Any]],
    snr_db: np.ndarray,
    enabled: bool = True,
) -> None:
    if not enabled:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("  [WARN] matplotlib is not installed; skipped Python figure output.")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    pa_order = ["LocalGNN", "UGNN", "DCGNN", "GNN", "DDPG", "DQN", "DWMMSE", "WMMSE", "FPCP", "EPA", "random", "baseline"]
    pa_labels = ["Local-GNN", "U-GNN", "DCGNN", "GNN", "DDPG", "DQN", "D-WMMSE", "WMMSE", "FPCP", "EPA", "Random", "Baseline"]
    colors = plt.cm.tab20(np.linspace(0, 1, len(pa_order)))

    plt.figure(figsize=(10, 6))
    for pa_key, label, color in zip(pa_order, pa_labels, colors):
        candidates = [i for i, a in enumerate(algo_table) if a["pa"] == pa_key and a["mode"] == "DCC"]
        if not candidates:
            continue
        best = max(candidates, key=lambda i: float(np.mean(ESR_mean[i, :])))
        plt.plot(snr_db, ESR_mean[best, :], marker="o", linewidth=2, color=color, label=f"{label}+{algo_table[best]['pc']}")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Ergodic Sum Rate (bit/s/Hz)")
    plt.title("Best Precoder per Power Allocation Method (Python)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_dir / "Fig1_Best_PA_ESR_python.png", dpi=160)
    plt.close()

    fixed_pc = "RMMSE"
    plt.figure(figsize=(10, 6))
    for pa_key, label, color in zip(pa_order, pa_labels, colors):
        idx = next((i for i, a in enumerate(algo_table) if a["pa"] == pa_key and a["pc"] == fixed_pc and a["mode"] == "DCC"), None)
        if idx is None:
            continue
        plt.plot(snr_db, ESR_mean[idx, :], marker="o", linewidth=2, color=color, label=label)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Ergodic Sum Rate (bit/s/Hz)")
    plt.title("Power Allocation Comparison with R-MMSE Precoding (Python)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_dir / "Fig2_RMMSE_PA_Comparison_python.png", dpi=160)
    plt.close()


def build_sync_ablation_metrics(
    ESR_mean: np.ndarray,
    algo_table: list[dict[str, Any]],
    snr_db: np.ndarray,
    perf: dict[str, Any],
    L: int,
    K: int,
    N: int,
    nbr: int,
    n_iter: int,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    num_algos = len(algo_table)
    num_snr = len(snr_db)
    sync_delay = np.zeros((num_algos, num_snr))
    pc_sync_delay = np.zeros((num_algos, num_snr))
    pa_sync_delay = np.zeros((num_algos, num_snr))
    feature_delay = np.zeros((num_algos, num_snr))
    model_inference_delay = np.zeros((num_algos, num_snr))
    pc_control_delay = np.zeros((num_algos, num_snr))
    pa_control_delay = np.zeros((num_algos, num_snr))
    compute_delay = np.full((num_algos, num_snr), np.nan)
    pc_compute_delay = np.full((num_algos, num_snr), np.nan)
    control_delay = np.zeros((num_algos, num_snr))
    sync_bytes = np.zeros((num_algos, num_snr))
    pc_sync_bytes = np.zeros((num_algos, num_snr))
    pa_sync_bytes = np.zeros((num_algos, num_snr))
    feature_bytes_mat = np.zeros((num_algos, num_snr))
    sync_rounds = np.zeros((num_algos, num_snr))
    pc_sync_rounds = np.zeros((num_algos, num_snr))
    pa_sync_rounds = np.zeros((num_algos, num_snr))
    feature_rounds_mat = np.zeros((num_algos, num_snr))
    avg_esr = np.mean(ESR_mean, axis=1)

    for ai, algo in enumerate(algo_table):
        pc_rounds, pc_bytes = estimate_pc_sync(algo["pc"], L, K, N, nbr, n_iter, cfg["dccPayloadRatio"])
        pa_rounds, pa_bytes = estimate_pa_sync(algo["pa"], L, K, cfg, cfg["dccPayloadRatio"])
        for si in range(num_snr):
            feature_bytes, feature_rounds = lookup_feature_collection_bytes(algo["pa"], algo["mode"], si, L, K, perf, cfg)
            model_ms = estimate_model_inference_ms(algo["pa"], L, K, cfg, perf)
            pc_sync_ms = pc_rounds * cfg["syncRttMs"] + bytes_to_ms(pc_bytes, cfg["fronthaulMbps"])
            pa_sync_ms = pa_rounds * cfg["syncRttMs"] + bytes_to_ms(pa_bytes, cfg["fronthaulMbps"])
            feature_ms = feature_rounds * cfg["syncRttMs"] + bytes_to_ms(feature_bytes, cfg["fronthaulMbps"])
            total_bytes = pc_bytes + pa_bytes + feature_bytes
            total_rounds = pc_rounds + pa_rounds + feature_rounds
            sync_delay[ai, si] = pc_sync_ms + pa_sync_ms + feature_ms
            pc_sync_delay[ai, si] = pc_sync_ms
            pa_sync_delay[ai, si] = pa_sync_ms
            feature_delay[ai, si] = feature_ms
            model_inference_delay[ai, si] = model_ms
            sync_bytes[ai, si] = total_bytes
            pc_sync_bytes[ai, si] = pc_bytes
            pa_sync_bytes[ai, si] = pa_bytes
            feature_bytes_mat[ai, si] = feature_bytes
            sync_rounds[ai, si] = total_rounds
            pc_sync_rounds[ai, si] = pc_rounds
            pa_sync_rounds[ai, si] = pa_rounds
            feature_rounds_mat[ai, si] = feature_rounds
            compute_delay[ai, si] = lookup_compute_delay_ms(algo["pa"], algo["mode"], si, perf)
            pc_compute_delay[ai, si] = lookup_pc_compute_delay_ms(algo["pc"], algo["mode"], si, perf)
            pa_compute = np.nansum([compute_delay[ai, si]])
            pc_compute = np.nansum([pc_compute_delay[ai, si]])
            if cfg.get("includeModelInferenceEstimate", True):
                pa_compute = max(pa_compute, model_ms)
            pa_control_delay[ai, si] = pa_sync_ms + feature_ms + (pa_compute if cfg["includeComputeTime"] else 0.0)
            pc_control_delay[ai, si] = pc_sync_ms + (pc_compute if cfg["includeComputeTime"] else 0.0)
            control_delay[ai, si] = pc_control_delay[ai, si] + pa_control_delay[ai, si]

    return {
        "SNR_dB": snr_db,
        "algoTable": algo_table,
        "ESR_mean": ESR_mean,
        "avgESR": avg_esr,
        "sync_delay_ms": sync_delay,
        "pc_sync_delay_ms": pc_sync_delay,
        "pa_sync_delay_ms": pa_sync_delay,
        "feature_collection_delay_ms": feature_delay,
        "model_inference_est_ms": model_inference_delay,
        "pc_control_delay_ms": pc_control_delay,
        "pa_control_delay_ms": pa_control_delay,
        "compute_delay_ms": compute_delay,
        "pc_compute_delay_ms": pc_compute_delay,
        "control_delay_ms": control_delay,
        "sync_bytes": sync_bytes,
        "pc_sync_bytes": pc_sync_bytes,
        "pa_sync_bytes": pa_sync_bytes,
        "feature_collection_bytes": feature_bytes_mat,
        "sync_rounds": sync_rounds,
        "pc_sync_rounds": pc_sync_rounds,
        "pa_sync_rounds": pa_sync_rounds,
        "feature_collection_rounds": feature_rounds_mat,
        "config": cfg,
    }


def estimate_pc_sync(pc_name: str, L: int, K: int, N: int, nbr: int, n_iter: int, payload_ratio: float) -> tuple[float, float]:
    lk_bytes = L * K * 8.0 * payload_ratio
    if pc_name == "MR":
        return 0.0, 0.0
    if pc_name == "LMMSE":
        return 1.0, lk_bytes
    if pc_name == "RMMSE":
        return float(max(n_iter, 1)), lk_bytes * max(n_iter, 1)
    if pc_name == "LMMSE_G":
        return 1.0, L * N * K * nbr * 16.0 * payload_ratio
    return 0.0, 0.0


def estimate_pa_sync(pa_name: str, L: int, K: int, cfg: dict[str, Any], payload_ratio: float) -> tuple[float, float]:
    lk_complex_bytes = L * K * 16.0 * payload_ratio
    if pa_name == "WMMSE":
        return float(cfg["wmmseRounds"]), lk_complex_bytes * cfg["wmmseRounds"]
    if pa_name == "DWMMSE":
        return float(cfg["dwmmseRounds"]), K * 8.0 * cfg["dwmmseRounds"]
    return 0.0, 0.0


def bytes_to_ms(num_bytes: float, fronthaul_mbps: float) -> float:
    return float(num_bytes) * 8.0 / (fronthaul_mbps * 1e6) * 1e3


def lookup_feature_collection_bytes(
    pa_name: str,
    mode_name: str,
    snr_idx: int,
    L: int,
    K: int,
    perf: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[float, float]:
    if pa_name == "LocalGNN":
        return 0.0, 0.0
    method = {"GNN": "GNN", "DCGNN": "DCGNN", "UGNN": "U-GNN", "DQN": "DQN", "DDPG": "DDPG"}.get(pa_name)
    if method is None:
        return 0.0, 0.0
    if pa_name == "DCGNN":
        z = int(cfg.get("dcgnnTopZ", 15))
        edge_count = min(L * K, z * (L + K))
        bytes_value = float(edge_count * cfg.get("edgeFeatureBytes", 8.0) + (L + K) * 8.0)
    elif pa_name in {"DQN", "DDPG"}:
        bytes_value = float(L * K * 12.0)
    else:
        bytes_value = float(L * K * 16.0)
    try:
        mi = perf["modeNames"].index(mode_name)
        pi = perf["methodNames"].index(method)
        measured = float(perf["comm_bytes"][pi, snr_idx, mi])
        if np.isfinite(measured) and measured > 0:
            bytes_value = max(bytes_value, measured)
    except Exception:
        pass
    return bytes_value, 1.0


def estimate_model_inference_ms(pa_name: str, L: int, K: int, cfg: dict[str, Any], perf: dict[str, Any]) -> float:
    """Architecture-aware PA inference estimate in milliseconds."""

    if not cfg.get("includeModelInferenceEstimate", True):
        return 0.0
    hidden = int(cfg.get("gnnHiddenDim", 128))
    layers = int(cfg.get("gnnLayers", 3))
    heads = int(cfg.get("gnnHeads", 4))
    edge_count = L * K
    gops = float(cfg.get("inferenceGops", 80.0))
    controller_gops = float(cfg.get("controllerGops", 200.0))

    if pa_name in {"GNN", "UGNN"}:
        param_ops = lookup_model_param_count(pa_name, perf) * 2.0
        message_ops = edge_count * layers * hidden * max(heads, 1) * 2.0
        if pa_name == "UGNN":
            message_ops *= 1.15
        return ops_to_ms(param_ops + message_ops, controller_gops)
    if pa_name == "DCGNN":
        z = int(cfg.get("dcgnnTopZ", 15))
        dc_edges = min(edge_count, z * (L + K))
        param_ops = lookup_model_param_count(pa_name, perf) * 2.0
        message_ops = dc_edges * layers * hidden * max(heads, 1) * 2.0
        return ops_to_ms(param_ops + message_ops, controller_gops)
    if pa_name == "LocalGNN":
        local_hidden = int(cfg.get("localHiddenDim", 96))
        local_layers = int(cfg.get("localLayers", 3))
        input_dim = 2 * K + 6
        ops_per_ap = input_dim * local_hidden + max(local_layers - 1, 0) * local_hidden * local_hidden + local_hidden * K
        return ops_to_ms(2.0 * L * ops_per_ap, gops)
    if pa_name in {"DQN", "DDPG"}:
        hidden_dim = int(cfg.get("rlHiddenDim", 256))
        input_dim = 4 * L * K + 4
        output_dim = 6 if pa_name == "DQN" else L * K
        layers = int(cfg.get("rlLayers", 3))
        ops = input_dim * hidden_dim + max(layers - 2, 0) * hidden_dim * hidden_dim + hidden_dim * output_dim
        if pa_name == "DDPG":
            ops *= 1.25
        return ops_to_ms(2.0 * ops, controller_gops)
    return 0.0


def lookup_model_param_count(pa_name: str, perf: dict[str, Any]) -> float:
    counts = perf.get("model_param_count", {}) if isinstance(perf, dict) else {}
    key_map = {"GNN": "GNN", "DCGNN": "DCGNN", "UGNN": "UGNN", "LocalGNN": "Local-GNN", "DQN": "DQN", "DDPG": "DDPG"}
    value = counts.get(key_map.get(pa_name, pa_name), 0.0)
    if value and np.isfinite(value):
        return float(value)
    return {
        "GNN": 400_000.0,
        "DCGNN": 400_000.0,
        "UGNN": 420_000.0,
        "LocalGNN": 110_000.0,
        "DQN": 2_100_000.0,
        "DDPG": 5_200_000.0,
    }.get(pa_name, 0.0)


def ops_to_ms(ops: float, gops: float) -> float:
    if gops <= 0:
        return 0.0
    return float(ops) / (gops * 1e9) * 1e3


def lookup_pc_compute_delay_ms(pc_name: str, mode_name: str, snr_idx: int, perf: dict[str, Any]) -> float:
    try:
        pc_order = ["MR", "LMMSE", "RMMSE", "LMMSE_G"]
        ci = pc_order.index(pc_name)
        mi = perf["modeNames"].index(mode_name)
        return float(perf["time_pc_sec"][ci, snr_idx, mi] * 1000.0)
    except Exception:
        return 0.0


def lookup_compute_delay_ms(pa_name: str, mode_name: str, snr_idx: int, perf: dict[str, Any]) -> float:
    if pa_name in {"EPA", "random"}:
        return 0.0
    method = pa_to_method(pa_name)
    if method is None:
        return float("nan")
    try:
        mi = perf["modeNames"].index(mode_name)
        pi = perf["methodNames"].index(method)
        return float(perf["time_pa_sec"][pi, snr_idx, mi] * 1000.0)
    except Exception:
        return float("nan")


def write_sync_ablation_csv(output_file: Path, ablation: dict[str, Any]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    metric_names = [
        "ESR_mean",
        "sync_delay_ms",
        "control_delay_ms",
        "pc_control_delay_ms",
        "pa_control_delay_ms",
        "pc_sync_delay_ms",
        "pa_sync_delay_ms",
        "feature_collection_delay_ms",
        "model_inference_est_ms",
        "compute_delay_ms",
        "pc_compute_delay_ms",
        "sync_bytes",
        "pc_sync_bytes",
        "pa_sync_bytes",
        "feature_collection_bytes",
        "sync_rounds",
        "pc_sync_rounds",
        "pa_sync_rounds",
        "feature_collection_rounds",
    ]
    with output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "table",
            "id",
            "algorithm",
            "pa",
            "pc",
            "mode",
            "pc_arch",
            "pa_arch",
            "is_distributed",
            "snr_db",
            "metric",
            "value",
            "avg_esr",
        ])
        for i, algo in enumerate(ablation["algoTable"]):
            base = [
                algo["id"],
                algo["name"],
                algo["pa"],
                algo["pc"],
                algo["mode"],
                algo["pcArch"],
                algo["paArch"],
                int(algo["isDistributed"]),
            ]
            for metric in metric_names:
                values = np.asarray(ablation[metric])[i, :]
                avg_value = np.nanmean(values)
                for si, snr in enumerate(ablation["SNR_dB"]):
                    writer.writerow(["detail", *base, f"{float(snr):.8g}", metric, f"{values[si]:.8g}", f"{ablation['avgESR'][i]:.8g}"])
                writer.writerow(["summary", *base, "", f"{metric}_avg", f"{avg_value:.8g}", f"{ablation['avgESR'][i]:.8g}"])
