#!/usr/bin/env python3
"""Pure-Python main simulation entrypoint for CF_downlink_sim."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from simulator import (
    build_algo_table,
    build_sync_ablation_metrics,
    channel_estimates,
    compute_rho_distributed_wmmse,
    compute_rho_dist,
    compute_rho_epa,
    compute_rho_fpcp,
    compute_rho_gnn,
    compute_rho_local_gnn,
    compute_rho_random,
    compute_rho_rl,
    compute_se,
    db2pow,
    default_params,
    fingerprint,
    generate_setup,
    method_names,
    pa_to_method,
    precoding_lmmse,
    precoding_lmmse_global,
    precoding_mr,
    precoding_robust_mmse,
    print_final_results,
    write_simulation_results_csv,
    write_sync_ablation_csv,
)
from plots import plot_all_ablation, plot_all_esr, plot_scenario_setup
from config import FIGURE_DIR, MODEL_DIR, PROJECT_ROOT, SIMULATION_DATA_DIR


TRAD_PA = ["baseline", "random", "EPA", "FPCP", "DWMMSE"]
LEARNED_PA = ["LocalGNN", "DCGNN", "DQN", "DDPG"]
ALL_PA = TRAD_PA + LEARNED_PA
ALL_PC = ["MR", "LMMSE", "RMMSE", "LMMSE_G"]


def parse_csv_list(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CF downlink simulation fully in Python")
    parser.add_argument("--num-scenarios", type=int, default=None)
    parser.add_argument("--realizations", type=int, default=None)
    parser.add_argument("--snr-db", type=float, nargs="+", default=None, help="Override SNR grid, e.g. --snr-db 5 10")
    parser.add_argument("--run-stage", type=int, choices=[1, 2, 3], default=None, help="1=traditional, 2=learning, 3=all")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-fig", action="store_true")
    parser.add_argument("--no-data", action="store_true")
    parser.add_argument("--no-sync-ablation", action="store_true")
    parser.add_argument("--pa", type=str, default="", help=f"Comma-separated PA filter. Choices: {','.join(ALL_PA)}")
    parser.add_argument("--pc", type=str, default="", help=f"Comma-separated PC filter. Choices: {','.join(ALL_PC)}")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--clean-figures", action="store_true")
    parser.add_argument("--verbose-algo", action="store_true")
    return parser.parse_args()


def resolved_model_paths(params: dict[str, Any]) -> dict[str, Path]:
    return {
        "LocalGNN": PROJECT_ROOT / params["gnn"]["localModelFile"],
        "DCGNN": PROJECT_ROOT / params["gnn"]["dcgnnModelFile"],
        "DQN": PROJECT_ROOT / params["rl"]["dqnModelFile"],
        "DDPG": PROJECT_ROOT / params["rl"]["ddpgModelFile"],
    }


def count_model_parameters(model_path: Path) -> int:
    if not model_path.is_file():
        return 0
    try:
        import torch

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint)
        if "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        if "actor_state_dict" in checkpoint:
            state = checkpoint["actor_state_dict"]
        if not isinstance(state, dict):
            return 0
        total = 0
        for value in state.values():
            if hasattr(value, "numel"):
                total += int(value.numel())
        return total
    except Exception:
        return 0


def selected_pa_keys(args: argparse.Namespace, params: dict[str, Any]) -> list[str]:
    stage = int(params["runtime"]["runStage"])
    if stage == 1:
        keys = TRAD_PA.copy()
    elif stage == 2:
        keys = LEARNED_PA.copy()
    else:
        keys = ALL_PA.copy()
    requested = parse_csv_list(args.pa, keys)
    unknown = [key for key in requested if key not in ALL_PA]
    if unknown:
        raise SystemExit(f"Unknown PA key(s): {unknown}. Choices: {ALL_PA}")
    return [key for key in keys if key in requested]


def selected_pc_keys(args: argparse.Namespace) -> list[str]:
    requested = parse_csv_list(args.pc, ALL_PC)
    unknown = [key for key in requested if key not in ALL_PC]
    if unknown:
        raise SystemExit(f"Unknown PC key(s): {unknown}. Choices: {ALL_PC}")
    return [key for key in ALL_PC if key in requested]


def _shape_to_text(shape: tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in shape)


def _text_to_shape(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("x") if part)


def _scenario_array_names() -> set[str]:
    return {
        "gainOverNoise",
        "R",
        "pilotIndex",
        "D_dcc",
        "Hhat",
        "H_ideal",
        "C",
        "H",
        "APpositions",
        "UEpositions",
    }


def load_scenario_cache(cache_file: Path, scenario_idx: int, expected_fp: str) -> dict[str, np.ndarray] | None:
    if not cache_file.is_file():
        return None
    try:
        arrays: dict[str, dict[str, Any]] = {}
        with cache_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["scenario"]) != scenario_idx or row["paramFingerprint"] != expected_fp:
                    continue
                name = row["array"]
                if name not in _scenario_array_names():
                    continue
                entry = arrays.setdefault(
                    name,
                    {
                        "shape": _text_to_shape(row["shape"]),
                        "dtype": row["dtype"],
                        "is_complex": int(row["is_complex"]),
                        "values": {},
                    },
                )
                real = float(row["value_real"])
                imag = float(row["value_imag"] or 0.0)
                entry["values"][int(row["index"])] = real + 1j * imag if entry["is_complex"] else real
        if not arrays:
            return None

        scenario: dict[str, np.ndarray] = {}
        for name, entry in arrays.items():
            dtype = np.complex128 if entry["is_complex"] else np.dtype(entry["dtype"])
            expected_size = int(np.prod(entry["shape"]))
            if len(entry["values"]) != expected_size:
                return None
            values = [entry["values"][idx] for idx in range(expected_size)]
            arr = np.asarray(values, dtype=dtype).reshape(entry["shape"])
            scenario[name] = arr
        missing = _scenario_array_names() - set(scenario)
        if missing:
            return None
        return scenario
    except Exception:
        return None


def save_scenario_cache(cache_file: Path, scenario_idx: int, fp: str, scenario: dict[str, np.ndarray]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_file.with_suffix(".tmp")
    fieldnames = [
        "scenario",
        "paramFingerprint",
        "array",
        "shape",
        "dtype",
        "is_complex",
        "index",
        "value_real",
        "value_imag",
    ]
    with tmp_file.open("w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        if cache_file.is_file():
            with cache_file.open("r", encoding="utf-8", newline="") as in_f:
                reader = csv.DictReader(in_f)
                for row in reader:
                    try:
                        if int(row["scenario"]) == scenario_idx:
                            continue
                    except Exception:
                        continue
                    writer.writerow(row)

        for name in sorted(_scenario_array_names()):
            arr = np.asarray(scenario[name])
            flat = arr.ravel(order="C")
            is_complex = int(np.iscomplexobj(arr))
            for idx, value in enumerate(flat):
                writer.writerow(
                    {
                        "scenario": scenario_idx,
                        "paramFingerprint": fp,
                        "array": name,
                        "shape": _shape_to_text(arr.shape),
                        "dtype": str(arr.dtype),
                        "is_complex": is_complex,
                        "index": idx,
                        "value_real": f"{float(np.real(value)):.17g}",
                        "value_imag": f"{float(np.imag(value)):.17g}" if is_complex else "",
                    }
                )
    tmp_file.replace(cache_file)


def _model_cache_signature(model_paths: dict[str, Path]) -> list[tuple[str, str, float, int]]:
    signature: list[tuple[str, str, float, int]] = []
    for key in sorted(model_paths):
        path = model_paths[key]
        if path.is_file():
            stat = path.stat()
            signature.append((key, str(path), stat.st_mtime, stat.st_size))
        else:
            signature.append((key, str(path), 0.0, 0))
    return signature


def snr_cache_fingerprint(
    scenario_fp: str,
    snr: float,
    pa_keys: list[str],
    pc_keys: list[str],
    modes: list[str],
    params: dict[str, Any],
    model_paths: dict[str, Path],
) -> str:
    """Fingerprint the SNR-level simulation work for safe result reuse."""

    return fingerprint(
        "python_snr_cache_v1",
        scenario_fp,
        float(snr),
        pa_keys,
        pc_keys,
        modes,
        params["runtime"]["runStage"],
        params["dwmmse"],
        params["fpcp"],
        params["csi"],
        _model_cache_signature(model_paths),
    )


def load_snr_cache(cache_file: Path, expected_fp: str, num_algos: int, perf_shape: tuple[int, int]) -> dict[str, np.ndarray] | None:
    if not cache_file.is_file():
        return None
    try:
        with np.load(cache_file, allow_pickle=False) as data:
            stored_fp = str(data["fingerprint"])
            if stored_fp != expected_fp:
                return None
            esr = np.asarray(data["esr"], dtype=np.float64)
            if esr.shape != (num_algos,):
                return None
            time_pa = np.asarray(data["time_pa_sec"], dtype=np.float64)
            time_core = np.asarray(data["time_core_sec"], dtype=np.float64)
            comm = np.asarray(data["comm_bytes"], dtype=np.float64)
            time_pc = np.asarray(data["time_pc_sec"], dtype=np.float64)
            if time_pa.shape != perf_shape or time_core.shape != perf_shape or comm.shape != perf_shape:
                return None
            return {
                "esr": esr,
                "time_pa_sec": time_pa,
                "time_core_sec": time_core,
                "comm_bytes": comm,
                "time_pc_sec": time_pc,
            }
    except Exception:
        return None


def save_snr_cache(
    cache_file: Path,
    fp: str,
    esr: np.ndarray,
    time_pa: np.ndarray,
    time_core: np.ndarray,
    comm: np.ndarray,
    time_pc: np.ndarray,
) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_file.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp_file,
        fingerprint=np.asarray(fp),
        esr=np.asarray(esr, dtype=np.float64),
        time_pa_sec=np.asarray(time_pa, dtype=np.float64),
        time_core_sec=np.asarray(time_core, dtype=np.float64),
        comm_bytes=np.asarray(comm, dtype=np.float64),
        time_pc_sec=np.asarray(time_pc, dtype=np.float64),
    )
    tmp_file.replace(cache_file)


def compute_precoder(
    pc_key: str,
    Hhat: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    nbr: int,
    N: int,
    K: int,
    L: int,
    p: float,
    Pt: float,
    sigma_e: float,
    n_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    if pc_key == "MR":
        return precoding_mr(Hhat, nbr, N, K, L)
    if pc_key == "LMMSE":
        return precoding_lmmse(Hhat, D, C, nbr, N, K, L, p)
    if pc_key == "RMMSE":
        return precoding_robust_mmse(Hhat, D, nbr, N, K, L, Pt, sigma_e, n_iter)
    if pc_key == "LMMSE_G":
        return precoding_lmmse_global(Hhat, D, C, nbr, N, K, L, p)
    raise ValueError(f"Unknown precoder: {pc_key}")


def compute_pa_once(
    pa_key: str,
    D: np.ndarray,
    gain: np.ndarray,
    Hhat: np.ndarray,
    Pt: float,
    N: int,
    K: int,
    L: int,
    nbr: int,
    params: dict[str, Any],
    model_paths: dict[str, Path],
) -> tuple[np.ndarray | None, dict[str, float]]:
    t0 = time.perf_counter()
    if pa_key == "baseline":
        rho = compute_rho_dist(D, gain, Pt)
        return rho, {"total_sec": time.perf_counter() - t0, "forward_sec": time.perf_counter() - t0}
    if pa_key == "EPA":
        rho = compute_rho_epa(D, Pt)
        return rho, {"total_sec": time.perf_counter() - t0, "forward_sec": time.perf_counter() - t0}
    if pa_key == "FPCP":
        rho = compute_rho_fpcp(D, gain, Pt, params["fpcp"]["alpha"])
        return rho, {"total_sec": time.perf_counter() - t0, "forward_sec": time.perf_counter() - t0}
    if pa_key == "DWMMSE":
        rho, info = compute_rho_distributed_wmmse(D, gain, Pt, params["dwmmse"]["rounds"], params["dwmmse"]["damping"])
        timing = {"total_sec": time.perf_counter() - t0, "forward_sec": time.perf_counter() - t0}
        timing["messageBytes"] = float(info.get("messageBytes", 0.0))
        return rho, timing
    if pa_key == "LocalGNN":
        return compute_rho_local_gnn(D, gain, Pt, model_paths["LocalGNN"], params["csi"]["sigma_e"])
    if pa_key == "DCGNN":
        return compute_rho_gnn(D, gain, Pt, model_paths["DCGNN"], params["csi"]["sigma_e"])
    if pa_key == "DQN":
        return compute_rho_rl(D, gain, Pt, model_paths["DQN"], params["csi"]["sigma_e"])
    if pa_key == "DDPG":
        return compute_rho_rl(D, gain, Pt, model_paths["DDPG"], params["csi"]["sigma_e"])
    if pa_key == "random":
        return None, {"total_sec": 0.0, "forward_sec": 0.0}
    raise ValueError(f"Unknown PA key: {pa_key}")


def update_perf(
    perf: dict[str, Any],
    pa_key: str,
    snr_idx: int,
    mode_idx: int,
    timing: dict[str, float],
    comm_bytes: float = 0.0,
) -> None:
    method = pa_to_method(pa_key)
    if method is None:
        return
    try:
        pi = perf["methodNames"].index(method)
    except ValueError:
        return
    perf["time_pa_sec"][pi, snr_idx, mode_idx] = float(timing.get("total_sec", 0.0))
    core = float(timing.get("forward_sec", timing.get("python_total_sec", timing.get("total_sec", 0.0))))
    perf["time_core_sec"][pi, snr_idx, mode_idx] = core
    perf["comm_bytes"][pi, snr_idx, mode_idx] = float(max(comm_bytes, timing.get("messageBytes", 0.0)))


def run_simulation(args: argparse.Namespace) -> None:
    params = default_params()
    if args.num_scenarios is not None:
        params["simulation"]["numScenarios"] = args.num_scenarios
    if args.realizations is not None:
        params["simulation"]["nbrOfRealizations"] = args.realizations
    if args.snr_db is not None:
        params["power"]["SNR_dB"] = list(args.snr_db)
    if args.run_stage is not None:
        params["runtime"]["runStage"] = args.run_stage
    if args.no_cache:
        params["runtime"]["useCache"] = False
    if args.no_fig:
        params["output"]["isSaveFig"] = False
    if args.no_data:
        params["output"]["isSaveData"] = False
    if args.no_sync_ablation:
        params["syncAblation"]["enable"] = False
    if args.seed is not None:
        params["simulation"]["seed"] = args.seed
    if args.verbose_algo:
        params["runtime"]["verboseAlgo"] = True

    L = int(params["system"]["L"])
    K = int(params["system"]["K"])
    N = int(params["system"]["N"])
    tau_c = int(params["system"]["tau_c"])
    tau_p = int(params["system"]["tau_p"])
    p = float(params["power"]["p"])
    snr_db = np.asarray(params["power"]["SNR_dB"], dtype=float)
    sigma_e = float(params["csi"]["sigma_e"])
    n_iter = int(params["csi"]["nIter"])
    nbr = int(params["simulation"]["nbrOfRealizations"])
    num_scenarios = int(params["simulation"]["numScenarios"])
    modes = list(params["simulation"]["accessModes"])
    use_cache = bool(params["runtime"]["useCache"])

    pa_keys = selected_pa_keys(args, params)
    pc_keys = selected_pc_keys(args)
    algo_table = build_algo_table(pa_keys, pc_keys, modes)
    model_paths = resolved_model_paths(params)

    figure_dir = FIGURE_DIR
    data_dir = SIMULATION_DATA_DIR
    if params["output"]["isSaveFig"] and (args.clean_figures or params["output"]["cleanOldFigures"]):
        if figure_dir.exists():
            shutil.rmtree(figure_dir)
        figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n=====================================================================")
    print("       Cell-Free Downlink Simulation  (pure Python)")
    print("=====================================================================")
    print(f"  System:  L={L} APs  N={N} antennas  K={K} UEs")
    print(f"  Channel: tau_c={tau_c}  tau_p={tau_p}  ASD={np.degrees(params['channel']['ASD_varphi']):.0f} deg")
    print(f"  CSI Error: sigma_e={sigma_e:.2f}  nIter={n_iter}")
    print(f"  SNR: {snr_db.tolist()} dB")
    print(f"  Scenarios: {num_scenarios} x {nbr} realizations")
    print(f"  Algorithms: {len(algo_table)}  PA={pa_keys}  PC={pc_keys}")
    print(f"  Stage: {params['runtime']['runStage']}  |  Cache: {use_cache}")
    print(f"  Models: {MODEL_DIR}")
    print("=====================================================================")

    num_snr = len(snr_db)
    num_algos = len(algo_table)
    ESR_acc = np.zeros((num_algos, num_snr), dtype=np.float64)
    perf = {
        "methodNames": method_names(),
        "modeNames": modes,
        "SNR_dB": snr_db,
        "time_pa_sec": np.zeros((len(method_names()), num_snr, len(modes))),
        "time_core_sec": np.zeros((len(method_names()), num_snr, len(modes))),
        "comm_bytes": np.zeros((len(method_names()), num_snr, len(modes))),
        "time_pc_sec": np.zeros((len(ALL_PC), num_snr, len(modes))),
        "model_param_count": {
            "Local-GNN": count_model_parameters(model_paths["LocalGNN"]),
            "DCGNN": count_model_parameters(model_paths["DCGNN"]),
            "DQN": count_model_parameters(model_paths["DQN"]),
            "DDPG": count_model_parameters(model_paths["DDPG"]),
        },
    }

    scenario_fp = fingerprint(L, K, N, tau_p, tau_c, params["channel"]["ASD_varphi"], params["channel"]["ASD_theta"], nbr, sigma_e, p, n_iter)
    base_seed = int(params["simulation"]["seed"])

    total_iters = num_scenarios * num_snr * num_algos
    completed = 0
    print(f"\n  STARTING MAIN LOOP | {num_scenarios} scenarios x {num_snr} SNR x {num_algos} algos = {total_iters} iters")

    for s in range(1, num_scenarios + 1):
        print(f"\n  === Scenario {s}/{num_scenarios} ===")
        cache_file = data_dir / "Scenario_Cache_python.csv"
        scenario = load_scenario_cache(cache_file, s, scenario_fp) if use_cache else None

        if scenario is None:
            print("  [1/3] Generating scenario layout and channel estimates...")
            setup = generate_setup(
                L,
                K,
                N,
                tau_p,
                1,
                s,
                float(params["channel"]["ASD_varphi"]),
                float(params["channel"]["ASD_theta"]),
                params["channel"],
            )
            R = setup["R"][:, :, :, :, 0]
            pilot_index = setup["pilotIndex"][:, 0]
            D_dcc = setup["D"][:, :, 0]
            gain = setup["gainOverNoise"][:, :, 0]
            rng = np.random.default_rng(base_seed + 1000 * s)
            Hhat, H_ideal, _, C = channel_estimates(R, nbr, L, K, N, tau_p, pilot_index, p, rng)
            H = Hhat + np.sqrt(sigma_e**2 / 2.0) * (
                rng.standard_normal(Hhat.shape) + 1j * rng.standard_normal(Hhat.shape)
            )
            scenario = {
                "gainOverNoise": gain,
                "R": R,
                "pilotIndex": pilot_index,
                "D_dcc": D_dcc,
                "Hhat": Hhat,
                "H_ideal": H_ideal,
                "C": C,
                "H": H,
                "APpositions": setup["APpositions"],
                "UEpositions": setup["UEpositions"],
            }
            if use_cache:
                save_scenario_cache(cache_file, s, scenario_fp, scenario)
        else:
            print("  [CACHE HIT] Scenario loaded")

        # Plot scenario layout for first scenario
        if s <= 1:
            ap_pos = scenario.get("APpositions", np.array([]))
            ue_pos = scenario.get("UEpositions", np.array([]))
            if ap_pos.size > 0 and ue_pos.size > 0:
                plot_scenario_setup(ap_pos, ue_pos, s, figure_dir,
                                    enabled=bool(params["output"]["isSaveFig"]))

        gain = scenario["gainOverNoise"]
        D = scenario["D_dcc"]
        Hhat = scenario["Hhat"]
        H = scenario["H"]
        C = scenario["C"]

        print(f"  [2/3] SNR scan ({num_snr} points)")
        for si, snr in enumerate(snr_db):
            Pt = float(db2pow(snr))
            print(f"  |   SNR {si + 1:2d}/{num_snr:2d} ({snr:5.1f} dB)")
            mode = modes[0]
            mode_idx = 0
            snr_fp = snr_cache_fingerprint(scenario_fp, float(snr), pa_keys, pc_keys, modes, params, model_paths)
            snr_cache_file = data_dir / f"cache_snr_python_s{s}_sn{snr:.0f}.npz"
            perf_shape = (len(method_names()), len(modes))
            cached_snr = load_snr_cache(snr_cache_file, snr_fp, num_algos, perf_shape) if use_cache else None
            if cached_snr is not None:
                ESR_acc[:, si] += cached_snr["esr"]
                perf["time_pa_sec"][:, si, :] += cached_snr["time_pa_sec"]
                perf["time_core_sec"][:, si, :] += cached_snr["time_core_sec"]
                perf["comm_bytes"][:, si, :] += cached_snr["comm_bytes"]
                perf["time_pc_sec"][:, si, :] += cached_snr["time_pc_sec"]
                completed += num_algos
                print("  |   [CACHE HIT] SNR result loaded")
                continue

            esr_local = np.zeros(num_algos, dtype=np.float64)
            time_pa_before = perf["time_pa_sec"][:, si, :].copy()
            time_core_before = perf["time_core_sec"][:, si, :].copy()
            comm_before = perf["comm_bytes"][:, si, :].copy()
            time_pc_before = perf["time_pc_sec"][:, si, :].copy()

            precoders: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for pc_key in pc_keys:
                pc_t0 = time.perf_counter()
                precoders[pc_key] = compute_precoder(pc_key, Hhat, D, C, nbr, N, K, L, p, Pt, sigma_e, n_iter)
                try:
                    pc_idx = ALL_PC.index(pc_key)
                    perf["time_pc_sec"][pc_idx, si, mode_idx] = time.perf_counter() - pc_t0
                except ValueError:
                    pass

            pa_cache: dict[str, tuple[np.ndarray | None, dict[str, float]]] = {}
            for pa_key in pa_keys:
                if pa_key == "random":
                    continue
                rho, timing = compute_pa_once(pa_key, D, gain, Hhat, Pt, N, K, L, nbr, params, model_paths)
                pa_cache[pa_key] = (rho, timing)
                centralized_learned = {"DCGNN", "DQN", "DDPG"}
                update_perf(perf, pa_key, si, mode_idx, timing, comm_bytes=L * K * 16 if pa_key in centralized_learned else 0.0)

            for ai, algo in enumerate(algo_table):
                if algo["mode"] != mode:
                    continue
                V, scaling = precoders[algo["pc"]]
                if algo["pa"] == "random":
                    rng = np.random.default_rng(base_seed + s * 100000 + si * 100 + ai)
                    rho = compute_rho_random(scaling, D, Pt, rng)
                    timing = {"total_sec": 0.0, "forward_sec": 0.0}
                else:
                    rho, timing = pa_cache[algo["pa"]]
                    if rho is None:
                        raise RuntimeError(f"PA cache unexpectedly empty for {algo['pa']}")

                se = compute_se(H, V, scaling, D, tau_c, tau_p, nbr, N, K, L, rho)
                esr = float(np.sum(se))
                ESR_acc[ai, si] += esr
                esr_local[ai] = esr
                if params["runtime"]["verboseAlgo"]:
                    print(f"  |   |  {algo['name']:<28} ESR={esr:8.3f} PA={timing.get('total_sec', 0.0) * 1000:.3f} ms")
            if use_cache:
                save_snr_cache(
                    snr_cache_file,
                    snr_fp,
                    esr_local,
                    perf["time_pa_sec"][:, si, :] - time_pa_before,
                    perf["time_core_sec"][:, si, :] - time_core_before,
                    perf["comm_bytes"][:, si, :] - comm_before,
                    perf["time_pc_sec"][:, si, :] - time_pc_before,
                )
            completed += num_algos

    ESR_mean = ESR_acc / max(num_scenarios, 1)
    ESR_best = np.max(ESR_mean, axis=0)
    best_idx = np.argmax(ESR_mean, axis=0)
    ESR_best_algo = [algo_table[int(idx)]["name"] for idx in best_idx]

    perf["time_pa_sec"] /= max(num_scenarios, 1)
    perf["time_core_sec"] /= max(num_scenarios, 1)
    perf["comm_bytes"] /= max(num_scenarios, 1)
    perf["time_pc_sec"] /= max(num_scenarios, 1)

    print("\n=====================================================================")
    print(f"  SIMULATION COMPLETED | Total: {completed} iterations")
    print("=====================================================================")
    print_final_results(ESR_mean, algo_table, snr_db, num_scenarios, nbr)

    if params["output"]["isSaveData"]:
        results_file = data_dir / "Simulation_Results_v2_python.csv"
        write_simulation_results_csv(results_file, ESR_mean, ESR_best, ESR_best_algo, algo_table, snr_db, perf)
        print(f"[INFO] Python simulation results saved to CSV: {results_file}")

    fig_enabled = bool(params["output"]["isSaveFig"])
    plot_all_esr(ESR_mean, algo_table, snr_db, perf, figure_dir, enabled=fig_enabled)

    if params["syncAblation"]["enable"] and params["output"]["isSaveData"]:
        sync_cfg = dict(params["syncAblation"])
        sync_cfg["dwmmseRounds"] = params["dwmmse"]["rounds"]
        ablation = build_sync_ablation_metrics(ESR_mean, algo_table, snr_db, perf, L, K, N, nbr, n_iter, sync_cfg)
        ablation_file = data_dir / "Sync_Ablation_Results_python.csv"
        write_sync_ablation_csv(ablation_file, ablation)
        print(f"[INFO] Python sync ablation results saved to CSV: {ablation_file}")
        plot_all_ablation(ablation, figure_dir, enabled=fig_enabled)

    print(f"\n>>> Python run done. Figures: {figure_dir}")
    print(f">>> Python data: {data_dir}")


def main() -> int:
    args = parse_args()
    run_simulation(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
