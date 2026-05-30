"""Pure-Python visualization module matching MATLAB plots.

Generates the same set of figures as the MATLAB visualization/ scripts:
- Scenario layout (AP/UE positions)
- ESR vs SNR curves (best PA, fixed R-MMSE)
- Learning-family gap to D-WMMSE (bar chart)
- Method summary table
- Timing comparison (2x2)
- Sync ablation: PA ranking, heatmap, per-PC dual-axis plots
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# ── Shared style constants (match MATLAB paColors / paMarkers) ──────────

PA_ORDER = ["LocalGNN", "DCGNN", "DDPG", "DQN",
            "DWMMSE", "FPCP", "EPA", "random", "baseline"]
PA_LABELS = ["Local-GNN", "DCGNN", "DDPG", "DQN",
             "D-WMMSE", "FPCP", "EPA", "Random", "Baseline"]
PA_COLORS = [
    (0.00, 0.52, 0.58), (0.10, 0.35, 0.95),
    (0.80, 0.25, 0.65), (0.35, 0.45, 0.90),
    (0.95, 0.55, 0.10), (0.10, 0.55, 0.25),
    (0.00, 0.60, 0.00), (0.55, 0.55, 0.55), (0.20, 0.20, 0.20),
]
PA_MARKERS = ['x', '^', '+', '*', 'h', '>', 'd', 'v', '<']

PC_ORDER = ["MR", "LMMSE", "RMMSE", "LMMSE_G"]
PC_LABELS = ["MR", "L-MMSE", "R-MMSE", "L-MMSE-G"]

DWMMSE_COLOR = (0.95, 0.55, 0.10)
LEFT_COLOR = (0.00, 0.45, 0.74)
RIGHT_COLOR = (0.85, 0.33, 0.10)

FS_AXIS = 12
FS_TITLE = 13
FS_LEG = 10
LW_GNN = 2.8
LW_BASE = 1.8


# ── Helpers ─────────────────────────────────────────────────────────────

def _pc_display(key: str) -> str:
    return {"LMMSE": "L-MMSE", "RMMSE": "R-MMSE",
            "LMMSE_G": "L-MMSE-G"}.get(key, key)


def _find_exact(algo_table, pa_key, pc_key, mode_key):
    for i, a in enumerate(algo_table):
        if a["pa"] == pa_key and a["pc"] == pc_key and a["mode"] == mode_key:
            return i
    return None


def _get_best_curve(ESR_mean, algo_table, pa_key, mode_key):
    idxs = [i for i, a in enumerate(algo_table)
            if a["pa"] == pa_key and a["mode"] == mode_key]
    if not idxs:
        return None
    best = max(idxs, key=lambda i: float(np.mean(ESR_mean[i, :])))
    return best


def _save(fig, save_dir: Path, name: str):
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f"{name}.png", dpi=160, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass


# ── Fig 0: Scenario Layout ─────────────────────────────────────────────

def plot_scenario_setup(ap_positions: np.ndarray, ue_positions: np.ndarray,
                        scenario_idx: int, save_dir: Path, enabled: bool = True):
    if not enabled:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(ap_positions.real, ap_positions.imag, 'b^',
            markersize=8, markeredgewidth=1.5, label='APs')
    ax.plot(ue_positions.real, ue_positions.imag, 'ro',
            markersize=6, markerfacecolor='r', label='UEs')
    ax.set_xlabel('Horizontal Position (m)')
    ax.set_ylabel('Vertical Position (m)')
    ax.set_title(f'Scenario {scenario_idx} Layout: '
                 f'{len(ap_positions)} APs, {len(ue_positions)} UEs')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.grid(True)
    ax.set_aspect('equal')
    fig.tight_layout()
    _save(fig, save_dir, f'Scenario_{scenario_idx}_Layout')


# ── Fig 1: Best PC per PA ───────────────────────────────────────────────

def plot_fig1_best_pa_esr(ESR_mean, algo_table, snr_db, save_dir, enabled=True):
    if not enabled:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    legend_labels = []
    for pi, pa_key in enumerate(PA_ORDER):
        idx = _get_best_curve(ESR_mean, algo_table, pa_key, 'DCC')
        if idx is None:
            continue
        is_learning = pa_key in {'LocalGNN', 'DCGNN', 'DQN', 'DDPG'}
        lw = LW_GNN if is_learning else LW_BASE
        ax.plot(snr_db, ESR_mean[idx, :], f'-{PA_MARKERS[pi]}',
                color=PA_COLORS[pi], linewidth=lw, markersize=7)
        a = algo_table[idx]
        legend_labels.append(f'{PA_LABELS[pi]} + {_pc_display(a["pc"])} ({a["mode"]})')
    ax.set_xlabel('SNR (dB)', fontsize=FS_AXIS)
    ax.set_ylabel('Ergodic Sum Rate (bit/s/Hz)', fontsize=FS_AXIS)
    ax.set_title('Best Precoder per Power Allocation Method', fontsize=FS_TITLE)
    ax.legend(legend_labels, loc='lower right', fontsize=FS_LEG)
    ax.grid(True)
    ax.set_box_aspect(0.6)
    fig.tight_layout()
    _save(fig, save_dir, 'Fig1_Best_PA_ESR')


# ── Fig 2: Fixed R-MMSE PA comparison ──────────────────────────────────

def plot_fig2_rmmse_pa(ESR_mean, algo_table, snr_db, save_dir, enabled=True):
    if not enabled:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    legend_labels = []
    fixed_pc = 'RMMSE'
    for pi, pa_key in enumerate(PA_ORDER):
        idx = _find_exact(algo_table, pa_key, fixed_pc, 'DCC')
        if idx is None:
            continue
        is_learning = pa_key in {'LocalGNN', 'DCGNN', 'DQN', 'DDPG'}
        lw = LW_GNN if is_learning else LW_BASE
        ax.plot(snr_db, ESR_mean[idx, :], f'-{PA_MARKERS[pi]}',
                color=PA_COLORS[pi], linewidth=lw, markersize=7)
        legend_labels.append(f'{PA_LABELS[pi]} ({algo_table[idx]["mode"]})')
    ax.set_xlabel('SNR (dB)', fontsize=FS_AXIS)
    ax.set_ylabel('Ergodic Sum Rate (bit/s/Hz)', fontsize=FS_AXIS)
    ax.set_title('Power Allocation Comparison with R-MMSE Precoding', fontsize=FS_TITLE)
    ax.legend(legend_labels, loc='lower right', fontsize=FS_LEG)
    ax.grid(True)
    ax.set_box_aspect(0.6)
    fig.tight_layout()
    _save(fig, save_dir, 'Fig2_RMMSE_PA_Comparison')


# ── Fig 3: Learning-family gap to D-WMMSE ──────────────────────────────

def plot_fig3_gnn_dwmmse_gap(ESR_mean, algo_table, snr_db, save_dir, enabled=True):
    if not enabled:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(9, 5.6))
    fixed_pc = 'RMMSE'
    dwmmse_idx = _find_exact(algo_table, 'DWMMSE', fixed_pc, 'DCC')
    if dwmmse_idx is None:
        ax.text(0.5, 0.5, 'D-WMMSE R-MMSE DCC reference unavailable',
                transform=ax.transAxes, ha='center', fontsize=FS_AXIS)
        fig.tight_layout()
        _save(fig, save_dir, 'Fig3_GNN_DWMMSE_Gap')
        return

    gnn_keys = ['LocalGNN', 'DCGNN', 'DDPG', 'DQN']
    gnn_labels = ['Local-GNN', 'DCGNN', 'DDPG', 'DQN']
    gap_data, gap_labels, gap_colors = [], [], []
    for gi, gk in enumerate(gnn_keys):
        idx = _find_exact(algo_table, gk, fixed_pc, 'DCC')
        if idx is None:
            continue
        pct = (ESR_mean[idx, :] - ESR_mean[dwmmse_idx, :]) / np.maximum(ESR_mean[dwmmse_idx, :], np.finfo(float).eps) * 100
        gap_data.append(pct)
        gap_labels.append(gnn_labels[gi])
        gap_colors.append(PA_COLORS[gi])

    if not gap_data:
        ax.text(0.5, 0.5, 'Learning-family R-MMSE DCC results unavailable',
                transform=ax.transAxes, ha='center', fontsize=FS_AXIS)
    else:
        gap_mat = np.column_stack(gap_data)
        x = np.arange(len(snr_db))
        width = 0.68 / len(gap_data)
        for gi in range(len(gap_data)):
            ax.bar(x + gi * width - 0.34 + width / 2, gap_mat[:, gi],
                   width, label=gap_labels[gi], color=gap_colors[gi],
                   edgecolor=(0.15, 0.15, 0.15))
        ax.axhline(0, color='k', linestyle='--', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{s:.0f}' for s in snr_db])
        ax.legend(loc='best', fontsize=FS_LEG)

    ax.set_xlabel('SNR (dB)', fontsize=FS_AXIS)
    ax.set_ylabel('ESR Gap to D-WMMSE (%)', fontsize=FS_AXIS)
    ax.set_title('Learning-family Gap to D-WMMSE (R-MMSE, DCC)', fontsize=FS_TITLE)
    ax.grid(True)
    fig.tight_layout()
    _save(fig, save_dir, 'Fig3_GNN_DWMMSE_Gap')


# ── Fig 4: Method Summary Table ─────────────────────────────────────────

def plot_fig4_method_summary(ESR_mean, algo_table, save_dir, enabled=True):
    if not enabled:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    notes = {
        'LocalGNN': 'AP-local learned split',
        'DCGNN': 'dynamic graph learned split',
        'DDPG': 'DDPG reward-trained split', 'DQN': 'DQN reward-trained alpha',
        'DWMMSE': 'fixed-round distributed update',
        'FPCP': 'fractional power control', 'EPA': 'equal power',
        'random': 'random baseline', 'baseline': 'large-scale baseline',
    }

    dwmmse_idx = _get_best_curve(ESR_mean, algo_table, 'DWMMSE', 'DCC')
    dwmmse_avg = float(np.mean(ESR_mean[dwmmse_idx, :])) if dwmmse_idx is not None else 0.0

    rows = []
    for pa_key in PA_ORDER:
        idx = _get_best_curve(ESR_mean, algo_table, pa_key, 'DCC')
        if idx is None:
            continue
        a = algo_table[idx]
        avg = float(np.mean(ESR_mean[idx, :]))
        rows.append([PA_LABELS[PA_ORDER.index(pa_key)],
                      _pc_display(a["pc"]), a["mode"],
                      f'{avg:.2f}', f'{avg - dwmmse_avg:+.2f}',
                      notes.get(pa_key, '')])

    fig, ax = plt.subplots(figsize=(11, max(3, 0.45 * len(rows) + 1.5)))
    ax.axis('off')
    headers = ['PA', 'Best PC', 'Mode', 'Avg ESR', 'vs D-WMMSE', 'Notes']
    table = ax.table(cellText=rows, colLabels=headers, loc='center',
                     cellLoc='left', colLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    # Bold header
    for ci in range(len(headers)):
        table[0, ci].set_text_props(fontweight='bold')
    # Highlight first data row
    if rows:
        for ci in range(len(headers)):
            table[1, ci].set_facecolor((0.96, 0.92, 1.00))
    ax.set_title('Current Method Summary', fontsize=FS_TITLE, pad=20)
    fig.tight_layout()
    _save(fig, save_dir, 'Fig4_Method_Summary')


# ── Fig 5: Timing Comparison (2x2) ──────────────────────────────────────

def plot_fig5_timing(perf, snr_db, save_dir, enabled=True):
    if not enabled:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    methods_avail = perf.get('methodNames', [])
    time_pa = perf.get('time_pa_sec')
    time_core = perf.get('time_core_sec')
    if time_pa is None or not methods_avail:
        return

    def _get_idx(name):
        try:
            return methods_avail.index(name)
        except ValueError:
            return None

    d_idx = _get_idx('D-WMMSE')
    if d_idx is None:
        return

    learn_keys = ['Local-GNN', 'DCGNN', 'DDPG', 'DQN']
    learn_colors = [PA_COLORS[0], PA_COLORS[1], PA_COLORS[2], PA_COLORS[3]]
    active, active_colors = [], []
    for i, nm in enumerate(learn_keys):
        mi = _get_idx(nm)
        if mi is not None:
            active.append((nm, mi))
            active_colors.append(learn_colors[i])
    if not active:
        return

    def _curve(arr, idx):
        v = np.squeeze(np.mean(arr[idx, :, :], axis=1))
        return np.maximum(v.ravel(), np.finfo(float).eps)

    dwmmse_e2e = _curve(time_pa, d_idx)
    e2e = {nm: _curve(time_pa, mi) for nm, mi in active}
    dwmmse_core = _curve(time_core, d_idx) if time_core is not None else dwmmse_e2e
    core = {nm: _curve(time_core, mi) for nm, mi in active} if time_core is not None else e2e

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2))
    fig.suptitle(f'D-WMMSE vs Learning-family Timing: '
                 f'D-WMMSE E2E {np.mean(dwmmse_e2e) * 1000:.3f} ms, '
                 f'Core {np.mean(dwmmse_core) * 1000:.3f} ms',
                 fontsize=FS_TITLE)

    def _plot_time(ax, dwmmse_t, method_t, title, subtitle):
        ax.semilogy(snr_db, dwmmse_t * 1000, '-s',
                    color=DWMMSE_COLOR, linewidth=2.2, markersize=7, label='D-WMMSE')
        for (nm, _), c in zip(active, active_colors):
            ax.semilogy(snr_db, method_t[nm] * 1000, '-o',
                        color=c, linewidth=2.6, markersize=7, label=nm)
        ax.set_xlabel('SNR (dB)', fontsize=FS_AXIS)
        ax.set_ylabel('Time (ms)', fontsize=FS_AXIS)
        ax.set_title(f'{title}\n{subtitle}', fontsize=FS_TITLE)
        ax.legend(loc='best', fontsize=FS_LEG)
        ax.grid(True)

    def _plot_speedup(ax, speedups, title):
        x = np.arange(len(snr_db))
        width = 0.68 / len(active)
        for gi, ((nm, _), c) in enumerate(zip(active, active_colors)):
            ax.bar(x + gi * width - 0.34 + width / 2, speedups[nm],
                   width, label=nm, color=c, edgecolor=(0.15, 0.15, 0.15))
        ax.axhline(1, color='k', linestyle='--', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{s:.0f}' for s in snr_db])
        ax.set_xlabel('SNR (dB)', fontsize=FS_AXIS)
        ax.set_ylabel('D-WMMSE / method time', fontsize=FS_AXIS)
        ax.set_title(title, fontsize=FS_TITLE)
        ax.legend(loc='best', fontsize=FS_LEG)
        ax.grid(True)

    speedup_e2e = {nm: dwmmse_e2e / t for nm, t in e2e.items()}
    speedup_core = {nm: dwmmse_core / t for nm, t in core.items()}

    _plot_time(axes[0, 0], dwmmse_e2e, e2e,
               'End-to-End PA Time', 'Includes bridge overhead')
    _plot_speedup(axes[0, 1], speedup_e2e, 'End-to-End Speedup')
    _plot_time(axes[1, 0], dwmmse_core, core,
               'Core Compute Time', 'D-WMMSE update vs learned-model forward')
    _plot_speedup(axes[1, 1], speedup_core, 'Core Compute Speedup')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, save_dir, 'Fig5_Time_Overhead_Comparison')


# ── Sync ablation plots ─────────────────────────────────────────────────

def plot_fig_a5_2_pa_ranking(ablation, save_dir, enabled=True):
    """PA ranking by best achievable ESR (distributed precoders only)."""
    if not enabled:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    algo_table = ablation['algoTable']
    avg_esr = ablation['avgESR']
    avg_ctrl = np.mean(ablation['control_delay_ms'], axis=1)

    best_esr, best_labels, best_colors = [], [], []
    for pi, pa_key in enumerate(PA_ORDER):
        candidates = [i for i, a in enumerate(algo_table)
                      if a['pa'] == pa_key and a['pcArch'] == 'distributed' and a['mode'] == 'DCC']
        if not candidates:
            candidates = [i for i, a in enumerate(algo_table)
                          if a['pa'] == pa_key and a['pcArch'] == 'distributed']
        if not candidates:
            best_esr.append(np.nan)
            best_labels.append('')
            best_colors.append((0.5, 0.5, 0.5))
            continue
        best_i = max(candidates, key=lambda i: avg_esr[i])
        best_esr.append(avg_esr[best_i])
        best_labels.append(f'{PA_LABELS[pi]}+{_pc_display(algo_table[best_i]["pc"])}')
        best_colors.append(PA_COLORS[pi])

    best_esr = np.array(best_esr)
    valid = np.isfinite(best_esr)
    rank = np.argsort(best_esr[valid])[::-1]
    valid_idx = np.where(valid)[0][rank]

    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    bars = ax.bar(range(len(valid_idx)), best_esr[valid_idx], 0.62,
                  color=[best_colors[i] for i in valid_idx])
    ax.set_xticks(range(len(valid_idx)))
    ax.set_xticklabels([best_labels[i] for i in valid_idx], rotation=20, fontsize=11)
    ax.set_ylabel('Average ESR (bit/s/Hz)')
    ax.set_xlabel('Best DCC + distributed-precode combination for each PA')
    ax.set_title('A5_2 PA Ranking by Best Achievable ESR\n'
                 'Higher bar = stronger PA under same distributed-precode constraint')
    ax.grid(True)
    fig.tight_layout()
    _save(fig, save_dir, 'FigA5_2_PA_Best_ESR_Ranking')


def plot_fig_a6_heatmap(ablation, save_dir, enabled=True):
    """PA x PC control delay heatmap."""
    if not enabled:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    algo_table = ablation['algoTable']
    avg_ctrl = np.mean(ablation['control_delay_ms'], axis=1)

    heat = np.full((len(PA_ORDER), len(PC_ORDER)), np.nan)
    for pi, pa_key in enumerate(PA_ORDER):
        for ci, pc_key in enumerate(PC_ORDER):
            idx = next((i for i, a in enumerate(algo_table)
                        if a['pa'] == pa_key and a['pc'] == pc_key and a['mode'] == 'DCC'), None)
            if idx is None:
                idx = next((i for i, a in enumerate(algo_table)
                            if a['pa'] == pa_key and a['pc'] == pc_key), None)
            if idx is not None:
                heat[pi, ci] = avg_ctrl[idx]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    im = ax.imshow(np.log10(heat + 1e-3), aspect='auto', cmap='viridis')
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('log10(control delay in ms + 1e-3)')
    ax.set_xticks(range(len(PC_ORDER)))
    ax.set_xticklabels(PC_LABELS, fontsize=12)
    ax.set_yticks(range(len(PA_ORDER)))
    ax.set_yticklabels(PA_LABELS, fontsize=12)
    ax.set_xlabel('Precoding method')
    ax.set_ylabel('Power allocation method')
    ax.set_title('PA x PC Control Delay Ablation (DCC)\n'
                 'Cell text = modeled control delay (ms); color = log delay')
    for pi in range(len(PA_ORDER)):
        for ci in range(len(PC_ORDER)):
            if np.isfinite(heat[pi, ci]):
                ax.text(ci, pi, f'{heat[pi, ci]:.3g}',
                        ha='center', va='center', color='w', fontweight='bold', fontsize=9)
    fig.tight_layout()
    _save(fig, save_dir, 'FigA6_PA_PC_SyncLatency_Heatmap')


def _plot_pc_ablation(ax, ablation, pc_key, title):
    """Single dual-axis ablation bar+line plot for a fixed precoder."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    algo_table = ablation['algoTable']
    avg_esr = ablation['avgESR']
    avg_ctrl = np.mean(ablation['control_delay_ms'], axis=1)

    bar_delay, line_esr = [], []
    for pa_key in PA_ORDER:
        idx = next((i for i, a in enumerate(algo_table)
                    if a['pa'] == pa_key and a['pc'] == pc_key and a['mode'] == 'DCC'), None)
        if idx is None:
            idx = next((i for i, a in enumerate(algo_table)
                        if a['pa'] == pa_key and a['pc'] == pc_key), None)
        if idx is not None:
            bar_delay.append(avg_ctrl[idx])
            line_esr.append(avg_esr[idx])
        else:
            bar_delay.append(np.nan)
            line_esr.append(np.nan)

    x = np.arange(len(PA_ORDER))
    ax.bar(x, bar_delay, 0.58, color=LEFT_COLOR, edgecolor=(0.10, 0.25, 0.40))
    ax.set_ylabel('Control delay (ms)', color=LEFT_COLOR)
    ax.tick_params(axis='y', labelcolor=LEFT_COLOR)

    ax2 = ax.twinx()
    valid = np.isfinite(line_esr)
    if np.sum(valid) >= 3:
        xv = x[valid].astype(float)
        yv = np.array(line_esr)[valid]
        xq = np.linspace(xv.min(), xv.max(), 240)
        yq = np.interp(xq, xv, yv)
        ax2.plot(xq, yq, '-', color=RIGHT_COLOR, linewidth=2.8)
    ax2.plot(x[valid], np.array(line_esr)[valid], 'o',
             color=RIGHT_COLOR, markerfacecolor=RIGHT_COLOR,
             markeredgecolor='w', markeredgewidth=1.2, markersize=6)
    ax2.set_ylabel('Average ESR (bit/s/Hz)', color=RIGHT_COLOR)
    ax2.tick_params(axis='y', labelcolor=RIGHT_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(PA_LABELS, fontsize=12)
    ax.set_xlabel('Power allocation method')
    ax.set_title(title)
    ax.grid(True)


def plot_fig_a7_rmmse_ablation(ablation, save_dir, enabled=True):
    if not enabled:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    _plot_pc_ablation(ax, ablation, 'RMMSE',
                      'PA Ablation under R-MMSE Precoding: Control Delay vs ESR')
    fig.tight_layout()
    _save(fig, save_dir, 'FigA7_RMMSE_PA_Latency_Ablation')


def plot_fig_a8_lmmse_ablation(ablation, save_dir, enabled=True):
    if not enabled:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    _plot_pc_ablation(ax, ablation, 'LMMSE',
                      'PA Ablation under L-MMSE Precoding: Control Delay vs ESR')
    fig.tight_layout()
    _save(fig, save_dir, 'FigA8_LMMSE_PA_Latency_Ablation')


def plot_fig_a9_mr_ablation(ablation, save_dir, enabled=True):
    if not enabled:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    _plot_pc_ablation(ax, ablation, 'MR',
                      'PA Ablation under MR Precoding: Control Delay vs ESR')
    fig.tight_layout()
    _save(fig, save_dir, 'FigA9_MR_PA_Latency_Ablation')


# ── Convenience: generate all ESR plots ─────────────────────────────────

def plot_fig_a10_control_split(ablation, save_dir, enabled=True):
    """Stacked PC/PA control-time split for the preferred DCC precoder."""
    if not enabled:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    algo_table = ablation['algoTable']
    avg_esr = ablation['avgESR']
    avg_pc_ctrl = np.mean(ablation['pc_control_delay_ms'], axis=1)
    avg_pa_ctrl = np.mean(ablation['pa_control_delay_ms'], axis=1)
    avg_feature = np.mean(ablation.get('feature_collection_delay_ms', 0), axis=1)
    avg_model = np.mean(ablation.get('model_inference_est_ms', 0), axis=1)

    labels, pc_vals, pa_vals, feature_vals, model_vals, esr_vals = [], [], [], [], [], []
    for pi, pa_key in enumerate(PA_ORDER):
        candidates = []
        for pc_key in ['RMMSE', 'LMMSE', 'MR', 'LMMSE_G']:
            idx = next((i for i, a in enumerate(algo_table)
                        if a['pa'] == pa_key and a['pc'] == pc_key and a['mode'] == 'DCC'), None)
            if idx is not None:
                candidates.append(idx)
        if not candidates:
            continue
        chosen = candidates[0]
        labels.append(f'{PA_LABELS[pi]}+{_pc_display(algo_table[chosen]["pc"])}')
        pc_vals.append(avg_pc_ctrl[chosen])
        pa_vals.append(avg_pa_ctrl[chosen])
        feature_vals.append(avg_feature[chosen])
        model_vals.append(avg_model[chosen])
        esr_vals.append(avg_esr[chosen])

    if not labels:
        return

    x = np.arange(len(labels))
    pc_vals = np.asarray(pc_vals, dtype=float)
    pa_vals = np.asarray(pa_vals, dtype=float)
    feature_vals = np.asarray(feature_vals, dtype=float)
    model_vals = np.asarray(model_vals, dtype=float)
    esr_vals = np.asarray(esr_vals, dtype=float)

    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    ax.bar(x, pc_vals, 0.62, color=(0.18, 0.43, 0.76),
           edgecolor=(0.10, 0.20, 0.35), label='PC control')
    ax.bar(x, pa_vals, 0.62, bottom=pc_vals, color=(0.86, 0.48, 0.22),
           edgecolor=(0.40, 0.20, 0.10), label='PA control')

    ax2 = ax.twinx()
    ax2.plot(x, esr_vals, 'o-', color=(0.10, 0.55, 0.25),
             linewidth=2.2, markersize=5, label='Average ESR')
    ax2.set_ylabel('Average ESR (bit/s/Hz)', color=(0.10, 0.55, 0.25))
    ax2.tick_params(axis='y', labelcolor=(0.10, 0.55, 0.25))

    for xi, total, feat, model in zip(x, pc_vals + pa_vals, feature_vals, model_vals):
        if feat > 0 or model > 0:
            ax.text(xi, total, f'F {feat:.2g}\nM {model:.2g}',
                    ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=10)
    ax.set_ylabel('Control delay (ms)')
    ax.set_xlabel('Power allocation method with preferred DCC precoder')
    ax.set_title('PC/PA Control-Time Split\n'
                 'PA segment includes feature collection and architecture-aware inference estimate')
    ax.grid(True, axis='y')
    ax.legend(loc='upper left', fontsize=FS_LEG)
    ax2.legend(loc='upper right', fontsize=FS_LEG)
    fig.tight_layout()
    _save(fig, save_dir, 'FigA10_PC_PA_Control_Split')


def plot_all_esr(ESR_mean, algo_table, snr_db, perf, save_dir, enabled=True):
    """Generate all 5 ESR/timing figures (Fig1–Fig5)."""
    if not enabled:
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        return

    plot_fig1_best_pa_esr(ESR_mean, algo_table, snr_db, save_dir, enabled)
    plot_fig2_rmmse_pa(ESR_mean, algo_table, snr_db, save_dir, enabled)
    plot_fig3_gnn_dwmmse_gap(ESR_mean, algo_table, snr_db, save_dir, enabled)
    plot_fig4_method_summary(ESR_mean, algo_table, save_dir, enabled)
    plot_fig5_timing(perf, snr_db, save_dir, enabled)
    print('[INFO] 5 ESR/timing figures generated.')


def plot_all_ablation(ablation, save_dir, enabled=True):
    """Generate all 5 ablation figures (FigA5_2–FigA9)."""
    if not enabled:
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        return

    plot_fig_a5_2_pa_ranking(ablation, save_dir, enabled)
    plot_fig_a6_heatmap(ablation, save_dir, enabled)
    plot_fig_a7_rmmse_ablation(ablation, save_dir, enabled)
    plot_fig_a8_lmmse_ablation(ablation, save_dir, enabled)
    plot_fig_a9_mr_ablation(ablation, save_dir, enabled)
    plot_fig_a10_control_split(ablation, save_dir, enabled)
    print('[INFO] 6 ablation figures generated.')
