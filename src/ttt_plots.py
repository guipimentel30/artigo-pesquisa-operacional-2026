from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np

from .benchmark_suite import (
    RESULTS_DIR,
    PreparedInstance,
    save_audit_json,
    save_audit_markdown,
)
from .data_generator import (
    MUNICIPALITIES_TABLE3,
    build_coverage_matrix,
    generate_municipality_instance,
)
from .grasp_reativo_vns import run_grasp_reativo_vns
from .grasp_reativo_vns_pr import run_grasp_reativo_vns_pr
from .heuristics import run_grasp
from .sparse_utils import HeuristicResult, build_sparse_structures

DEFAULT_SEED_STRIDE = 1_000_003
DEFAULT_N_TTT_RUNS = 100
DEFAULT_TARGET_ORDER = ("easy", "medium", "hard")

TARGET_DISPLAY_LABELS = {
    "easy": "Alvo Fácil",
    "medium": "Alvo Médio",
    "hard": "Alvo Difícil",
}

@dataclass(frozen=True)
class AlgorithmSpec:

    key: str
    label: str
    short_label: str
    solver_fn: Callable[..., HeuristicResult]
    color: str
    linestyle: str
    marker: str

DEFAULT_ALGORITHMS: tuple[AlgorithmSpec, ...] = (
    AlgorithmSpec(
        key="grasp_classico",
        label="GRASP Clássico (Baseline)",
        short_label="GRASP Clássico",
        solver_fn=run_grasp,
        color="#D55E00",
        linestyle="--",
        marker="^",
    ),
    AlgorithmSpec(
        key="grasp_reativo_vns",
        label="GRASP Reativo + Basic VNS",
        short_label="GRASP Reativo + VNS",
        solver_fn=run_grasp_reativo_vns,
        color="#0072B2",
        linestyle="-.",
        marker="s",
    ),
    AlgorithmSpec(
        key="grasp_reativo_vns_pr",
        label="GRASP Reativo + Basic VNS + PR",
        short_label="GRASP Reativo + VNS + PR",
        solver_fn=run_grasp_reativo_vns_pr,
        color="#009E73",
        linestyle="-",
        marker="o",
    ),
)

def build_disjoint_seeds(
    base_seed: int,
    n_runs: int,
    stride: int = DEFAULT_SEED_STRIDE,
) -> List[int]:
    if n_runs <= 0:
        raise ValueError("n_runs deve ser positivo.")
    return [int(base_seed + idx * stride) for idx in range(n_runs)]

def prepare_relocation_instance(
    municipality: str,
    n_train_demands: int,
    n_eval_demands: int,
    max_candidates: int,
    seed: int,
) -> PreparedInstance:
    inst = generate_municipality_instance(
        municipality,
        n_train_demands=n_train_demands,
        n_eval_demands=n_eval_demands,
        max_candidates=max_candidates,
        seed=seed,
    )
    c_train = build_coverage_matrix(inst.train_demands_coords, inst.candidate_coords)
    c_eval = build_coverage_matrix(inst.eval_demands_coords, inst.candidate_coords)
    sparse = build_sparse_structures(c_train)
    return PreparedInstance(
        instance=inst,
        c_train=c_train,
        c_eval=c_eval,
        sparse=sparse,
        seed=seed,
    )

def warmup_solvers(
    prepared: PreparedInstance,
    algorithms: Sequence[AlgorithmSpec] = DEFAULT_ALGORITHMS,
) -> None:
    inst = prepared.instance
    for spec in algorithms:
        spec.solver_fn(
            prepared.c_train,
            inst.num_existing_aeds,
            base_cov_train=None,
            c_matrix_eval=prepared.c_eval,
            base_cov_eval=None,
            max_iterations=2,
            seed=1,
            sparse=prepared.sparse,
        )

def summarize_series(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "q25": None,
            "q75": None,
            "min": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }

def compute_slide94_targets(
    algorithm_summaries: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    means = {
        key: float(summary["train_coverage_pct"]["mean"])
        for key, summary in algorithm_summaries.items()
    }
    if not means:
        raise ValueError("algorithm_summaries não pode ser vazio.")
    return {
        "easy": float(min(means.values())),
        "medium": float(np.mean(list(means.values()))),
        "hard": float(max(means.values())),
    }

def compute_global_median_time(
    hit_times: Sequence[float],
    n_total_runs: int,
) -> Optional[float]:
    if n_total_runs <= 0:
        return None
    if n_total_runs % 2 == 1:
        idx = n_total_runs // 2
        return float(hit_times[idx]) if len(hit_times) > idx else None
    hi = n_total_runs // 2
    lo = hi - 1
    if len(hit_times) <= hi:
        return None
    return float(0.5 * (hit_times[lo] + hit_times[hi]))

def compute_empirical_ttt_curve(
    run_records: Sequence[Dict[str, Any]],
    n_total_runs: int,
) -> Dict[str, Any]:
    if n_total_runs <= 0:
        raise ValueError("n_total_runs deve ser positivo.")

    hit_records = [r for r in run_records if r["hit"]]
    hit_times = sorted(float(r["cpu_time_sec"]) for r in hit_records)
    probabilities = [(idx + 0.5) / n_total_runs for idx in range(len(hit_times))]

    hits = len(hit_times)
    censored = n_total_runs - hits
    all_times = sorted(float(r["cpu_time_sec"]) for r in run_records)

    return {
        "n_total_runs": int(n_total_runs),
        "hits": int(hits),
        "censored": int(censored),
        "hit_rate_pct": float(100.0 * hits / n_total_runs),
        "global_median_sec": compute_global_median_time(hit_times, n_total_runs),
        "hit_times_sec": hit_times,
        "probabilities": probabilities,
        "hit_time_stats": summarize_series(hit_times),
        "all_time_stats": summarize_series(all_times),
    }

def _run_calibration_algorithm(
    prepared: PreparedInstance,
    spec: AlgorithmSpec,
    cal_seeds: Sequence[int],
    cal_max_iterations: int,
) -> Dict[str, Any]:
    inst = prepared.instance
    runs: List[Dict[str, Any]] = []
    for idx, seed in enumerate(cal_seeds):
        res = spec.solver_fn(
            prepared.c_train,
            inst.num_existing_aeds,
            base_cov_train=None,
            c_matrix_eval=prepared.c_eval,
            base_cov_eval=None,
            max_iterations=cal_max_iterations,
            seed=seed,
            sparse=prepared.sparse,
        )
        runs.append(
            {
                "run_index": idx,
                "seed": int(seed),
                "train_coverage_pct": float(res.train_coverage_pct),
                "eval_coverage_pct": float(res.eval_coverage_pct)
                if res.eval_coverage_pct is not None
                else None,
                "cpu_time_sec": float(res.cpu_time_sec),
                "time_to_best_sec": float(res.time_to_best_sec)
                if res.time_to_best_sec is not None
                else None,
                "iterations_completed": int(res.iterations_completed),
            }
        )
        if (idx + 1) % 10 == 0 or idx + 1 == len(cal_seeds):
            print(
                f"[Calibração][{spec.short_label}] {idx + 1}/{len(cal_seeds)} "
                f"(última={res.train_coverage_pct:.4f}%, {res.cpu_time_sec:.2f}s)",
                flush=True,
            )

    return {
        "label": spec.label,
        "short_label": spec.short_label,
        "train_coverage_pct": summarize_series([r["train_coverage_pct"] for r in runs]),
        "eval_coverage_pct": summarize_series(
            [r["eval_coverage_pct"] for r in runs if r["eval_coverage_pct"] is not None]
        ),
        "cpu_time_sec": summarize_series([r["cpu_time_sec"] for r in runs]),
        "time_to_best_sec": summarize_series(
            [r["time_to_best_sec"] for r in runs if r["time_to_best_sec"] is not None]
        ),
        "runs": runs,
    }

def _run_target_algorithm(
    prepared: PreparedInstance,
    spec: AlgorithmSpec,
    target_name: str,
    target_value_pct: float,
    ttt_seeds: Sequence[int],
    ttt_max_iterations: int,
    ttt_time_limit_sec: Optional[float],
) -> Dict[str, Any]:
    inst = prepared.instance
    runs: List[Dict[str, Any]] = []
    for idx, seed in enumerate(ttt_seeds):
        res = spec.solver_fn(
            prepared.c_train,
            inst.num_existing_aeds,
            base_cov_train=None,
            c_matrix_eval=None,
            base_cov_eval=None,
            max_iterations=ttt_max_iterations,
            time_limit_sec=ttt_time_limit_sec,
            target_coverage_pct=target_value_pct,
            seed=seed,
            sparse=prepared.sparse,
        )
        hit = bool(res.train_coverage_pct >= target_value_pct - 1e-9)
        runs.append(
            {
                "run_index": idx,
                "seed": int(seed),
                "hit": hit,
                "achieved_train_pct": float(res.train_coverage_pct),
                "cpu_time_sec": float(res.cpu_time_sec),
                "iterations_completed": int(res.iterations_completed),
            }
        )
        if (idx + 1) % 25 == 0 or idx + 1 == len(ttt_seeds):
            hits_so_far = sum(1 for r in runs if r["hit"])
            print(
                f"[TTT-{target_name}][{spec.short_label}] "
                f"{idx + 1}/{len(ttt_seeds)} | acertos={hits_so_far}/{idx + 1} "
                f"| último={res.cpu_time_sec:.2f}s ({'OK' if hit else 'CENS'})",
                flush=True,
            )

    return {
        "label": spec.label,
        "short_label": spec.short_label,
        "curve": compute_empirical_ttt_curve(runs, len(ttt_seeds)),
        "runs": runs,
    }

def run_calibration_phase(
    prepared: PreparedInstance,
    cal_seeds: Sequence[int],
    cal_max_iterations: int,
    algorithms: Sequence[AlgorithmSpec] = DEFAULT_ALGORITHMS,
) -> Dict[str, Any]:
    per_algo = {
        spec.key: _run_calibration_algorithm(prepared, spec, cal_seeds, cal_max_iterations)
        for spec in algorithms
    }
    return {
        "n_cal_runs": len(cal_seeds),
        "cal_max_iterations": int(cal_max_iterations),
        "cal_seeds_base": int(cal_seeds[0]),
        "cal_seeds_stride": int(cal_seeds[1] - cal_seeds[0]) if len(cal_seeds) > 1 else 0,
        "algorithms": per_algo,
        "targets_pct": compute_slide94_targets(per_algo),
    }

def run_target_ttt_phase(
    prepared: PreparedInstance,
    target_name: str,
    target_value_pct: float,
    ttt_seeds: Sequence[int],
    ttt_max_iterations: int,
    ttt_time_limit_sec: Optional[float],
    algorithms: Sequence[AlgorithmSpec] = DEFAULT_ALGORITHMS,
) -> Dict[str, Any]:
    per_algo = {
        spec.key: _run_target_algorithm(
            prepared,
            spec,
            target_name,
            target_value_pct,
            ttt_seeds,
            ttt_max_iterations,
            ttt_time_limit_sec,
        )
        for spec in algorithms
    }
    return {
        "target_name": target_name,
        "target_label": TARGET_DISPLAY_LABELS.get(target_name, target_name),
        "target_value_pct": float(target_value_pct),
        "ttt_max_iterations": int(ttt_max_iterations),
        "ttt_time_limit_sec": float(ttt_time_limit_sec)
        if ttt_time_limit_sec is not None
        else None,
        "algorithms": per_algo,
    }

def _plot_single_target_axis(
    ax: plt.Axes,
    target_data: Dict[str, Any],
    algorithms: Sequence[AlgorithmSpec],
    municipality: str,
    show_ylabel: bool = True,
    show_legend: bool = True,
    title_override: Optional[str] = None,
) -> None:
    target_val = target_data["target_value_pct"]
    target_label = target_data["target_label"]

    for spec in algorithms:
        algo_entry = target_data["algorithms"][spec.key]
        curve = algo_entry["curve"]
        times = curve["hit_times_sec"]
        probs = curve["probabilities"]
        hits = curve["hits"]
        n_total = curve["n_total_runs"]
        p50 = curve.get("global_median_sec")
        if p50 is None:
            p50 = compute_global_median_time(times, n_total)
        t50_lbl = f"{p50:.2f}s" if p50 is not None else ">lim."
        legend_lbl = f"{spec.short_label} ({hits}/{n_total}, $t_{{50}}$={t50_lbl})"

        if not times:
            ax.plot([], [], color=spec.color, linestyle=spec.linestyle, label=legend_lbl)
            continue

        x_vals = [max(t, 1e-4) for t in times]
        mark_step = max(1, len(x_vals) // 10)
        mark_indices = list(range(0, len(x_vals), mark_step))
        if (len(x_vals) - 1) not in mark_indices:
            mark_indices.append(len(x_vals) - 1)

        ax.plot(
            x_vals,
            probs,
            color=spec.color,
            linestyle=spec.linestyle,
            linewidth=2.0,
            marker=spec.marker,
            markersize=5.2,
            markevery=mark_indices,
            label=legend_lbl,
            alpha=0.92,
        )

    ax.set_xscale("log")
    xmin, xmax = ax.get_xlim()
    candidate_ticks = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    visible_ticks = [t for t in candidate_ticks if xmin * 0.9 <= t <= xmax * 1.1]
    if visible_ticks:
        ax.xaxis.set_major_locator(FixedLocator(visible_ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:g}"))
    ax.set_ylim(-0.02, 1.05)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, which="major", linestyle="-", alpha=0.28, color="#888888")
    ax.grid(True, which="minor", linestyle=":", alpha=0.18, color="#aaaaaa")
    ax.tick_params(axis="both", which="major", labelsize=9.5)
    ax.set_xlabel("Tempo decorrido (s) [escala log]", fontsize=11.0)
    if show_ylabel:
        ax.set_ylabel("Probabilidade acumulada p(i)", fontsize=11.0)

    title = title_override or f"{target_label} ({municipality}: {target_val:.4f}%)"
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=8)

    if show_legend:
        ax.legend(loc="upper left", fontsize=9.0, frameon=True, framealpha=0.94)

def generate_ttt_figures(
    payload: Dict[str, Any],
    figures_dir: Optional[Path] = None,
    algorithms: Sequence[AlgorithmSpec] = DEFAULT_ALGORITHMS,
) -> Dict[str, List[str]]:
    if figures_dir is None:
        figures_dir = RESULTS_DIR / "figuras"
    figures_dir.mkdir(parents=True, exist_ok=True)

    meta = payload["metadata"]
    municipality = meta["municipality"]
    ttt_targets = payload["ttt_plots"]

    file_stems = {
        "easy": "ttt_alvo_facil",
        "medium": "ttt_alvo_medio",
        "hard": "ttt_alvo_dificil",
    }
    generated_files: Dict[str, List[str]] = {"individual": [], "consolidated": []}

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
        }
    )

    for target_key in DEFAULT_TARGET_ORDER:
        target_data = ttt_targets[target_key]
        fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=300)
        _plot_single_target_axis(ax, target_data, algorithms, municipality)
        fig.tight_layout()

        stem = file_stems[target_key]
        for ext in ("png", "pdf"):
            out_path = figures_dir / f"{stem}.{ext}"
            fig.savefig(out_path, bbox_inches="tight", dpi=300)
            generated_files["individual"].append(str(out_path))
        decorrido_path = figures_dir / f"{stem}_decorrido.pdf"
        fig.savefig(decorrido_path, bbox_inches="tight", dpi=300)
        generated_files["individual"].append(str(decorrido_path))
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5), dpi=300, sharey=True)
    for col_idx, target_key in enumerate(DEFAULT_TARGET_ORDER):
        target_data = ttt_targets[target_key]
        target_val = target_data["target_value_pct"]
        _plot_single_target_axis(
            axes[col_idx],
            target_data,
            algorithms,
            municipality,
            show_ylabel=(col_idx == 0),
            show_legend=True,
            title_override=f"{target_data['target_label']} (alvo = {target_val:.4f}%)",
        )
    fig.suptitle(
        f"Curvas Time-to-Target em {municipality} "
        f"($N = {meta['num_existing_aeds']}$, $|\\mathcal{{J}}^c| = {meta['n_candidates']}$, "
        f"$|\\mathcal{{I}}^{{\\mathrm{{train}}}}| = {meta['n_train_demands']}$, "
        f"{meta['n_ttt_runs']} execuções/alvo)",
        fontsize=12.0,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out_path = figures_dir / f"ttt_consolidado.{ext}"
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
        generated_files["consolidated"].append(str(out_path))
    plt.close(fig)

    return generated_files

def format_escala_ttt_markdown(payload: Dict[str, Any]) -> str:
    meta = payload["metadata"]
    cal = payload["calibration"]
    lines = [
        f"# Relatório Auditável — Escala Ampliada e TTT-Plots ({meta['municipality']})",
        "",
        "## Configuração da Instância e Protocolo",
        f"- Município (Tabela 3 de Buter et al., 2024): **{meta['municipality']}**",
        f"- AEDs a realocar (`N`): **{meta['num_existing_aeds']}**",
        f"- Candidatos selecionados (`|J^c|`): **{meta['n_candidates']}** (Tabela 3 de referência: {meta['table3_candidates']} pontos)",
        f"- Demandas de treino (`|I^train|`): **{meta['n_train_demands']}** | Demandas de avaliação (`|I^eval|`): **{meta['n_eval_demands']}**",
        f"- Semente da instância: `{meta['instance_seed']}` | Espaçamento entre sementes: `{meta['seed_stride']}`",
        f"- Calibração não censurada: **{cal['n_cal_runs']} execuções** por algoritmo (`max_iterations = {cal['cal_max_iterations']}`)",
        f"- TTT-plots (Slide 94): **{meta['n_ttt_runs']} execuções independentes** por algoritmo e por alvo "
        f"(`max_iterations = {meta['ttt_max_iterations']}`, `time_limit_sec = {meta['ttt_time_limit_sec']}`)",
        "",
        "## Fase 1 — Calibração Não Censurada (Base para Alvos do Slide 94)",
        "",
        "| Algoritmo | Cobertura Treino Média (%) | Desvio Padrão (%) | Mín (%) | Máx (%) | Avaliação Média (%) | Tempo Médio (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for spec in DEFAULT_ALGORITHMS:
        algo = cal["algorithms"][spec.key]
        tr = algo["train_coverage_pct"]
        ev = algo["eval_coverage_pct"]
        tm = algo["cpu_time_sec"]
        ev_mean = f"{ev['mean']:.4f}" if ev["mean"] is not None else "-"
        lines.append(
            f"| {algo['label']} | {tr['mean']:.4f} | {tr['std']:.4f} | "
            f"{tr['min']:.4f} | {tr['max']:.4f} | {ev_mean} | {tm['mean']:.3f} |"
        )

    targets = cal["targets_pct"]
    lines.extend(
        [
            "",
            "### Alvos Extraídos (Slide 94)",
            f"- **Alvo Fácil** (`min` das médias): **{targets['easy']:.6f}%**",
            f"- **Alvo Médio** (`média` das médias): **{targets['medium']:.6f}%**",
            f"- **Alvo Difícil** (`max` das médias): **{targets['hard']:.6f}%**",
            "",
            "## Fase 2 — Distribuição Empírica Time-to-Target (100 Execuções por Alvo)",
            "",
            "| Alvo | Valor (%) | Algoritmo | Acertos / 100 | Taxa de Acerto (%) | Mediana Global $t_{50}$ (s) | Mediana Acertos (s) | Q25 Acertos (s) | Q75 Acertos (s) | Média Acertos (s) |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for target_key in DEFAULT_TARGET_ORDER:
        tdata = payload["ttt_plots"][target_key]
        for spec in DEFAULT_ALGORITHMS:
            aentry = tdata["algorithms"][spec.key]
            curve = aentry["curve"]
            hts = curve["hit_time_stats"]
            p50 = curve.get("global_median_sec")
            if p50 is None:
                p50 = compute_global_median_time(
                    curve["hit_times_sec"], curve["n_total_runs"]
                )
            p50_txt = f"{p50:.3f}" if p50 is not None else ">lim."
            med = f"{hts['median']:.3f}" if hts["median"] is not None else "Cens."
            q25 = f"{hts['q25']:.3f}" if hts["q25"] is not None else "-"
            q75 = f"{hts['q75']:.3f}" if hts["q75"] is not None else "-"
            avg = f"{hts['mean']:.3f}" if hts["mean"] is not None else "-"
            lines.append(
                f"| {tdata['target_label']} | {tdata['target_value_pct']:.4f} | "
                f"{spec.short_label} | {curve['hits']}/100 | {curve['hit_rate_pct']:.1f}% | "
                f"{p50_txt} | {med} | {q25} | {q75} | {avg} |"
            )

    lines.append("")
    return "\n".join(lines)

def _save_checkpoint(path: Optional[Path], payload: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)

def run_large_scale_ttt_experiment(
    municipality: str = "Hollands Kroon",
    n_train_demands: int = 1500,
    n_eval_demands: int = 5000,
    max_candidates: int = 1200,
    instance_seed: int = 42,
    n_cal_runs: int = 30,
    cal_max_iterations: int = 30,
    cal_base_seed: int = 5000,
    n_ttt_runs: int = DEFAULT_N_TTT_RUNS,
    ttt_max_iterations: int = 96,
    ttt_time_limit_sec: Optional[float] = 20.0,
    ttt_base_seed: int = 100,
    seed_stride: int = DEFAULT_SEED_STRIDE,
    algorithms: Sequence[AlgorithmSpec] = DEFAULT_ALGORITHMS,
    checkpoint_filename: str = "escala_ttt_checkpoint.json",
    resume_checkpoint: bool = True,
    phases: Optional[Sequence[str]] = None,
    target_filter: Optional[Sequence[str]] = None,
    algo_filter: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    t_wall0 = time.perf_counter()
    active_phases = set(phases) if phases is not None else {"calibration", "ttt", "finalize"}
    active_targets = tuple(target_filter) if target_filter is not None else DEFAULT_TARGET_ORDER
    algo_filter_set = set(algo_filter) if algo_filter is not None else None

    prepared = prepare_relocation_instance(
        municipality=municipality,
        n_train_demands=n_train_demands,
        n_eval_demands=n_eval_demands,
        max_candidates=max_candidates,
        seed=instance_seed,
    )
    inst = prepared.instance
    active_algos = [
        spec for spec in algorithms if algo_filter_set is None or spec.key in algo_filter_set
    ]
    if active_phases & {"calibration", "ttt"}:
        warmup_solvers(prepared, algorithms=active_algos)

    config_signature = {
        "municipality": municipality,
        "num_existing_aeds": int(inst.num_existing_aeds),
        "n_candidates": int(len(inst.candidate_coords)),
        "table3_candidates": int(MUNICIPALITIES_TABLE3[municipality]["candidates"]),
        "n_train_demands": int(n_train_demands),
        "n_eval_demands": int(n_eval_demands),
        "max_candidates": int(max_candidates),
        "instance_seed": int(instance_seed),
        "n_cal_runs": int(n_cal_runs),
        "cal_max_iterations": int(cal_max_iterations),
        "cal_base_seed": int(cal_base_seed),
        "n_ttt_runs": int(n_ttt_runs),
        "ttt_max_iterations": int(ttt_max_iterations),
        "ttt_time_limit_sec": float(ttt_time_limit_sec)
        if ttt_time_limit_sec is not None
        else None,
        "ttt_base_seed": int(ttt_base_seed),
        "seed_stride": int(seed_stride),
    }

    checkpoint_path = RESULTS_DIR / checkpoint_filename
    payload: Dict[str, Any] = {"metadata": config_signature, "ttt_plots": {}}
    if resume_checkpoint and checkpoint_path.exists():
        loaded = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if loaded.get("metadata") == config_signature:
            payload = loaded
            print("[Checkpoint] Retomando execução compatível encontrada.", flush=True)

    cal_seeds = build_disjoint_seeds(cal_base_seed, n_cal_runs, stride=seed_stride)
    cal_block = payload.setdefault(
        "calibration",
        {
            "n_cal_runs": len(cal_seeds),
            "cal_max_iterations": int(cal_max_iterations),
            "cal_seeds_base": int(cal_seeds[0]),
            "cal_seeds_stride": int(cal_seeds[1] - cal_seeds[0]) if len(cal_seeds) > 1 else 0,
            "algorithms": {},
        },
    )
    if "calibration" in active_phases:
        for spec in active_algos:
            if spec.key in cal_block["algorithms"]:
                print(f"[Checkpoint][Calibração] '{spec.key}' já concluído.", flush=True)
                continue
            cal_block["algorithms"][spec.key] = _run_calibration_algorithm(
                prepared=prepared,
                spec=spec,
                cal_seeds=cal_seeds,
                cal_max_iterations=cal_max_iterations,
            )
            _save_checkpoint(checkpoint_path, payload)

        if all(spec.key in cal_block["algorithms"] for spec in algorithms):
            cal_block["targets_pct"] = compute_slide94_targets(cal_block["algorithms"])
            _save_checkpoint(checkpoint_path, payload)

    if "ttt" in active_phases:
        if "targets_pct" not in cal_block:
            raise RuntimeError("Calibração incompleta: execute todos os algoritmos antes dos TTT-plots.")
        targets = cal_block["targets_pct"]
        ttt_seeds = build_disjoint_seeds(ttt_base_seed, n_ttt_runs, stride=seed_stride)

        for target_name in active_targets:
            target_entry = payload["ttt_plots"].setdefault(
                target_name,
                {
                    "target_name": target_name,
                    "target_label": TARGET_DISPLAY_LABELS.get(target_name, target_name),
                    "target_value_pct": float(targets[target_name]),
                    "ttt_max_iterations": int(ttt_max_iterations),
                    "ttt_time_limit_sec": float(ttt_time_limit_sec)
                    if ttt_time_limit_sec is not None
                    else None,
                    "algorithms": {},
                },
            )
            for spec in active_algos:
                if spec.key in target_entry["algorithms"]:
                    print(f"[Checkpoint][TTT-{target_name}] '{spec.key}' já concluído.", flush=True)
                    continue
                target_entry["algorithms"][spec.key] = _run_target_algorithm(
                    prepared=prepared,
                    spec=spec,
                    target_name=target_name,
                    target_value_pct=targets[target_name],
                    ttt_seeds=ttt_seeds,
                    ttt_max_iterations=ttt_max_iterations,
                    ttt_time_limit_sec=ttt_time_limit_sec,
                )
                _save_checkpoint(checkpoint_path, payload)

    if "finalize" in active_phases:
        for target_name in DEFAULT_TARGET_ORDER:
            algos_done = payload["ttt_plots"].get(target_name, {}).get("algorithms", {})
            missing = [spec.key for spec in algorithms if spec.key not in algos_done]
            if missing:
                raise RuntimeError(f"Não é possível finalizar: '{target_name}' sem {missing}.")
            for spec in algorithms:
                c = algos_done[spec.key]["curve"]
                c["global_median_sec"] = compute_global_median_time(
                    c["hit_times_sec"], c["n_total_runs"]
                )

        figures = generate_ttt_figures(payload, algorithms=algorithms)
        payload["figures"] = figures

        cal_cpu = sum(
            float(r["cpu_time_sec"])
            for algo in payload["calibration"]["algorithms"].values()
            for r in algo["runs"]
        )
        ttt_cpu = sum(
            float(r["cpu_time_sec"])
            for target_data in payload["ttt_plots"].values()
            for algo in target_data["algorithms"].values()
            for r in algo["runs"]
        )
        payload["total_solver_cpu_time_sec"] = float(cal_cpu + ttt_cpu)
        payload["total_wall_time_sec"] = float(time.perf_counter() - t_wall0)

        json_path = save_audit_json("escala_ttt_plots.json", payload)
        md_path = save_audit_markdown(
            "relatorio_escala_ttt_plots.md",
            format_escala_ttt_markdown(payload),
        )
        payload["audit_files"] = {"json": str(json_path), "markdown": str(md_path)}
        save_audit_json("escala_ttt_plots.json", payload)

    return payload
