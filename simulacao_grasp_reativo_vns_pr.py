from typing import Any, Dict, List, Optional
import numpy as np

from src.heuristics import run_grasp
from src.sparse_utils import build_sparse_structures
from src.grasp_reativo_vns import DEFAULT_ALPHA_POOL, run_grasp_reativo_vns
from src.grasp_reativo_vns_pr import _path_relinking_one_way_numba, run_grasp_reativo_vns_pr
from src.benchmark_suite import (
    PreparedInstance,
    build_exp1_instances,
    build_shared_instances,
    format_pr_summary,
    format_vns_summary,
    load_audit_json,
    resolve_exp1_reference,
    resolve_exp2_reference,
    run_exp3_addition_sweep,
    save_audit_json,
    save_audit_markdown,
)

def warmup_numba():
    rng = np.random.default_rng(0)
    dummy_c = rng.uniform(0.0, 0.5, size=(30, 20))
    dummy_sp = build_sparse_structures(dummy_c)
    dummy_base = np.zeros(30, dtype=np.float64)

    _path_relinking_one_way_numba(
        dummy_sp.n_demands,
        dummy_sp.n_candidates,
        dummy_sp.col_ptr,
        dummy_sp.row_idx,
        dummy_sp.cov_vals,
        dummy_base,
        np.array([0, 1, 2, 3, 4], dtype=np.int32),
        np.array([5, 6, 7, 8, 9], dtype=np.int32),
    )

    run_grasp_reativo_vns_pr(dummy_c, n_to_open=5, r_max=3, max_iterations=14)

def run_experiment_1_comparison(
    exp1_instances: List[PreparedInstance],
    baseline_json: Optional[Dict[str, Any]],
    rvns_json: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    print("\n[EXPERIMENTO 1] Evolução Evolutiva em Heiloo (Tabela 4)")
    print("-" * 116)
    print(
        f"{'Método':<28} | {'Demandas':>8} | {'Treino (%)':>10} | {'Eval (%)':>10} | "
        f"{'Tempo (s)':>9} | {'PR (Cham/Skip/+SB)':>18} | {'VNS (LS | +k1/k2/k3)':>20}"
    )
    print("-" * 116)

    records: List[Dict[str, Any]] = []
    for idx, prep in enumerate(exp1_instances):
        inst, c_train, c_eval, sparse = prep.instance, prep.c_train, prep.c_eval, prep.sparse
        n_demands = len(inst.train_demands_coords)
        n_aeds = inst.num_existing_aeds

        b_train, b_eval, b_cpu = resolve_exp1_reference(
            baseline_json,
            idx,
            "grasp_classico",
            lambda: run_grasp(c_train, n_aeds, c_matrix_eval=c_eval, seed=42, sparse=sparse),
        )
        rv_train, rv_eval, rv_cpu = resolve_exp1_reference(
            rvns_json,
            idx,
            "grasp_reativo_vns",
            lambda: run_grasp_reativo_vns(
                c_train, n_aeds, c_matrix_eval=c_eval, r_max=3, max_iterations=96, seed=42, sparse=sparse
            ),
        )

        pr_res = run_grasp_reativo_vns_pr(
            c_matrix_train=c_train,
            n_to_open=n_aeds,
            c_matrix_eval=c_eval,
            r_max=3,
            max_iterations=96,
            seed=42,
            sparse=sparse,
        )
        vns = pr_res.vns_stats or {}
        prs = pr_res.pr_stats or {}
        pr_str = format_pr_summary(prs)

        print(f"{'1. GRASP Clássico':<28} | {n_demands:>8d} | {b_train:>9.2f}% | {b_eval:>9.2f}% | {b_cpu:>9.4f} | {'---':>18} | {'---':>20}")
        print(f"{'2. GRASP Reativo + VNS':<28} | {n_demands:>8d} | {rv_train:>9.2f}% | {rv_eval:>9.2f}% | {rv_cpu:>9.4f} | {'---':>18} | {'---':>20}")
        print(
            f"{'3. GRASP Reativo + VNS + PR':<28} | {n_demands:>8d} | {pr_res.train_coverage_pct:>9.2f}% | "
            f"{pr_res.eval_coverage_pct:>9.2f}% | {pr_res.cpu_time_sec:>9.4f} | {pr_str:>18} | {format_vns_summary(vns):>20}"
        )
        print("-" * 116)

        records.append({
            "municipality": inst.name,
            "n_train_demands": n_demands,
            "n_candidates": len(inst.candidate_coords),
            "n_aeds": n_aeds,
            "seed": prep.seed,
            "grasp_classico_ref": {"train_pct": round(b_train, 4), "eval_pct": round(b_eval, 4), "cpu_sec": round(b_cpu, 4)},
            "grasp_reativo_vns_ref": {"train_pct": round(rv_train, 4), "eval_pct": round(rv_eval, 4), "cpu_sec": round(rv_cpu, 4)},
            "grasp_reativo_vns_pr": {
                "train_pct": round(pr_res.train_coverage_pct, 4),
                "eval_pct": round(pr_res.eval_coverage_pct, 4),
                "cpu_sec": round(pr_res.cpu_time_sec, 4),
                "delta_train_vs_classico_pp": round(pr_res.train_coverage_pct - b_train, 4),
                "delta_eval_vs_classico_pp": round(pr_res.eval_coverage_pct - b_eval, 4),
                "delta_train_vs_rvns_pp": round(pr_res.train_coverage_pct - rv_train, 4),
                "vns_stats": vns,
                "pr_stats": prs,
                "alpha_probabilities": {
                    str(k): round(v, 4) for k, v in (pr_res.alpha_probabilities or {}).items()
                },
            },
        })
    return records

def run_experiment_2_comparison(
    shared_instances: Dict[str, PreparedInstance],
    baseline_json: Optional[Dict[str, Any]],
    rvns_json: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    print("\n[EXPERIMENTO 2] Realocação (Tabela 5) — Clássico -> Reativo+VNS -> Reativo+VNS+PR")
    print("-" * 120)
    print(
        f"{'Município':<12} | {'Clás Trn':>8} | {'RVNS Trn':>8} | {'PR Trn':>8} | "
        f"{'Clás Evl':>8} | {'RVNS Evl':>8} | {'PR Evl':>8} | {'t PR(s)':>7} | {'PR (Ch/Sk/+SB)':>14} | {'VNS (LS | +k1/k2/k3)':>20}"
    )
    print("-" * 120)

    base_map = {e["municipality"]: e for e in (baseline_json or {}).get("experiment_2_table5", [])}
    rvns_map = {e["municipality"]: e for e in (rvns_json or {}).get("experiment_2_table5", [])}

    records: List[Dict[str, Any]] = []
    for mun_name, prep in shared_instances.items():
        inst, c_train, c_eval, sparse = prep.instance, prep.c_train, prep.c_eval, prep.sparse

        b_train, b_eval, b_cpu = resolve_exp2_reference(
            base_map,
            mun_name,
            None,
            "relocated_eval_pct",
            lambda: run_grasp(c_train, inst.num_existing_aeds, c_matrix_eval=c_eval, seed=42, sparse=sparse),
        )
        rv_train, rv_eval, rv_cpu = resolve_exp2_reference(
            rvns_map,
            mun_name,
            "grasp_reativo_vns",
            "eval_pct",
            lambda: run_grasp_reativo_vns(
                c_train, inst.num_existing_aeds, c_matrix_eval=c_eval, r_max=3, max_iterations=96, seed=42, sparse=sparse
            ),
        )

        pr_res = run_grasp_reativo_vns_pr(
            c_matrix_train=c_train,
            n_to_open=inst.num_existing_aeds,
            c_matrix_eval=c_eval,
            r_max=3,
            max_iterations=96,
            seed=42,
            sparse=sparse,
        )
        vns = pr_res.vns_stats or {}
        prs = pr_res.pr_stats or {}
        pr_str = format_pr_summary(prs)

        print(
            f"{mun_name:<12} | {b_train:>7.2f}% | {rv_train:>7.2f}% | {pr_res.train_coverage_pct:>7.2f}% | "
            f"{b_eval:>7.2f}% | {rv_eval:>7.2f}% | {pr_res.eval_coverage_pct:>7.2f}% | "
            f"{pr_res.cpu_time_sec:>7.3f} | {pr_str:>14} | {format_vns_summary(vns):>20}"
        )

        records.append({
            "municipality": mun_name,
            "n_aeds": inst.num_existing_aeds,
            "n_candidates": len(inst.candidate_coords),
            "seed": prep.seed,
            "grasp_classico_ref": {"train_pct": round(b_train, 4), "eval_pct": round(b_eval, 4), "cpu_sec": round(b_cpu, 4)},
            "grasp_reativo_vns_ref": {"train_pct": round(rv_train, 4), "eval_pct": round(rv_eval, 4), "cpu_sec": round(rv_cpu, 4)},
            "grasp_reativo_vns_pr": {
                "train_pct": round(pr_res.train_coverage_pct, 4),
                "eval_pct": round(pr_res.eval_coverage_pct, 4),
                "cpu_sec": round(pr_res.cpu_time_sec, 4),
                "delta_train_vs_classico_pp": round(pr_res.train_coverage_pct - b_train, 4),
                "delta_eval_vs_classico_pp": round(pr_res.eval_coverage_pct - b_eval, 4),
                "delta_train_vs_rvns_pp": round(pr_res.train_coverage_pct - rv_train, 4),
                "vns_stats": vns,
                "pr_stats": prs,
                "alpha_probabilities": {str(k): round(v, 4) for k, v in (pr_res.alpha_probabilities or {}).items()},
                "alpha_counts": {str(k): v for k, v in (pr_res.alpha_counts or {}).items()},
                "alpha_avg_scores": {str(k): round(v, 4) for k, v in (pr_res.alpha_avg_scores or {}).items()},
                "selected_candidates": pr_res.selected_candidates.tolist(),
            },
        })
    print("-" * 120)
    return records

def run_experiment_3_comparison(
    shared_instances: Dict[str, PreparedInstance],
    baseline_json: Optional[Dict[str, Any]],
    rvns_json: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    print("\n[EXPERIMENTO 3] AEDs Adicionais via GRASP Reativo + VNS + PR (Tabela 6 — Eval %)")
    print("-" * 102)
    print(f"{'Município':<14} | {'N = 0':>9} | {'N = 5':>9} | {'N = 10':>9} | {'N = 20':>9} | {'N = 40':>9} | {'Tempo Total(s)':>14}")
    print("-" * 102)

    base_map = {e["municipality"]: e for e in (baseline_json or {}).get("experiment_3_table6", [])}
    rvns_map = {e["municipality"]: e for e in (rvns_json or {}).get("experiment_3_table6", [])}

    records = run_exp3_addition_sweep(
        shared_instances=shared_instances,
        solver_fn=lambda prep, n_add: run_grasp_reativo_vns_pr(
            c_matrix_train=prep.c_train,
            n_to_open=n_add,
            base_cov_train=prep.instance.base_cov_train,
            c_matrix_eval=prep.c_eval,
            base_cov_eval=prep.instance.base_cov_eval,
            r_max=3,
            max_iterations=96,
            seed=42,
            sparse=prep.sparse,
        ),
        collect_vns=True,
        collect_pr=True,
    )
    for rec in records:
        mun_name = rec["municipality"]
        cov_by_n = rec["eval_pct_by_n"]
        total_cpu = sum(rec["cpu_sec_by_n"].values())
        print(
            f"{mun_name:<14} | {cov_by_n['0']:>8.2f}% | {cov_by_n['5']:>8.2f}% | "
            f"{cov_by_n['10']:>8.2f}% | {cov_by_n['20']:>8.2f}% | {cov_by_n['40']:>8.2f}% | {total_cpu:>14.3f}"
        )
        base_entry = base_map.get(mun_name, {})
        rvns_entry = rvns_map.get(mun_name, {})
        rec["grasp_classico_ref_train_by_n"] = base_entry.get("train_pct_by_n")
        rec["grasp_classico_ref_eval_by_n"] = base_entry.get("eval_pct_by_n")
        rec["grasp_reativo_vns_ref_train_by_n"] = rvns_entry.get("train_pct_by_n")
        rec["grasp_reativo_vns_ref_eval_by_n"] = rvns_entry.get("eval_pct_by_n")
    print("=" * 102)
    return records

def build_markdown_report(
    exp1: List[Dict[str, Any]],
    exp2: List[Dict[str, Any]],
    exp3: List[Dict[str, Any]],
) -> str:
    lines = [
        "# Relatório Auditável — Etapa 2: GRASP Reativo + Basic VNS + Path Relinking (Slide 76)",
        "",
        "Comparação evolutiva nas mesmas instâncias padronizadas:",
        "`GRASP Clássico (Artigo)` $\\to$ `GRASP Reativo + Basic VNS` $\\to$ `GRASP Reativo + Basic VNS + Path Relinking (Slide 76)`.",
        "",
        "## Experimento 1 — Heiloo (Tabela 4)",
        "",
        "| Demandas | Clássico Treino | RVNS Treino | RVNS+PR Treino | Clássico Eval | RVNS Eval | RVNS+PR Eval | Tempo PR (s) | PR (Cham/Sk/+SB) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in exp1:
        b, rv, pr = r["grasp_classico_ref"], r["grasp_reativo_vns_ref"], r["grasp_reativo_vns_pr"]
        prs = pr["pr_stats"]
        lines.append(
            f"| {r['n_train_demands']} | {b['train_pct']:.2f}% | {rv['train_pct']:.2f}% | {pr['train_pct']:.2f}% | "
            f"{b['eval_pct']:.2f}% | {rv['eval_pct']:.2f}% | {pr['eval_pct']:.2f}% | {pr['cpu_sec']:.4f} | "
            f"{format_pr_summary(prs)} |"
        )

    lines.extend([
        "",
        "## Experimento 2 — Realocação de AEDs Existentes (Tabela 5)",
        "",
        "| Município | Clássico Treino | RVNS Treino | RVNS+PR Treino | Clássico Eval | RVNS Eval | RVNS+PR Eval | Tempo PR (s) | PR (Cham/Sk/+SB) | VNS (LS \\| +k1/k2/k3) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ])
    for r in exp2:
        b, rv, pr = r["grasp_classico_ref"], r["grasp_reativo_vns_ref"], r["grasp_reativo_vns_pr"]
        prs, vns = pr["pr_stats"], pr["vns_stats"]
        lines.append(
            f"| {r['municipality']} | {b['train_pct']:.2f}% | {rv['train_pct']:.2f}% | {pr['train_pct']:.2f}% | "
            f"{b['eval_pct']:.2f}% | {rv['eval_pct']:.2f}% | {pr['eval_pct']:.2f}% | {pr['cpu_sec']:.3f} | "
            f"{format_pr_summary(prs)} | "
            f"{format_vns_summary(vns)} |"
        )

    lines.extend([
        "",
        "## Experimento 3 — AEDs Adicionais (Tabela 6 — Treino % / Eval %)",
        "",
        "| Município | N = 5 (Eval) | N = 10 (Eval) | N = 20 (Eval) | N = 40 (Eval) | Treino N = 40 (Clás / RVNS / RVNS+PR) |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for r in exp3:
        c = r["eval_pct_by_n"]
        t_pr = r["train_pct_by_n"].get("40", 0.0)
        t_cl = (r.get("grasp_classico_ref_train_by_n") or {}).get("40", 0.0)
        t_rv = (r.get("grasp_reativo_vns_ref_train_by_n") or {}).get("40", 0.0)
        lines.append(
            f"| {r['municipality']} | {c['5']:.2f}% | {c['10']:.2f}% | {c['20']:.2f}% | {c['40']:.2f}% | "
            f"{t_cl:.2f}% / {t_rv:.2f}% / {t_pr:.2f}% |"
        )

    return "\n".join(lines) + "\n"

if __name__ == "__main__":
    print("Compilando kernels Numba (JIT) do GRASP Reativo + Basic VNS + Path Relinking...")
    warmup_numba()

    print("=" * 116)
    print(" SIMULAÇÃO ETAPA 2 — GRASP REATIVO + BASIC VNS + PATH RELINKING (SLIDE 76: SB -> SA)")
    print("=" * 116)

    baseline_data = load_audit_json("artigo_original.json")
    rvns_data = load_audit_json("grasp_reativo_vns.json")
    if baseline_data:
        print("Referência Clássico carregada de: codigo/resultados/artigo_original.json")
    if rvns_data:
        print("Referência RVNS carregada de    : codigo/resultados/grasp_reativo_vns.json")

    exp1_insts = build_exp1_instances()
    shared_insts = build_shared_instances()

    exp1_rec = run_experiment_1_comparison(exp1_insts, baseline_data, rvns_data)
    exp2_rec = run_experiment_2_comparison(shared_insts, baseline_data, rvns_data)
    exp3_rec = run_experiment_3_comparison(shared_insts, baseline_data, rvns_data)

    json_path = save_audit_json("grasp_reativo_vns_pr.json", {
        "method_family": "GRASP Reativo (Slide 17) + Basic VNS (Slide 28, r = 3) + Path Relinking (Slide 76: SB -> SA em top-4 abaixo de SB)",
        "alpha_pool": list(DEFAULT_ALPHA_POOL),
        "block_size": 15,
        "gamma": 4.0,
        "r_max": 3,
        "n_pr_targets": 4,
        "experiment_1_table4": exp1_rec,
        "experiment_2_table5": exp2_rec,
        "experiment_3_table6": exp3_rec,
    })
    md_path = save_audit_markdown(
        "relatorio_grasp_reativo_vns_pr.md",
        build_markdown_report(exp1_rec, exp2_rec, exp3_rec),
    )
    print(f"\nRelatórios auditáveis salvos em:\n  - {json_path}\n  - {md_path}")
