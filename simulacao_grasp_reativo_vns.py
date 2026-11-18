from typing import Any, Dict, List, Optional
import numpy as np

from src.heuristics import run_grasp
from src.grasp_reativo_vns import DEFAULT_ALPHA_POOL, run_grasp_reativo_vns
from src.benchmark_suite import (
    PreparedInstance,
    build_exp1_instances,
    build_shared_instances,
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
    run_grasp(dummy_c, n_to_open=5, max_iterations=2)
    run_grasp_reativo_vns(dummy_c, n_to_open=5, r_max=3, max_iterations=12)

def run_experiment_1_comparison(
    exp1_instances: List[PreparedInstance],
    baseline_json: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    print("\n[EXPERIMENTO 1] GRASP Clássico vs. GRASP Reativo + Basic VNS (Heiloo / Tabela 4)")
    print("-" * 106)
    print(
        f"{'Método':<24} | {'Demandas':>8} | {'Treino (%)':>10} | {'Eval (%)':>10} | "
        f"{'Tempo (s)':>9} | {'VNS (LS | +k1/+k2/+k3)':>25}"
    )
    print("-" * 106)

    records: List[Dict[str, Any]] = []
    for idx, prep in enumerate(exp1_instances):
        inst, c_train, c_eval, sparse = prep.instance, prep.c_train, prep.c_eval, prep.sparse
        n_demands = len(inst.train_demands_coords)
        n_aeds = inst.num_existing_aeds

        base_train, base_eval, base_cpu = resolve_exp1_reference(
            baseline_json,
            idx,
            "grasp_classico",
            lambda: run_grasp(c_train, n_aeds, c_matrix_eval=c_eval, seed=42, sparse=sparse),
        )

        rvns_res = run_grasp_reativo_vns(
            c_matrix_train=c_train,
            n_to_open=n_aeds,
            c_matrix_eval=c_eval,
            r_max=3,
            max_iterations=96,
            seed=42,
            sparse=sparse,
        )
        vns = rvns_res.vns_stats or {}

        print(
            f"{'GRASP Clássico':<24} | {n_demands:>8d} | {base_train:>9.2f}% | {base_eval:>9.2f}% | "
            f"{base_cpu:>9.4f} | {'---':>25}"
        )
        print(
            f"{'GRASP Reativo + VNS':<24} | {n_demands:>8d} | {rvns_res.train_coverage_pct:>9.2f}% | "
            f"{rvns_res.eval_coverage_pct:>9.2f}% | {rvns_res.cpu_time_sec:>9.4f} | {format_vns_summary(vns):>25}"
        )
        print("-" * 106)

        records.append({
            "municipality": inst.name,
            "n_train_demands": n_demands,
            "n_candidates": len(inst.candidate_coords),
            "n_aeds": n_aeds,
            "seed": prep.seed,
            "grasp_classico_ref": {
                "train_pct": round(base_train, 4),
                "eval_pct": round(base_eval, 4),
                "cpu_sec": round(base_cpu, 4),
            },
            "grasp_reativo_vns": {
                "train_pct": round(rvns_res.train_coverage_pct, 4),
                "eval_pct": round(rvns_res.eval_coverage_pct, 4),
                "cpu_sec": round(rvns_res.cpu_time_sec, 4),
                "delta_train_pp": round(rvns_res.train_coverage_pct - base_train, 4),
                "delta_eval_pp": round(rvns_res.eval_coverage_pct - base_eval, 4),
                "cpu_ratio_vs_classico": round(rvns_res.cpu_time_sec / max(base_cpu, 1e-9), 2),
                "iterations": rvns_res.iterations_completed,
                "vns_stats": vns,
                "alpha_probabilities": {
                    str(k): round(v, 4) for k, v in (rvns_res.alpha_probabilities or {}).items()
                },
                "alpha_counts": {str(k): v for k, v in (rvns_res.alpha_counts or {}).items()},
            },
        })
    return records

def run_experiment_2_comparison(
    shared_instances: Dict[str, PreparedInstance],
    baseline_json: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    print("\n[EXPERIMENTO 2] Realocação (Tabela 5) — Comparativo & Estatísticas Basic VNS")
    print("-" * 112)
    print(
        f"{'Município':<12} | {'Clás Trn':>8} | {'RVNS Trn':>8} | {'Clás Evl':>8} | "
        f"{'RVNS Evl':>8} | {'t Clás(s)':>9} | {'t RVNS(s)':>9} | {'VNS (LS | +k1/+k2/+k3)':>25}"
    )
    print("-" * 112)

    base_map = {e["municipality"]: e for e in (baseline_json or {}).get("experiment_2_table5", [])}

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

        rvns_res = run_grasp_reativo_vns(
            c_matrix_train=c_train,
            n_to_open=inst.num_existing_aeds,
            c_matrix_eval=c_eval,
            r_max=3,
            max_iterations=96,
            seed=42,
            sparse=sparse,
        )
        vns = rvns_res.vns_stats or {}

        print(
            f"{mun_name:<12} | {b_train:>7.2f}% | {rvns_res.train_coverage_pct:>7.2f}% | "
            f"{b_eval:>7.2f}% | {rvns_res.eval_coverage_pct:>7.2f}% | "
            f"{b_cpu:>9.3f} | {rvns_res.cpu_time_sec:>9.3f} | {format_vns_summary(vns):>25}"
        )

        records.append({
            "municipality": mun_name,
            "n_aeds": inst.num_existing_aeds,
            "n_candidates": len(inst.candidate_coords),
            "seed": prep.seed,
            "grasp_classico_ref": {
                "train_pct": round(b_train, 4),
                "eval_pct": round(b_eval, 4),
                "cpu_sec": round(b_cpu, 4),
            },
            "grasp_reativo_vns": {
                "train_pct": round(rvns_res.train_coverage_pct, 4),
                "eval_pct": round(rvns_res.eval_coverage_pct, 4),
                "cpu_sec": round(rvns_res.cpu_time_sec, 4),
                "delta_train_pp": round(rvns_res.train_coverage_pct - b_train, 4),
                "delta_eval_pp": round(rvns_res.eval_coverage_pct - b_eval, 4),
                "cpu_ratio_vs_classico": round(rvns_res.cpu_time_sec / max(b_cpu, 1e-9), 2),
                "vns_stats": vns,
                "alpha_probabilities": {
                    str(k): round(v, 4) for k, v in (rvns_res.alpha_probabilities or {}).items()
                },
                "alpha_counts": {str(k): v for k, v in (rvns_res.alpha_counts or {}).items()},
                "alpha_avg_scores": {
                    str(k): round(v, 4) for k, v in (rvns_res.alpha_avg_scores or {}).items()
                },
                "selected_candidates": rvns_res.selected_candidates.tolist(),
            },
        })
    print("-" * 112)

    zaanstad_rec = next((r for r in records if r["municipality"] == "Zaanstad"), None)
    if zaanstad_rec:
        rv = zaanstad_rec["grasp_reativo_vns"]
        print("\n[AUDITORIA GRASP REATIVO — Zaanstad] Probabilidades finais p_k e médias A_k por alpha_k:")
        print("-" * 80)
        print(f"{'alpha_k':>8} | {'Vezes usado':>11} | {'Média Treino A_k':>17} | {'Prob. Final p_k':>16}")
        print("-" * 80)
        for a in DEFAULT_ALPHA_POOL:
            k_str = str(float(a))
            cnt = rv["alpha_counts"].get(k_str, 0)
            avg_s = rv["alpha_avg_scores"].get(k_str, 0.0)
            prob = rv["alpha_probabilities"].get(k_str, 0.0)
            print(f"{a:>8.2f} | {cnt:>11d} | {avg_s:>16.2f}% | {prob * 100.0:>15.2f}%")
        print("-" * 80)

    return records

def run_experiment_3_comparison(
    shared_instances: Dict[str, PreparedInstance],
    baseline_json: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    print("\n[EXPERIMENTO 3] AEDs Adicionais via GRASP Reativo + Basic VNS (Tabela 6)")
    print("-" * 96)
    print(
        f"{'Município':<14} | {'N = 0':>9} | {'N = 5':>9} | {'N = 10':>9} | "
        f"{'N = 20':>9} | {'N = 40':>9} | {'Tempo Total(s)':>14}"
    )
    print("-" * 96)

    base_map = {e["municipality"]: e for e in (baseline_json or {}).get("experiment_3_table6", [])}
    records = run_exp3_addition_sweep(
        shared_instances=shared_instances,
        solver_fn=lambda prep, n_add: run_grasp_reativo_vns(
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
        rec["grasp_classico_ref_eval_by_n"] = base_entry.get("eval_pct_by_n")
        rec["grasp_classico_ref_train_by_n"] = base_entry.get("train_pct_by_n")
        rec["grasp_classico_ref_cpu_by_n"] = base_entry.get("cpu_sec_by_n")
    print("=" * 96)
    return records

def build_markdown_report(
    exp1: List[Dict[str, Any]],
    exp2: List[Dict[str, Any]],
    exp3: List[Dict[str, Any]],
) -> str:
    lines = [
        "# Relatório Auditável — Etapa 1: GRASP Reativo + Basic VNS",
        "",
        "Comparação direta sobre as mesmas instâncias padronizadas do baseline (**GRASP Clássico**, Buter et al., 2024).",
        "Vizinhanças Basic VNS (Slide 28–29): $N^{(1)}$ (`1-swap`), $N^{(2)}$ (`2-swap`) e $N^{(3)}$ (`3-swap`) com `BuscaLocal = 1-swap`.",
        "",
        "## Experimento 1 — Heiloo (Qualidade, Tempo e Basic VNS)",
        "",
        "| Demandas | Clássico Treino | RVNS Treino | Clássico Eval | RVNS Eval | t Clássico (s) | t RVNS (s) | VNS (LS \\| +k1/+k2/+k3) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in exp1:
        b = r["grasp_classico_ref"]
        rv = r["grasp_reativo_vns"]
        lines.append(
            f"| {r['n_train_demands']} | {b['train_pct']:.2f}% | {rv['train_pct']:.2f}% | "
            f"{b['eval_pct']:.2f}% | {rv['eval_pct']:.2f}% | {b['cpu_sec']:.4f} | {rv['cpu_sec']:.4f} | "
            f"{format_vns_summary(rv['vns_stats'])} |"
        )

    lines.extend([
        "",
        "## Experimento 2 — Realocação de AEDs Existentes (Tabela 5)",
        "",
        "| Município | Clássico Treino | RVNS Treino | Clássico Eval | RVNS Eval | t Clássico (s) | t RVNS (s) | VNS (LS \\| +k1/+k2/+k3) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for r in exp2:
        b = r["grasp_classico_ref"]
        rv = r["grasp_reativo_vns"]
        lines.append(
            f"| {r['municipality']} | {b['train_pct']:.2f}% | {rv['train_pct']:.2f}% | "
            f"{b['eval_pct']:.2f}% | {rv['eval_pct']:.2f}% | {b['cpu_sec']:.3f} | {rv['cpu_sec']:.3f} | "
            f"{format_vns_summary(rv['vns_stats'])} |"
        )

    zaan = next((r for r in exp2 if r["municipality"] == "Zaanstad"), None)
    if zaan:
        rv = zaan["grasp_reativo_vns"]
        lines.extend([
            "",
            "### Auditoria das Probabilidades Reativas Aprendidas (Zaanstad)",
            "",
            "| $\\alpha_k$ | Iterações | Média Treino $A_k$ | Probabilidade Final $p_k$ |",
            "| ---: | ---: | ---: | ---: |",
        ])
        for a in DEFAULT_ALPHA_POOL:
            k_str = str(float(a))
            lines.append(
                f"| {a:.2f} | {rv['alpha_counts'][k_str]} | {rv['alpha_avg_scores'][k_str]:.2f}% | "
                f"{rv['alpha_probabilities'][k_str] * 100.0:.2f}% |"
            )

    lines.extend([
        "",
        "## Experimento 3 — AEDs Adicionais (Tabela 6 — Eval %)",
        "",
        "| Município | N = 0 | N = 5 (RVNS) | N = 10 (RVNS) | N = 20 (RVNS) | N = 40 (RVNS) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for r in exp3:
        c = r["eval_pct_by_n"]
        lines.append(
            f"| {r['municipality']} | {c['0']:.2f}% | {c['5']:.2f}% | {c['10']:.2f}% | {c['20']:.2f}% | {c['40']:.2f}% |"
        )

    return "\n".join(lines) + "\n"

if __name__ == "__main__":
    print("Compilando kernels Numba (JIT) do GRASP Reativo + Basic VNS...")
    warmup_numba()

    print("=" * 106)
    print(" SIMULAÇÃO ETAPA 1 — GRASP REATIVO (SLIDE 17) + BASIC VNS (SLIDES 28-29, k = 1, 2, 3)")
    print("=" * 106)

    baseline_data = load_audit_json("artigo_original.json")
    if baseline_data:
        print("Referência carregada de: codigo/resultados/artigo_original.json")
    else:
        print("Aviso: artigo_original.json não encontrado; executando GRASP Clássico em linha para comparação.")

    exp1_insts = build_exp1_instances()
    shared_insts = build_shared_instances()

    exp1_rec = run_experiment_1_comparison(exp1_insts, baseline_data)
    exp2_rec = run_experiment_2_comparison(shared_insts, baseline_data)
    exp3_rec = run_experiment_3_comparison(shared_insts, baseline_data)

    json_path = save_audit_json("grasp_reativo_vns.json", {
        "method_family": "GRASP Reativo (Slide 17) + Basic VNS (Slide 28, r = 3 com k = 1, 2, 3)",
        "alpha_pool": list(DEFAULT_ALPHA_POOL),
        "block_size": 15,
        "gamma": 4.0,
        "r_max": 3,
        "experiment_1_table4": exp1_rec,
        "experiment_2_table5": exp2_rec,
        "experiment_3_table6": exp3_rec,
    })
    md_path = save_audit_markdown(
        "relatorio_grasp_reativo_vns.md",
        build_markdown_report(exp1_rec, exp2_rec, exp3_rec),
    )
    print(f"\nRelatórios auditáveis salvos em:\n  - {json_path}\n  - {md_path}")
