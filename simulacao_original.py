from typing import Any, Dict, List
import numpy as np

from src.coverage import (
    MODE_WEIGHTS,
    CUTOFF_DISTANCES_M,
    DISTANCE_MULTIPLIERS,
    coverage_piecewise_linear,
    coverage_sigmoid,
    coverage_binary,
    coverage_exponential,
)
from src.data_generator import build_coverage_matrix
from src.models_bip import solve_bip_exact
from src.heuristics import run_greedy, run_grasp
from src.sparse_utils import evaluate_solution
from src.benchmark_suite import (
    PreparedInstance,
    build_exp1_instances,
    build_shared_instances,
    run_exp3_addition_sweep,
    save_audit_json,
    save_audit_markdown,
)

def warmup_numba():
    dummy_c = np.random.uniform(0.0, 0.5, size=(20, 15))
    run_greedy(dummy_c, n_to_open=3)
    run_grasp(dummy_c, n_to_open=3, max_iterations=2)

def print_header():
    print("=" * 88)
    print(" SIMULAÇÃO FIEL DO ARTIGO BASE — BUTER ET AL. (2024)")
    print(" Strategic placement of volunteer responder system defibrillators")
    print("=" * 88)
    print("Parâmetros multimodais carregados (Tabelas 1 e 2):")
    for mode in ["foot", "bicycle", "car"]:
        print(
            f"  - Modal {mode:7s}: peso w_t = {MODE_WEIGHTS[mode]:.2f} | "
            f"multiplicador q_t = {DISTANCE_MULTIPLIERS[mode]:.3f} | "
            f"cutoff r_t = {CUTOFF_DISTANCES_M[mode]:.0f} m"
        )
    print("-" * 88)

def run_experiment_1_table4(exp1_instances: List[PreparedInstance]) -> List[Dict[str, Any]]:
    print("\n[EXPERIMENTO 1] Comparação BIP (Exato) vs. GRASP vs. Greedy (Seção 5.2.1 / Tabela 4)")
    print("-" * 88)
    print(
        f"{'Método':<18} | {'Demandas':>8} | {'Treino (%)':>10} | "
        f"{'Avaliação (%)':>13} | {'Tempo CPU (s)':>13} | {'Gap p/ BIP':>10}"
    )
    print("-" * 88)

    records: List[Dict[str, Any]] = []
    for prep in exp1_instances:
        inst, c_train, c_eval, sparse = prep.instance, prep.c_train, prep.c_eval, prep.sparse
        n_demands = len(inst.train_demands_coords)
        n_aeds = inst.num_existing_aeds

        bip_res = solve_bip_exact(c_train, n_to_open=n_aeds, time_limit_sec=60.0)
        bip_eval_cov = evaluate_solution(bip_res.selected_candidates, c_eval)

        grasp_res = run_grasp(
            c_matrix_train=c_train,
            n_to_open=n_aeds,
            c_matrix_eval=c_eval,
            alpha_start=0.95,
            alpha_end=0.00,
            alpha_step=0.01,
            epsilon=5e-6,
            seed=42,
            sparse=sparse,
        )
        grasp_gap = (bip_res.train_coverage_pct - grasp_res.train_coverage_pct) / bip_res.train_coverage_pct * 100.0

        greedy_res = run_greedy(
            c_matrix_train=c_train,
            n_to_open=n_aeds,
            c_matrix_eval=c_eval,
            sparse=sparse,
        )
        greedy_gap = (bip_res.train_coverage_pct - greedy_res.train_coverage_pct) / bip_res.train_coverage_pct * 100.0

        print(
            f"{'BIP (Exato)':<18} | {n_demands:>8d} | {bip_res.train_coverage_pct:>9.2f}% | "
            f"{bip_eval_cov:>12.2f}% | {bip_res.cpu_time_sec:>13.4f} | "
            f"{bip_res.optimality_gap_pct:>9.3f}%"
        )
        print(
            f"{'GRASP (Alg. 2)':<18} | {n_demands:>8d} | {grasp_res.train_coverage_pct:>9.2f}% | "
            f"{grasp_res.eval_coverage_pct:>12.2f}% | {grasp_res.cpu_time_sec:>13.4f} | "
            f"{grasp_gap:>9.3f}%"
        )
        print(
            f"{'Greedy (Alg. 1)':<18} | {n_demands:>8d} | {greedy_res.train_coverage_pct:>9.2f}% | "
            f"{greedy_res.eval_coverage_pct:>12.2f}% | {greedy_res.cpu_time_sec:>13.4f} | "
            f"{greedy_gap:>9.3f}%"
        )
        print("-" * 88)

        records.append({
            "municipality": inst.name,
            "n_train_demands": n_demands,
            "n_candidates": len(inst.candidate_coords),
            "n_aeds": n_aeds,
            "seed": prep.seed,
            "bip": {
                "train_pct": round(bip_res.train_coverage_pct, 4),
                "eval_pct": round(bip_eval_cov, 4),
                "cpu_sec": round(bip_res.cpu_time_sec, 4),
                "optimality_gap_pct": round(bip_res.optimality_gap_pct, 4),
            },
            "grasp_classico": {
                "train_pct": round(grasp_res.train_coverage_pct, 4),
                "eval_pct": round(grasp_res.eval_coverage_pct, 4),
                "cpu_sec": round(grasp_res.cpu_time_sec, 4),
                "gap_to_bip_pct": round(grasp_gap, 4),
                "iterations": grasp_res.iterations_completed,
            },
            "greedy": {
                "train_pct": round(greedy_res.train_coverage_pct, 4),
                "eval_pct": round(greedy_res.eval_coverage_pct, 4),
                "cpu_sec": round(greedy_res.cpu_time_sec, 4),
                "gap_to_bip_pct": round(greedy_gap, 4),
            },
        })
    return records

def _weighted_mu_mean(values: List[float], weights: List[int]) -> float:
    w = np.asarray(weights, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    return float(np.sum(w * v) / np.sum(w))

def run_experiment_2_table5(shared_instances: Dict[str, PreparedInstance]) -> List[Dict[str, Any]]:
    print("\n[EXPERIMENTO 2] Realocação de AEDs Existentes via GRASP (Seção 5.3 / Tabela 5)")
    print("-" * 88)
    print(
        f"{'Município':<16} | {'AEDs':>4} | {'OHCAs':>5} | {'Atual (%)':>10} | "
        f"{'Realocado (%)':>13} | {'Melhoria Rel.':>13} | {'Tempo (s)':>9}"
    )
    print("-" * 88)

    records: List[Dict[str, Any]] = []
    for mun_name, prep in shared_instances.items():
        inst, c_train, c_eval, sparse = prep.instance, prep.c_train, prep.c_eval, prep.sparse
        current_cov_pct = 100.0 * float(np.mean(inst.base_cov_eval))

        grasp_res = run_grasp(
            c_matrix_train=c_train,
            n_to_open=inst.num_existing_aeds,
            base_cov_train=None,
            c_matrix_eval=c_eval,
            base_cov_eval=None,
            seed=42,
            sparse=sparse,
        )
        rel_improv = ((grasp_res.eval_coverage_pct - current_cov_pct) / max(current_cov_pct, 1e-9)) * 100.0

        print(
            f"{mun_name:<16} | {inst.num_existing_aeds:>4d} | {inst.historic_ohcas_count:>5d} | "
            f"{current_cov_pct:>9.2f}% | {grasp_res.eval_coverage_pct:>12.2f}% | "
            f"{rel_improv:>+12.1f}% | {grasp_res.cpu_time_sec:>9.3f}"
        )

        records.append({
            "municipality": mun_name,
            "n_aeds": inst.num_existing_aeds,
            "historic_ohcas": inst.historic_ohcas_count,
            "n_candidates": len(inst.candidate_coords),
            "seed": prep.seed,
            "current_eval_pct": round(current_cov_pct, 4),
            "train_pct": round(grasp_res.train_coverage_pct, 4),
            "relocated_eval_pct": round(grasp_res.eval_coverage_pct, 4),
            "rel_improvement_pct": round(rel_improv, 2),
            "cpu_sec": round(grasp_res.cpu_time_sec, 4),
            "iterations": grasp_res.iterations_completed,
            "selected_candidates": grasp_res.selected_candidates.tolist(),
        })
    weights = [r["historic_ohcas"] for r in records]
    ov_cur = _weighted_mu_mean([r["current_eval_pct"] for r in records], weights)
    ov_trn = _weighted_mu_mean([r["train_pct"] for r in records], weights)
    ov_rel = _weighted_mu_mean([r["relocated_eval_pct"] for r in records], weights)
    ov_imp = ((ov_rel - ov_cur) / max(ov_cur, 1e-9)) * 100.0
    print("-" * 88)
    print(
        f"{'Overall (Eq.4)':<16} | {sum(r['n_aeds'] for r in records):>4d} | "
        f"{sum(weights):>5d} | {ov_cur:>9.2f}% | {ov_rel:>12.2f}% | {ov_imp:>+12.1f}% | {'---':>9}"
    )
    print("-" * 88)
    return records

def run_experiment_3_table6(shared_instances: Dict[str, PreparedInstance]) -> List[Dict[str, Any]]:
    print("\n[EXPERIMENTO 3] Implantação de AEDs Adicionais mantendo Existentes Fixos (Seção 5.4 / Tabela 6)")
    print("-" * 88)
    print(f"{'Município':<16} | {'N = 0':>9} | {'N = 5':>9} | {'N = 10':>9} | {'N = 20':>9} | {'N = 40':>9}")
    print("-" * 88)

    records = run_exp3_addition_sweep(
        shared_instances=shared_instances,
        solver_fn=lambda prep, n_add: run_grasp(
            c_matrix_train=prep.c_train,
            n_to_open=n_add,
            base_cov_train=prep.instance.base_cov_train,
            c_matrix_eval=prep.c_eval,
            base_cov_eval=prep.instance.base_cov_eval,
            seed=42,
            sparse=prep.sparse,
        ),
    )
    for rec in records:
        cov_by_n = rec["eval_pct_by_n"]
        print(
            f"{rec['municipality']:<16} | {cov_by_n['0']:>8.2f}% | {cov_by_n['5']:>8.2f}% | "
            f"{cov_by_n['10']:>8.2f}% | {cov_by_n['20']:>8.2f}% | {cov_by_n['40']:>8.2f}%"
        )
    w_exp3 = [shared_instances[r["municipality"]].instance.historic_ohcas_count for r in records]
    ov3 = {n_k: _weighted_mu_mean([r["eval_pct_by_n"][n_k] for r in records], w_exp3) for n_k in ["0", "5", "10", "20", "40"]}
    print("-" * 88)
    print(
        f"{'Overall (Eq.4)':<16} | {ov3['0']:>8.2f}% | {ov3['5']:>8.2f}% | "
        f"{ov3['10']:>8.2f}% | {ov3['20']:>8.2f}% | {ov3['40']:>8.2f}%"
    )
    print("-" * 88)
    return records

def run_experiment_4_table7(shared_instances: Dict[str, PreparedInstance]) -> Dict[str, Any]:
    print("\n[EXPERIMENTO 4] Análise de Sensibilidade da Função de Cobertura — Zaanstad (Seção 5.5 / Tabela 7)")
    print("-" * 88)

    inst = shared_instances["Zaanstad"].instance
    cov_functions = [
        ("Piecewise linear", coverage_piecewise_linear),
        ("Sigmoid",          coverage_sigmoid),
        ("Binary",           coverage_binary),
        ("Exponential",      coverage_exponential),
    ]

    eval_matrices = {
        name: build_coverage_matrix(inst.eval_demands_coords, inst.candidate_coords, coverage_fn=fn)
        for name, fn in cov_functions
    }

    solutions = {}
    for name, fn in cov_functions:
        c_train_fn = build_coverage_matrix(inst.train_demands_coords, inst.candidate_coords, coverage_fn=fn)
        res = run_grasp(
            c_matrix_train=c_train_fn,
            n_to_open=inst.num_existing_aeds,
            base_cov_train=None,
            c_matrix_eval=eval_matrices[name],
            base_cov_eval=None,
            seed=42,
        )
        solutions[name] = res.selected_candidates

    diag_vals = {
        true_name: evaluate_solution(solutions[true_name], eval_matrices[true_name])
        for true_name, _ in cov_functions
    }

    print(
        f"{'Assumida \\ Verdadeira':<20} | {'Piecewise lin.':>16} | "
        f"{'Sigmoid':>16} | {'Binary':>16} | {'Exponential':>16}"
    )
    print("-" * 88)

    matrix_records: Dict[str, Dict[str, Dict[str, float]]] = {}
    for assumed_name, _ in cov_functions:
        matrix_records[assumed_name] = {}
        row_cells = []
        for true_name, _ in cov_functions:
            val = evaluate_solution(solutions[assumed_name], eval_matrices[true_name])
            rel = (val / max(diag_vals[true_name], 1e-9)) * 100.0
            row_cells.append(f"{val:5.2f}% ({rel:6.2f}%)")
            matrix_records[assumed_name][true_name] = {
                "eval_pct": round(val, 4),
                "relative_to_diag_pct": round(rel, 4),
            }
        print(
            f"{assumed_name:<20} | {row_cells[0]:>16} | {row_cells[1]:>16} | "
            f"{row_cells[2]:>16} | {row_cells[3]:>16}"
        )
    print("=" * 88)
    return matrix_records

def build_markdown_report(
    exp1: List[Dict[str, Any]],
    exp2: List[Dict[str, Any]],
    exp3: List[Dict[str, Any]],
    exp4: Dict[str, Any],
) -> str:
    lines = [
        "# Relatório Auditável — Baselines do Artigo Original (Buter et al., 2024)",
        "",
        "Métodos avaliados: **BIP (Exato)**, **Greedy (Algoritmo 1)** e **GRASP Clássico (Algoritmo 2)**.",
        "",
        "## Experimento 1 — BIP vs. GRASP Clássico vs. Greedy (Tabela 4 / Heiloo)",
        "",
        "| Método | Demandas | Treino (%) | Avaliação (%) | Tempo CPU (s) | Gap p/ BIP (%) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in exp1:
        n = r["n_train_demands"]
        b, g, gr = r["bip"], r["grasp_classico"], r["greedy"]
        lines.append(f"| BIP (Exato) | {n} | {b['train_pct']:.2f}% | {b['eval_pct']:.2f}% | {b['cpu_sec']:.4f} | {b['optimality_gap_pct']:.3f}% |")
        lines.append(f"| GRASP Clássico (Alg. 2) | {n} | {g['train_pct']:.2f}% | {g['eval_pct']:.2f}% | {g['cpu_sec']:.4f} | {g['gap_to_bip_pct']:.3f}% |")
        lines.append(f"| Greedy (Alg. 1) | {n} | {gr['train_pct']:.2f}% | {gr['eval_pct']:.2f}% | {gr['cpu_sec']:.4f} | {gr['gap_to_bip_pct']:.3f}% |")

    lines.extend([
        "",
        "## Experimento 2 — Realocação de AEDs Existentes via GRASP Clássico (Tabela 5)",
        "",
        "| Município | AEDs | OHCAs | Candidatos | Atual Eval (%) | Treino (%) | Realocado Eval (%) | Melhoria Rel. | Tempo (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for r in exp2:
        lines.append(
            f"| {r['municipality']} | {r['n_aeds']} | {r['historic_ohcas']} | {r['n_candidates']} | "
            f"{r['current_eval_pct']:.2f}% | {r['train_pct']:.2f}% | {r['relocated_eval_pct']:.2f}% | "
            f"{r['rel_improvement_pct']:+.1f}% | {r['cpu_sec']:.3f} |"
        )
    w2 = [r["historic_ohcas"] for r in exp2]
    ov2_cur = _weighted_mu_mean([r["current_eval_pct"] for r in exp2], w2)
    ov2_trn = _weighted_mu_mean([r["train_pct"] for r in exp2], w2)
    ov2_rel = _weighted_mu_mean([r["relocated_eval_pct"] for r in exp2], w2)
    ov2_imp = ((ov2_rel - ov2_cur) / max(ov2_cur, 1e-9)) * 100.0
    lines.append(
        f"| **Overall (Eq. 4)** | {sum(r['n_aeds'] for r in exp2)} | {sum(w2)} | --- | "
        f"{ov2_cur:.2f}% | {ov2_trn:.2f}% | {ov2_rel:.2f}% | {ov2_imp:+.1f}% | --- |"
    )

    lines.extend([
        "",
        "## Experimento 3 — Implantação de AEDs Adicionais via GRASP Clássico (Tabela 6)",
        "",
        "| Município | N = 0 | N = 5 | N = 10 | N = 20 | N = 40 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    ohca_by_mun = {r["municipality"]: r["historic_ohcas"] for r in exp2}
    for r in exp3:
        c = r["eval_pct_by_n"]
        lines.append(
            f"| {r['municipality']} | {c['0']:.2f}% | {c['5']:.2f}% | {c['10']:.2f}% | {c['20']:.2f}% | {c['40']:.2f}% |"
        )
    w3 = [ohca_by_mun[r["municipality"]] for r in exp3]
    ov3 = {k: _weighted_mu_mean([r["eval_pct_by_n"][k] for r in exp3], w3) for k in ["0", "5", "10", "20", "40"]}
    lines.append(
        f"| **Overall (Eq. 4)** | {ov3['0']:.2f}% | {ov3['5']:.2f}% | {ov3['10']:.2f}% | {ov3['20']:.2f}% | {ov3['40']:.2f}% |"
    )

    lines.extend([
        "",
        "## Experimento 4 — Sensibilidade da Função de Cobertura em Zaanstad (Tabela 7)",
        "",
        "| Assumida \\ Verdadeira | Piecewise linear | Sigmoid | Binary | Exponential |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for assumed, row in exp4.items():
        cells = [
            f"{row[t]['eval_pct']:.2f}% ({row[t]['relative_to_diag_pct']:.2f}%)"
            for t in ["Piecewise linear", "Sigmoid", "Binary", "Exponential"]
        ]
        lines.append(f"| {assumed} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")

    return "\n".join(lines) + "\n"

if __name__ == "__main__":
    print("Compilando kernels Numba (JIT)...")
    warmup_numba()
    print_header()

    exp1_insts = build_exp1_instances()
    shared_insts = build_shared_instances()

    exp1_rec = run_experiment_1_table4(exp1_insts)
    exp2_rec = run_experiment_2_table5(shared_insts)
    exp3_rec = run_experiment_3_table6(shared_insts)
    exp4_rec = run_experiment_4_table7(shared_insts)

    w2 = [r["historic_ohcas"] for r in exp2_rec]
    ohca_by_mun = {r["municipality"]: r["historic_ohcas"] for r in exp2_rec}
    w3 = [ohca_by_mun[r["municipality"]] for r in exp3_rec]

    json_path = save_audit_json("artigo_original.json", {
        "method_family": "Baselines do Artigo Original (Buter et al., 2024)",
        "experiment_1_table4": exp1_rec,
        "experiment_2_table5": exp2_rec,
        "experiment_2_overall_weighted_mu_m": {
            "current_eval_pct": round(_weighted_mu_mean([r["current_eval_pct"] for r in exp2_rec], w2), 4),
            "train_pct": round(_weighted_mu_mean([r["train_pct"] for r in exp2_rec], w2), 4),
            "relocated_eval_pct": round(_weighted_mu_mean([r["relocated_eval_pct"] for r in exp2_rec], w2), 4),
        },
        "experiment_3_table6": exp3_rec,
        "experiment_3_overall_weighted_mu_m": {
            k: round(_weighted_mu_mean([r["eval_pct_by_n"][k] for r in exp3_rec], w3), 4)
            for k in ["0", "5", "10", "20", "40"]
        },
        "experiment_4_table7": exp4_rec,
    })
    md_path = save_audit_markdown(
        "relatorio_artigo_original.md",
        build_markdown_report(exp1_rec, exp2_rec, exp3_rec, exp4_rec),
    )
    print(f"\nRelatórios auditáveis salvos em:\n  - {json_path}\n  - {md_path}")
