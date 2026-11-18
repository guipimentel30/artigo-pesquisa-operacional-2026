from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ttt_plots import (  # noqa: E402
    DEFAULT_ALGORITHMS,
    DEFAULT_TARGET_ORDER,
    run_large_scale_ttt_experiment,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimento 5: Escala Ampliada e Curvas Time-to-Target (TTT-plots)."
    )
    parser.add_argument("--municipality", default="Hollands Kroon")
    parser.add_argument("--max-candidates", type=int, default=1200)
    parser.add_argument("--n-train", type=int, default=1500)
    parser.add_argument("--n-eval", type=int, default=5000)
    parser.add_argument("--instance-seed", type=int, default=42)
    parser.add_argument("--n-cal", type=int, default=30)
    parser.add_argument("--cal-iters", type=int, default=30)
    parser.add_argument("--cal-base-seed", type=int, default=5000)
    parser.add_argument("--n-ttt", type=int, default=100)
    parser.add_argument("--ttt-max-iters", type=int, default=96)
    parser.add_argument("--ttt-time-limit", type=float, default=20.0)
    parser.add_argument("--ttt-base-seed", type=int, default=100)
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["calibration", "ttt", "finalize"],
        default=["calibration", "ttt", "finalize"],
        help="Fases a executar neste chamado (padrão: todas).",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=list(DEFAULT_TARGET_ORDER),
        default=None,
        help="Subconjunto opcional de alvos na fase ttt.",
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        choices=[spec.key for spec in DEFAULT_ALGORITHMS],
        default=None,
        help="Subconjunto opcional de algoritmos.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignora checkpoint anterior e reexecuta todas as fases.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    print(
        f"Iniciando Experimento 5 ({args.municipality}, |J^c|={args.max_candidates}, "
        f"|I^train|={args.n_train}, cal={args.n_cal}x{args.cal_iters}, "
        f"ttt={args.n_ttt}x{args.ttt_max_iters} cap={args.ttt_time_limit}s, "
        f"fases={args.phases})...",
        flush=True,
    )
    payload = run_large_scale_ttt_experiment(
        municipality=args.municipality,
        n_train_demands=args.n_train,
        n_eval_demands=args.n_eval,
        max_candidates=args.max_candidates,
        instance_seed=args.instance_seed,
        n_cal_runs=args.n_cal,
        cal_max_iterations=args.cal_iters,
        cal_base_seed=args.cal_base_seed,
        n_ttt_runs=args.n_ttt,
        ttt_max_iterations=args.ttt_max_iters,
        ttt_time_limit_sec=args.ttt_time_limit,
        ttt_base_seed=args.ttt_base_seed,
        resume_checkpoint=not args.no_resume,
        phases=args.phases,
        target_filter=args.targets,
        algo_filter=args.algos,
    )

    if "targets_pct" in payload.get("calibration", {}):
        print("\n=== Alvos da Calibração (Slide 94) ===", flush=True)
        for k, v in payload["calibration"]["targets_pct"].items():
            print(f"  - {k:6s}: {v:.6f}%", flush=True)

    if "audit_files" in payload:
        print("\n=== Resumo Final TTT-plots ===", flush=True)
        for target_key in DEFAULT_TARGET_ORDER:
            tdata = payload["ttt_plots"][target_key]
            print(f"Alvo {tdata['target_label']} ({tdata['target_value_pct']:.4f}%):", flush=True)
            for spec in DEFAULT_ALGORITHMS:
                c = tdata["algorithms"][spec.key]["curve"]
                med = c["hit_time_stats"]["median"]
                med_txt = f"{med:.3f}s" if med is not None else "Censurado"
                print(
                    f"  - {spec.short_label:22s}: {c['hits']:3d}/{c['n_total_runs']} "
                    f"| mediana acertos = {med_txt}",
                    flush=True,
                )

        print(
            f"\nTempo acumulado dos resolvedores: {payload['total_solver_cpu_time_sec']:.1f}s "
            f"(etapa atual: {payload['total_wall_time_sec']:.1f}s)",
            flush=True,
        )
        print(f"JSON: {payload['audit_files']['json']}", flush=True)
        print(f"Markdown: {payload['audit_files']['markdown']}", flush=True)
        print(
            f"Figuras: {len(payload['figures']['individual']) + len(payload['figures']['consolidated'])} arquivos",
            flush=True,
        )

if __name__ == "__main__":
    main()
