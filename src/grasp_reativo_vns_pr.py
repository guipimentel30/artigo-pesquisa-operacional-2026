import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from numba import njit

from .sparse_utils import (
    HeuristicResult,
    SparseCoverageStructures,
    build_sparse_structures,
    validate_solver_inputs,
    _recompute_best1_best2_owner,
    evaluate_solution,
)
from .grasp_reativo_vns import (
    DEFAULT_ALPHA_POOL,
    ReactiveAlphaController,
    execute_reactive_vns_iteration,
    init_vns_stats,
)

@njit(cache=True)
def _path_relinking_one_way_numba(
    n_demands: int,
    n_candidates: int,
    col_ptr: np.ndarray,
    row_idx: np.ndarray,
    cov_vals: np.ndarray,
    base_cov: np.ndarray,
    sol_start: np.ndarray,
    sol_target: np.ndarray,
):
    n_open = len(sol_start)
    y_target = np.zeros(n_candidates, dtype=np.uint8)
    y_start = np.zeros(n_candidates, dtype=np.uint8)
    for idx in range(n_open):
        y_start[sol_start[idx]] = 1
        y_target[sol_target[idx]] = 1

    out_pos = np.empty(n_open, dtype=np.int32)
    in_cands = np.empty(n_open, dtype=np.int32)
    n_out = 0
    n_in = 0

    for pos in range(n_open):
        j = sol_start[pos]
        if y_target[j] == 0:
            out_pos[n_out] = pos
            n_out += 1

    for idx in range(n_open):
        j = sol_target[idx]
        if y_start[j] == 0:
            in_cands[n_in] = j
            n_in += 1

    d = n_out
    if d < 2 or d != n_in:
        return sol_start.copy(), -1.0

    curr_sel = sol_start.copy()
    best_sel = sol_start.copy()
    best_total = -1.0

    best1 = np.empty(n_demands, dtype=np.float64)
    best2 = np.empty(n_demands, dtype=np.float64)
    owner = np.empty(n_demands, dtype=np.int32)

    rem_d = d
    for _ in range(d - 1):
        _recompute_best1_best2_owner(
            n_demands, col_ptr, row_idx, cov_vals, base_cov, curr_sel, best1, best2, owner
        )

        total_curr = 0.0
        for i in range(n_demands):
            total_curr += best1[i]

        best_delta = -1e18
        best_a = 0
        best_b = 0

        for a in range(rem_d):
            pos = out_pos[a]
            j1 = curr_sel[pos]

            loss_j1 = 0.0
            for k in range(col_ptr[j1], col_ptr[j1 + 1]):
                i = row_idx[k]
                if owner[i] == j1:
                    loss_j1 += (best1[i] - best2[i])

            for b in range(rem_d):
                j2 = in_cands[b]
                gain_j2 = 0.0
                for k2 in range(col_ptr[j2], col_ptr[j2 + 1]):
                    i = row_idx[k2]
                    v2 = cov_vals[k2]
                    phi_wo = best2[i] if owner[i] == j1 else best1[i]
                    if v2 > phi_wo:
                        gain_j2 += (v2 - phi_wo)

                delta = gain_j2 - loss_j1
                if delta > best_delta:
                    best_delta = delta
                    best_a = a
                    best_b = b

        chosen_pos = out_pos[best_a]
        chosen_in = in_cands[best_b]
        curr_sel[chosen_pos] = chosen_in
        new_total = total_curr + best_delta

        out_pos[best_a] = out_pos[rem_d - 1]
        in_cands[best_b] = in_cands[rem_d - 1]
        rem_d -= 1

        if new_total > best_total:
            best_total = new_total
            best_sel[:] = curr_sel[:]

    return best_sel, best_total

def _update_ranked_solutions(
    ranked: List[Tuple[float, Tuple[int, ...], np.ndarray]],
    cand_obj: float,
    cand_sel: np.ndarray,
    max_ranked: int = 16,
) -> None:
    cand_key = tuple(sorted(cand_sel.tolist()))
    for _, existing_key, _ in ranked:
        if existing_key == cand_key:
            return

    ranked.append((cand_obj, cand_key, cand_sel.copy()))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if len(ranked) > max_ranked:
        del ranked[max_ranked:]

def _select_slide76_target(
    ranked: List[Tuple[float, Tuple[int, ...], np.ndarray]],
    rng: np.random.Generator,
    n_targets: int = 4,
) -> Optional[Tuple[float, np.ndarray]]:
    if len(ranked) < 2:
        return None

    sb_obj = ranked[0][0]
    strict_below = [
        (obj, sel) for (obj, _, sel) in ranked[1:]
        if obj < sb_obj - 1e-10
    ][:n_targets]

    candidates = strict_below if strict_below else [(obj, sel) for (obj, _, sel) in ranked[1:1 + n_targets]]
    pick = int(rng.integers(0, len(candidates)))
    return candidates[pick]

def run_grasp_reativo_vns_pr(
    c_matrix_train: np.ndarray,
    n_to_open: int,
    base_cov_train: Optional[np.ndarray] = None,
    c_matrix_eval: Optional[np.ndarray] = None,
    base_cov_eval: Optional[np.ndarray] = None,
    alpha_pool: Sequence[float] = DEFAULT_ALPHA_POOL,
    block_size: int = 15,
    gamma: float = 4.0,
    min_prob: float = 0.01,
    epsilon: float = 5e-6,
    r_max: int = 3,
    n_pr_targets: int = 4,
    max_iterations: int = 96,
    time_limit_sec: Optional[float] = None,
    target_coverage_pct: Optional[float] = None,
    seed: int = 42,
    sparse: Optional[SparseCoverageStructures] = None,
) -> HeuristicResult:
    _, n_candidates, base_tr, base_ev = validate_solver_inputs(
        c_matrix_train, n_to_open, base_cov_train, c_matrix_eval, base_cov_eval, max_iterations
    )
    if r_max < 1:
        raise ValueError(f"r_max deve ser >= 1, recebido {r_max}.")
    if n_pr_targets < 1:
        raise ValueError(f"n_pr_targets deve ser >= 1, recebido {n_pr_targets}.")

    if sparse is None:
        sparse = build_sparse_structures(c_matrix_train)

    alpha_rng = np.random.default_rng(seed)
    pr_rng = np.random.default_rng(seed + 7919)
    alpha_ctrl = ReactiveAlphaController(
        alpha_pool=alpha_pool,
        block_size=block_size,
        gamma=gamma,
        min_prob=min_prob,
        rng=alpha_rng,
    )

    ranked_solutions: List[Tuple[float, Tuple[int, ...], np.ndarray]] = []
    max_ranked = max(16, 1 + n_pr_targets * 3)

    best_obj = -np.inf
    best_selected = np.empty(0, dtype=np.int32)
    best_y = np.zeros(n_candidates, dtype=np.uint8)
    history: List[float] = []
    time_to_best = 0.0

    vns_totals = init_vns_stats(r_max)
    pr_calls = 0
    pr_skipped_close = 0
    pr_improvements_over_sb = 0

    t0 = time.perf_counter()
    iters_done = 0

    for it in range(max_iterations):
        if time_limit_sec is not None and (time.perf_counter() - t0) >= time_limit_sec and it > 0:
            break

        sel, y_act, obj_vns = execute_reactive_vns_iteration(
            sparse=sparse,
            base_cov=base_tr,
            n_to_open=n_to_open,
            alpha_ctrl=alpha_ctrl,
            vns_totals=vns_totals,
            epsilon=epsilon,
            r_max=r_max,
            seed=seed,
            iteration=it,
        )

        if obj_vns > best_obj + 1e-12:
            best_obj = obj_vns
            best_selected = sel.copy()
            best_y = y_act.copy()
            time_to_best = time.perf_counter() - t0

        _update_ranked_solutions(ranked_solutions, obj_vns, sel, max_ranked=max_ranked)

        hit_target = target_coverage_pct is not None and best_obj >= target_coverage_pct - 1e-9

        if not hit_target:
            target = _select_slide76_target(ranked_solutions, pr_rng, n_targets=n_pr_targets)
            if target is not None:
                _, sa_sel = target
                sb_sel = ranked_solutions[0][2]
                pr_calls += 1

                si_sel, si_tot = _path_relinking_one_way_numba(
                    sparse.n_demands,
                    sparse.n_candidates,
                    sparse.col_ptr,
                    sparse.row_idx,
                    sparse.cov_vals,
                    base_tr,
                    sb_sel,
                    sa_sel,
                )

                if si_tot < 0.0:
                    pr_skipped_close += 1
                else:
                    si_obj = 100.0 * (si_tot / sparse.n_demands)
                    if si_obj > best_obj + 1e-12:
                        pr_improvements_over_sb += 1
                        best_obj = si_obj
                        best_selected = si_sel.copy()
                        best_y = np.zeros(n_candidates, dtype=np.uint8)
                        best_y[best_selected] = 1
                        time_to_best = time.perf_counter() - t0
                        _update_ranked_solutions(ranked_solutions, si_obj, si_sel, max_ranked=max_ranked)

        history.append(best_obj)
        iters_done += 1

        if target_coverage_pct is not None and best_obj >= target_coverage_pct - 1e-9:
            break

    cpu_time = time.perf_counter() - t0
    eval_cov_pct = (
        evaluate_solution(best_selected, c_matrix_eval, base_ev)
        if c_matrix_eval is not None
        else best_obj
    )

    alpha_probs_dict, alpha_counts_dict, alpha_avgs_dict = alpha_ctrl.export_dicts()

    pr_stats_dict: Dict[str, Any] = {
        "pr_calls": pr_calls,
        "pr_skipped_distance_lt_2": pr_skipped_close,
        "pr_improvements_over_sb": pr_improvements_over_sb,
        "ranked_pool_size": len(ranked_solutions),
        "n_pr_targets_list": n_pr_targets,
    }

    return HeuristicResult(
        method_name="GRASP Reativo + VNS + PR",
        selected_candidates=best_selected.astype(int),
        y_vector=best_y.astype(int),
        train_coverage_pct=best_obj,
        eval_coverage_pct=eval_cov_pct,
        cpu_time_sec=cpu_time,
        time_to_best_sec=time_to_best,
        iterations_completed=iters_done,
        iteration_history=history,
        alpha_probabilities=alpha_probs_dict,
        alpha_counts=alpha_counts_dict,
        alpha_avg_scores=alpha_avgs_dict,
        vns_stats=vns_totals,
        pr_stats=pr_stats_dict,
    )
