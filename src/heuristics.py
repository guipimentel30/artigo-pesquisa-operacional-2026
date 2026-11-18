import time
from typing import List, Optional
import numpy as np

from .sparse_utils import (
    HeuristicResult,
    SparseCoverageStructures,
    build_sparse_structures,
    validate_solver_inputs,
    _construct_solution_numba,
    _local_search_1swap_numba,
    evaluate_solution,
)

def run_greedy(
    c_matrix_train: np.ndarray,
    n_to_open: int,
    base_cov_train: Optional[np.ndarray] = None,
    c_matrix_eval: Optional[np.ndarray] = None,
    base_cov_eval: Optional[np.ndarray] = None,
    sparse: Optional[SparseCoverageStructures] = None,
) -> HeuristicResult:
    n_demands, n_candidates, base_tr, base_ev = validate_solver_inputs(
        c_matrix_train, n_to_open, base_cov_train, c_matrix_eval, base_cov_eval
    )

    if sparse is None:
        sparse = build_sparse_structures(c_matrix_train)

    t0 = time.perf_counter()
    selected, y_active, phi_train = _construct_solution_numba(
        n_demands=n_demands,
        n_candidates=n_candidates,
        col_ptr=sparse.col_ptr,
        row_idx=sparse.row_idx,
        cov_vals=sparse.cov_vals,
        row_ptr=sparse.row_ptr,
        col_idx=sparse.col_idx,
        row_vals=sparse.row_vals,
        base_cov=base_tr,
        n_to_open=n_to_open,
        alpha=1.0,
        deterministic=True,
        rng_seed=0,
    )
    cpu_time = time.perf_counter() - t0

    train_cov_pct = 100.0 * float(np.mean(phi_train))
    eval_cov_pct = evaluate_solution(selected, c_matrix_eval, base_ev) if c_matrix_eval is not None else train_cov_pct

    return HeuristicResult(
        method_name="Greedy (Alg. 1)",
        selected_candidates=selected.astype(int),
        y_vector=y_active.astype(int),
        train_coverage_pct=train_cov_pct,
        eval_coverage_pct=eval_cov_pct,
        cpu_time_sec=cpu_time,
        time_to_best_sec=cpu_time,
        iterations_completed=1,
        iteration_history=[train_cov_pct],
    )

def run_grasp(
    c_matrix_train: np.ndarray,
    n_to_open: int,
    base_cov_train: Optional[np.ndarray] = None,
    c_matrix_eval: Optional[np.ndarray] = None,
    base_cov_eval: Optional[np.ndarray] = None,
    alpha_start: float = 0.95,
    alpha_end: float = 0.00,
    alpha_step: float = 0.01,
    epsilon: float = 5e-6,
    max_iterations: Optional[int] = None,
    time_limit_sec: Optional[float] = None,
    target_coverage_pct: Optional[float] = None,
    seed: int = 42,
    sparse: Optional[SparseCoverageStructures] = None,
) -> HeuristicResult:
    n_alphas = int(round((alpha_start - alpha_end) / alpha_step)) + 1
    alpha_schedule = np.array([max(0.0, round(alpha_start - k * alpha_step, 4)) for k in range(n_alphas)])

    if max_iterations is None:
        max_iterations = len(alpha_schedule)

    n_demands, n_candidates, base_tr, base_ev = validate_solver_inputs(
        c_matrix_train, n_to_open, base_cov_train, c_matrix_eval, base_cov_eval, max_iterations
    )

    if sparse is None:
        sparse = build_sparse_structures(c_matrix_train)

    best_obj = -np.inf
    best_selected = np.empty(0, dtype=np.int32)
    best_y = np.zeros(n_candidates, dtype=np.uint8)
    history: List[float] = []
    time_to_best = 0.0

    t0 = time.perf_counter()
    iters_done = 0

    for it in range(max_iterations):
        if time_limit_sec is not None and (time.perf_counter() - t0) >= time_limit_sec and it > 0:
            break

        alpha_k = float(alpha_schedule[it % len(alpha_schedule)])
        iter_seed = seed + it * 1009

        sel, y_act, _ = _construct_solution_numba(
            n_demands=n_demands,
            n_candidates=n_candidates,
            col_ptr=sparse.col_ptr,
            row_idx=sparse.row_idx,
            cov_vals=sparse.cov_vals,
            row_ptr=sparse.row_ptr,
            col_idx=sparse.col_idx,
            row_vals=sparse.row_vals,
            base_cov=base_tr,
            n_to_open=n_to_open,
            alpha=alpha_k,
            deterministic=False,
            rng_seed=iter_seed,
        )

        sel, y_act, phi_after_ls = _local_search_1swap_numba(
            n_demands=n_demands,
            n_candidates=n_candidates,
            col_ptr=sparse.col_ptr,
            row_idx=sparse.row_idx,
            cov_vals=sparse.cov_vals,
            row_ptr=sparse.row_ptr,
            col_idx=sparse.col_idx,
            row_vals=sparse.row_vals,
            base_cov=base_tr,
            selected=sel,
            y_active=y_act,
            epsilon=epsilon,
        )

        obj_val = 100.0 * float(np.mean(phi_after_ls))
        if obj_val > best_obj + 1e-12:
            best_obj = obj_val
            best_selected = sel.copy()
            best_y = y_act.copy()
            time_to_best = time.perf_counter() - t0

        history.append(best_obj)
        iters_done += 1

        if target_coverage_pct is not None and best_obj >= target_coverage_pct - 1e-9:
            break

    cpu_time = time.perf_counter() - t0
    eval_cov_pct = evaluate_solution(best_selected, c_matrix_eval, base_ev) if c_matrix_eval is not None else best_obj

    return HeuristicResult(
        method_name="GRASP Clássico (Alg. 2)",
        selected_candidates=best_selected.astype(int),
        y_vector=best_y.astype(int),
        train_coverage_pct=best_obj,
        eval_coverage_pct=eval_cov_pct,
        cpu_time_sec=cpu_time,
        time_to_best_sec=time_to_best,
        iterations_completed=iters_done,
        iteration_history=history,
    )
