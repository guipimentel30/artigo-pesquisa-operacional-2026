import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from numba import njit

from .sparse_utils import (
    HeuristicResult,
    SparseCoverageStructures,
    build_sparse_structures,
    validate_solver_inputs,
    _construct_solution_numba,
    _local_search_1swap_numba,
    evaluate_solution,
)

DEFAULT_ALPHA_POOL = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95)

class ReactiveAlphaController:

    def __init__(
        self,
        alpha_pool: Sequence[float],
        block_size: int,
        gamma: float,
        min_prob: float,
        rng: np.random.Generator,
    ) -> None:
        self.alphas = np.array(alpha_pool, dtype=np.float64)
        self.m = len(self.alphas)
        if self.m == 0:
            raise ValueError("alpha_pool não pode ser vazio.")
        if block_size < 1:
            raise ValueError(f"block_size deve ser >= 1, recebido {block_size}.")
        if min_prob < 0.0:
            raise ValueError(f"min_prob deve ser >= 0, recebido {min_prob}.")

        self.block_size = block_size
        self.gamma = gamma
        self.min_prob = min(float(min_prob), 0.5 / self.m)
        self.rng = rng

        self.probs = np.full(self.m, 1.0 / self.m, dtype=np.float64)
        self.sum_scores = np.zeros(self.m, dtype=np.float64)
        self.counts = np.zeros(self.m, dtype=np.int32)
        self.best_seen = -np.inf
        self.worst_seen = np.inf

    def select_alpha(self, iteration: int) -> Tuple[int, float]:
        if iteration < self.m:
            k_idx = iteration
        else:
            k_idx = int(self.rng.choice(self.m, p=self.probs))
        return k_idx, float(self.alphas[k_idx])

    def record_and_maybe_update(self, k_idx: int, obj_val: float, iteration: int) -> None:
        self.sum_scores[k_idx] += obj_val
        self.counts[k_idx] += 1
        if obj_val > self.best_seen:
            self.best_seen = obj_val
        if obj_val < self.worst_seen:
            self.worst_seen = obj_val

        if (iteration + 1) >= self.m and ((iteration + 1) % self.block_size == 0):
            avg_scores = np.where(
                self.counts > 0,
                self.sum_scores / np.maximum(self.counts, 1),
                self.best_seen,
            )
            denom = (self.best_seen - self.worst_seen) + 1e-4
            q = np.power(np.maximum((avg_scores - self.worst_seen + 1e-4) / denom, 0.0), self.gamma)
            q_sum = float(np.sum(q))
            if q_sum > 0.0:
                q_norm = q / q_sum
                self.probs = self.min_prob + (1.0 - self.m * self.min_prob) * q_norm
            else:
                self.probs = np.full(self.m, 1.0 / self.m, dtype=np.float64)

    def export_dicts(self) -> Tuple[Dict[float, float], Dict[float, int], Dict[float, float]]:
        probs_dict = {float(self.alphas[i]): float(self.probs[i]) for i in range(self.m)}
        counts_dict = {float(self.alphas[i]): int(self.counts[i]) for i in range(self.m)}
        avgs_dict = {
            float(self.alphas[i]): (
                float(self.sum_scores[i] / self.counts[i]) if self.counts[i] > 0 else 0.0
            )
            for i in range(self.m)
        }
        return probs_dict, counts_dict, avgs_dict

@njit(cache=True)
def _shake_k_swap_numba(
    n_candidates: int,
    selected: np.ndarray,
    y_active: np.ndarray,
    k: int,
):
    n_open = len(selected)
    n_inactive = n_candidates - n_open
    k_eff = min(k, min(n_open, n_inactive))

    new_sel = selected.copy()
    new_y = y_active.copy()
    if k_eff <= 0:
        return new_sel, new_y

    pos_pool = np.arange(n_open, dtype=np.int32)
    for t in range(k_eff):
        pick = t + np.random.randint(0, n_open - t)
        tmp = pos_pool[t]
        pos_pool[t] = pos_pool[pick]
        pos_pool[pick] = tmp

    inactive_pool = np.empty(n_inactive, dtype=np.int32)
    idx_in = 0
    for j in range(n_candidates):
        if y_active[j] == 0:
            inactive_pool[idx_in] = j
            idx_in += 1

    for t in range(k_eff):
        pick = t + np.random.randint(0, n_inactive - t)
        tmp = inactive_pool[t]
        inactive_pool[t] = inactive_pool[pick]
        inactive_pool[pick] = tmp

    for t in range(k_eff):
        pos = pos_pool[t]
        j_out = new_sel[pos]
        j_in = inactive_pool[t]
        new_y[j_out] = 0
        new_y[j_in] = 1
        new_sel[pos] = j_in

    return new_sel, new_y

@njit(cache=True)
def _basic_vns_numba(
    n_demands: int,
    n_candidates: int,
    col_ptr: np.ndarray,
    row_idx: np.ndarray,
    cov_vals: np.ndarray,
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    row_vals: np.ndarray,
    base_cov: np.ndarray,
    selected_init: np.ndarray,
    y_active_init: np.ndarray,
    epsilon: float,
    r_max: int,
    rng_seed: int,
):
    np.random.seed(rng_seed)

    sel_curr = selected_init.copy()
    y_curr = y_active_init.copy()
    sel_curr, y_curr, phi_curr = _local_search_1swap_numba(
        n_demands, n_candidates, col_ptr, row_idx, cov_vals,
        row_ptr, col_idx, row_vals, base_cov, sel_curr, y_curr, epsilon,
    )
    ls_calls = 1

    total_curr = 0.0
    for i in range(n_demands):
        total_curr += phi_curr[i]

    shakes_by_k = np.zeros(r_max, dtype=np.int32)
    improv_by_k = np.zeros(r_max, dtype=np.int32)

    n_open = len(sel_curr)
    n_inactive = n_candidates - n_open
    r_eff = min(r_max, min(n_open, n_inactive))

    k = 1
    while k <= r_eff:
        sel_p, y_p = _shake_k_swap_numba(n_candidates, sel_curr, y_curr, k)
        shakes_by_k[k - 1] += 1

        sel_pp, y_pp, phi_pp = _local_search_1swap_numba(
            n_demands, n_candidates, col_ptr, row_idx, cov_vals,
            row_ptr, col_idx, row_vals, base_cov, sel_p, y_p, epsilon,
        )
        ls_calls += 1

        total_pp = 0.0
        for i in range(n_demands):
            total_pp += phi_pp[i]

        if total_pp > total_curr + 1e-10:
            sel_curr = sel_pp
            y_curr = y_pp
            phi_curr = phi_pp
            total_curr = total_pp
            improv_by_k[k - 1] += 1
            k = 1
        else:
            k += 1

    return sel_curr, y_curr, phi_curr, ls_calls, shakes_by_k, improv_by_k

def run_basic_vns(
    sparse: SparseCoverageStructures,
    base_cov: np.ndarray,
    selected: np.ndarray,
    y_active: np.ndarray,
    epsilon: float = 5e-6,
    r_max: int = 3,
    rng_seed: int = 0,
):
    sel_out, y_out, phi_out, ls_calls, shakes_k, improv_k = _basic_vns_numba(
        n_demands=sparse.n_demands,
        n_candidates=sparse.n_candidates,
        col_ptr=sparse.col_ptr,
        row_idx=sparse.row_idx,
        cov_vals=sparse.cov_vals,
        row_ptr=sparse.row_ptr,
        col_idx=sparse.col_idx,
        row_vals=sparse.row_vals,
        base_cov=base_cov,
        selected_init=selected,
        y_active_init=y_active,
        epsilon=epsilon,
        r_max=r_max,
        rng_seed=rng_seed,
    )
    return sel_out, y_out, phi_out, int(ls_calls), shakes_k, improv_k

def accumulate_vns_stats(
    totals: Dict[str, Any],
    ls_calls: int,
    shakes_k: np.ndarray,
    improv_k: np.ndarray,
) -> None:
    totals["ls_calls"] += ls_calls
    for k_idx in range(len(shakes_k)):
        k = k_idx + 1
        totals["shakes_by_k"][f"k{k}"] += int(shakes_k[k_idx])
        totals["improv_by_k"][f"k{k}"] += int(improv_k[k_idx])

def init_vns_stats(r_max: int) -> Dict[str, Any]:
    return {
        "ls_calls": 0,
        "shakes_by_k": {f"k{k}": 0 for k in range(1, r_max + 1)},
        "improv_by_k": {f"k{k}": 0 for k in range(1, r_max + 1)},
    }

def execute_reactive_vns_iteration(
    sparse: SparseCoverageStructures,
    base_cov: np.ndarray,
    n_to_open: int,
    alpha_ctrl: ReactiveAlphaController,
    vns_totals: Dict[str, Any],
    epsilon: float,
    r_max: int,
    seed: int,
    iteration: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    k_idx, alpha_k = alpha_ctrl.select_alpha(iteration)
    iter_seed = seed + iteration * 1009
    vns_seed = seed + iteration * 1009 + 503

    sel, y_act, _ = _construct_solution_numba(
        n_demands=sparse.n_demands,
        n_candidates=sparse.n_candidates,
        col_ptr=sparse.col_ptr,
        row_idx=sparse.row_idx,
        cov_vals=sparse.cov_vals,
        row_ptr=sparse.row_ptr,
        col_idx=sparse.col_idx,
        row_vals=sparse.row_vals,
        base_cov=base_cov,
        n_to_open=n_to_open,
        alpha=alpha_k,
        deterministic=False,
        rng_seed=iter_seed,
    )

    sel, y_act, phi_after_vns, ls_calls, shakes_k, improv_k = run_basic_vns(
        sparse=sparse,
        base_cov=base_cov,
        selected=sel,
        y_active=y_act,
        epsilon=epsilon,
        r_max=r_max,
        rng_seed=vns_seed,
    )
    accumulate_vns_stats(vns_totals, ls_calls, shakes_k, improv_k)

    obj_val = 100.0 * float(np.mean(phi_after_vns))
    alpha_ctrl.record_and_maybe_update(k_idx, obj_val, iteration)
    return sel, y_act, obj_val

def run_grasp_reativo_vns(
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

    if sparse is None:
        sparse = build_sparse_structures(c_matrix_train)

    rng = np.random.default_rng(seed)
    alpha_ctrl = ReactiveAlphaController(
        alpha_pool=alpha_pool,
        block_size=block_size,
        gamma=gamma,
        min_prob=min_prob,
        rng=rng,
    )

    best_obj = -np.inf
    best_selected = np.empty(0, dtype=np.int32)
    best_y = np.zeros(n_candidates, dtype=np.uint8)
    history: List[float] = []
    time_to_best = 0.0
    vns_totals = init_vns_stats(r_max)

    t0 = time.perf_counter()
    iters_done = 0

    for it in range(max_iterations):
        if time_limit_sec is not None and (time.perf_counter() - t0) >= time_limit_sec and it > 0:
            break

        sel, y_act, obj_val = execute_reactive_vns_iteration(
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
    eval_cov_pct = (
        evaluate_solution(best_selected, c_matrix_eval, base_ev)
        if c_matrix_eval is not None
        else best_obj
    )

    alpha_probs_dict, alpha_counts_dict, alpha_avgs_dict = alpha_ctrl.export_dicts()

    return HeuristicResult(
        method_name="GRASP Reativo + VNS",
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
    )
