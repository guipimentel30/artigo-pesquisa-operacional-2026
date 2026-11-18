from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
from numba import njit

@dataclass
class HeuristicResult:
    method_name: str
    selected_candidates: np.ndarray
    y_vector: np.ndarray
    train_coverage_pct: float
    eval_coverage_pct: float
    cpu_time_sec: float
    iterations_completed: int = 1
    iteration_history: List[float] = field(default_factory=list)
    time_to_best_sec: float = 0.0
    alpha_probabilities: Optional[Dict[float, float]] = None
    alpha_counts: Optional[Dict[float, int]] = None
    alpha_avg_scores: Optional[Dict[float, float]] = None
    vns_stats: Optional[Dict[str, Any]] = None
    pr_stats: Optional[Dict[str, Any]] = None

@dataclass
class SparseCoverageStructures:
    n_demands: int
    n_candidates: int
    col_ptr: np.ndarray
    row_idx: np.ndarray
    cov_vals: np.ndarray
    row_ptr: np.ndarray
    col_idx: np.ndarray
    row_vals: np.ndarray

def validate_solver_inputs(
    c_matrix_train: np.ndarray,
    n_to_open: int,
    base_cov_train: Optional[np.ndarray] = None,
    c_matrix_eval: Optional[np.ndarray] = None,
    base_cov_eval: Optional[np.ndarray] = None,
    max_iterations: int = 1,
) -> tuple:
    if c_matrix_train.ndim != 2:
        raise ValueError(f"c_matrix_train deve ser 2D, recebido ndim={c_matrix_train.ndim}.")
    n_demands, n_candidates = c_matrix_train.shape
    if n_demands <= 0 or n_candidates <= 0:
        raise ValueError(f"c_matrix_train vazia: shape={c_matrix_train.shape}.")
    if n_to_open < 0 or n_to_open > n_candidates:
        raise ValueError(
            f"n_to_open={n_to_open} fora do intervalo válido [0, {n_candidates}]."
        )
    if max_iterations < 1:
        raise ValueError(f"max_iterations deve ser >= 1, recebido {max_iterations}.")

    if base_cov_train is None:
        base_tr = np.zeros(n_demands, dtype=np.float64)
    else:
        base_tr = np.asarray(base_cov_train, dtype=np.float64)
        if base_tr.shape != (n_demands,):
            raise ValueError(f"base_cov_train deve ter shape ({n_demands},), recebido {base_tr.shape}.")

    base_ev = None
    if c_matrix_eval is not None:
        if c_matrix_eval.ndim != 2 or c_matrix_eval.shape[1] != n_candidates:
            raise ValueError(
                f"c_matrix_eval incompatível: esperado (?, {n_candidates}), recebido {c_matrix_eval.shape}."
            )
        n_eval = c_matrix_eval.shape[0]
        if base_cov_eval is not None:
            base_ev = np.asarray(base_cov_eval, dtype=np.float64)
            if base_ev.shape != (n_eval,):
                raise ValueError(f"base_cov_eval deve ter shape ({n_eval},), recebido {base_ev.shape}.")

    return n_demands, n_candidates, base_tr, base_ev

def build_sparse_structures(c_matrix: np.ndarray) -> SparseCoverageStructures:
    n_demands, n_candidates = c_matrix.shape

    col_ptr = np.zeros(n_candidates + 1, dtype=np.int32)
    rows_list = []
    vals_list = []
    for j in range(n_candidates):
        col = c_matrix[:, j]
        nz = np.flatnonzero(col > 0.0).astype(np.int32)
        col_ptr[j + 1] = col_ptr[j] + len(nz)
        if len(nz) > 0:
            rows_list.append(nz)
            vals_list.append(col[nz].astype(np.float64))

    if len(rows_list) > 0:
        row_idx = np.concatenate(rows_list)
        cov_vals = np.concatenate(vals_list)
    else:
        row_idx = np.zeros(0, dtype=np.int32)
        cov_vals = np.zeros(0, dtype=np.float64)

    row_ptr = np.zeros(n_demands + 1, dtype=np.int32)
    cols_list = []
    rvals_list = []
    for i in range(n_demands):
        row = c_matrix[i, :]
        nz = np.flatnonzero(row > 0.0).astype(np.int32)
        row_ptr[i + 1] = row_ptr[i] + len(nz)
        if len(nz) > 0:
            cols_list.append(nz)
            rvals_list.append(row[nz].astype(np.float64))

    if len(cols_list) > 0:
        col_idx = np.concatenate(cols_list)
        row_vals = np.concatenate(rvals_list)
    else:
        col_idx = np.zeros(0, dtype=np.int32)
        row_vals = np.zeros(0, dtype=np.float64)

    return SparseCoverageStructures(
        n_demands=n_demands,
        n_candidates=n_candidates,
        col_ptr=col_ptr,
        row_idx=row_idx,
        cov_vals=cov_vals,
        row_ptr=row_ptr,
        col_idx=col_idx,
        row_vals=row_vals,
    )

@njit(cache=True)
def _compute_phi_numba(
    n_demands: int,
    col_ptr: np.ndarray,
    row_idx: np.ndarray,
    cov_vals: np.ndarray,
    base_cov: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    phi = base_cov.copy()
    for idx in range(len(selected)):
        j = selected[idx]
        for k in range(col_ptr[j], col_ptr[j + 1]):
            i = row_idx[k]
            if cov_vals[k] > phi[i]:
                phi[i] = cov_vals[k]
    return phi

@njit(cache=True)
def _recompute_best1_best2_owner(
    n_demands: int,
    col_ptr: np.ndarray,
    row_idx: np.ndarray,
    cov_vals: np.ndarray,
    base_cov: np.ndarray,
    selected: np.ndarray,
    best1: np.ndarray,
    best2: np.ndarray,
    owner: np.ndarray,
):
    for i in range(n_demands):
        best1[i] = base_cov[i]
        best2[i] = base_cov[i]
        owner[i] = -1

    n_open = len(selected)
    for idx in range(n_open):
        j = selected[idx]
        for k in range(col_ptr[j], col_ptr[j + 1]):
            i = row_idx[k]
            v = cov_vals[k]
            if v > best1[i]:
                best2[i] = best1[i]
                best1[i] = v
                owner[i] = j
            elif v > best2[i]:
                best2[i] = v

@njit(cache=True)
def _construct_solution_numba(
    n_demands: int,
    n_candidates: int,
    col_ptr: np.ndarray,
    row_idx: np.ndarray,
    cov_vals: np.ndarray,
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    row_vals: np.ndarray,
    base_cov: np.ndarray,
    n_to_open: int,
    alpha: float,
    deterministic: bool,
    rng_seed: int,
):
    np.random.seed(rng_seed)

    y_active = np.zeros(n_candidates, dtype=np.uint8)
    selected = np.empty(n_to_open, dtype=np.int32)

    phi = base_cov.copy()

    phi_0 = np.zeros(n_candidates, dtype=np.float64)
    for j in range(n_candidates):
        gain = 0.0
        for k in range(col_ptr[j], col_ptr[j + 1]):
            i = row_idx[k]
            diff = cov_vals[k] - phi[i]
            if diff > 0.0:
                gain += diff
        phi_0[j] = gain

    rcl = np.empty(n_candidates, dtype=np.int32)

    for step in range(n_to_open):
        phi_min = np.inf
        phi_max = -np.inf
        best_j = -1

        for j in range(n_candidates):
            if y_active[j] == 1:
                continue
            g = phi_0[j]
            if g < phi_min:
                phi_min = g
            if g > phi_max:
                phi_max = g
                best_j = j

        if deterministic:
            j_star = best_j
        else:
            threshold = phi_min + alpha * (phi_max - phi_min) - 1e-12
            rcl_count = 0
            for j in range(n_candidates):
                if y_active[j] == 0 and phi_0[j] >= threshold:
                    rcl[rcl_count] = j
                    rcl_count += 1
            if rcl_count == 0:
                j_star = best_j
            else:
                rand_idx = np.random.randint(0, rcl_count)
                j_star = rcl[rand_idx]

        y_active[j_star] = 1
        phi_0[j_star] = 0.0
        selected[step] = j_star

        for k in range(col_ptr[j_star], col_ptr[j_star + 1]):
            i = row_idx[k]
            new_phi = cov_vals[k]
            old_phi = phi[i]
            if new_phi > old_phi:
                phi[i] = new_phi
                for kr in range(row_ptr[i], row_ptr[i + 1]):
                    j = col_idx[kr]
                    if y_active[j] == 0:
                        c_ij = row_vals[kr]
                        if c_ij > old_phi:
                            old_contrib = c_ij - old_phi
                            new_contrib = c_ij - new_phi if c_ij > new_phi else 0.0
                            updated = phi_0[j] - (old_contrib - new_contrib)
                            phi_0[j] = updated if updated > 0.0 else 0.0

    return selected, y_active, phi

@njit(cache=True)
def _local_search_1swap_numba(
    n_demands: int,
    n_candidates: int,
    col_ptr: np.ndarray,
    row_idx: np.ndarray,
    cov_vals: np.ndarray,
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    row_vals: np.ndarray,
    base_cov: np.ndarray,
    selected: np.ndarray,
    y_active: np.ndarray,
    epsilon: float,
):
    n_open = len(selected)
    if n_open == 0 or n_open == n_candidates:
        phi = _compute_phi_numba(n_demands, col_ptr, row_idx, cov_vals, base_cov, selected)
        return selected, y_active, phi

    best1 = np.empty(n_demands, dtype=np.float64)
    best2 = np.empty(n_demands, dtype=np.float64)
    owner = np.empty(n_demands, dtype=np.int32)
    phi_0 = np.zeros(n_candidates, dtype=np.float64)
    extra_gain = np.zeros(n_candidates, dtype=np.float64)
    touched = np.empty(n_candidates, dtype=np.int32)
    is_touched = np.zeros(n_candidates, dtype=np.uint8)

    while True:
        _recompute_best1_best2_owner(
            n_demands, col_ptr, row_idx, cov_vals, base_cov, selected, best1, best2, owner
        )

        max_phi0_1 = -1.0
        max_phi0_j1 = -1
        for j2 in range(n_candidates):
            if y_active[j2] == 1:
                phi_0[j2] = -1.0
                continue
            g = 0.0
            for k in range(col_ptr[j2], col_ptr[j2 + 1]):
                i = row_idx[k]
                diff = cov_vals[k] - best1[i]
                if diff > 0.0:
                    g += diff
            phi_0[j2] = g
            if g > max_phi0_1:
                max_phi0_1 = g
                max_phi0_j1 = j2

        best_delta = 0.0
        best_pos_out = -1
        best_j2_in = -1

        for pos in range(n_open):
            j1 = selected[pos]
            loss_j1 = 0.0
            n_touched = 0

            for k in range(col_ptr[j1], col_ptr[j1 + 1]):
                i = row_idx[k]
                if owner[i] == j1:
                    b1 = best1[i]
                    b2 = best2[i]
                    loss_j1 += (b1 - b2)
                    for kr in range(row_ptr[i], row_ptr[i + 1]):
                        j2 = col_idx[kr]
                        if y_active[j2] == 1:
                            continue
                        v2 = row_vals[kr]
                        if v2 > b2:
                            old_c = v2 - b1 if v2 > b1 else 0.0
                            new_c = v2 - b2
                            if is_touched[j2] == 0:
                                is_touched[j2] = 1
                                touched[n_touched] = j2
                                n_touched += 1
                            extra_gain[j2] += (new_c - old_c)

            if max_phi0_j1 != -1:
                delta_non_overlap = max_phi0_1 - loss_j1
                if delta_non_overlap > best_delta:
                    best_delta = delta_non_overlap
                    best_pos_out = pos
                    best_j2_in = max_phi0_j1

            for t_idx in range(n_touched):
                j2 = touched[t_idx]
                delta = phi_0[j2] + extra_gain[j2] - loss_j1
                if delta > best_delta:
                    best_delta = delta
                    best_pos_out = pos
                    best_j2_in = j2
                extra_gain[j2] = 0.0
                is_touched[j2] = 0

        if (best_delta / n_demands) < epsilon or best_pos_out == -1:
            break

        j1_out = selected[best_pos_out]
        y_active[j1_out] = 0
        y_active[best_j2_in] = 1
        selected[best_pos_out] = best_j2_in

    return selected, y_active, best1

def evaluate_solution(
    selected_candidates: np.ndarray,
    c_matrix_eval: np.ndarray,
    base_cov_eval: Optional[np.ndarray] = None,
) -> float:
    n_eval = c_matrix_eval.shape[0]
    if base_cov_eval is None:
        cov = np.zeros(n_eval, dtype=np.float64)
    else:
        cov = np.asarray(base_cov_eval, dtype=np.float64).copy()

    if len(selected_candidates) > 0:
        best_new = np.max(c_matrix_eval[:, selected_candidates], axis=1)
        cov = np.maximum(cov, best_new)

    return 100.0 * float(np.mean(cov))
