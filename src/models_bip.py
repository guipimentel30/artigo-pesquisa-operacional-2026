import time
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

from .sparse_utils import validate_solver_inputs

@dataclass
class BIPResult:
    selected_candidates: np.ndarray
    y_vector: np.ndarray
    train_coverage_pct: float
    cpu_time_sec: float
    optimality_gap_pct: float
    status_message: str

def solve_bip_exact(
    c_matrix: np.ndarray,
    n_to_open: int,
    base_coverage: Optional[np.ndarray] = None,
    time_limit_sec: float = 300.0,
    mip_rel_gap: float = 1e-4,
) -> BIPResult:
    t0 = time.perf_counter()
    n_demands, n_candidates, base_cov, _ = validate_solver_inputs(
        c_matrix_train=c_matrix,
        n_to_open=n_to_open,
        base_cov_train=base_coverage,
    )

    eff_gain = np.maximum(c_matrix - base_cov[:, None], 0.0)
    rows_i, cols_j = np.nonzero(eff_gain > 0.0)
    n_x_vars = len(rows_i)

    total_vars = n_candidates + n_x_vars

    obj_coeffs = np.zeros(total_vars, dtype=np.float64)
    obj_coeffs[n_candidates:] = -eff_gain[rows_i, cols_j]

    n_constraints = 1 + n_x_vars + n_demands
    ub = np.zeros(n_constraints, dtype=np.float64)
    lb = np.full(n_constraints, -np.inf, dtype=np.float64)
    ub[0] = float(n_to_open)
    ub[1 + n_x_vars:] = 1.0

    r_1b = np.zeros(n_candidates, dtype=np.int32)
    c_1b = np.arange(n_candidates, dtype=np.int32)
    d_1b = np.ones(n_candidates, dtype=np.float64)

    k_idx = np.arange(n_x_vars, dtype=np.int32)
    r_1d_y = 1 + k_idx
    c_1d_y = cols_j.astype(np.int32)
    d_1d_y = np.full(n_x_vars, -1.0, dtype=np.float64)

    r_1d_x = 1 + k_idx
    c_1d_x = n_candidates + k_idx
    d_1d_x = np.ones(n_x_vars, dtype=np.float64)

    r_1e = (1 + n_x_vars) + rows_i.astype(np.int32)
    c_1e = n_candidates + k_idx
    d_1e = np.ones(n_x_vars, dtype=np.float64)

    all_rows = np.concatenate([r_1b, r_1d_y, r_1d_x, r_1e])
    all_cols = np.concatenate([c_1b, c_1d_y, c_1d_x, c_1e])
    all_vals = np.concatenate([d_1b, d_1d_y, d_1d_x, d_1e])
    A_csc = coo_matrix((all_vals, (all_rows, all_cols)), shape=(n_constraints, total_vars)).tocsc()
    constraints = LinearConstraint(A_csc, lb, ub)
    integrality = np.zeros(total_vars, dtype=int)
    integrality[:n_candidates] = 1
    bounds = Bounds(np.zeros(total_vars), np.ones(total_vars))

    res = milp(
        c=obj_coeffs,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={"time_limit": time_limit_sec, "mip_rel_gap": mip_rel_gap},
    )

    cpu_time = time.perf_counter() - t0

    if res.x is None:
        return BIPResult(
            selected_candidates=np.array([], dtype=int),
            y_vector=np.zeros(n_candidates, dtype=int),
            train_coverage_pct=0.0,
            cpu_time_sec=cpu_time,
            optimality_gap_pct=100.0,
            status_message=res.message,
        )

    y_sol = np.rint(res.x[:n_candidates]).astype(int)
    selected = np.flatnonzero(y_sol == 1)

    if len(selected) > 0:
        best_cand_cov = np.max(c_matrix[:, selected], axis=1)
        total_cov = np.maximum(base_cov, best_cand_cov)
    else:
        total_cov = base_cov.copy()

    train_cov_pct = 100.0 * float(np.mean(total_cov))
    gap_pct = float(getattr(res, "mip_gap", 0.0) or 0.0) * 100.0

    return BIPResult(
        selected_candidates=selected,
        y_vector=y_sol,
        train_coverage_pct=train_cov_pct,
        cpu_time_sec=cpu_time,
        optimality_gap_pct=gap_pct,
        status_message=res.message,
    )
