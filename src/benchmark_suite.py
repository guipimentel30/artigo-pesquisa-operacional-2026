import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from .data_generator import (
    MunicipalityInstance,
    generate_municipality_instance,
    build_coverage_matrix,
)
from .sparse_utils import SparseCoverageStructures, build_sparse_structures

RESULTS_DIR = Path(__file__).resolve().parent.parent / "resultados"

EXP1_MUNICIPALITY = "Heiloo"
EXP1_DEMAND_SIZES = [250, 500, 1000]
EXP1_MAX_CANDIDATES = 500
EXP1_EVAL_DEMANDS = 5000
EXP1_SEED = 42

SHARED_MUNICIPALITIES = ["Aalsmeer", "Uithoorn", "Heiloo", "Enkhuizen", "Zaanstad"]
EXP3_MUNICIPALITIES = ["Aalsmeer", "Uithoorn", "Zaanstad"]
EXP3_ADD_COUNTS = [5, 10, 20, 40]
SHARED_TRAIN_DEMANDS = 1500
SHARED_EVAL_DEMANDS = 5000
SHARED_MAX_CANDIDATES = 1000
SHARED_BASE_SEED = 100

@dataclass
class PreparedInstance:
    instance: MunicipalityInstance
    c_train: np.ndarray
    c_eval: np.ndarray
    sparse: SparseCoverageStructures
    seed: int

def build_exp1_instances() -> List[PreparedInstance]:
    prepared = []
    for n_demands in EXP1_DEMAND_SIZES:
        inst = generate_municipality_instance(
            municipality_name=EXP1_MUNICIPALITY,
            n_train_demands=n_demands,
            n_eval_demands=EXP1_EVAL_DEMANDS,
            max_candidates=EXP1_MAX_CANDIDATES,
            seed=EXP1_SEED,
        )
        c_train = build_coverage_matrix(inst.train_demands_coords, inst.candidate_coords)
        c_eval = build_coverage_matrix(inst.eval_demands_coords, inst.candidate_coords)
        sparse = build_sparse_structures(c_train)
        prepared.append(PreparedInstance(inst, c_train, c_eval, sparse, EXP1_SEED))
    return prepared

def build_shared_instances() -> Dict[str, PreparedInstance]:
    shared: Dict[str, PreparedInstance] = {}
    for idx, mun_name in enumerate(SHARED_MUNICIPALITIES):
        seed = SHARED_BASE_SEED + idx
        inst = generate_municipality_instance(
            municipality_name=mun_name,
            n_train_demands=SHARED_TRAIN_DEMANDS,
            n_eval_demands=SHARED_EVAL_DEMANDS,
            max_candidates=SHARED_MAX_CANDIDATES,
            seed=seed,
        )
        c_train = build_coverage_matrix(inst.train_demands_coords, inst.candidate_coords)
        c_eval = build_coverage_matrix(inst.eval_demands_coords, inst.candidate_coords)
        sparse = build_sparse_structures(c_train)
        shared[mun_name] = PreparedInstance(inst, c_train, c_eval, sparse, seed)
    return shared

def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR

def save_audit_json(filename: str, payload: Dict[str, Any]) -> Path:
    out_dir = ensure_results_dir()
    path = out_dir / filename
    enriched = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        **payload,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)
    return path

def save_audit_markdown(filename: str, content: str) -> Path:
    out_dir = ensure_results_dir()
    path = out_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path

def load_audit_json(filename: str) -> Optional[Dict[str, Any]]:
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def format_vns_summary(vns: Dict[str, Any]) -> str:
    imp = vns.get("improv_by_k", {})
    ls = vns.get("ls_calls", 0)
    return f"LS:{ls} | +{imp.get('k1', 0)}/{imp.get('k2', 0)}/{imp.get('k3', 0)}"

def format_pr_summary(prs: Dict[str, Any]) -> str:
    return (
        f"{prs.get('pr_calls', 0)}/"
        f"{prs.get('pr_skipped_distance_lt_2', 0)}/"
        f"+{prs.get('pr_improvements_over_sb', 0)}"
    )

def run_exp3_addition_sweep(
    shared_instances: Dict[str, PreparedInstance],
    solver_fn: Callable[[PreparedInstance, int], Any],
    collect_vns: bool = False,
    collect_pr: bool = False,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for mun_name in EXP3_MUNICIPALITIES:
        prep = shared_instances[mun_name]
        inst = prep.instance

        eval_by_n: Dict[str, float] = {"0": round(100.0 * float(np.mean(inst.base_cov_eval)), 4)}
        train_by_n: Dict[str, float] = {}
        cpu_by_n: Dict[str, float] = {}
        vns_by_n: Dict[str, Dict[str, Any]] = {}
        pr_by_n: Dict[str, Dict[str, Any]] = {}

        for n_add in EXP3_ADD_COUNTS:
            res = solver_fn(prep, n_add)
            eval_by_n[str(n_add)] = round(res.eval_coverage_pct, 4)
            train_by_n[str(n_add)] = round(res.train_coverage_pct, 4)
            cpu_by_n[str(n_add)] = round(res.cpu_time_sec, 4)
            if collect_vns:
                vns_by_n[str(n_add)] = res.vns_stats or {}
            if collect_pr:
                pr_by_n[str(n_add)] = res.pr_stats or {}

        record: Dict[str, Any] = {
            "municipality": mun_name,
            "n_candidates": len(inst.candidate_coords),
            "seed": prep.seed,
            "eval_pct_by_n": eval_by_n,
            "train_pct_by_n": train_by_n,
            "cpu_sec_by_n": cpu_by_n,
        }
        if collect_vns:
            record["vns_stats_by_n"] = vns_by_n
        if collect_pr:
            record["pr_stats_by_n"] = pr_by_n
        records.append(record)
    return records

def resolve_exp1_reference(
    audit_json: Optional[Dict[str, Any]],
    idx: int,
    method_key: str,
    fallback_fn: Callable[[], Any],
) -> Tuple[float, float, float]:
    entries = (audit_json or {}).get("experiment_1_table4", [])
    if idx < len(entries) and method_key in entries[idx]:
        entry = entries[idx][method_key]
        return float(entry["train_pct"]), float(entry["eval_pct"]), float(entry["cpu_sec"])
    res = fallback_fn()
    return res.train_coverage_pct, res.eval_coverage_pct, res.cpu_time_sec

def resolve_exp2_reference(
    audit_map: Dict[str, Dict[str, Any]],
    mun_name: str,
    nested_key: Optional[str],
    eval_field: str,
    fallback_fn: Callable[[], Any],
) -> Tuple[float, float, float]:
    if mun_name in audit_map:
        entry = audit_map[mun_name] if nested_key is None else audit_map[mun_name][nested_key]
        return float(entry["train_pct"]), float(entry[eval_field]), float(entry["cpu_sec"])
    res = fallback_fn()
    return res.train_coverage_pct, res.eval_coverage_pct, res.cpu_time_sec
