from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial import cKDTree

from .coverage import (
    coverage_piecewise_linear,
    coverage_binary,
    CUTOFF_DISTANCES_M,
    MAX_CUTOFF_RADIUS_M,
)

MUNICIPALITIES_TABLE3: Dict[str, Dict[str, float]] = {
    "Aalsmeer":       {"existing_aeds": 3,   "ohcas": 99,  "ratio": 33.0, "candidates": 5600},
    "Alkmaar":        {"existing_aeds": 118, "ohcas": 392, "ratio": 3.3,  "candidates": 14452},
    "Beemster":       {"existing_aeds": 28,  "ohcas": 34,  "ratio": 1.2,  "candidates": 7793},
    "Bergen":         {"existing_aeds": 42,  "ohcas": 171, "ratio": 4.1,  "candidates": 10010},
    "Castricum":      {"existing_aeds": 49,  "ohcas": 150, "ratio": 3.1,  "candidates": 6328},
    "Den Helder":     {"existing_aeds": 50,  "ohcas": 253, "ratio": 5.1,  "candidates": 6351},
    "Diemen":         {"existing_aeds": 2,   "ohcas": 100, "ratio": 50.0, "candidates": 2183},
    "Drechterland":   {"existing_aeds": 26,  "ohcas": 55,  "ratio": 2.1,  "candidates": 8047},
    "Edam-Volendam":  {"existing_aeds": 84,  "ohcas": 146, "ratio": 1.7,  "candidates": 7819},
    "Enkhuizen":      {"existing_aeds": 24,  "ohcas": 76,  "ratio": 3.2,  "candidates": 2381},
    "Heerhugowaard":  {"existing_aeds": 39,  "ohcas": 181, "ratio": 4.6,  "candidates": 5525},
    "Heiloo":         {"existing_aeds": 16,  "ohcas": 97,  "ratio": 6.1,  "candidates": 4103},
    "Hollands Kroon": {"existing_aeds": 80,  "ohcas": 199, "ratio": 2.5,  "candidates": 30387},
    "Hoorn":          {"existing_aeds": 84,  "ohcas": 222, "ratio": 2.6,  "candidates": 4595},
    "Koggenland":     {"existing_aeds": 56,  "ohcas": 77,  "ratio": 1.4,  "candidates": 9680},
    "Landsmeer":      {"existing_aeds": 5,   "ohcas": 32,  "ratio": 6.4,  "candidates": 3293},
    "Langedijk":      {"existing_aeds": 23,  "ohcas": 65,  "ratio": 2.8,  "candidates": 5915},
    "Medemblik":      {"existing_aeds": 80,  "ohcas": 167, "ratio": 2.1,  "candidates": 16188},
    "Oostzaan":       {"existing_aeds": 1,   "ohcas": 40,  "ratio": 40.0, "candidates": 2195},
    "Opmeer":         {"existing_aeds": 26,  "ohcas": 49,  "ratio": 1.9,  "candidates": 6582},
    "Ouder-Amstel":   {"existing_aeds": 2,   "ohcas": 62,  "ratio": 31.0, "candidates": 5211},
    "Purmerend":      {"existing_aeds": 47,  "ohcas": 312, "ratio": 6.6,  "candidates": 4237},
    "Schagen":        {"existing_aeds": 91,  "ohcas": 209, "ratio": 2.3,  "candidates": 21602},
    "Stede Broec":    {"existing_aeds": 16,  "ohcas": 62,  "ratio": 3.9,  "candidates": 2276},
    "Texel":          {"existing_aeds": 65,  "ohcas": 108, "ratio": 1.7,  "candidates": 16603},
    "Uithoorn":       {"existing_aeds": 7,   "ohcas": 105, "ratio": 15.0, "candidates": 3601},
    "Waterland":      {"existing_aeds": 35,  "ohcas": 76,  "ratio": 2.2,  "candidates": 8571},
    "Wormerland":     {"existing_aeds": 6,   "ohcas": 74,  "ratio": 12.3, "candidates": 5749},
    "Zaanstad":       {"existing_aeds": 44,  "ohcas": 616, "ratio": 14.0, "candidates": 11989},
}

@dataclass
class MunicipalityInstance:
    name: str
    num_existing_aeds: int
    historic_ohcas_count: int
    historic_ohcas_coords: np.ndarray
    existing_aeds_coords: np.ndarray
    candidate_coords: np.ndarray
    train_demands_coords: np.ndarray
    eval_demands_coords: np.ndarray
    base_cov_train: np.ndarray
    base_cov_eval: np.ndarray

def _estimate_botev_like_bandwidth_factor(points: np.ndarray) -> float:
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=min(6, len(points)))
    local_scale = float(np.median(dists[:, -1]))
    global_scale = float(np.mean(np.std(points, axis=0)))
    if global_scale <= 1e-9:
        return 0.2
    target_bw_m = np.clip(local_scale * 0.85, 160.0, 320.0)
    return float(np.clip(target_bw_m / global_scale, 0.04, 0.35))

def generate_municipality_instance(
    municipality_name: str = "Zaanstad",
    n_train_demands: int = 2000,
    n_eval_demands: int = 5000,
    grid_spacing_m: float = 100.0,
    max_candidates: Optional[int] = None,
    seed: int = 42,
) -> MunicipalityInstance:
    if municipality_name not in MUNICIPALITIES_TABLE3:
        raise ValueError(f"Município '{municipality_name}' não encontrado na Tabela 3.")
    if n_train_demands < 1 or n_eval_demands < 1:
        raise ValueError(
            f"n_train_demands ({n_train_demands}) e n_eval_demands ({n_eval_demands}) devem ser >= 1."
        )
    if grid_spacing_m <= 0.0:
        raise ValueError(f"grid_spacing_m deve ser > 0, recebido {grid_spacing_m}.")
    if max_candidates is not None and max_candidates < 1:
        raise ValueError(f"max_candidates deve ser >= 1 ou None, recebido {max_candidates}.")

    info = MUNICIPALITIES_TABLE3[municipality_name]
    mu_m = int(info["ohcas"])
    num_existing = int(info["existing_aeds"])
    real_candidates = int(info["candidates"])

    rng = np.random.default_rng(seed)

    area_m2 = real_candidates * (grid_spacing_m ** 2) * 1.35
    side_m = float(np.sqrt(area_m2))

    n_hotspots = max(3, min(10, 2 + mu_m // 50))
    hotspot_centers = rng.uniform(0.18 * side_m, 0.82 * side_m, size=(n_hotspots, 2))
    hotspot_weights = rng.dirichlet(np.full(n_hotspots, 1.5))
    hotspot_stds = rng.uniform(250.0, 550.0, size=n_hotspots)

    chosen_hotspots = rng.choice(n_hotspots, size=mu_m, p=hotspot_weights)
    historic_ohcas = np.zeros((mu_m, 2), dtype=np.float64)
    for idx in range(mu_m):
        h = chosen_hotspots[idx]
        historic_ohcas[idx] = rng.normal(loc=hotspot_centers[h], scale=hotspot_stds[h], size=2)
    historic_ohcas = np.clip(historic_ohcas, 0.0, side_m)

    bw_factor = _estimate_botev_like_bandwidth_factor(historic_ohcas)
    kde = gaussian_kde(historic_ohcas.T, bw_method=bw_factor)

    train_demands = kde.resample(size=n_train_demands, seed=int(rng.integers(1, 1_000_000))).T
    eval_demands = kde.resample(size=n_eval_demands, seed=int(rng.integers(1, 1_000_000))).T

    min_x = float(np.floor(min(historic_ohcas[:, 0].min(), train_demands[:, 0].min()) / grid_spacing_m) * grid_spacing_m)
    max_x = float(np.ceil(max(historic_ohcas[:, 0].max(), train_demands[:, 0].max()) / grid_spacing_m) * grid_spacing_m)
    min_y = float(np.floor(min(historic_ohcas[:, 1].min(), train_demands[:, 1].min()) / grid_spacing_m) * grid_spacing_m)
    max_y = float(np.ceil(max(historic_ohcas[:, 1].max(), train_demands[:, 1].max()) / grid_spacing_m) * grid_spacing_m)

    xs = np.arange(min_x, max_x + grid_spacing_m, grid_spacing_m)
    ys = np.arange(min_y, max_y + grid_spacing_m, grid_spacing_m)
    gx, gy = np.meshgrid(xs, ys)
    raw_grid = np.column_stack([gx.ravel(), gy.ravel()])

    tree_hist = cKDTree(historic_ohcas)
    dist_to_nearest_hist, _ = tree_hist.query(raw_grid, k=1)
    valid_mask = dist_to_nearest_hist <= MAX_CUTOFF_RADIUS_M
    candidate_coords = raw_grid[valid_mask]

    if max_candidates is not None and len(candidate_coords) > max_candidates:
        valid_dists = dist_to_nearest_hist[valid_mask]
        top_idx = np.argpartition(valid_dists, max_candidates - 1)[:max_candidates]
        candidate_coords = candidate_coords[np.sort(top_idx)]

    aed_cluster_idx = rng.choice(n_hotspots, size=num_existing)
    existing_aeds = np.zeros((num_existing, 2), dtype=np.float64)
    for idx in range(num_existing):
        c_center = hotspot_centers[aed_cluster_idx[idx]]
        existing_aeds[idx] = rng.normal(loc=c_center, scale=750.0, size=2)
    existing_aeds = np.clip(existing_aeds, 0.0, side_m)

    base_cov_train = compute_coverage_from_coords(train_demands, existing_aeds)
    base_cov_eval = compute_coverage_from_coords(eval_demands, existing_aeds)

    return MunicipalityInstance(
        name=municipality_name,
        num_existing_aeds=num_existing,
        historic_ohcas_count=mu_m,
        historic_ohcas_coords=historic_ohcas,
        existing_aeds_coords=existing_aeds,
        candidate_coords=candidate_coords,
        train_demands_coords=train_demands,
        eval_demands_coords=eval_demands,
        base_cov_train=base_cov_train,
        base_cov_eval=base_cov_eval,
    )

def _resolve_compact_cutoff(coverage_fn, cutoff_m: Optional[float]) -> Optional[float]:
    if cutoff_m is not None:
        return cutoff_m
    if coverage_fn is coverage_piecewise_linear:
        return MAX_CUTOFF_RADIUS_M
    if coverage_fn is coverage_binary:
        return CUTOFF_DISTANCES_M["foot"]
    return None

def compute_coverage_from_coords(
    demand_coords: np.ndarray,
    facility_coords: np.ndarray,
    coverage_fn=coverage_piecewise_linear,
    cutoff_m: Optional[float] = None,
) -> np.ndarray:
    n_demands = len(demand_coords)
    if len(facility_coords) == 0:
        return np.zeros(n_demands, dtype=np.float64)

    resolved_cutoff = _resolve_compact_cutoff(coverage_fn, cutoff_m)
    tree = cKDTree(facility_coords)
    if resolved_cutoff is not None:
        min_dists, _ = tree.query(demand_coords, k=1, distance_upper_bound=resolved_cutoff)
        cov = np.zeros(n_demands, dtype=np.float64)
        reachable = np.isfinite(min_dists) & (min_dists <= resolved_cutoff)
        if np.any(reachable):
            cov[reachable] = coverage_fn(min_dists[reachable])
        return cov

    min_dists, _ = tree.query(demand_coords, k=1)
    return coverage_fn(min_dists)

def build_coverage_matrix(
    demand_coords: np.ndarray,
    candidate_coords: np.ndarray,
    coverage_fn=coverage_piecewise_linear,
    cutoff_m: Optional[float] = None,
) -> np.ndarray:
    diff_x = demand_coords[:, 0:1] - candidate_coords[:, 0]
    diff_y = demand_coords[:, 1:2] - candidate_coords[:, 1]
    dists = np.hypot(diff_x, diff_y)

    resolved_cutoff = _resolve_compact_cutoff(coverage_fn, cutoff_m)
    if resolved_cutoff is not None:
        c_matrix = np.zeros_like(dists, dtype=np.float64)
        mask = dists <= resolved_cutoff
        c_matrix[mask] = coverage_fn(dists[mask])
        return c_matrix

    return coverage_fn(dists)
