import numpy as np

MODE_WEIGHTS = {
    "foot": 0.22,
    "bicycle": 0.33,
    "car": 0.45,
}

DISTANCE_MULTIPLIERS = {
    "foot": 1.383,
    "bicycle": 1.519,
    "car": 1.961,
}

TRAVEL_SPEEDS_KMH = {
    "foot": 8.0,
    "bicycle": 16.9,
    "car": 16.4,
}

CUTOFF_DISTANCES_M = {
    "foot": 310.0,
    "bicycle": 710.0,
    "car": 470.0,
}

MAX_CUTOFF_RADIUS_M = max(CUTOFF_DISTANCES_M.values())

EXPONENTIAL_BETAS = {
    "foot": 0.02,
    "bicycle": 0.0085,
    "car": 0.0012,
}

def coverage_piecewise_linear(distances: np.ndarray) -> np.ndarray:
    d = np.asarray(distances, dtype=np.float64)
    cov = np.zeros_like(d, dtype=np.float64)
    for mode, w_t in MODE_WEIGHTS.items():
        r_t = CUTOFF_DISTANCES_M[mode]
        cov += w_t * np.maximum(1.0 - d / r_t, 0.0)
    return cov

def coverage_binary(distances: np.ndarray, cutoff_m: float = 310.0) -> np.ndarray:
    d = np.asarray(distances, dtype=np.float64)
    return (d <= cutoff_m).astype(np.float64)

def coverage_exponential(distances: np.ndarray) -> np.ndarray:
    d = np.asarray(distances, dtype=np.float64)
    cov = np.zeros_like(d, dtype=np.float64)
    for mode, w_t in MODE_WEIGHTS.items():
        beta_t = EXPONENTIAL_BETAS[mode]
        cov += w_t * np.exp(-beta_t * d)
    return cov

def coverage_sigmoid(distances: np.ndarray) -> np.ndarray:
    d = np.asarray(distances, dtype=np.float64)
    cov = np.zeros_like(d, dtype=np.float64)
    for mode, w_t in MODE_WEIGHTS.items():
        r_t = CUTOFF_DISTANCES_M[mode]
        exponent = np.clip(12.0 * (d / r_t) - 6.0, -500.0, 500.0)
        cov += w_t / (1.0 + np.exp(exponent))
    return cov
