"""
UAF Information Metrics
========================
Closes GAP 6: информационные метрики поверх ODE/SDE.

Метрики
-------
1. Surprisal  — log P(state):  насколько маловероятно текущее состояние
2. Complexity — KL[Q||P]:      сколько информации закодировано относительно prior
3. Mutual Information I(Γ_k; Γ_l) — реальная корреляция между уровнями
4. Integrated Information Φ — мера "целостности" системы (упрощённая)
5. Coherence bandwidth — диапазон частот с высоким Γ
6. Time-horizon estimate — как далеко вперёд система "видит"

Физическая интерпретация
------------------------
  Surprisal  = -log P ~ 1/Γ (низкая когерентность → высокий surprisal)
  Complexity = количество информации, сохранённое в α (integration)
  MI(l,k)    = реальная статистическая зависимость между уровнями,
               должна коррелировать с Φ_{k→l} из UAF core

  Integrated Information Φ:
  Метрика Тонони — мера того, насколько система "больше суммы частей".
  В UAF: система с высоким Φ не разбивается на независимые подсистемы
  → это и есть "матрёшечная интеграция".
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional

EPS = 1e-12


# ---------------------------------------------------------------------------
# 1. Surprisal
# ---------------------------------------------------------------------------

def surprisal(gamma: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Surprisal per level:  -log P(Γ_l)
    Approximated as -log(Γ_l + eps) — high coherence → low surprise.

    This follows directly from UAF: Γ IS the probability of "open future".
    A system with Γ=0.9 at level l is 9x less surprised than one with Γ=0.1.
    """
    return -np.log(gamma + EPS)


def total_surprisal(gamma: np.ndarray, alpha: np.ndarray,
                    weights: Optional[np.ndarray] = None) -> float:
    """Weighted sum of per-level surprisal."""
    S = surprisal(gamma, alpha)
    if weights is None:
        weights = np.ones(len(S)) / len(S)
    return float(np.dot(weights, S))


# ---------------------------------------------------------------------------
# 2. Complexity (KL from uniform prior)
# ---------------------------------------------------------------------------

def complexity(alpha: np.ndarray) -> float:
    """
    How far α is from the uninformative prior α=0.5.
    KL[Bernoulli(α) || Bernoulli(0.5)] per level, summed.

    α close to 0 or 1 → high complexity (system has strong "beliefs")
    α = 0.5           → zero complexity (maximally uncertain)
    """
    a = np.clip(alpha, EPS, 1 - EPS)
    kl = a * np.log(2 * a) + (1 - a) * np.log(2 * (1 - a))
    return float(np.sum(kl))


# ---------------------------------------------------------------------------
# 3. Mutual Information between levels (from trajectory)
# ---------------------------------------------------------------------------

def mutual_information_levels(traj_gamma: np.ndarray,
                               level_i: int, level_j: int,
                               n_bins: int = 20) -> float:
    """
    I(Γ_i ; Γ_j) estimated via histogram from trajectory.

    traj_gamma: shape (T, n_levels)
    Returns MI in nats.
    """
    x = traj_gamma[:, level_i]
    y = traj_gamma[:, level_j]

    # Joint histogram
    H_xy, edges_x, edges_y = np.histogram2d(x, y, bins=n_bins, density=True)
    dx = edges_x[1] - edges_x[0]
    dy = edges_y[1] - edges_y[0]

    H_x = H_xy.sum(axis=1) * dy   # marginal of x
    H_y = H_xy.sum(axis=0) * dx   # marginal of y

    # MI = sum p(x,y) log p(x,y)/(p(x)p(y))
    outer = np.outer(H_x, H_y)
    mask = (H_xy > 0) & (outer > 0)
    mi = float(np.sum(H_xy[mask] * np.log(H_xy[mask] / (outer[mask] + EPS))
                      * dx * dy))
    return max(0.0, mi)


def mutual_information_matrix(traj_gamma: np.ndarray,
                               n_bins: int = 20) -> np.ndarray:
    """Full MI matrix between all level pairs. Shape (n_levels, n_levels)."""
    L = traj_gamma.shape[1]
    M = np.zeros((L, L))
    for i in range(L):
        for j in range(i + 1, L):
            mi = mutual_information_levels(traj_gamma, i, j, n_bins)
            M[i, j] = M[j, i] = mi
    return M


# ---------------------------------------------------------------------------
# 4. Integrated Information Φ (simplified)
# ---------------------------------------------------------------------------

def integrated_information(traj_gamma: np.ndarray) -> float:
    """
    Simplified Φ: how much information is lost when we
    partition the system into independent halves.

    Full Φ (Tononi) is NP-hard; this uses:
      Φ ≈ MI(whole) - max_partition MI(parts)
        = I(Γ_all_pairs) - best_split

    High Φ → system is truly integrated (matryoshka coherence)
    Low  Φ → system factorises into independent levels
    """
    L = traj_gamma.shape[1]
    if L < 2:
        return 0.0

    M = mutual_information_matrix(traj_gamma)

    # Total MI of the whole system = sum of all pairs / normalisation
    total_mi = float(M.sum()) / 2   # symmetric, count once

    # Best bipartition: find split that minimises cut MI
    best_cut = np.inf
    for mask in _bipartitions(L):
        cut = 0.0
        for i in range(L):
            for j in range(L):
                if mask[i] != mask[j]:
                    cut += M[i, j]
        cut /= 2
        best_cut = min(best_cut, cut)

    phi = total_mi - best_cut
    return max(0.0, float(phi))


def _bipartitions(n: int):
    """All non-trivial bipartitions of n elements."""
    for k in range(1, 2**(n-1)):
        mask = np.array([(k >> i) & 1 for i in range(n)])
        if mask.sum() < n:   # avoid all-same
            yield mask


# ---------------------------------------------------------------------------
# 5. Coherence Bandwidth
# ---------------------------------------------------------------------------

def coherence_bandwidth(traj_gamma: np.ndarray,
                        level: int = -1,
                        threshold: float = 0.5) -> float:
    """
    Fraction of time Γ_l > threshold.
    High bandwidth → system maintains coherence robustly over time.
    """
    g = traj_gamma[:, level]
    return float(np.mean(g > threshold))


# ---------------------------------------------------------------------------
# 6. Time-Horizon Estimate
# ---------------------------------------------------------------------------

def time_horizon(traj_gamma: np.ndarray,
                 dtau: float = 0.01) -> float:
    """
    Estimate how far "into the future" the system maintains coherence.

    Method: autocorrelation decay time of Γ.
    τ_corr = integral of normalised autocorrelation R(τ)/R(0)
    until R drops below 1/e.

    High τ_corr → system has "long memory" = large t_future
    """
    g = traj_gamma[:, -1]   # top level
    g_centered = g - np.mean(g)
    var = np.var(g)
    if var < EPS:
        return 0.0

    acf = np.correlate(g_centered, g_centered, mode='full')
    acf = acf[len(acf)//2:]   # keep positive lags
    acf /= acf[0]

    # Find first crossing of 1/e
    threshold = 1.0 / np.e
    crossings = np.where(acf < threshold)[0]
    if len(crossings) == 0:
        return float(len(acf)) * dtau
    return float(crossings[0]) * dtau


# ---------------------------------------------------------------------------
# Combined report
# ---------------------------------------------------------------------------

def information_report(traj_gamma: np.ndarray,
                       traj_alpha: np.ndarray,
                       dtau: float = 0.01) -> dict:
    """
    Full information-theoretic summary of a UAF trajectory.

    Parameters
    ----------
    traj_gamma: shape (T, n_levels)
    traj_alpha: shape (T, n_levels)
    """
    final_gamma = traj_gamma[-1]
    final_alpha = traj_alpha[-1]

    mi_matrix = mutual_information_matrix(traj_gamma)
    phi = integrated_information(traj_gamma)
    bandwidth = {f"level_{l}": coherence_bandwidth(traj_gamma, l)
                 for l in range(traj_gamma.shape[1])}
    tau = time_horizon(traj_gamma, dtau)

    return {
        "final_surprisal": surprisal(final_gamma, final_alpha).tolist(),
        "total_surprisal": total_surprisal(final_gamma, final_alpha),
        "complexity": complexity(final_alpha),
        "mutual_information_matrix": mi_matrix.tolist(),
        "integrated_information_phi": phi,
        "coherence_bandwidth": bandwidth,
        "time_horizon_tau": tau,
        # Summary interpretation
        "interpretation": {
            "phi_strong": phi > 0.05,
            "time_horizon_long": tau > 0.5,
            "high_coherence": float(np.mean(final_gamma)) > 0.6,
            "low_surprisal": total_surprisal(final_gamma, final_alpha) < 2.0,
        }
    }
