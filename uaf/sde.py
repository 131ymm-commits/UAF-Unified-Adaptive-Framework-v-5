"""
UAF-SDE — Stochastic UAF
=========================
Closes GAP 1: детерминированная ODE → стохастическая SDE (Langevin).

Физическая мотивация
--------------------
Флуктуационно-диссипационная теорема (FDT):
  dΓ = f(Γ,α) dτ + σ_Γ · dW_Γ
  dα = g(Γ,α) dτ + σ_α · dW_α

где dW — стандартный Wiener процесс.

Это прямо соответствует "particular physics" Фристона (2019, 2023):
системы с Markov blanket живут как случайные аттракторы в пространстве
состояний, а не как точечные траектории.

Новые возможности
-----------------
• Моделирование критических переходов (noise-induced bifurcations)
• Оценка uncertainty системы через ensemble sampling
• Связь с термодинамической интерпретацией: σ² ~ kT
• Правильная стрела времени (Second Law через FDT)

Метод интегрирования: Euler-Maruyama (порядок 0.5 по сильной сходимости).
Для более точных симуляций: Milstein scheme (добавлена опционально).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .core import UAFParams, UAFState, _derivatives, EPS


# ---------------------------------------------------------------------------
# SDE parameters
# ---------------------------------------------------------------------------

@dataclass
class SDEParams:
    """Noise intensities for Γ and α at each level."""
    sigma_gamma: float = 0.05   # noise on coherence (thermal fluctuations)
    sigma_alpha: float = 0.02   # noise on integration (slower diffusion)
    milstein: bool = False       # use Milstein correction (more accurate)


# ---------------------------------------------------------------------------
# Euler-Maruyama step
# ---------------------------------------------------------------------------

def _em_step(state: UAFState, p: UAFParams, sde: SDEParams,
             dtau: float, delta_S_env: float,
             rng: np.random.Generator) -> UAFState:
    """
    One Euler-Maruyama step:
      X(t+dt) = X(t) + f(X) dt + σ √dt ξ,   ξ ~ N(0,1)
    """
    L = len(state.gamma)
    dg, da = _derivatives(state, p, delta_S_env)

    sqrt_dt = np.sqrt(dtau)
    noise_g = rng.standard_normal(L) * sde.sigma_gamma * sqrt_dt
    noise_a = rng.standard_normal(L) * sde.sigma_alpha * sqrt_dt

    new_gamma = state.gamma + dtau * dg + noise_g
    new_alpha = state.alpha + dtau * da + noise_a

    # Milstein correction (adds dσ/dX term for multiplicative noise)
    # Here noise is additive so Milstein = EM; but kept as option for
    # subclasses with state-dependent σ.
    if sde.milstein:
        pass  # additive noise: no correction needed

    return UAFState(
        gamma=np.clip(new_gamma, 0.0, 1.0),
        alpha=np.clip(new_alpha, 0.0, 1.0),
        tau=state.tau + dtau,
    )


# ---------------------------------------------------------------------------
# Stochastic UAF System
# ---------------------------------------------------------------------------

class UAFStochasticSystem:
    """
    UAF with Langevin noise.

    Usage
    -----
    sys = UAFStochasticSystem(UAFParams(n_levels=3), SDEParams(sigma_gamma=0.05))
    traj = sys.run(steps=1000, dtau=0.005)
    ensemble = sys.ensemble(n_runs=100, steps=500)
    """

    def __init__(self, params: Optional[UAFParams] = None,
                 sde: Optional[SDEParams] = None,
                 seed: int = 42):
        self.p = params or UAFParams()
        self.sde = sde or SDEParams()
        self.rng = np.random.default_rng(seed)
        self.state = UAFState.initial(self.p.n_levels)
        self.history: List[UAFState] = [self.state.copy()]

    def step(self, dtau: float = 0.005, delta_S_env: float = 0.0):
        self.state = _em_step(self.state, self.p, self.sde,
                               dtau, delta_S_env, self.rng)
        self.history.append(self.state.copy())

    def run(self, steps: int = 1000, dtau: float = 0.005,
            delta_S_env: float = 0.0) -> List[UAFState]:
        for _ in range(steps):
            self.step(dtau, delta_S_env)
        return self.history

    def ensemble(self, n_runs: int = 50, steps: int = 500,
                 dtau: float = 0.005, delta_S_env: float = 0.0,
                 seed_offset: int = 0) -> np.ndarray:
        """
        Run n_runs independent realisations.
        Returns array shape (n_runs, steps+1, n_levels, 2)
        where dim 3 = [gamma, alpha].
        """
        L = self.p.n_levels
        out = np.zeros((n_runs, steps + 1, L, 2))
        for r in range(n_runs):
            sys = UAFStochasticSystem(
                self.p, self.sde, seed=r + seed_offset
            )
            sys.run(steps, dtau, delta_S_env)
            for t, s in enumerate(sys.history):
                out[r, t, :, 0] = s.gamma
                out[r, t, :, 1] = s.alpha
        return out

    # ── statistics over ensemble ─────────────────────────────────────────

    def coherence_uncertainty(self, n_runs: int = 50, steps: int = 200,
                              dtau: float = 0.005) -> dict:
        """
        Returns mean ± std of final Γ across ensemble.
        Quantifies how much noise affects the coherence outcome.
        """
        ens = self.ensemble(n_runs=n_runs, steps=steps, dtau=dtau)
        final_gamma = ens[:, -1, :, 0]   # (n_runs, n_levels)
        return {
            "mean": float(np.mean(final_gamma)),
            "std": float(np.std(final_gamma)),
            "per_level_mean": final_gamma.mean(axis=0).tolist(),
            "per_level_std": final_gamma.std(axis=0).tolist(),
        }

    # ── phase transition detection ───────────────────────────────────────

    def scan_noise_bifurcation(self, sigma_range: np.ndarray,
                               n_runs: int = 30, steps: int = 300,
                               dtau: float = 0.005) -> List[dict]:
        """
        Scan noise intensity σ_Γ and measure:
          • mean final coherence
          • variance (susceptibility — peaks at phase transition)
          • bistability index

        Returns list of dicts, one per sigma value.
        """
        results = []
        for sigma in sigma_range:
            sde = SDEParams(sigma_gamma=float(sigma),
                            sigma_alpha=float(sigma) * 0.4)
            sys = UAFStochasticSystem(self.p, sde, seed=0)
            ens = sys.ensemble(n_runs=n_runs, steps=steps, dtau=dtau)
            final = ens[:, -1, :, 0].mean(axis=1)   # mean Γ per run

            results.append({
                "sigma": float(sigma),
                "mean_gamma": float(np.mean(final)),
                "var_gamma": float(np.var(final)),     # susceptibility χ
                "bimodality": _bimodality_index(final),
            })
        return results


def _bimodality_index(x: np.ndarray) -> float:
    """
    Bimodality coefficient BC = (skewness² + 1) / kurtosis.
    BC > 0.555 suggests bimodal (two-state) distribution.
    """
    if len(x) < 4:
        return 0.0
    n = len(x)
    m = np.mean(x)
    s2 = np.var(x)
    if s2 < EPS:
        return 0.0
    skew = float(np.mean((x - m)**3) / (s2**1.5 + EPS))
    kurt = float(np.mean((x - m)**4) / (s2**2 + EPS))
    return float((skew**2 + 1) / (kurt + EPS))
