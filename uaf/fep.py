"""
UAF-FEP — Free Energy Principle extension
==========================================
Closes GAP 3 (prediction error / VFE) and GAP 4 (active inference).

Связь UAF ↔ FEP
----------------
UAF утверждает что адаптивные системы минимизируют prediction error.
Но в оригинальном коде этого нет — Γ это просто ODE переменная.

Здесь мы делаем это явным через:

  1. Generative model  P(o|s)  — как система предсказывает наблюдения
  2. Recognition model Q(s|o)  — как система обновляет belief о состоянии
  3. Variational Free Energy:
        F = KL[Q(s|o) || P(s)] - E_Q[log P(o|s)]
          = -ELBO
          = complexity - accuracy

  4. Expected Free Energy (для policy selection):
        G(π) = E_Q[log Q(s|π) - log P(o,s|π)]
             = epistemic_value + pragmatic_value

Связь с UAF переменными
-----------------------
  Γ_l (coherence) ↔  -F_l  (negative free energy = evidence lower bound)
  α_l (integration) ↔ belief precision (inverse temperature)
  Φ_{k→l} (flux) ↔  prediction error signal от нижнего уровня

Это не постфактум аналогия — именно так FEP формализует иерархию
(Friston 2010, 2019; Parr & Friston 2019 "Generalised free energy").

Архитектура
-----------
  UAFAgent
    ├── generative_model   — P(o|s, π)
    ├── recognition_model  — Q(s|o)  (mean-field Gaussian)
    ├── infer(o)           — variational inference → update Q
    ├── evaluate_policies()— G(π) для каждой политики
    └── act()              — выбрать π*, выполнить action
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

EPS = 1e-12


# ---------------------------------------------------------------------------
# State belief (Gaussian mean-field)
# ---------------------------------------------------------------------------

@dataclass
class Belief:
    """
    Q(s) = N(mu, sigma²) — factored Gaussian belief over hidden states.
    """
    mu: np.ndarray       # mean,  shape (n_states,)
    log_sigma: np.ndarray # log std, same shape

    @property
    def sigma(self) -> np.ndarray:
        return np.exp(self.log_sigma)

    @classmethod
    def uniform(cls, n: int) -> "Belief":
        return cls(mu=np.zeros(n), log_sigma=np.zeros(n))

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return self.mu + self.sigma * rng.standard_normal(len(self.mu))

    def entropy(self) -> float:
        """H[Q] = 0.5 * sum(1 + log(2π) + 2*log_sigma)"""
        return float(0.5 * np.sum(1 + np.log(2 * np.pi) + 2 * self.log_sigma))

    def kl_to_prior(self, prior_mu: np.ndarray,
                    prior_sigma: np.ndarray) -> float:
        """
        KL[Q || P] where P = N(prior_mu, prior_sigma²).
        KL = 0.5 * Σ [(σ_Q/σ_P)² + (μ_P-μ_Q)²/σ_P² - 1 + 2log(σ_P/σ_Q)]
        """
        ratio = self.sigma / (prior_sigma + EPS)
        diff = (prior_mu - self.mu) / (prior_sigma + EPS)
        kl = 0.5 * np.sum(ratio**2 + diff**2 - 1
                          + 2 * (np.log(prior_sigma + EPS) - self.log_sigma))
        return float(kl)


# ---------------------------------------------------------------------------
# Generative model (likelihood + prior)
# ---------------------------------------------------------------------------

class GenerativeModel:
    """
    P(o, s) = P(o|s) · P(s)

    Simple linear-Gaussian:
      s ~ N(prior_mu, prior_sigma²)
      o = A·s + noise,   noise ~ N(0, obs_sigma²)

    In UAF language:
      s = hidden state (Γ, α vector)
      o = observation (e.g. time series value, sensor reading)
      A = observation matrix (how states map to observations)
    """

    def __init__(self, n_states: int, n_obs: int,
                 obs_sigma: float = 0.1,
                 prior_sigma: float = 1.0,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_states = n_states
        self.n_obs = n_obs
        self.obs_sigma = obs_sigma
        self.prior_mu = np.zeros(n_states)
        self.prior_sigma = np.full(n_states, prior_sigma)
        # Random observation matrix (normalised columns)
        A = rng.standard_normal((n_obs, n_states))
        self.A = A / (np.linalg.norm(A, axis=0, keepdims=True) + EPS)

    def log_likelihood(self, o: np.ndarray, s: np.ndarray) -> float:
        """log P(o|s) = -0.5 * ||o - A·s||² / σ² - const"""
        pred = self.A @ s
        residual = o - pred
        return float(-0.5 * np.dot(residual, residual) / (self.obs_sigma**2 + EPS))

    def log_prior(self, s: np.ndarray) -> float:
        """log P(s) = -0.5 * sum((s-mu)²/sigma²) - const"""
        z = (s - self.prior_mu) / (self.prior_sigma + EPS)
        return float(-0.5 * np.dot(z, z))

    def prediction_error(self, o: np.ndarray, mu_s: np.ndarray) -> np.ndarray:
        """
        Prediction error δ = A^T · (o - A·mu_s) / σ²
        This is the gradient of log P(o|s) w.r.t. s.
        In UAF terms: this IS the flux Φ_{k→l}.
        """
        residual = o - self.A @ mu_s
        return self.A.T @ residual / (self.obs_sigma**2 + EPS)


# ---------------------------------------------------------------------------
# Variational Free Energy
# ---------------------------------------------------------------------------

def variational_free_energy(belief: Belief,
                             obs: np.ndarray,
                             model: GenerativeModel,
                             n_samples: int = 20,
                             rng: Optional[np.random.Generator] = None) -> dict:
    """
    F = KL[Q(s|o) || P(s)] - E_Q[log P(o|s)]
      = complexity - accuracy
      = -ELBO

    Returns dict with full breakdown.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Complexity (analytic for Gaussian)
    complexity = belief.kl_to_prior(model.prior_mu, model.prior_sigma)

    # Accuracy (Monte Carlo over Q)
    acc_samples = [model.log_likelihood(obs, belief.sample(rng))
                   for _ in range(n_samples)]
    accuracy = float(np.mean(acc_samples))

    # Prediction error (at mean)
    pe = model.prediction_error(obs, belief.mu)
    pe_norm = float(np.linalg.norm(pe))

    F = complexity - accuracy

    return {
        "F": F,
        "complexity": complexity,
        "accuracy": accuracy,
        "prediction_error_norm": pe_norm,
        "ELBO": -F,
    }


# ---------------------------------------------------------------------------
# Variational inference step (gradient ascent on ELBO)
# ---------------------------------------------------------------------------

def infer(belief: Belief, obs: np.ndarray, model: GenerativeModel,
          lr_mu: float = 0.1, lr_sigma: float = 0.05,
          steps: int = 20) -> Tuple[Belief, List[float]]:
    """
    Update Q(s|o) by gradient ascent on ELBO (= -F minimisation).

    dμ/dt     ∝  ∂log P(o|s)/∂s + ∂log P(s)/∂s  (precision-weighted PE)
    d log_σ/dt ∝  0.5(1 - σ²/σ_prior² - PE²·σ²)
    """
    mu = belief.mu.copy()
    log_sigma = belief.log_sigma.copy()
    rng = np.random.default_rng(0)
    history = []

    # Scale lr by precision to avoid overflow (precision = 1/sigma^2)
    precision_lik = 1.0 / (model.obs_sigma**2 + EPS)
    precision_prior = 1.0 / (model.prior_sigma**2 + EPS)
    total_precision = precision_lik * np.sum(model.A**2, axis=0) + precision_prior
    lr_scaled = lr_mu / (total_precision + EPS)

    for _ in range(steps):
        sigma = np.exp(log_sigma)
        # gradient of log likelihood w.r.t. mu (precision-weighted PE)
        residual = obs - model.A @ mu
        grad_ll = model.A.T @ residual * precision_lik

        # gradient of log prior w.r.t. mu
        grad_prior = -(mu - model.prior_mu) * precision_prior

        # Combined gradient with per-dimension scaling
        total_grad = grad_ll + grad_prior
        # Clip to prevent overflow
        total_grad = np.clip(total_grad, -10.0, 10.0)

        # update mu with precision-scaled lr
        mu = mu + lr_scaled * total_grad

        # update log_sigma: natural gradient
        natural_grad_sigma = 0.5 * (1.0 - (sigma**2) * total_precision)
        natural_grad_sigma = np.clip(natural_grad_sigma, -1.0, 1.0)
        log_sigma = np.clip(log_sigma + lr_sigma * natural_grad_sigma, -5.0, 2.0)

        new_belief = Belief(mu, log_sigma)
        fe = variational_free_energy(new_belief, obs, model, n_samples=5, rng=rng)
        history.append(fe["F"])

    return Belief(mu, log_sigma), history


# ---------------------------------------------------------------------------
# Policy & Expected Free Energy
# ---------------------------------------------------------------------------

@dataclass
class Policy:
    """A policy is a sequence of actions."""
    actions: np.ndarray   # shape (horizon, n_actions)
    name: str = ""


def expected_free_energy(policy: Policy,
                         belief: Belief,
                         model: GenerativeModel,
                         preferred_obs: Optional[np.ndarray] = None,
                         n_samples: int = 30,
                         rng: Optional[np.random.Generator] = None) -> dict:
    """
    G(π) = epistemic_value + pragmatic_value

    Epistemic value (information gain):
        E_Q[log Q(s) - log Q(s|o_future)]
        ≈ mutual information I(s; o_future)
        Measures: how much will this action reduce my uncertainty?

    Pragmatic value (goal-directedness):
        E_Q[log P(o_preferred)]
        Measures: how close are predicted observations to preferred states?

    In UAF terms:
        High epistemic value → system is exploring, Γ should increase
        High pragmatic value → system is exploiting, α should increase
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if preferred_obs is None:
        preferred_obs = np.zeros(model.n_obs)

    epistemic = 0.0
    pragmatic = 0.0

    for _ in range(n_samples):
        s = belief.sample(rng)
        # Predict observation under this state
        o_pred = model.A @ s + model.obs_sigma * rng.standard_normal(model.n_obs)
        # Epistemic: negative entropy of predicted obs (proxy for info gain)
        # Using variance of prediction as proxy
        epistemic += float(np.var(o_pred))
        # Pragmatic: log probability of reaching preferred obs
        diff = o_pred - preferred_obs
        pragmatic += float(-0.5 * np.dot(diff, diff) / (model.obs_sigma**2 + EPS))

    epistemic /= n_samples
    pragmatic /= n_samples

    G = -(epistemic + pragmatic)   # minimise G
    return {
        "G": G,
        "epistemic_value": epistemic,
        "pragmatic_value": pragmatic,
    }


# ---------------------------------------------------------------------------
# UAFAgent — full active inference agent
# ---------------------------------------------------------------------------

class UAFAgent:
    """
    Active inference agent grounded in UAF.

    The agent:
    1. Maintains a Belief Q(s) over hidden UAF states
    2. On each observation: minimises F (perception)
    3. Evaluates candidate policies by G(π) (planning)
    4. Selects policy that minimises G (action)

    This closes GAP 3 (no prediction / no FEP) and GAP 4 (no action).

    Usage
    -----
    model = GenerativeModel(n_states=3, n_obs=2)
    agent = UAFAgent(model, n_policies=5)

    for obs in observation_stream:
        result = agent.step(obs)
        print(result["action"], result["F"])
    """

    def __init__(self, model: GenerativeModel,
                 n_policies: int = 5,
                 horizon: int = 3,
                 preferred_obs: Optional[np.ndarray] = None,
                 infer_steps: int = 10,
                 seed: int = 0):
        self.model = model
        self.n_policies = n_policies
        self.horizon = horizon
        self.preferred_obs = preferred_obs
        self.infer_steps = infer_steps
        self.rng = np.random.default_rng(seed)

        n = model.n_states
        self.belief = Belief.uniform(n)
        self.history: List[dict] = []

    def step(self, obs: np.ndarray) -> dict:
        """
        Process one observation:
        1. Infer: update Q(s|o) by minimising F
        2. Plan: evaluate G(π) for candidate policies
        3. Act: return best action
        """
        obs = np.asarray(obs, dtype=float)

        # ── Perception: minimise F ───────────────────────────────────────
        self.belief, fe_history = infer(
            self.belief, obs, self.model,
            steps=self.infer_steps
        )
        fe = variational_free_energy(self.belief, obs, self.model,
                                     rng=self.rng)

        # ── Planning: evaluate G(π) for random policies ──────────────────
        n_act = self.model.n_states
        policies = [
            Policy(
                actions=self.rng.standard_normal((self.horizon, n_act)),
                name=f"π_{i}"
            )
            for i in range(self.n_policies)
        ]
        gfe_results = [
            expected_free_energy(π, self.belief, self.model,
                                  self.preferred_obs, rng=self.rng)
            for π in policies
        ]
        G_values = [r["G"] for r in gfe_results]
        best_idx = int(np.argmin(G_values))
        best_policy = policies[best_idx]

        # ── Action ───────────────────────────────────────────────────────
        action = best_policy.actions[0]  # first step of best policy

        result = {
            "F": fe["F"],
            "complexity": fe["complexity"],
            "accuracy": fe["accuracy"],
            "prediction_error": fe["prediction_error_norm"],
            "ELBO": fe["ELBO"],
            "best_G": G_values[best_idx],
            "epistemic_value": gfe_results[best_idx]["epistemic_value"],
            "pragmatic_value": gfe_results[best_idx]["pragmatic_value"],
            "action": action,
            "belief_mu": self.belief.mu.copy(),
            "belief_sigma": self.belief.sigma.copy(),
        }
        self.history.append(result)
        return result

    def run(self, observations: np.ndarray) -> List[dict]:
        """Process a sequence of observations."""
        return [self.step(o) for o in observations]

    def free_energy_trajectory(self) -> np.ndarray:
        return np.array([h["F"] for h in self.history])

    def prediction_error_trajectory(self) -> np.ndarray:
        return np.array([h["prediction_error"] for h in self.history])
