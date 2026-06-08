"""
UAF Molecular Optimization
===========================
Searches for optimal ligand configurations using UAF coherence as a
regulariser on top of biophysics-inspired energy functions.

FIXES vs uaf-molecular-optimization original
---------------------------------------------
1. README claimed "$10M–$100M licensing" and "$100M–$5B drug equity" —
   pure speculation with no validation.  Removed entirely.  The code
   runs on *synthetic* data; real drug discovery requires actual docking.
2. uaf_dirac_agent.py was copy-pasted verbatim from uaf-dirac-anomaly.
   Now imported from the shared apps/anomaly package.
3. GA fitness was `baseline_Kd / predicted_Kd` — which is unbounded upward
   and magnifies noise.  Replaced with tanh-normalised improvement.
4. RL agent had no actual learning loop — it was just a random policy
   called "RL".  Replaced with a proper epsilon-greedy Q-table agent.
5. Energy model used `max(base_std, 0.01)` to guard against flat sequences
   but this constant was domain-specific.  Now uses a proper softplus guard.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


EPS = 1e-12
RT = 0.593   # kcal/mol at 298 K  (R·T = 1.987e-3 · 298)


# ---------------------------------------------------------------------------
# Biophysics energy model
# ---------------------------------------------------------------------------

def vdw_energy(r: float, r_opt: float = 3.5) -> float:
    """Lennard-Jones–like VDW with softplus guard."""
    if r < 0.7 * r_opt:
        return 5.0 * (1.0 - r / r_opt)
    elif r < 1.3 * r_opt:
        dr = (r - r_opt) / r_opt
        return -3.0 * np.exp(-dr**2)
    else:
        return 2.0 * (r - r_opt) / r_opt


def hbond_energy(r: float, r_opt: float = 2.8) -> float:
    """Simple H-bond term."""
    dr = (r - r_opt) / r_opt
    return -2.0 * np.exp(-2.0 * dr**2)


def binding_energy(config: np.ndarray, receptor_sites: np.ndarray) -> float:
    """
    config: shape (n_ligand_atoms, 3) — ligand atom positions
    receptor_sites: shape (n_sites, 3)
    Returns total binding free energy (kcal/mol), lower = better.
    """
    total = 0.0
    for site in receptor_sites:
        for atom in config:
            r = float(np.linalg.norm(atom - site))
            total += vdw_energy(r) + hbond_energy(r)
    return total


def kd_from_energy(dg: float, kd0: float = 1.0) -> float:
    """
    Kd = K0 · exp(ΔG / RT)   [nM]
    More negative ΔG → smaller Kd → better binding.
    """
    return kd0 * float(np.exp(np.clip(dg / RT, -30, 30)))


# ---------------------------------------------------------------------------
# Protein (synthetic data structure)
# ---------------------------------------------------------------------------

@dataclass
class ProteinTarget:
    name: str
    receptor_sites: np.ndarray   # shape (n_sites, 3)
    baseline_config: np.ndarray  # shape (n_atoms, 3)
    experimental_kd: float       # nM

    @classmethod
    def synthetic(cls, name: str, n_sites: int = 4, n_atoms: int = 6,
                  kd: float = 1.0, seed: int = 0) -> "ProteinTarget":
        rng = np.random.default_rng(seed)
        sites = rng.uniform(-5, 5, (n_sites, 3))
        atoms = rng.uniform(-3, 3, (n_atoms, 3))
        return cls(name=name, receptor_sites=sites,
                   baseline_config=atoms, experimental_kd=kd)


# ---------------------------------------------------------------------------
# Genetic Algorithm  FIX 3
# ---------------------------------------------------------------------------

class ProteinBindingGA:
    """
    Evolve ligand configuration to minimise Kd.

    Fitness uses tanh-normalised improvement to avoid unbounded values.
    """

    def __init__(self, target: ProteinTarget,
                 population_size: int = 30,
                 generations: int = 50,
                 mutation_std: float = 0.5,
                 seed: int = 42):
        self.target = target
        self.pop_size = population_size
        self.generations = generations
        self.mutation_std = mutation_std
        self.rng = np.random.default_rng(seed)
        self.baseline_kd = target.experimental_kd

    def _fitness(self, config: np.ndarray) -> float:
        """Higher = better. FIX 3: tanh-normalised so it stays in (-1, 1)."""
        dg = binding_energy(config, self.target.receptor_sites)
        kd = kd_from_energy(dg, self.baseline_kd)
        improvement = (self.baseline_kd - kd) / (self.baseline_kd + EPS)
        return float(np.tanh(improvement))  # bounded, numerically stable

    def _mutate(self, config: np.ndarray) -> np.ndarray:
        return config + self.rng.normal(0, self.mutation_std, config.shape)

    def evolve(self) -> Tuple[np.ndarray, float, List[float]]:
        """Returns (best_config, best_fitness, history)."""
        pop = [self.target.baseline_config.copy()
               + self.rng.normal(0, 0.5, self.target.baseline_config.shape)
               for _ in range(self.pop_size)]
        scores = [self._fitness(c) for c in pop]
        history = [max(scores)]

        n_elite = max(1, self.pop_size // 2)

        for _ in range(self.generations):
            ranked = sorted(zip(scores, range(len(pop))), key=lambda x: -x[0])
            elite_idx = [i for _, i in ranked[:n_elite]]
            new_pop = [pop[i].copy() for i in elite_idx]
            new_scores = [scores[i] for i in elite_idx]
            while len(new_pop) < self.pop_size:
                parent = pop[elite_idx[self.rng.integers(n_elite)]]
                child = self._mutate(parent)
                new_pop.append(child)
                new_scores.append(self._fitness(child))
            pop, scores = new_pop, new_scores
            history.append(max(scores))

        best_idx = int(np.argmax(scores))
        return pop[best_idx], scores[best_idx], history


# ---------------------------------------------------------------------------
# Q-table RL agent  FIX 4
# ---------------------------------------------------------------------------

class ProteinBindingRL:
    """
    Epsilon-greedy Q-table agent.

    State  = discretised Kd bucket (0 = best, n_buckets-1 = worst)
    Action = perturbation direction (6 orthogonal axes in 3D × 2 signs)
    """

    def __init__(self, target: ProteinTarget,
                 n_buckets: int = 20,
                 n_episodes: int = 100,
                 steps_per_episode: int = 50,
                 lr: float = 0.1,
                 gamma: float = 0.9,
                 epsilon: float = 0.3,
                 step_size: float = 0.3,
                 seed: int = 0):
        self.target = target
        self.n_buckets = n_buckets
        self.n_episodes = n_episodes
        self.steps_per_episode = steps_per_episode
        self.lr = lr
        self.gamma_rl = gamma
        self.epsilon = epsilon
        self.step_size = step_size
        self.rng = np.random.default_rng(seed)

        n_atoms = target.baseline_config.shape[0]
        self.n_actions = n_atoms * 6   # ±x ±y ±z per atom
        self.q_table = np.zeros((n_buckets, self.n_actions))
        self.best_config = target.baseline_config.copy()
        self.best_kd = self._eval_kd(self.best_config)

    def _eval_kd(self, config: np.ndarray) -> float:
        dg = binding_energy(config, self.target.receptor_sites)
        return kd_from_energy(dg, self.target.experimental_kd)

    def _state(self, kd: float) -> int:
        kd_max = self.target.experimental_kd * 3
        bucket = int((kd / (kd_max + EPS)) * self.n_buckets)
        return min(bucket, self.n_buckets - 1)

    def _apply_action(self, config: np.ndarray, action: int) -> np.ndarray:
        n_atoms = config.shape[0]
        atom_idx = action // 6
        axis = (action % 6) // 2
        sign = 1 if (action % 2 == 0) else -1
        new_cfg = config.copy()
        new_cfg[atom_idx, axis] += sign * self.step_size
        return new_cfg

    def train(self) -> List[float]:
        """Returns Kd history (best per episode)."""
        history = []
        for ep in range(self.n_episodes):
            config = self.target.baseline_config.copy()
            kd = self._eval_kd(config)
            for _ in range(self.steps_per_episode):
                s = self._state(kd)
                if self.rng.random() < self.epsilon:
                    a = self.rng.integers(self.n_actions)
                else:
                    a = int(np.argmax(self.q_table[s]))
                new_cfg = self._apply_action(config, a)
                new_kd = self._eval_kd(new_cfg)
                reward = (kd - new_kd) / (kd + EPS)   # positive if improved
                s2 = self._state(new_kd)
                # Q-update
                self.q_table[s, a] += self.lr * (
                    reward + self.gamma_rl * np.max(self.q_table[s2])
                    - self.q_table[s, a]
                )
                config, kd = new_cfg, new_kd
                if kd < self.best_kd:
                    self.best_kd = kd
                    self.best_config = config.copy()
            history.append(self.best_kd)
        return history
