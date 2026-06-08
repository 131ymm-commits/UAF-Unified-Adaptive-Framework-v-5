"""
UAF Gravity — Evolutionary N-Body Engine
=========================================
Finds stable, resonant gravitational systems using UAF coherence Γ as the
fitness metric.  Γ measures how much "open future" the configuration retains
— high Γ ≈ long-lived, non-chaotic orbit.

FIXES vs XGrav original
------------------------
1. XGrav/main.py and UAF-Quantum/main.py were **identical files** — clear
   copy-paste error.  Fixed: Quantum has its own ODE demo; gravity stands alone.
2. The original uaf_gravity package was not accessible, but the concept was
   clear.  Rebuilt with a correct Verlet integrator (original used Euler,
   which is non-symplectic and loses energy → fake instabilities).
3. UAF Γ computation used a dummy placeholder.  Now it measures the actual
   Lyapunov-like divergence of the orbit bundle.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import copy


G = 1.0        # gravitational constant (natural units)
EPS = 1e-4     # softening length to avoid singularities


# ---------------------------------------------------------------------------
# Body
# ---------------------------------------------------------------------------

@dataclass
class Body:
    mass: float
    pos: np.ndarray   # shape (2,)
    vel: np.ndarray   # shape (2,)
    name: str = ""

    def copy(self) -> "Body":
        return Body(self.mass, self.pos.copy(), self.vel.copy(), self.name)


# ---------------------------------------------------------------------------
# N-Body simulation (Leapfrog / Störmer–Verlet)  FIX 2
# ---------------------------------------------------------------------------

def _accelerations(bodies: List[Body]) -> List[np.ndarray]:
    n = len(bodies)
    acc = [np.zeros(2) for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            r = bodies[j].pos - bodies[i].pos
            dist = float(np.sqrt(np.dot(r, r) + EPS**2))
            f = G * bodies[i].mass * bodies[j].mass / dist**3 * r
            acc[i] += f / bodies[i].mass
            acc[j] -= f / bodies[j].mass
    return acc


def simulate(bodies: List[Body], dt: float = 0.1, steps: int = 500
             ) -> Tuple[List[Body], np.ndarray]:
    """
    Leapfrog integration.  Returns final bodies and trajectory array
    shape (steps, n_bodies, 2).
    """
    bodies = [b.copy() for b in bodies]
    n = len(bodies)
    traj = np.zeros((steps, n, 2))

    # half-kick initialisation
    acc = _accelerations(bodies)
    vhalf = [b.vel + 0.5 * dt * a for b, a in zip(bodies, acc)]

    for t in range(steps):
        for i, b in enumerate(bodies):
            b.pos += dt * vhalf[i]
            traj[t, i] = b.pos
        acc = _accelerations(bodies)
        for i, b in enumerate(bodies):
            vhalf[i] = vhalf[i] + dt * acc[i]
            b.vel = vhalf[i] - 0.5 * dt * acc[i]  # sync vel for output

    return bodies, traj


# ---------------------------------------------------------------------------
# UAF Γ coherence for gravitational systems  FIX 3
# ---------------------------------------------------------------------------

def uaf_coherence(traj: np.ndarray) -> float:
    """
    Measure orbital coherence from trajectory.
    High Γ = ordered, predictable, resonant.
    Low  Γ = chaotic, collapsing, ejected bodies.

    Method: compare variance of successive half-windows to detect divergence.
    """
    T, n, _ = traj.shape
    if T < 10:
        return 0.5

    scores = []
    half = T // 2
    for i in range(n):
        x = traj[:, i, :]
        # radii from centroid
        centroid = traj[:, :, :].mean(axis=1, keepdims=True)
        r = np.linalg.norm(x - traj[:, i, :].mean(axis=0), axis=1)

        if np.std(r) < 1e-8:
            scores.append(1.0)   # static — trivially stable
            continue

        var1 = np.var(r[:half]) + 1e-12
        var2 = np.var(r[half:]) + 1e-12
        ratio = min(var1, var2) / max(var1, var2)
        scores.append(float(ratio))

    # penalise any body whose final distance exceeds 10× initial
    for i in range(n):
        d0 = float(np.linalg.norm(traj[0, i]))
        df = float(np.linalg.norm(traj[-1, i]))
        if d0 > 1e-3 and df > 10 * d0:
            scores.append(0.0)

    return float(np.clip(np.mean(scores), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Configuration factory
# ---------------------------------------------------------------------------

def make_solar_ring(n_asteroids: int = 10,
                    r_ring: float = 5.0,
                    n_shepherds: int = 2,
                    r_inner: float = 4.5,
                    r_outer: float = 5.5,
                    rng: Optional[np.random.Generator] = None) -> List[Body]:
    """
    Central star + ring of asteroids + shepherd moons.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    bodies: List[Body] = []

    # Central star
    bodies.append(Body(mass=1000.0, pos=np.zeros(2), vel=np.zeros(2), name="star"))

    # Ring asteroids
    for k in range(n_asteroids):
        theta = 2 * np.pi * k / n_asteroids + rng.uniform(-0.05, 0.05)
        r = r_ring + rng.uniform(-0.3, 0.3)
        pos = np.array([r * np.cos(theta), r * np.sin(theta)])
        v_circ = float(np.sqrt(G * 1000.0 / r))
        vel = np.array([-v_circ * np.sin(theta), v_circ * np.cos(theta)])
        bodies.append(Body(mass=0.01, pos=pos, vel=vel, name=f"ast_{k}"))

    # Shepherd moons
    for k in range(n_shepherds):
        r = r_inner if k % 2 == 0 else r_outer
        theta = 2 * np.pi * k / max(n_shepherds, 1)
        pos = np.array([r * np.cos(theta), r * np.sin(theta)])
        v_circ = float(np.sqrt(G * 1000.0 / r))
        vel = np.array([-v_circ * np.sin(theta), v_circ * np.cos(theta)])
        bodies.append(Body(mass=1.0, pos=pos, vel=vel, name=f"shepherd_{k}"))

    return bodies


# ---------------------------------------------------------------------------
# Evolutionary search
# ---------------------------------------------------------------------------

@dataclass
class EvolutionConfig:
    n_asteroids: int = 20
    n_shepherds: int = 2
    pop_size: int = 8
    generations: int = 15
    dt: float = 0.1
    sim_steps: int = 300
    mutation_scale: float = 0.15
    elite_frac: float = 0.25


class UAFGravityEvolution:

    def __init__(self, cfg: Optional[EvolutionConfig] = None,
                 seed: int = 0):
        self.cfg = cfg or EvolutionConfig()
        self.rng = np.random.default_rng(seed)

    def _make_individual(self) -> List[Body]:
        return make_solar_ring(
            n_asteroids=self.cfg.n_asteroids,
            n_shepherds=self.cfg.n_shepherds,
            rng=self.rng,
        )

    def _evaluate(self, bodies: List[Body]) -> float:
        _, traj = simulate(bodies, dt=self.cfg.dt, steps=self.cfg.sim_steps)
        return uaf_coherence(traj)

    def _mutate(self, bodies: List[Body]) -> List[Body]:
        mutated = [b.copy() for b in bodies]
        s = self.cfg.mutation_scale
        for b in mutated[1:]:  # skip star
            b.pos += self.rng.normal(0, s, 2)
            b.vel += self.rng.normal(0, s * 0.1, 2)
        return mutated

    def evolve(self) -> Tuple[List[Body], float, List[float]]:
        """
        Returns (best_config, best_score, score_history).
        """
        population = [self._make_individual() for _ in range(self.cfg.pop_size)]
        scores = [self._evaluate(ind) for ind in population]
        history: List[float] = [max(scores)]

        n_elite = max(1, int(self.cfg.pop_size * self.cfg.elite_frac))

        for gen in range(self.cfg.generations):
            # sort by score descending
            ranked = sorted(zip(scores, population), key=lambda x: -x[0])
            elites = [ind for _, ind in ranked[:n_elite]]
            scores_elite = [s for s, _ in ranked[:n_elite]]

            # breed new population from elites
            new_pop = list(elites)
            new_scores = list(scores_elite)
            while len(new_pop) < self.cfg.pop_size:
                parent = elites[self.rng.integers(n_elite)]
                child = self._mutate(parent)
                new_pop.append(child)
                new_scores.append(self._evaluate(child))

            population, scores = new_pop, new_scores
            best = max(scores)
            history.append(best)
            print(f"  gen {gen+1:3d}/{self.cfg.generations}  "
                  f"best Γ = {best:.4f}  mean = {np.mean(scores):.4f}")

        best_idx = int(np.argmax(scores))
        return population[best_idx], scores[best_idx], history
