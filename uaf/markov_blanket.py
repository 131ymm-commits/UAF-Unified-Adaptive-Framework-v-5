"""
UAF Markov Blanket
==================
Closes GAP 2: явная структура internal / blanket / external состояний.

Что такое Markov Blanket в UAF контексте
-----------------------------------------
В FEP (Friston 2019, 2023) система существует через Markov blanket —
статистическую границу, которая делает внутренние состояния условно
независимыми от внешних.

В UAF иерархии:
  Level l имеет:
    internal_states  = (Γ_l, α_l)  — то, что "внутри" уровня
    sensory_states   = входящие потоки Φ_{k→l} от нижних уровней
    active_states    = исходящие потоки Φ_{l→m} к верхним уровням
    external_states  = всё остальное (другие ветки иерархии, среда)

  blanket_states = sensory_states ∪ active_states

Это операционализирует α_l:
  α_l = степень "закрытости" blanket
  α_l = 0: blanket прозрачный, всё проходит
  α_l = 1: blanket непрозрачный, система изолирована

Реализация
----------
  MarkovBlanket      — структура данных для одного уровня
  UAFHierarchyMB     — полная иерархия с явными MB
  emergent_blankets  — алгоритм обнаружения MB из data (coupling threshold)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

EPS = 1e-12


# ---------------------------------------------------------------------------
# Markov Blanket state partition
# ---------------------------------------------------------------------------

@dataclass
class MarkovBlanketState:
    """
    State partition for one level of the UAF hierarchy.

    Corresponds to FEP partition:
      μ (internal)  ← optimises free energy
      b (blanket)   ← sensory ∪ active
      η (external)  ← not directly accessible
    """
    level: int

    # Internal states (optimised by the level itself)
    gamma: float = 0.8      # future coherence
    alpha: float = 0.1      # integration degree

    # Sensory states (incoming from lower levels)
    sensory: np.ndarray = field(default_factory=lambda: np.array([0.0]))

    # Active states (outgoing to higher levels)
    active: np.ndarray = field(default_factory=lambda: np.array([0.0]))

    # Precision (= inverse variance = how sharp the blanket is)
    # High precision → strong blanket → high α
    precision: float = 1.0

    def blanket_norm(self) -> float:
        """||blanket||² = ||sensory||² + ||active||²"""
        return float(np.dot(self.sensory, self.sensory)
                     + np.dot(self.active, self.active))

    def self_evidence(self) -> float:
        """
        log P(blanket | internal) — how well internal states explain the blanket.
        Proxy: -surprisal of gamma weighted by precision.
        """
        return float(self.precision * np.log(self.gamma + EPS))

    def markov_blanket_integrity(self) -> float:
        """
        How "closed" is the blanket?
        High α → blanket is tight → high integrity.
        Returns value in [0, 1].
        """
        return float(np.clip(self.alpha, 0, 1))


# ---------------------------------------------------------------------------
# UAF Hierarchy with explicit Markov Blankets
# ---------------------------------------------------------------------------

class UAFHierarchyMB:
    """
    UAF hierarchy where each level has an explicit Markov Blanket.

    Key difference from UAFSystem (core.py):
      - Flux Φ_{k→l} now flows through blanket states explicitly
      - α_l directly controls the blanket "tightness"
      - Free energy is computed per-level based on blanket statistics

    This makes the connection to FEP/Active Inference explicit.
    """

    def __init__(self, n_levels: int = 3,
                 coupling: float = 0.5,
                 decoherence: float = 0.3):
        self.n_levels = n_levels
        self.coupling = coupling
        self.decoherence = decoherence

        self.levels = [
            MarkovBlanketState(
                level=l,
                gamma=0.8 - 0.1 * l,   # lower levels more coherent initially
                alpha=0.1 + 0.05 * l,
                sensory=np.zeros(max(1, l)),
                active=np.zeros(1),
                precision=1.0 + 0.5 * l,  # higher levels have higher precision
            )
            for l in range(n_levels)
        ]

        self.history: List[List[MarkovBlanketState]] = [
            self._snapshot()
        ]

    def _snapshot(self) -> List[MarkovBlanketState]:
        return [
            MarkovBlanketState(
                level=s.level, gamma=s.gamma, alpha=s.alpha,
                sensory=s.sensory.copy(), active=s.active.copy(),
                precision=s.precision,
            )
            for s in self.levels
        ]

    def step(self, dtau: float = 0.01,
             external_input: Optional[np.ndarray] = None):
        """
        One step of MB-aware UAF dynamics.

        The blanket of level l is driven by:
          sensory_l  ← active_{l-1}   (what comes from below)
          active_l   → sensory_{l+1}  (what goes up)

        Internal update:
          dΓ/dτ = η · ||sensory|| · (1-α)² - λ · α
          dα/dτ = κ · α · (1-α) · (||sensory|| - c) - δ · (1-α) · Γ
        """
        new_levels = [s for s in self.levels]

        for l in range(self.n_levels):
            s = self.levels[l]

            # ── Sensory: incoming signal ──────────────────────────────────
            if l == 0:
                # Bottom level: sensory = external input or noise
                inp = (external_input if external_input is not None
                       else np.array([0.0]))
                sensory_strength = float(np.linalg.norm(inp))
            else:
                sensory_strength = float(np.linalg.norm(
                    self.levels[l-1].active
                ))

            # ── Internal dynamics (blanket-mediated) ─────────────────────
            # Growth: incoming flux × (1-α)² × blanket integrity of source
            if l > 0:
                source_integrity = self.levels[l-1].markov_blanket_integrity()
            else:
                source_integrity = 1.0

            growth = (self.coupling * sensory_strength
                      * source_integrity * (1 - s.alpha)**2)
            decay  = self.decoherence * s.alpha * s.gamma

            new_gamma = float(np.clip(s.gamma + dtau * (growth - decay),
                                      0.0, 1.0))

            # Alpha: tightens when sensory input is low (isolation)
            energy = sensory_strength * source_integrity
            cost = 0.3
            d_alpha = (0.3 * s.alpha * (1 - s.alpha) * (energy - cost)
                       - 0.1 * (1 - s.alpha) * s.gamma)
            new_alpha = float(np.clip(s.alpha + dtau * d_alpha, 0.0, 1.0))

            # ── Active: what this level sends upward ─────────────────────
            new_active = np.array([new_gamma * (1 - new_alpha)])

            new_levels[l] = MarkovBlanketState(
                level=l,
                gamma=new_gamma,
                alpha=new_alpha,
                sensory=np.array([sensory_strength]),
                active=new_active,
                precision=s.precision,
            )

        self.levels = new_levels
        self.history.append(self._snapshot())

    def run(self, steps: int = 500, dtau: float = 0.01,
            external_signal: Optional[np.ndarray] = None):
        for t in range(steps):
            ext = (external_signal[t:t+1]
                   if external_signal is not None else None)
            self.step(dtau, ext)

    def blanket_integrity_trajectory(self) -> np.ndarray:
        """Returns shape (T, n_levels)."""
        return np.array([
            [s.markov_blanket_integrity() for s in snap]
            for snap in self.history
        ])

    def self_evidence_trajectory(self) -> np.ndarray:
        """Returns shape (T, n_levels)."""
        return np.array([
            [s.self_evidence() for s in snap]
            for snap in self.history
        ])


# ---------------------------------------------------------------------------
# Emergent blanket detection from coupling data (GAP 5 preview)
# ---------------------------------------------------------------------------

def detect_emergent_blankets(coupling_matrix: np.ndarray,
                              threshold: float = 0.3) -> List[List[int]]:
    """
    Given a coupling matrix C[i,j] (strength of influence of i on j),
    detect clusters that behave as Markov blankets.

    Algorithm: greedy community detection on coupling graph.
    Nodes with C[i,j] > threshold are in the same blanket.

    This is a preview of GAP 5 (dynamic hierarchy formation).
    """
    n = len(coupling_matrix)
    visited = [False] * n
    blankets = []

    for start in range(n):
        if visited[start]:
            continue
        cluster = [start]
        visited[start] = True
        queue = [start]
        while queue:
            node = queue.pop(0)
            for neighbor in range(n):
                if not visited[neighbor]:
                    if coupling_matrix[node, neighbor] > threshold:
                        visited[neighbor] = True
                        cluster.append(neighbor)
                        queue.append(neighbor)
        blankets.append(sorted(cluster))

    return blankets
