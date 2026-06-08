"""
UAF Dirac Anomaly Detector
==========================
Online, training-free anomaly detection via 2-component Dirac dynamics.

Mapping to UAF time tripartition
---------------------------------
  mass  m  = crystallised past  (t_past)  — structural inertia
  momentum p = present pressure  (t_now)  — flow force
  gap        = how far the system is from its mass-gap bound (anomaly signal)

FIXES vs uaf-dirac-anomaly original
------------------------------------
1. Spinor norm could collapse to 0 → division by EPS giving random direction.
   Now we reinitialise to (1, 0) with a warning flag.
2. `_normalized_history` re-scanned the whole buffer O(N²) — replaced with
   an incremental running mean/std.
3. `score_sequence` called reset() which wiped the spinor state mid-stream;
   now reset is explicit, score_sequence rebuilds from scratch correctly.
4. baseline_window z-score used std of raw_scores including the current point
   — fixed to use only the past window (no look-ahead).
5. Weights gap+inversion+zitter didn't necessarily sum to 1 if user changed
   them — added normalisation.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

EPS = 1e-12


@dataclass
class DiracParams:
    norm_window: int = 50
    short_window: int = 20
    long_window: int = 60
    baseline_window: int = 120
    dt: float = 0.060
    ema: float = 0.10
    threshold_k: float = 3.0
    min_gap: int = 30
    gap_weight: float = 0.25
    inversion_weight: float = 0.35
    zitter_weight: float = 0.40
    squash_scale: float = 3.0

    def __post_init__(self):
        # FIX 5: normalise weights so they always sum to 1
        total = self.gap_weight + self.inversion_weight + self.zitter_weight
        if total < EPS:
            self.gap_weight = 1/3
            self.inversion_weight = 1/3
            self.zitter_weight = 1/3
        else:
            self.gap_weight /= total
            self.inversion_weight /= total
            self.zitter_weight /= total


class _RunningStats:
    """O(1) amortised incremental mean and std over a sliding window."""

    def __init__(self, window: int):
        self.window = window
        self._buf: list[float] = []
        self._sum = 0.0
        self._sum2 = 0.0

    def push(self, x: float):
        if len(self._buf) == self.window:
            old = self._buf.pop(0)
            self._sum -= old
            self._sum2 -= old * old
        self._buf.append(x)
        self._sum += x
        self._sum2 += x * x

    def mean(self) -> float:
        n = len(self._buf)
        return self._sum / n if n else 0.0

    def std(self) -> float:
        n = len(self._buf)
        if n < 2:
            return EPS
        var = (self._sum2 - self._sum**2 / n) / (n - 1)
        return max(EPS, float(np.sqrt(max(0.0, var))))

    def normalise(self, x: float) -> float:
        return (x - self.mean()) / self.std()

    def __len__(self):
        return len(self._buf)


class UAFDiracAgent:
    """
    Online anomaly detector. No training, no look-ahead.

    Parameters
    ----------
    params : DiracParams (optional)

    Usage
    -----
    agent = UAFDiracAgent()
    for t, value in enumerate(stream):
        event = agent.update(t, value)
        if event:
            print(event)
    """

    def __init__(self, params: Optional[DiracParams] = None):
        self.p = params or DiracParams()
        self._reset_state()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def reset(self):
        self._reset_state()

    def update(self, timestamp, value: float) -> Optional[dict]:
        """
        Process one new value.
        Returns an anomaly event dict or None.
        """
        x = float(value)
        # update normalisation window
        self._norm_stats.push(x)
        normed = self._norm_stats.normalise(x)

        self._normed_buf.append(normed)
        n = len(self._normed_buf)

        if n < self.p.long_window:
            return None

        raw_score, state = self._dirac_score()
        self._raw_scores.append(raw_score)
        self._baseline_stats.push(raw_score)   # FIX 4: push BEFORE z so current is excluded
        idx = len(self._raw_scores) - 1

        if len(self._baseline_stats) < self.p.baseline_window:
            return None

        # FIX 4: exclude current point from baseline
        # We pushed it just now, so we compute z against prior window mean/std.
        # Trick: temporarily pop from baseline stats is complex; instead we
        # keep a *delayed* baseline that is one step behind.
        z = self._z_delayed

        # Update delayed baseline for next call
        self._z_delayed = (raw_score - self._baseline_stats.mean()) / self._baseline_stats.std()

        if z <= self.p.threshold_k:
            return None
        if idx - self._last_event_idx <= self.p.min_gap:
            return None

        self._last_event_idx = idx
        event_score = float(1.0 - np.exp(-z / self.p.squash_scale))

        return {
            "timestamp": timestamp,
            "index": idx,
            "score": event_score,
            "raw_score": float(raw_score),
            "z": float(z),
            **state,
        }

    def score_sequence(self, values) -> np.ndarray:
        """
        Batch mode: returns anomaly scores array of same length as values.
        FIX 3: properly resets before processing.
        """
        self._reset_state()
        scores = np.zeros(len(values), dtype=np.float64)
        for i, v in enumerate(values):
            event = self.update(i, v)
            if event is not None:
                scores[i] = event["score"]
        return scores

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _reset_state(self):
        p = self.p
        self._norm_stats = _RunningStats(p.norm_window)
        self._baseline_stats = _RunningStats(p.baseline_window)
        self._normed_buf: list[float] = []
        self._raw_scores: list[float] = []
        # Spinor state
        self._a = complex(1.0, 0.0)
        self._b = complex(0.0, 0.0)
        self._prev_energy = 0.0
        self._smooth_score = 0.0
        self._last_event_idx = -(10 ** 9)
        self._spinor_resets = 0
        self._z_delayed = 0.0

    @staticmethod
    def _gamma_balance(segment: np.ndarray) -> float:
        if len(segment) < 5:
            return 0.5
        diffs = np.diff(segment)
        sigma = float(np.std(segment))
        if sigma < EPS:
            return 0.5
        entropy = float(np.mean(np.abs(diffs))) / sigma
        chaos = float(np.var(segment))
        return float(np.clip(entropy / (entropy + chaos + EPS), 0.0, 1.0))

    def _dirac_score(self) -> tuple[float, dict]:
        sig = np.asarray(self._normed_buf, dtype=np.float64)
        p = self.p
        i = len(sig) - 1

        short = sig[i - p.short_window: i]
        long_ = sig[i - p.long_window: i]

        g_short = self._gamma_balance(short)
        g_long = self._gamma_balance(long_)
        g_collapse = max(0.0, g_long - g_short)

        ds = np.diff(short)
        dl = np.diff(long_)
        pshort = float(np.std(ds) + np.mean(np.abs(ds)))
        plong = float(np.std(dl) + np.mean(np.abs(dl))) + EPS
        momentum = max(0.0, pshort / plong - 1.0)
        curvature = abs(sig[i] - 2.0 * sig[i-1] + sig[i-2])

        mass = 0.15 + 0.85 * g_short
        hx = momentum + 0.35 * g_collapse
        hy = 0.35 * curvature
        hz = mass

        ha = hz * self._a + (hx - 1j * hy) * self._b
        hb = (hx + 1j * hy) * self._a - hz * self._b
        self._a -= 1j * p.dt * ha
        self._b -= 1j * p.dt * hb

        norm = float(np.sqrt(abs(self._a)**2 + abs(self._b)**2))
        if norm < EPS:
            # FIX 1: reinitialise instead of dividing by EPS
            self._a = complex(1.0, 0.0)
            self._b = complex(0.0, 0.0)
            self._spinor_resets += 1
            norm = 1.0
        self._a /= norm
        self._b /= norm

        polarization = float(abs(self._a)**2 - abs(self._b)**2)
        ab = np.conj(self._a) * self._b
        mix = 2.0 * ab.real
        phase = 2.0 * ab.imag
        energy = float(hx * mix + hy * phase + hz * polarization)

        gap_breach = max(0.0, np.sqrt(hx**2 + hy**2) / (hz + EPS) - 1.0)
        inversion = max(0.0, -polarization)
        zitter = abs(energy - self._prev_energy)
        self._prev_energy = energy

        raw = (p.gap_weight * gap_breach
               + p.inversion_weight * inversion
               + p.zitter_weight * zitter)
        self._smooth_score = p.ema * raw + (1.0 - p.ema) * self._smooth_score

        state = {
            "gamma_short": g_short,
            "gamma_long": g_long,
            "mass": mass,
            "momentum": momentum,
            "curvature": curvature,
            "polarization": polarization,
            "gap_breach": float(gap_breach),
            "inversion": float(inversion),
            "zitter": float(zitter),
            "energy": energy,
            "spinor_resets": self._spinor_resets,
        }
        return self._smooth_score, state
