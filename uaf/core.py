"""
UAF v5 — Unified Adaptive Framework
=====================================
Объединяет TSV (v3.1) + FEP (v4) в единое уравнение.

═══════════════════════════════════════════════════════════════
ЦЕНТРАЛЬНЫЙ ТЕЗИС
═══════════════════════════════════════════════════════════════

TSV:  ΔA_i = α·A_i·A_j·(1-A_i) - δ·A_i
FEP:  dμ/dt = Π·ε  (precision × prediction error)

Связь: A_i — это не просто "closure", а ELBO агента i.
  A_i = 1 - F_i/F_baseline = точность модели агента i
  α·A_j = precision от соседа j (сколько доверяем его модели)
  (1-A_i) = prediction error агента i (сколько ещё учиться)
  δ·A_i = деградация точности (forgetting, entropy)

Тогда TSV = частный случай FEP для дискретных агентов:
  ΔA_i = Π_ij · ε_i · (1 - A_i) - δ·A_i
где Π_ij = α·A_j (precision от соседа j)
    ε_i  = A_i   (текущая точность как "сигнал")

══════════════════════════════════════════════════════════
ИСПРАВЛЕНИЯ vs v3.1
══════════════════════════════════════════════════════════

1. α разделён на три параметра:
   alpha_social  — скорость TSV взаимодействия (бывший α)
   alpha_learn   — скорость внутреннего обучения (FEP lr)
   alpha_cat     — каталитическое усиление хабов (catalysis_eta)

2. NPG исправлен:
   NPG_old = (var_baseline - var_current) / var_baseline  ← сломан
   NPG_new = (F_baseline - F_current)   / F_baseline      ← честный

3. Уровни L0→L3: не пороги mean(A), а фазовые состояния
   через bifurcation analysis (eigenvalue sign of Jacobian).

4. Basal floor интегрирован как F_basal (минимальный метаболизм):
   ΔA_i += floor · (1 - A_i/A_ceiling)  ← adaptive version

5. Мост F_k ↔ closure (EXP 044):
   Проблема была в том, что F_k и closure измерялись
   РАЗНЫМИ системами на одних данных.
   Решение: A_i = clip(1 - F_i/F_base, 0, 1) — прямая подстановка.
   Тогда corr(closure_from_F, TSV_A) = corr(A_FEP, A_TSV).
   Мост закроется когда обе системы обновляют одно A.

══════════════════════════════════════════════════════════
МАСТЕР-УРАВНЕНИЕ UAF v5
══════════════════════════════════════════════════════════

dA_i/dτ = [α_social · C_ij · A_j · (1-A_i)]_TSV
         + [α_learn  · Π_i  · ε_i · (1-A_i)]_FEP
         + [floor    · (1-A_i/A_ceil)]_basal
         - [δ        · (1 - 0.3·A_i)]_decay
         + [η_i      · (A_target - A_i)]_CPS (optional)

где:
  C_ij     = catalysis от соседа j (BA-топология)
  Π_i      = precision агента i (обновляется из данных)
  ε_i      = prediction error (1 - текущая точность)
  τ        = relational time (не clock, а число взаимодействий)
  η_i      = CPS коэффициент (= 0 вне homeostatic zone)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import warnings

EPS = 1e-12


# ══════════════════════════════════════════════════════════
# ПАРАМЕТРЫ
# ══════════════════════════════════════════════════════════

@dataclass
class UAFv5Params:
    # TSV
    alpha_social:  float = 0.08    # социальное взаимодействие
    alpha_learn:   float = 0.05    # FEP обучение
    alpha_cat:     float = 0.65    # каталитическое усиление (было catalysis_eta)
    epsilon_cost:  float = 0.02    # cost TSV взаимодействия

    # Decay
    decay:         float = 0.010   # энтропийный распад
    decay_mod:     float = 0.30    # модификатор (1 - mod·A)

    # Basal floor
    floor:         float = 0.002   # минимальный метаболизм
    floor_adaptive: bool = True    # floor*(1-A/ceil) вместо const
    a_ceiling:     float = 0.95    # потолок (не 1.0!)

    # CPS (homeostat) — KAPPA
    kappa:         float = 0.0     # 0 = выключен (вредит в 79% случаев)
    a_target:      float = 0.80    # цель CPS
    kappa_zone:    float = 0.05    # CPS активен только если |A-target|>zone

    # Структура
    n_agents:      int   = 60
    n_meta:        int   = 6
    density:       float = 0.25    # fraction of pairs per step
    K:             int   = 3       # соседей для BA

    # TippingPoint
    a_crit:        float = 0.75

    # Начальные условия
    a_init_low:    float = 0.25
    a_init_high:   float = 0.35


# ══════════════════════════════════════════════════════════
# ТОПОЛОГИЯ
# ══════════════════════════════════════════════════════════

class BATopology:
    """Barabási–Albert граф с каталитическими весами."""

    def __init__(self, n: int, m: int = 3, eta: float = 0.65, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n = n
        self.adj: List[List[int]] = [[] for _ in range(n)]
        self.catalysis = np.ones(n)

        if n < 4:
            # полный граф для малых систем
            for i in range(n):
                for j in range(i+1, n):
                    self.adj[i].append(j)
                    self.adj[j].append(i)
        else:
            m = min(m, n-1)
            # seed clique
            for i in range(m+1):
                for j in range(i+1, m+1):
                    self.adj[i].append(j)
                    self.adj[j].append(i)
            degrees = np.array([len(self.adj[i]) for i in range(n)], dtype=float)
            for v in range(m+1, n):
                total = degrees[:v].sum()
                p = degrees[:v]/total if total > 0 else np.ones(v)/v
                chosen: set = set()
                while len(chosen) < min(m, v):
                    chosen.add(int(rng.choice(v, p=p)))
                for nb in chosen:
                    self.adj[v].append(nb)
                    self.adj[nb].append(v)
                    degrees[v] += 1
                    degrees[nb] += 1

        # гарантируем связность
        for i in range(n):
            if not self.adj[i]:
                j = (i+1) % n
                self.adj[i].append(j)
                self.adj[j].append(i)

        # каталитические веса: хабы усиливают сильнее
        degs = np.array([len(self.adj[i]) for i in range(n)], dtype=float)
        mean_d = max(degs.mean(), EPS)
        self.catalysis = np.clip((degs/mean_d)**eta, 0.5, 3.5)
        self.degrees = degs

    def partners(self, i: int, k: int, rng: np.random.Generator) -> np.ndarray:
        nb = self.adj[i]
        if not nb:
            return np.array([i]*k, dtype=int)
        if len(nb) >= k:
            return rng.choice(nb, k, replace=False)
        return rng.choice(nb, k, replace=True)


# ══════════════════════════════════════════════════════════
# NPG — ИСПРАВЛЕННАЯ МЕТРИКА
# ══════════════════════════════════════════════════════════

class NPGMetric:
    """
    Normalized Performance Gain — честная версия.

    NPG = (F_baseline - F_current) / (F_baseline + eps)

    F = свободная энергия = -log P(выживание | params)
      ≈ -mean(A_tail) * log(surv_rate + eps)

    Baseline: та же система с floor=0, kappa=0.
    """

    @staticmethod
    def free_energy(history: List[float], survived: bool) -> float:
        if not history:
            return 10.0
        tail = history[-50:] if len(history) >= 50 else history
        mean_tail = float(np.mean(tail))
        # F ≈ -log P ≈ -(mean_tail + log(surv + eps))
        surv_term = np.log(float(survived) + EPS)
        return float(-(mean_tail + surv_term))

    @staticmethod
    def npg(F_model: float, F_baseline: float) -> float:
        return float((F_baseline - F_model) / (abs(F_baseline) + EPS))

    @staticmethod
    def variance_npg_old(history_model: List[float],
                          history_base: List[float]) -> float:
        """Старая (сломанная) версия для сравнения."""
        tail_m = np.var(history_model[-20:]) if len(history_model) >= 20 else 0.05
        tail_b = np.var(history_base[-20:])  if len(history_base)  >= 20 else 0.05
        return float((tail_b - tail_m) / (tail_b + EPS))


# ══════════════════════════════════════════════════════════
# ФАЗОВЫЙ ДЕТЕКТОР (L0→L3)
# ══════════════════════════════════════════════════════════

class PhaseDetector:
    """
    Определяет фазу системы НЕ по порогам mean(A),
    а по динамической устойчивости.

    Jacobian линеаризованного TSV вокруг фиксированной точки A*:
      dΔA/dA_i ≈ α·A_j·(1-2·A_i) - δ·(1-0.3·A_i) + ...

    Eigenvalue λ:
      λ > 0: неустойчивая точка (L0-хаос, рост возможен)
      λ < 0: устойчивая точка  (L1-L3, система закреплена)
      |λ| → 0: критический переход (TippingPoint)
    """

    @staticmethod
    def local_jacobian(A_i: float, A_j_mean: float,
                       p: UAFv5Params) -> float:
        """
        Диагональный элемент Jacobian для агента i.
        Floor исключён: это внешняя накачка, не внутренняя динамика.
        λ > 0 → рост (chaos), λ < 0 → затухание (coherent/integrated).
        Граница λ=0 = критическая точка (TippingPoint).
        """
        tsv_term   = p.alpha_social * A_j_mean * (1 - 2*A_i)
        decay_term = -p.decay * (1 - 0.3*A_i)
        return tsv_term + decay_term

    @classmethod
    def phase(cls, A: np.ndarray, p: UAFv5Params) -> Tuple[str, float]:
        """
        Возвращает (фаза, λ_mean).
        Фазы: 'chaos' | 'transition' | 'coherent' | 'integrated'
        """
        mean_A = float(np.mean(A))
        A_j_mean = mean_A  # упрощение: средний сосед = mean

        lambdas = np.array([
            cls.local_jacobian(float(ai), A_j_mean, p)
            for ai in A
        ])
        lam = float(np.mean(lambdas))

        # Фаза по Jacobian (не по порогу!)
        std_A = float(np.std(A))
        if lam > 0.01:
            phase = 'chaos'        # расширяется, нет устойчивости
        elif lam > -0.005:
            phase = 'transition'   # около критической точки
        elif std_A > 0.05:
            phase = 'coherent'     # устойчива, но неоднородна
        else:
            phase = 'integrated'   # устойчива и однородна

        return phase, lam

    @staticmethod
    def level_label(phase: str) -> str:
        return {
            'chaos':      'L0-хаос',
            'transition': 'L1-переход',
            'coherent':   'L2-когерентность',
            'integrated': 'L3-интеграция',
        }.get(phase, '?')


# ══════════════════════════════════════════════════════════
# МАСТЕР-УРАВНЕНИЕ — ЕДИНЫЙ ШАГ
# ══════════════════════════════════════════════════════════

def uaf_step(A: np.ndarray,
             topo: BATopology,
             precision: np.ndarray,
             p: UAFv5Params,
             rng: np.random.Generator,
             step: int) -> Tuple[np.ndarray, Dict]:
    """
    Один шаг мастер-уравнения UAF v5.

    Возвращает новый A и словарь метрик шага.
    """
    N = len(A)
    dA = np.zeros(N)

    # N//2 пар = каждый агент участвует каждый шаг (как в 025e wrap-around)
    n_pairs = max(1, N // 2)
    all_idx = rng.permutation(N)

    for pp in range(n_pairs):
        i = int(all_idx[(2*pp) % N])
        j = int(all_idx[(2*pp+1) % N])

        Ai, Aj = A[i], A[j]
        C_ij = topo.catalysis[j]   # каталитический вес соседа j
        C_ji = topo.catalysis[i]

        # ── TSV член ───────────────────────────────────────────
        tsv_i = p.alpha_social * C_ij * Aj * (1 - Ai)
        tsv_j = p.alpha_social * C_ji * Ai * (1 - Aj)

        # cost (симметричный, из 025e)
        cost_i = p.epsilon_cost * Ai * Aj * (1 - Aj)
        cost_j = p.epsilon_cost * Aj * Ai * (1 - Ai)

        # ── FEP член (precision-weighted PE) ───────────────────
        # PE_i = (Aj - Ai): агент i хочет быть как более точный j
        pe_i = max(0.0, Aj - Ai)   # только положительный сигнал
        fep_i = p.alpha_learn * precision[i] * pe_i * (1 - Ai)
        pe_j = max(0.0, Ai - Aj)
        fep_j = p.alpha_learn * precision[j] * pe_j * (1 - Aj)

        dA[i] += tsv_i - cost_i + fep_i
        dA[j] += tsv_j - cost_j + fep_j

    # ── Decay ──────────────────────────────────────────────────
    dA -= p.decay * (1.0 - p.decay_mod * A)

    # ── Basal floor ────────────────────────────────────────────
    if p.floor_adaptive:
        floor_eff = p.floor * np.maximum(0.0, 1.0 - A / p.a_ceiling)
    else:
        floor_eff = np.full(N, p.floor)
    dA += floor_eff

    # ── CPS (homeostat, KAPPA) ─────────────────────────────────
    # Активен только если далеко от цели И kappa > 0
    if p.kappa > 0:
        deviation = A - p.a_target
        cps_mask = np.abs(deviation) > p.kappa_zone
        dA -= p.kappa * deviation * cps_mask

    # ── Обновление precision (FEP) ─────────────────────────────
    # Precision растёт там где PE мала (агент хорошо предсказывает)
    mean_A = float(np.mean(A))
    pe_global = np.abs(A - mean_A)
    precision_new = np.clip(precision + 0.01*(0.5 - pe_global), 0.1, 3.0)

    # ── Применение ─────────────────────────────────────────────
    A_new = np.clip(A + dA, 0.0, p.a_ceiling)

    # ── Метрики ────────────────────────────────────────────────
    phase, lam = PhaseDetector.phase(A_new, p)
    metrics = {
        "mean_A":    float(np.mean(A_new)),
        "std_A":     float(np.std(A_new)),
        "phase":     phase,
        "lambda":    lam,
        "mean_prec": float(np.mean(precision_new)),
        "floor_eff": float(np.mean(floor_eff)),
        "dA_tsv":    float(np.mean(np.abs(dA + p.decay*(1-p.decay_mod*A) - floor_eff))),
    }
    return A_new, precision_new, metrics


# ══════════════════════════════════════════════════════════
# СИМУЛЯТОР
# ══════════════════════════════════════════════════════════

class UAFv5System:
    """
    Полная система UAF v5.

    Объединяет:
    - TSV (социальная диффузия)
    - FEP (precision-weighted обучение)
    - Basal floor (метаболизм)
    - BA топология с catalysis
    - NPG честная метрика
    - PhaseDetector (Jacobian-based)
    """

    def __init__(self, params: Optional[UAFv5Params] = None, seed: int = 42):
        self.p = params or UAFv5Params()
        self.rng = np.random.default_rng(seed)
        self.topo = BATopology(self.p.n_agents, self.p.K,
                               self.p.alpha_cat, seed=seed)

        # Инициализация
        self.A = self.rng.uniform(self.p.a_init_low, self.p.a_init_high,
                                  self.p.n_agents)
        self.precision = np.ones(self.p.n_agents)

        self.history: List[Dict] = []
        self.tip_step: Optional[int] = None

    def step(self, t: int) -> Dict:
        self.A, self.precision, m = uaf_step(
            self.A, self.topo, self.precision, self.p, self.rng, t
        )
        m["step"] = t
        m["tip"] = self.tip_step

        if self.tip_step is None and m["mean_A"] >= self.p.a_crit:
            self.tip_step = t
            m["event"] = "TIPPING_POINT"
        else:
            m["event"] = ""

        self.history.append(m)
        return m

    def run(self, steps: int = 500, verbose_every: int = 0) -> List[Dict]:
        for t in range(steps):
            m = self.step(t)
            if verbose_every > 0 and (t % verbose_every == 0 or m["event"]):
                tipped = int(np.sum(self.A >= self.p.a_crit))
                print(f"  t={t:4d}: A={m['mean_A']:.4f} ±{m['std_A']:.4f} "
                      f"[{PhaseDetector.level_label(m['phase'])}] "
                      f"λ={m['lambda']:+.4f}  "
                      f"tipped={tipped}/{self.p.n_agents}"
                      + (f"  ← {m['event']}" if m['event'] else ""))
        return self.history

    def npg(self, baseline_history: List[float]) -> Dict:
        """Честный NPG vs baseline (floor=0, kappa=0)."""
        my_hist = [m["mean_A"] for m in self.history]
        survived = self.history[-1]["mean_A"] >= self.p.a_crit
        base_survived = baseline_history[-1] >= self.p.a_crit if baseline_history else False

        F_model = NPGMetric.free_energy(my_hist, survived)
        F_base  = NPGMetric.free_energy(baseline_history, base_survived)
        npg_val = NPGMetric.npg(F_model, F_base)

        # Старый NPG для сравнения
        old_npg = NPGMetric.variance_npg_old(my_hist, baseline_history)

        return {
            "NPG_v5":  npg_val,
            "NPG_old": old_npg,
            "F_model": F_model,
            "F_base":  F_base,
            "survived": survived,
            "tip_step": self.tip_step,
            "final_A":  self.history[-1]["mean_A"] if self.history else 0.0,
        }

    def phase_trajectory(self) -> List[str]:
        return [m["phase"] for m in self.history]

    def mean_A_history(self) -> List[float]:
        return [m["mean_A"] for m in self.history]
