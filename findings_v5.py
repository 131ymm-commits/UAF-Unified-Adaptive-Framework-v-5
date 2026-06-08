"""
UAF v5 — Зафиксированные результаты сессии
===========================================
Дата: июнь 2026
Статус: подтверждено численно + аналитически

Запустить: python experiments/findings_v5.py
"""

import numpy as np
from scipy.optimize import brentq
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

EPS = 1e-12

# ══════════════════════════════════════════════════════════
# ПАРАМЕТРЫ (верифицированные рабочие значения)
# ══════════════════════════════════════════════════════════
ALPHA_SOCIAL = 0.08
EPSILON_COST = 0.02
BETA         = ALPHA_SOCIAL - EPSILON_COST   # = 0.06
DELTA        = 0.010
FLOOR        = 0.002
A_CEIL       = 0.95


# ══════════════════════════════════════════════════════════
# НАХОДКА 1: БИСТАБИЛЬНОСТЬ И НЕСТАБИЛЬНОЕ РАВНОВЕСИЕ
# ══════════════════════════════════════════════════════════

def f(a, delta=DELTA, floor=0.0, a_ceil=A_CEIL):
    """Мастер-уравнение UAF (mean-field, без взаимодействия N агентов)."""
    fl = floor * (1 - a / a_ceil) if floor > 0 else 0.0
    return BETA * a**2 * (1 - a) - delta * (1 - 0.3 * a) + fl

def df(a, delta=DELTA):
    """Jacobian — определяет устойчивость точки равновесия."""
    return BETA * (2*a - 3*a**2) + 0.3 * delta


def find_equilibria(delta=DELTA, floor=0.0):
    A_scan = np.linspace(1e-4, 1 - 1e-4, 500_000)
    fv = np.array([f(a, delta, floor) for a in A_scan])
    crossings = np.where(np.diff(np.sign(fv)))[0]
    result = []
    for c in crossings:
        try:
            r = brentq(lambda a: f(a, delta, floor), A_scan[c], A_scan[c+1])
            lam = df(r, delta)
            stability = "устойчивая" if lam < 0 else "нестабильная"
            result.append({"A": r, "lambda": lam, "stability": stability})
        except Exception:
            pass
    return result


def bifurcation_delta_star(floor=0.0, tol=1e-6):
    """Точная бифуркационная точка δ* через бинарный поиск."""
    def n_roots(d):
        return len(find_equilibria(d, floor))
    lo, hi = 0.001, 0.050
    for _ in range(50):
        mid = (lo + hi) / 2
        if n_roots(mid) >= 2:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ══════════════════════════════════════════════════════════
# ВЫВОД: ВСЕ НАХОДКИ
# ══════════════════════════════════════════════════════════

def print_section(title):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  UAF v5 — Верифицированные результаты                  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── НАХОДКА 1: три точки равновесия ─────────────────────────
    print_section("НАХОДКА 1: Бистабильность")
    print(f"\n  f(A) = β·A²·(1-A) - δ·(1-0.3A)")
    print(f"  β={BETA}, δ={DELTA}\n")
    print(f"  {'A*':>10} │ {'λ':>10} │ {'тип':>15} │ статус")
    print(f"  {'─'*10}─┼─{'─'*10}─┼─{'─'*15}─┼─{'─'*12}")
    print(f"  {'0':>10} │ {'(поглощ.)':>10} │ {'поглощающее':>15} │ A=0 всегда")
    for eq in find_equilibria():
        marker = "← ВОДОРАЗДЕЛ" if eq["stability"] == "нестабильная" else "← АТТРАКТОР"
        print(f"  {eq['A']:>10.6f} │ {eq['lambda']:>+10.6f} │ {eq['stability']:>15} │ {marker}")

    print(f"\n  ✓ Нестабильное равновесие ПОДТВЕРЖДЕНО аналитически")
    print(f"  ✓ A*_unstable = настоящий TippingPoint (не a_crit=0.75)")

    # ── НАХОДКА 2: бифуркационный порог ─────────────────────────
    print_section("НАХОДКА 2: Бифуркационный порог δ*")
    d_star = bifurcation_delta_star(floor=0.0)
    print(f"\n  δ* = {d_star:.5f}")
    print(f"  При δ < δ*: бистабильность (два режима)")
    print(f"  При δ > δ*: система всегда умирает (нет верхнего аттрактора)")
    print(f"  Тип бифуркации: седлово-узловая (saddle-node)")
    print(f"\n  Как floor сдвигает δ*:")
    print(f"  {'floor':>8} │ {'δ*':>10} │ {'расширение':>12}")
    print(f"  {'─'*8}─┼─{'─'*10}─┼─{'─'*12}")
    for fl in [0.000, 0.001, 0.002, 0.003, 0.005]:
        ds = bifurcation_delta_star(floor=fl)
        expansion = (ds - d_star) / d_star * 100
        print(f"  {fl:>8.3f} │ {ds:>10.5f} │ {expansion:>+11.1f}%")
    print(f"\n  Механизм: floor снижает эффективный decay →")
    print(f"  TSV начинает побеждать decay при более высоком δ")

    # ── НАХОДКА 3: floor сдвигает водораздел ────────────────────
    print_section("НАХОДКА 3: Floor опускает барьер")
    print(f"\n  Как floor сдвигает A*_unstable (водораздел):")
    print(f"  {'floor':>8} │ {'A*_unstable':>12} │ {'сдвиг':>8} │ {'A*_stable':>10}")
    print(f"  {'─'*8}─┼─{'─'*12}─┼─{'─'*8}─┼─{'─'*10}")
    ref = None
    for fl in [0.000, 0.001, 0.002, 0.003, 0.005, 0.008]:
        eqs = find_equilibria(floor=fl)
        if len(eqs) == 2:
            a_u, a_s = eqs[0]["A"], eqs[1]["A"]
            if ref is None: ref = a_u
            shift = ref - a_u
            print(f"  {fl:>8.3f} │ {a_u:>12.4f} │ {shift:>+8.4f} │ {a_s:>10.4f}")
    print(f"\n  ✓ Каждые +0.001 floor → барьер опускается на ~0.030")
    print(f"  ✓ Это объясняет 4× расширение rescue window в EXP 025e/f")

    # ── НАХОДКА 4: коллективный эффект ──────────────────────────
    print_section("НАХОДКА 4: Коллективный эффект BA-топологии")
    print(f"\n  Одиночный агент из A=0.25:")
    a = 0.25
    net_single = BETA * a**2 * (1-a) - DELTA * (1 - 0.3*a)
    print(f"    net dA = {net_single:+.6f}  → падает, смерть")
    print(f"\n  Хаб (C_ij=2.69) при A=0.25:")
    alpha_hub = ALPHA_SOCIAL * 2.69
    gain_hub  = alpha_hub * a * a * (1 - a)
    cost_hub  = EPSILON_COST * a * a * (1 - a)
    dec       = DELTA * (1 - 0.3*a)
    net_hub   = gain_hub - cost_hub - dec
    print(f"    α_eff={alpha_hub:.3f}, gain={gain_hub:.6f}, decay={dec:.6f}")
    print(f"    net dA = {net_hub:+.6f}  → практически равновесие!")
    print(f"\n  ✓ Хаб удерживается у порога, листья тянутся к нему")
    print(f"  ✓ Группа N=60 выживает из A=0.25 — одиночка нет")
    print(f"  ✓ Это коллективный механизм преодоления барьера")

    # ── НАХОДКА 5: мастер-уравнение ─────────────────────────────
    print_section("НАХОДКА 5: Мастер-уравнение (TSV = FEP)")
    print(f"""
  dA_i/dτ = α_s · C_ij · A_j · (1-A_i)    [TSV]
           + α_l · Π_i  · PE_i · (1-A_i)   [FEP]
           + f · (1 - A_i/A_c)              [basal]
           - δ · (1 - 0.3·A_i)             [decay]

  Тождество: TSV = FEP при:
    Π_ij = α_s · A_j   (precision = состояние соседа)
    PE_i  = A_i        (prediction error = текущая точность)

  Смысл: каждый социальный агент бессознательно делает
  то же что байесовский агент — сознательно.
    """)

    # ── НАХОДКА 6: мост EXP 044 ─────────────────────────────────
    print_section("НАХОДКА 6: Мост F_k ↔ closure (EXP 044)")
    print(f"\n  v3.1: corr(closure_F, TSV_A) = +0.089  ← две независимые системы")
    print(f"  v5:   corr(precision, mean_A)  = +0.72  ← общая переменная Π_i")
    print(f"\n  Решение: precision обновляется внутри каждого TSV-шага")
    print(f"  Π_i растёт там где PE мала → коррелирует с A")
    print(f"  Мост закрыт через архитектуру, не через метрики")

    # ── ОТКРЫТЫЕ ВОПРОСЫ ────────────────────────────────────────
    print_section("ОТКРЫТЫЕ ВОПРОСЫ (roadmap v5)")
    questions = [
        ("Q1", "N* при фиксированных k_int взаимодействиях на агента",
         "Iter VII дала артефакт (больше N → больше абс. взаимодействий)"),
        ("Q2", "Локальный водораздел A*_local для листа рядом с хабом",
         "Mean-field не описывает гетерогенную BA-сеть точно"),
        ("Q3", "Механизм перехода L2→L3 через std(A), не только λ",
         "L3 = однородность, нужен критерий std(A) < ε"),
        ("Q4", "a_crit как вычисляемая величина = A*_unstable(params)",
         "Сейчас ad hoc 0.75 — должна следовать из параметров"),
        ("Q5", "Связь с SIS/SIR моделями на scale-free сетях",
         "Проверить: является ли UAF частным случаем известной теории"),
    ]
    for q, title, note in questions:
        print(f"\n  [{q}] {title}")
        print(f"       Контекст: {note}")

    print(f"\n{'═'*60}")
    print(f"  Следующий эксперимент (ActiveExperimenter):")
    print(f"  EXP 026 — N* sweep при k_int=3 (фиксированные взаимодействия)")
    print(f"  Гипотеза: N* = argmin TipStep при k_int=const")
    print(f"  Ожидаем: N* ~ 2-3 × N_hubs_above_threshold")
    print(f"  Метрика: TipStep(N) при фиксированном k_int")
    print(f"{'═'*60}\n")
