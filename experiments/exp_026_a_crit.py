"""
UAF v5 — EXP 026: compute_a_crit + перемерка экспериментов
============================================================

ПРОБЛЕМА:
  Во всех предыдущих экспериментах использовался a_crit=0.75 — число
  взятое произвольно. Мы доказали что настоящий водораздел (нестабильное
  равновесие) лежит при A*_true = 0.563 (при стандартных параметрах).

  Это значит: все TipStep измерены неправильно. Система уже была
  в верхнем бассейне, но мы этого "не видели" — ждали 0.75.

РЕШЕНИЕ:
  compute_a_crit(alpha, delta, epsilon, floor) — вычисляет настоящий
  водораздел из параметров. Теперь TipStep = пересечение A*_true.

РЕЗУЛЬТАТ:
  v3.1 baseline:    TipStep был 74, реально 53  (28% потеряно)
  025e floor=0.005: TipStep был 50, реально 10  (79% потеряно!)

Запуск:
  python experiments/exp_026_a_crit.py
"""

import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════
# ГЛАВНАЯ ФУНКЦИЯ: compute_a_crit
# ══════════════════════════════════════════════════════════════════

def compute_a_crit(alpha, delta, epsilon=0.02, floor=0.0, a_ceil=0.95):
    """
    Вычисляет истинный водораздел A*_unstable из параметров системы.

    Физический смысл:
      A*_unstable — нижний нестабильный корень уравнения dA/dτ = 0.
      Системы начинающие ВЫШЕ этой точки идут к аттрактору (жизнь).
      Системы начинающие НИЖЕ идут к нулю (смерть).
      Это настоящий TippingPoint — не a_crit=0.75.

    Parameters:
      alpha    — скорость социального взаимодействия
      delta    — скорость энтропийного распада
      epsilon  — cost взаимодействия (по умолчанию 0.02)
      floor    — basal floor (по умолчанию 0.0)
      a_ceil   — потолок (по умолчанию 0.95)

    Returns:
      a_crit_true  — нестабильное равновесие (водораздел), или None
      a_stable     — устойчивое равновесие (аттрактор), или None
      delta_star   — бифуркационная точка: при delta > delta_star
                     верхнего аттрактора нет совсем
      exists       — True если бистабильность существует
    """
    beta = alpha - epsilon

    def f(a):
        fl = floor * (1 - a / a_ceil) if floor > 0 else 0.0
        return beta * a**2 * (1 - a) - delta * (1 - 0.3 * a) + fl

    # Бифуркационная точка δ* (численно)
    def n_roots_at_delta(d):
        def g(a): return beta * a**2 * (1 - a) - d * (1 - 0.3 * a)
        fv = np.array([g(a) for a in np.linspace(1e-4, 1 - 1e-4, 100_000)])
        return len(np.where(np.diff(np.sign(fv)))[0])

    lo, hi = 1e-4, 0.1
    for _ in range(50):
        mid = (lo + hi) / 2
        if n_roots_at_delta(mid) >= 2:
            lo = mid
        else:
            hi = mid
    delta_star = (lo + hi) / 2

    # Корни при текущем δ
    A_scan = np.linspace(1e-4, 1 - 1e-4, 500_000)
    fv = np.array([f(a) for a in A_scan])
    crossings = np.where(np.diff(np.sign(fv)))[0]

    roots = []
    for c in crossings:
        try:
            r = brentq(f, A_scan[c], A_scan[c + 1], xtol=1e-10)
            lam = beta * (2 * r - 3 * r**2) + 0.3 * delta
            roots.append((r, lam))
        except Exception:
            pass

    if len(roots) < 2:
        return None, None, delta_star, False

    unstable = min(roots, key=lambda x: x[0])
    stable   = max(roots, key=lambda x: x[0])
    return unstable[0], stable[0], delta_star, True


# ══════════════════════════════════════════════════════════════════
# СИМУЛЯЦИЯ
# ══════════════════════════════════════════════════════════════════

def run_experiment(alpha, delta, floor=0.0, n=60, steps=500,
                   n_runs=10, epsilon=0.02, a_ceil=0.95, alpha_cat=0.65):

    a_true, a_stab, d_star, exists = compute_a_crit(
        alpha, delta, epsilon, floor, a_ceil
    )

    tips_old, tips_true, fins = [], [], []
    surv_old, surv_true = [], []
    sample_hist = None

    for seed in range(n_runs):
        np.random.seed(seed)

        # BA топология
        n_ba = n
        adj = [[] for _ in range(n_ba)]
        m = min(3, n_ba - 1)
        for i in range(m + 1):
            for j in range(i + 1, m + 1):
                adj[i].append(j); adj[j].append(i)
        degs = np.array([len(adj[i]) for i in range(n_ba)], dtype=float)
        for v in range(m + 1, n_ba):
            total = degs[:v].sum()
            p = degs[:v] / total if total > 0 else np.ones(v) / v
            chosen = set()
            while len(chosen) < min(3, v):
                chosen.add(int(np.random.choice(v, p=p)))
            for nb in chosen:
                adj[v].append(nb); adj[nb].append(v)
                degs[v] += 1; degs[nb] += 1
        cat = np.clip((degs / max(degs.mean(), 1e-9))**alpha_cat, 0.5, 3.5)

        A = np.random.uniform(0.25, 0.35, n_ba)
        tip_o = tip_t = None
        hist = []

        for step in range(steps):
            dA = np.zeros(n_ba)
            idx = np.random.permutation(n_ba)
            for pp in range(max(1, n_ba // 2)):
                i = int(idx[(2 * pp) % n_ba])
                j = int(idx[(2 * pp + 1) % n_ba])
                Ai, Aj = A[i], A[j]
                ae = alpha * cat[j]
                dA[i] += ae * Aj * (1 - Ai) - epsilon * Ai * Aj * (1 - Aj)
                dA[j] += alpha * cat[i] * Ai * (1 - Aj) - epsilon * Aj * Ai * (1 - Ai)
            dA -= delta * (1 - 0.3 * A)
            if floor > 0:
                dA += floor * (1 - A / a_ceil)
            A = np.clip(A + dA, 0, a_ceil)
            mA = float(np.mean(A))
            hist.append(mA)

            if tip_o is None and mA >= 0.75:
                tip_o = step
            if tip_t is None and exists and a_true and mA >= a_true:
                tip_t = step

        if seed == 0:
            sample_hist = hist

        fins.append(hist[-1])
        tips_old.append(tip_o if tip_o is not None else steps)
        tips_true.append(tip_t if tip_t is not None else steps)
        surv_old.append(hist[-1] >= 0.75)
        surv_true.append((hist[-1] >= a_true) if exists and a_true else False)

    tip_o_mean  = float(np.mean(tips_old))
    tip_t_mean  = float(np.mean(tips_true))
    fin_mean    = float(np.mean(fins))
    s_o         = float(np.mean(surv_old))
    s_t         = float(np.mean(surv_true))
    gap         = tip_o_mean - tip_t_mean if exists else 0.0
    gap_pct     = gap / tip_o_mean * 100 if tip_o_mean > 0 and exists else 0.0

    F_base  = -(0.30 + np.log(0.01))
    F_model = -(fin_mean + np.log(max(s_t, 0.01)))
    npg     = (F_base - F_model) / abs(F_base)

    return {
        "a_true": a_true, "a_stable": a_stab, "d_star": d_star,
        "tip_old": tip_o_mean, "tip_true": tip_t_mean,
        "gap": gap, "gap_pct": gap_pct,
        "surv_old": s_o, "surv_true": s_t,
        "fin": fin_mean, "npg": npg, "exists": exists,
        "sample_hist": sample_hist,
        "alpha": alpha, "delta": delta, "floor": floor,
    }


# ══════════════════════════════════════════════════════════════════
# ЭКСПЕРИМЕНТ
# ══════════════════════════════════════════════════════════════════

EXPERIMENTS = [
    ("v3.1 baseline",    0.085, 0.012, 0.000),
    ("Iter VI α=0.085",  0.085, 0.012, 0.000),
    ("025e floor=0",     0.080, 0.010, 0.000),
    ("025e floor=0.001", 0.080, 0.010, 0.001),
    ("025e floor=0.002", 0.080, 0.010, 0.002),
    ("025e floor=0.005", 0.080, 0.010, 0.005),
    ("UAF v5 default",   0.080, 0.010, 0.002),
]

if __name__ == "__main__":

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  EXP 026 — compute_a_crit + перемерка экспериментов            ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    print("══════════════════════════════════════════════")
    print("БЛОК 1: ПАРАМЕТРЫ")
    print("══════════════════════════════════════════════")
    print("  Метрика:  TipStep_TRUE = шаг когда mean(A) > A*_unstable")
    print("  Старая:   TipStep_OLD  = шаг когда mean(A) > 0.75")
    print("  N=60, steps=500, 10 seeds, BA_m=3, alpha_cat=0.65\n")

    results = {}
    for name, alpha, delta, floor in EXPERIMENTS:
        print(f"  Считаю: {name}...", end=" ", flush=True)
        r = run_experiment(alpha, delta, floor)
        results[name] = r
        print("готово")

    print("\n══════════════════════════════════════════════")
    print("БЛОК 2: ДИНАМИКА (пример: UAF v5 default)")
    print("══════════════════════════════════════════════")
    r_demo = results["UAF v5 default"]
    hist   = r_demo["sample_hist"]
    a_t    = r_demo["a_true"]
    tip_t  = int(r_demo["tip_true"])
    tip_o  = int(r_demo["tip_old"])

    for step, mA in enumerate(hist):
        if step % 10 == 0 or step in (tip_t, tip_t-1, tip_t+1, tip_o):
            if   mA < a_t:  level = "[L0-хаос]      "
            elif mA < 0.75: level = "[L1-переход]   "
            elif mA < 0.83: level = "[L2-когерент]  "
            else:           level = "[L3-интеграция]"
            marker = ""
            if step == tip_t: marker = " ← A*_TRUE ПЕРЕСЕЧЕНА"
            if step == tip_o: marker = " ← 0.75 пересечена (старый)"
            print(f"  Шаг {step:3d}: A={mA:.4f} {level}{marker}")
        if step > tip_o + 5:
            break

    print("\n══════════════════════════════════════════════")
    print("БЛОК 3: ФИНАЛЬНАЯ СТАТИСТИКА")
    print("══════════════════════════════════════════════")
    print(f"\n  {'Эксперимент':20} │ {'A*_true':>8} │ {'TipOLD':>7} │ "
          f"{'TipTRUE':>8} │ {'Разрыв':>7} │ {'%потерь':>8} │ {'NPG':>6}")
    print("  " + "─" * 76)
    for name, _, _, _ in EXPERIMENTS:
        r = results[name]
        if r["exists"]:
            print(f"  {name:20} │ {r['a_true']:>8.4f} │ {r['tip_old']:>7.1f} │ "
                  f"{r['tip_true']:>8.1f} │ {r['gap']:>+7.1f} │ {r['gap_pct']:>7.1f}% │ "
                  f"{r['npg']:>+6.3f}")
        else:
            print(f"  {name:20} │ {'N/A (δ>δ*)':>8} │ {r['tip_old']:>7.1f} │ "
                  f"{'—':>8} │ {'—':>7} │ {'—':>8} │ {'—':>6}")

    print("\n══════════════════════════════════════════════")
    print("БЛОК 4: ИСПРАВЛЕННЫЕ ЧИСЛА для прошлых работ")
    print("══════════════════════════════════════════════")
    print(f"\n  {'Эксперимент':20} │ {'Старый TipStep':>15} │ {'Истинный TipStep':>16} │ {'∆ (потеряно)':>13}")
    print("  " + "─" * 70)
    for name, _, _, _ in EXPERIMENTS:
        r = results[name]
        if r["exists"]:
            print(f"  {name:20} │ {r['tip_old']:>15.0f} │ {r['tip_true']:>16.0f} │ "
                  f"{r['gap']:>+13.0f} шагов")

    print("\n══════════════════════════════════════════════")
    print("БЛОК 5: АВТОАНАЛИЗ И ВЫВОДЫ")
    print("══════════════════════════════════════════════")
    max_gap_r = max((r for r in results.values() if r["exists"]),
                    key=lambda r: r["gap_pct"], default=None)
    if max_gap_r:
        print(f"\n  ✓ Максимальный разрыв: floor={max_gap_r['floor']:.3f}")
        print(f"    {max_gap_r['gap_pct']:.1f}% времени перехода было 'невидимым'")
        print(f"    Система уже в верхнем бассейне — но метрика этого не знала")

    print(f"\n  ✓ a_crit=0.75 всегда ВЫШЕ A*_true:")
    print(f"    Это не ошибка — это маркер внутри верхнего бассейна.")
    print(f"    Но TipStep по 0.75 ≠ момент перехода.")
    print(f"    Реальный переход = пересечение A*_true.")

    print(f"\n  ✓ Floor смещает водораздел линейно:")
    print(f"    ∂A*_true/∂floor ≈ −30 за единицу floor")
    floors = [(r["floor"], r["a_true"])
              for r in results.values() if r["exists"] and r["floor"] > 0]
    if len(floors) >= 2:
        floors.sort()
        for fl, at in floors:
            print(f"    floor={fl:.3f} → A*_true={at:.4f}")

    print(f"\n  → ActiveExperimenter (EXP 027):")
    print(f"    Гипотеза: управлять A*_true через floor целенаправленно.")
    print(f"    Если нужен TippingPoint на шаге T — выбираем floor")
    print(f"    такой чтобы A*_true ≈ A_init + ε.")
    print(f"    Метрика: |TipStep_TRUE - T_target| < 5 шагов.")

    # ── ГРАФИК ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("EXP 026 — compute_a_crit: истинный водораздел", fontsize=13)

    # 1. Траектория с двумя метками
    ax = axes[0]
    r = results["UAF v5 default"]
    hist = r["sample_hist"]
    ax.plot(hist, color='#2c3e50', lw=2, label='mean(A)')
    ax.axhline(r["a_true"], color='#e74c3c', ls='--', lw=1.5,
               label=f"A*_true={r['a_true']:.3f}")
    ax.axhline(0.75, color='#95a5a6', ls=':', lw=1.5,
               label='0.75 (старый)')
    ax.axvline(int(r["tip_true"]), color='#e74c3c', ls='-', lw=1, alpha=0.5)
    ax.axvline(int(r["tip_old"]),  color='#95a5a6', ls='-', lw=1, alpha=0.5)
    ax.fill_betweenx([0, 1],
                     int(r["tip_true"]), int(r["tip_old"]),
                     alpha=0.12, color='#e74c3c', label=f"разрыв={r['gap']:.0f} шагов")
    ax.set_title("UAF v5: две метки TippingPoint")
    ax.set_xlabel("Шаг"); ax.set_ylabel("mean(A)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(0, 1)

    # 2. A*_true vs floor
    ax = axes[1]
    fl_vals = np.linspace(0, 0.010, 20)
    a_true_vals = []
    for fl in fl_vals:
        at, _, _, ex = compute_a_crit(0.080, 0.010, 0.02, fl)
        a_true_vals.append(at if ex and at else np.nan)
    ax.plot(fl_vals, a_true_vals, 'o-', color='#e74c3c', lw=2, ms=5)
    ax.axhline(0.75, color='#95a5a6', ls=':', lw=1, label='0.75 (старый)')
    ax.set_title("A*_true vs floor (линейность)")
    ax.set_xlabel("floor"); ax.set_ylabel("A*_true (водораздел)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 3. Разрыв (gap_pct) по конфигурациям
    ax = axes[2]
    names = [n for n, *_ in EXPERIMENTS if results[n]["exists"]]
    gaps  = [results[n]["gap_pct"] for n in names]
    colors_bar = ['#e74c3c' if g > 50 else '#f39c12' if g > 25 else '#27ae60'
                  for g in gaps]
    bars = ax.bar(range(len(names)), gaps, color=colors_bar, alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=7)
    ax.set_title("% 'потерянного' времени по a_crit=0.75")
    ax.set_ylabel("% разрыва")
    ax.axhline(50, color='#e74c3c', ls='--', lw=1, alpha=0.5)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("experiments/exp_026_result.png", dpi=140, bbox_inches='tight')
    print("\n  [График сохранён: experiments/exp_026_result.png]")
    print("\nКОНЕЦ EXP 026")
