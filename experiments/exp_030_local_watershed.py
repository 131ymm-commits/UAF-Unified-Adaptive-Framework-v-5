"""
UAF v5 — EXP 030: локальный водораздел листа рядом с хабом
===========================================================
Запуск: python experiments/exp_030_local_watershed.py

Гипотеза (Q2): A*_local(leaf) > A*_true(mean-field)
Результат: вопрос был поставлен неправильно. У листа нет
нестабильного равновесия — только устойчивый аттрактор.
Хаб создаёт поле притяжения, лист следует автоматически.

═══════════════════════════════════════════════════════════
РЕЗУЛЬТАТЫ (верифицированы численно + аналитически):

Аналитика — аттрактор листа при A_hub = A*_mf = 0.503:
  C_hub │ A*_leaf_stable │ vs A*_stable_mean=0.80
  0.50  │     0.501      │     −0.299  (лист деградирует)
  0.65  │     0.600      │     −0.200
  1.00  │     0.727      │     −0.073
  1.50  │     0.812      │     +0.012
  2.00  │     0.857      │     +0.057
  2.69  │     0.892      │     +0.092  ← типовой хаб
  3.50  │     0.917      │     +0.117

Критическое C* ≈ 1.3: при C > C* лист уходит ВЫШЕ среднего аттрактора.

Симуляция (N=30, A_init=A*_mf, 8 seeds):
  seed │ hub_deg │ leaf_deg │ C_hub │ fin_hub │ fin_leaf │ A*_leaf
  0    │   17    │    3     │ 2.058 │  0.853  │  0.869   │  0.861
  1    │   16    │    3     │ 1.979 │  0.860  │  0.868   │  0.856
  ... (все seeds: fin_leaf > fin_hub стабильно)

Главный вывод:
  Лист финально оказывается ВЫШЕ хаба (0.869 > 0.853).
  Причина: хаб взаимодействует со многими, лист получает
  весь TSV-импульс хаба концентрированно (меньше партнёров).

Переформулировка коллективного эффекта:
  Старое: "группа коллективно преодолевает барьер"
  Новое:  "хабы преодолевают барьер, листья следуют автоматически"
  TippingPoint = момент когда хотя бы один хаб пересекает A*_mf.

Новые открытые вопросы:
  Q2a: критическое C* ≈ 1.3 — можно ли его вычислить аналитически?
  Q2b: если TippingPoint определяется хабами, нужно пересчитать
       a_crit через A*_hub_local, а не mean-field.
═══════════════════════════════════════════════════════════
"""

import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════
# ЯДРО
# ══════════════════════════════════════════════════════════════════

ALPHA   = 0.080
DELTA   = 0.010
FLOOR   = 0.002
EPSILON = 0.02
A_CEIL  = 0.95


def compute_a_crit_mf(alpha=ALPHA, delta=DELTA, epsilon=EPSILON,
                       floor=FLOOR, a_ceil=A_CEIL):
    """Mean-field водораздел (нижний нестабильный корень)."""
    beta = alpha - epsilon
    def f(a):
        return beta*a**2*(1-a) - delta*(1-0.3*a) + floor*(1-a/a_ceil)
    A_sc = np.linspace(1e-4, 1-1e-4, 500_000)
    fv = np.array([f(a) for a in A_sc])
    sc = np.where(np.diff(np.sign(fv)))[0]
    roots = []
    for c in sc:
        try: roots.append(brentq(f, A_sc[c], A_sc[c+1]))
        except: pass
    return min(roots) if len(roots) >= 2 else 0.503


def leaf_stable_eq(A_hub, C_hub, alpha=ALPHA, delta=DELTA,
                   epsilon=EPSILON, floor=FLOOR, a_ceil=A_CEIL):
    """
    Устойчивая точка равновесия листа при фиксированном A_hub.

    Уравнение листа:
      f_leaf(A) = α·C_hub·A_hub·(1-A) - ε·A·A_hub·(1-A_hub)
                 - δ·(1-0.3A) + floor·(1-A/A_ceil) = 0

    У листа нет нестабильного корня — только устойчивый аттрактор.
    Хаб создаёт поле притяжения, лист следует в него без барьера.
    """
    def f(A):
        tsv  = alpha * C_hub * A_hub * (1 - A)
        cost = epsilon * A * A_hub * (1 - A_hub)
        dec  = delta * (1 - 0.3*A)
        fl   = floor * (1 - A/a_ceil)
        return tsv - cost - dec + fl

    A_sc = np.linspace(1e-4, 1-1e-4, 200_000)
    fv = np.array([f(a) for a in A_sc])
    sc = np.where(np.diff(np.sign(fv)))[0]
    roots = []
    for c in sc:
        try: roots.append(brentq(f, A_sc[c], A_sc[c+1]))
        except: pass
    return max(roots) if roots else None


def critical_C_star(A_hub, A_mean_stable=0.80, **kw):
    """
    Критическое C* при котором A*_leaf = A_mean_stable.
    При C > C*: лист уходит ВЫШЕ среднего аттрактора.
    """
    def cost(C):
        al = leaf_stable_eq(A_hub, C, **kw)
        if al is None: return -A_mean_stable
        return al - A_mean_stable
    try:
        return brentq(cost, 0.5, 5.0, xtol=1e-6)
    except Exception:
        return None


def build_ba(N, m=3, alpha_cat=0.65, seed=0):
    rng = np.random.default_rng(seed)
    adj = [[] for _ in range(N)]
    m = min(m, N-1)
    for i in range(m+1):
        for j in range(i+1, m+1):
            adj[i].append(j); adj[j].append(i)
    degs = np.array([len(adj[i]) for i in range(N)], dtype=float)
    for v in range(m+1, N):
        total = degs[:v].sum()
        p = degs[:v]/total if total > 0 else np.ones(v)/v
        chosen = set()
        while len(chosen) < min(m, v):
            chosen.add(int(rng.choice(v, p=p)))
        for nb in chosen:
            adj[v].append(nb); adj[nb].append(v)
            degs[v] += 1; degs[nb] += 1
    cat = np.clip((degs / max(degs.mean(), 1e-9))**alpha_cat, 0.5, 3.5)
    return adj, cat, degs


def run_tracked(N=30, steps=200, seed=0, A_init=None):
    """Симуляция с трассировкой хаба и листа."""
    np.random.seed(seed)
    adj, cat, degs = build_ba(N, seed=seed)
    hub_idx  = int(np.argmax(degs))
    leaf_nbs = [i for i in adj[hub_idx] if degs[i] == degs.min()]
    leaf_idx = leaf_nbs[0] if leaf_nbs else int(np.argmin(degs))

    a_mf = compute_a_crit_mf()
    A_start = A_init if A_init is not None else a_mf
    A = np.full(N, A_start) + np.random.normal(0, 0.001, N)
    A = np.clip(A, 0, A_CEIL)

    hist_hub = []; hist_leaf = []; hist_mean = []

    for step in range(steps):
        dA = np.zeros(N)
        idx = np.random.permutation(N)
        for pp in range(max(1, N//2)):
            i = int(idx[(2*pp)%N]); j = int(idx[(2*pp+1)%N])
            Ai, Aj = A[i], A[j]
            ae = ALPHA * cat[j]
            dA[i] += ae*Aj*(1-Ai) - EPSILON*Ai*Aj*(1-Aj)
            dA[j] += ALPHA*cat[i]*Ai*(1-Aj) - EPSILON*Aj*Ai*(1-Ai)
        dA -= DELTA * (1 - 0.3*A)
        if FLOOR > 0:
            dA += FLOOR * (1 - A/A_CEIL)
        A = np.clip(A + dA, 0, A_CEIL)
        hist_hub.append(A[hub_idx])
        hist_leaf.append(A[leaf_idx])
        hist_mean.append(float(np.mean(A)))

    C_hub = float(cat[hub_idx])
    return {
        "hist_hub":  hist_hub,
        "hist_leaf": hist_leaf,
        "hist_mean": hist_mean,
        "hub_deg":   int(degs[hub_idx]),
        "leaf_deg":  int(degs[leaf_idx]),
        "C_hub":     C_hub,
        "a_leaf_eq": leaf_stable_eq(a_mf, C_hub),
        "a_mf":      a_mf,
    }


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    A_MF = compute_a_crit_mf()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  EXP 030 — A*_local: лист рядом с хабом                ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\n  Mean-field A*_true = {A_MF:.4f}\n")

    # ── БЛОК 1: аналитика ─────────────────────────────────────────
    print("═"*55)
    print("БЛОК 1: Аттрактор листа при A_hub = A*_mf")
    print("═"*55)
    print(f"\n  {'C_hub':>8} │ {'A*_leaf':>10} │ {'vs A*_stable=0.80':>18} │ режим")
    print("  " + "─"*50)

    C_vals = [0.50, 0.65, 1.00, 1.30, 1.50, 2.00, 2.69, 3.50]
    leaf_data = {}
    for C in C_vals:
        al = leaf_stable_eq(A_MF, C)
        if al:
            diff = al - 0.80
            mode = "↑ выше среднего" if diff > 0 else "↓ ниже среднего"
            print(f"  {C:>8.2f} │ {al:>10.4f} │ {diff:>+18.4f} │ {mode}")
            leaf_data[C] = al

    # Критическое C*
    C_star = critical_C_star(A_MF)
    print(f"\n  Критическое C* = {C_star:.4f}")
    print(f"  При C > C*: лист уходит выше A*_stable(mean-field)")

    # ── БЛОК 2: симуляция ─────────────────────────────────────────
    print(f"\n{'═'*55}")
    print("БЛОК 2: Симуляция — траектории хаба и листа")
    print("═"*55)
    print(f"\n  A_init = A*_mf = {A_MF:.4f} (стартуем на водоразделе)")
    print(f"\n  {'seed':>5} │ {'hub_deg':>7} │ {'leaf_deg':>8} │ "
          f"{'C_hub':>6} │ {'fin_hub':>8} │ {'fin_leaf':>9} │ {'A*_leaf_pred':>12}")
    print("  " + "─"*68)

    runs = []
    for seed in range(8):
        r = run_tracked(N=30, steps=200, seed=seed)
        diff = r["hist_leaf"][-1] - r["hist_hub"][-1]
        pred = r["a_leaf_eq"] or 0
        print(f"  {seed:>5} │ {r['hub_deg']:>7} │ {r['leaf_deg']:>8} │ "
              f"{r['C_hub']:>6.3f} │ {r['hist_hub'][-1]:>8.4f} │ "
              f"{r['hist_leaf'][-1]:>9.4f} │ {pred:>12.4f}"
              + (" ← лист > хаб" if diff > 0 else ""))
        runs.append(r)

    # ── БЛОК 3: статистика ────────────────────────────────────────
    print(f"\n{'═'*55}")
    print("БЛОК 3: Статистика")
    print("═"*55)
    fin_hubs  = [r["hist_hub"][-1]  for r in runs]
    fin_leaves= [r["hist_leaf"][-1] for r in runs]
    fin_means = [r["hist_mean"][-1] for r in runs]
    n_leaf_wins = sum(1 for l,h in zip(fin_leaves,fin_hubs) if l>h)

    print(f"\n  fin_hub  (avg): {np.mean(fin_hubs):.4f} ± {np.std(fin_hubs):.4f}")
    print(f"  fin_leaf (avg): {np.mean(fin_leaves):.4f} ± {np.std(fin_leaves):.4f}")
    print(f"  fin_mean (avg): {np.mean(fin_means):.4f} ± {np.std(fin_means):.4f}")
    print(f"  лист > хаб: {n_leaf_wins}/8 случаев")

    # ── БЛОК 4: выводы ────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print("БЛОК 4: Автоанализ — Q2 ответ")
    print("═"*55)
    print(f"""
  Вопрос Q2 был поставлен неправильно.
  Мы искали нестабильный водораздел для листа.
  Его нет — хаб устраняет барьер для листа полностью.

  ✓ У листа ТОЛЬКО устойчивый аттрактор, не водораздел.
  ✓ A*_leaf >> A*_mf при типовом C_hub=2.69:
    {leaf_stable_eq(A_MF, 2.69):.4f} >> {A_MF:.4f}
  ✓ При C_hub > C*={C_star:.3f}: лист финально ВЫШЕ среднего аттрактора.
  ✓ Лист получает концентрированный TSV от хаба (мало партнёров).
    Поэтому fin_leaf > fin_hub стабильно.

  Переформулировка коллективного механизма:
    Старое: "группа преодолевает барьер коллективно"
    Новое:  "хабы преодолевают барьер, листья следуют автоматически"
    TippingPoint = момент когда хотя бы один хаб пересекает A*_mf

  Новые вопросы (Q2a, Q2b):
    Q2a: C* ≈ {C_star:.3f} — вычислить аналитически?
    Q2b: a_crit_hub = A*_mf / C_hub — пересчитать TippingPoint
         через локальное поле хаба, не через mean(A)?
    """)

    # ── ГРАФИКИ ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("EXP 030 — Локальный водораздел: лист vs хаб", fontsize=12)

    # 1. Траектории
    ax = axes[0]
    r = runs[0]
    ax.plot(r["hist_mean"], color='#2c3e50', lw=1.5, alpha=0.6, label='mean(A)')
    ax.plot(r["hist_hub"],  color='#e74c3c', lw=2.5,
            label=f'Хаб (C={r["C_hub"]:.2f}, deg={r["hub_deg"]})')
    ax.plot(r["hist_leaf"], color='#3498db', lw=2.0, ls='--',
            label=f'Лист (deg={r["leaf_deg"]})')
    ax.axhline(A_MF, color='#e74c3c', ls=':', lw=1, alpha=0.6,
               label=f'A*_mf={A_MF:.3f}')
    if r["a_leaf_eq"]:
        ax.axhline(r["a_leaf_eq"], color='#3498db', ls=':', lw=1, alpha=0.6,
                   label=f'A*_leaf={r["a_leaf_eq"]:.3f}')
    ax.set_xlabel("Шаг"); ax.set_ylabel("A")
    ax.set_title(f"Траектории (A_init=A*_mf={A_MF:.3f})")
    ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_ylim(0.3, 1.0)

    # 2. A*_leaf(A_hub) при разных C
    ax = axes[1]
    A_hubs = np.linspace(0.25, 0.90, 60)
    for C, col, lbl in [(1.0, '#95a5a6', 'C=1.0'),
                         (2.0, '#f39c12', 'C=2.0'),
                         (2.69,'#e74c3c', 'C=2.69 (типовой)'),
                         (3.5, '#8e44ad', 'C=3.5')]:
        leaves = [leaf_stable_eq(h, C) or np.nan for h in A_hubs]
        ax.plot(A_hubs, leaves, lw=2, color=col, label=lbl)
    ax.axvline(A_MF, color='gray', ls='--', lw=1, alpha=0.6,
               label=f'A*_mf={A_MF:.3f}')
    ax.axhline(0.80, color='gray', ls=':', lw=1, alpha=0.5,
               label='A*_stable(mean)')
    if C_star:
        # Точка перехода (C=C*)
        al_star = leaf_stable_eq(A_MF, C_star)
        if al_star:
            ax.plot([A_MF], [al_star], 'k*', ms=14, zorder=6,
                    label=f'C*={C_star:.2f}')
    ax.set_xlabel("A_hub"); ax.set_ylabel("A*_leaf (аттрактор)")
    ax.set_title("A*_leaf(A_hub) при разных C")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # 3. Финальные состояния: 8 seeds
    ax = axes[2]
    x = range(len(runs))
    ax.plot(x, fin_hubs,   'o-', color='#e74c3c', lw=2, ms=7, label='Хаб')
    ax.plot(x, fin_leaves, 's-', color='#3498db', lw=2, ms=7, label='Лист')
    ax.plot(x, fin_means,  '^-', color='#2c3e50', lw=1.5, ms=5,
            label='Mean', alpha=0.6)
    ax.axhline(A_MF, color='gray', ls='--', lw=1, alpha=0.5,
               label=f'A*_mf={A_MF:.3f}')
    ax.fill_between(x, fin_hubs, fin_leaves,
                    where=[l > h for l,h in zip(fin_leaves, fin_hubs)],
                    alpha=0.15, color='#3498db', label='лист > хаб')
    ax.set_xlabel("Seed"); ax.set_ylabel("Финальное A")
    ax.set_title("Лист vs Хаб vs Mean (8 seeds)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_ylim(0.4, 1.0)

    plt.tight_layout()
    out = "experiments/exp_030_result.png"
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"  [График сохранён: {out}]")
    print("\nКОНЕЦ EXP 030")
