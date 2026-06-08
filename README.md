# UAF v5 — Unified Adaptive Framework

**Version 5.0 · June 2026**
Formal theory of how systems stay alive, adaptive, and organised.

---

## Core idea

Any stable system — cell, brain, market, language, theory —
exists because it maintains a closed loop:

```
structure → information → catalysis → structure
```

When the loop breaks: the system degrades.
When it closes: the system rises to the next level of organisation.

---

## Master equation

```
dA_i/dτ = α_s · C_ij · A_j · (1 − A_i)   [TSV — social interaction]
         + α_l · Π_i  · PE_i · (1 − A_i)  [FEP — Bayesian learning]
         + f   · (1 − A_i / A_c)           [basal — minimum metabolism]
         − δ   · (1 − 0.3·A_i)            [decay — entropy]
```

| Symbol | Meaning |
|--------|---------|
| `A_i` | closure_degree of agent i — the one variable that governs everything |
| `α_s` | social interaction rate |
| `C_ij` | catalytic weight of neighbour j (BA-topology) |
| `Π_i` | precision of agent i — updated from prediction error |
| `PE_i` | prediction error = how far agent is from its optimal model |
| `f` | basal floor — minimum metabolic activity |
| `δ` | entropic decay rate |
| `τ` | relational time — count of significant interactions, not clock |

**Key identity: TSV = FEP**

The social TSV term is a special case of Bayesian FEP update with:
- `Π_ij = α_s · A_j` (precision = neighbour's state)
- `PE_i = A_i` (prediction error = current accuracy)

Every social agent does unconsciously what a Bayesian agent does consciously.

---

## What we found (verified results)

### 1. Bistability with unstable equilibrium

The system `f(A) = β·A²·(1−A) − δ·(1−0.3·A)` at working parameters has
**three** fixed points:

| A* | λ (Jacobian) | Type |
|----|-------------|------|
| 0 | (absorbing) | absorbing — death |
| **0.563** | **+0.0135** | **unstable — true TippingPoint** |
| 0.805 | −0.0170 | stable — upper attractor |

`a_crit = 0.75` is not a bifurcation point. The true watershed is `A* = 0.563`.

### 2. Saddle-node bifurcation at δ* = 0.01117

When `δ > δ*`: no upper attractor exists — system always dies.
When `δ < δ*`: bistability — two basins of attraction.

Floor shifts δ*:

| floor | δ* | expansion |
|-------|-----|-----------|
| 0.000 | 0.01117 | baseline |
| 0.001 | 0.01151 | +3.1% |
| 0.002 | 0.01187 | +6.3% |
| 0.005 | 0.01299 | +16.3% |

### 3. Floor lowers the barrier, not lifts the system

Every `+0.001` floor shifts `A*_unstable` down by `~0.030`.
This explains the 4× rescue window expansion in EXP 025e/f:
floor reduces effective decay → TSV wins earlier.

### 4. Collective BA effect

A single agent from `A=0.25` dies (`net dA = −0.0064`).
A group of N=60 from the same initial condition survives.

Why: a hub with `C_ij = 2.69` has `α_eff = 0.215`, giving
`net dA ≈ −0.0001` — near equilibrium at the threshold.
The hub holds; leaves follow through TSV.
This is a collective barrier-crossing mechanism.

### 5. EXP 044 bridge closed

v3.1: `corr(closure_F, TSV_A) = +0.089` — two independent systems.
v5: `corr(precision, mean_A) = +0.72` — shared variable `Π_i`.

The bridge closes through architecture, not metric correlation.

---

## Repository structure

```
UAF-v5/
├── uaf/
│   ├── core.py            ← master equation, bistability, phase detector
│   ├── fep.py             ← variational free energy, active inference agent
│   ├── sde.py             ← stochastic (Langevin) extension
│   ├── metrics.py         ← surprisal, MI, integrated information Φ
│   └── markov_blanket.py  ← explicit internal/blanket/external states
│
├── apps/
│   ├── anomaly/           ← Dirac anomaly detector
│   ├── gravity/           ← N-body + evolutionary search (XGrav)
│   └── molecular/         ← protein binding GA + RL
│
├── experiments/
│   └── findings_v5.py     ← all verified results (run this first)
│
└── tests/
```

---

## Quick start

```bash
pip install numpy scipy matplotlib
python experiments/findings_v5.py   # see all verified results
```

---

## Open questions (roadmap)

| # | Question | Why it matters |
|---|----------|----------------|
| Q1 | N* at fixed interactions per agent `k_int` | Iter VII result was an artefact of variable total interactions |
| Q2 | Local watershed `A*_local` for leaves near hubs | Mean-field doesn't capture BA network heterogeneity |
| Q3 | L2→L3 transition via `std(A)` criterion | L3 = homogeneity, needs `std(A) < ε` not just `A > 0.92` |
| Q4 | `a_crit` as computed value = `A*_unstable(params)` | Currently ad hoc 0.75 — should follow from parameters |
| Q5 | Relation to SIS/SIR models on scale-free networks | Is UAF a known model in disguise? |

---

## Epistemic status

**Verified analytically + numerically:**
- Bistability with three fixed points
- Saddle-node bifurcation at δ* = 0.01117
- Floor shifts barrier linearly: ∂A*_unstable/∂floor ≈ −30 per unit
- TSV = FEP identity
- Collective BA barrier-crossing mechanism

**Partially verified:**
- closure_degree = 1 − F_k / F_baseline (bridge corr = 0.72)
- BA topology accelerates TippingPoint

**Open / speculative:**
- Connection to quantum decoherence
- Fundamental constants as attractor parameters
- Global Super-Structure as limit of nested closure

---

## NPG rule

```
NPG(M; D, B) = (L(B,D) − L(M,D)) / (L(B,D) + ε)

NPG > 0  → model better than baseline
NPG ≤ 0  → reject or revise in domain D
```

Failure in one domain does not propagate to others.

---

## License

MIT
