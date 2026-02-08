# jaxCLASS Development Progress

## Status: Differentiable Boltzmann solver — ODE 0.02%, g(τ) 2.6%, C_l 5-8%

**End-to-end differentiable pipeline from cosmological parameters to P(k),
C_l^TT/EE/TE/BB, and lensed C_l. AD gradients verified to 0.03%.**

The perturbation ODE matches CLASS to **0.02%** (gauge-invariant quantities at
recombination). The remaining C_l accuracy gap is dominated by the **visibility
function g(τ) being 2.6% low** at τ_star, caused by MB95 recombination (~1% x_e
error at z≈1100). Sub-percent C_l requires upgrading to full RECFAST.

95+ tests passing, ~10K lines.

---

### Accuracy Summary

**Reference preset for all comparisons:** `fast_cl` (k_max=0.15, 15 k/decade,
l_max=25, rtol=1e-4). Frozen as the baseline for A/B testing.

#### C_l (fast_cl preset, nonIBP mode)

| Spectrum | l=30 | l=50 | l=100 | l=200 |
|----------|------|------|-------|-------|
| **TT** | 5.7% | 12.5% | **8.4%** | **3.8%** |

Previous TT at l=100: 17.5% → now **8.4%** (theta_b' fix + TCA dual criteria).
IBP and nonIBP modes agree to <0.3% at l=10-100 (consistency verified).

Note: l<20 has ~23% error (SW plateau, gauge-dependent source — excluded from
sub-% trajectory). EE/TE at fast_cl are resolution-limited; best EE (science_cl
on GPU): 2.9% at l=200.

#### P(k)

| k [Mpc⁻¹] | ratio | error |
|-----------|-------|-------|
| 0.001 | 0.970 | 3.0% |
| 0.010 | 0.986 | 1.4% |
| 0.050 | 0.984 | 1.6% |
| 0.100 | 1.013 | 1.3% |
| 0.300 | 0.966 | 3.5% |

#### Perturbation ODE (gauge-corrected, k=0.05 Mpc⁻¹)

| Quantity | Error at τ_star | Notes |
|----------|----------------|-------|
| phi (Φ) | **0.01%** | Newtonian potential |
| psi (Ψ) | **0.01%** | Gravitational slip |
| delta_g (Newt) | **0.23%** | After gauge correction |
| theta_b (Newt) | **0.02%** | = theta_b + k²α |
| delta_b (Newt) | **0.25%** | After gauge correction |
| delta_cdm | 2.1% → 0.3% | Converges at late times |
| shear_g | **0.06%** | At recombination |

CLASS outputs perturbation variables in **Newtonian gauge** by default (even
when solving in synchronous gauge). Raw sync-gauge variables differ by O(1)
from CLASS output — must apply gauge correction: delta_g → delta_g - 4aHα,
theta_b → theta_b + k²α.

#### Visibility function

| Quantity | Error at τ_star | Impact |
|----------|----------------|--------|
| x_e(z=1100) | -0.24% | Primary driver |
| x_e(z=1000) | -1.1% | Freeze-out tail |
| kappa_dot | 0.4% | Via x_e |
| **g(τ)** | **-2.6%** | g enters C_l squared → **~5% C_l** |

Root cause: MB95 simplified recombination has ~1% x_e error at z≈1100,
amplified through κ integral. MB95 code units are self-consistent (alpha, beta,
C factor calibrated together) — upgrading alpha_B alone breaks the balance.

---

### Phases 1-3: Background, Thermodynamics, Perturbations — COMPLETE ✓

- Background: H(z), distances, growth — **< 0.01%** vs CLASS
- Thermodynamics: z_star within 0.1%, x_e within 1-4%, Float64 required
- Perturbations: Full Boltzmann hierarchy in synchronous gauge with TCA
- P(k): **1-4%** at all k
- AD gradients: d(P(k))/d(params) matches finite differences to **0.03%**

### Phase 4: Transfer + C_l — VALIDATED ✓

- Bessel: j_l(x) with hybrid backward+upward recurrence for l=0-500
- C_l^TT/EE/TE: Line-of-sight integration with IBP and nonIBP source functions
- C_l^BB: Tensor modes within factor 2.3
- Lensing: Correlation function method, **<5%** with exact inputs
- HaloFit: σ(R), k_sigma, n_eff, P_NL(k) — 10 tests
- Shooting: θ_s→H0 via Newton + custom_vjp — 8 tests

### Phase 5-6: Gradients + API — COMPLETE ✓

- `compute()`, `compute_pk()`, `compute_cls_all()` API
- Sparse l-sampling (~100 points) + spline interpolation to l=2..2500
- Limber helpers (for lensing C_l^pp; NOT for primary TT/EE/TE)
- `science_cl()` preset: k_max=0.35, l_max=50, 60 k/decade

### Phase 7: Diagnostics + Bug Fixes — COMPLETE ✓

- Perturbation variable comparison vs CLASS at matched (k,τ) — **0.02% at τ_star**
- Visibility function g(τ) identified as dominant error source (2.6% low)
- Gauge convention clarified: CLASS outputs Newtonian gauge for perturbation variables
- IBP/nonIBP consistency verified after theta_b' fix (<0.3%)
- Source decomposition diagnostic (SW/ISW/Doppler subterms)
- Cross-term analysis at l=100

### Phase 8: Sub-percent accuracy — IN PROGRESS

See "Road to sub-percent" section below.

---

### Bugs Found and Fixed (18 total)

1. ncdm deg=1 → g*=2 (factor of 2 in density)
2. age: divide by Gyr_over_Mpc, not multiply
3. a_ini: 1e-14 → 1e-7 (ODE step count + high-k ICs)
4. adot = a²H (conformal time derivative)
5. Diffrax args: plain tuples, not custom classes
6. float() breaks JAX tracing → use jnp values
7. **h' must be CONSTRAINT** (CLASS perturbations.c:6612)
8. Monopole: -2/3*h', not -1/3*h'
9. C_l formula: ∫ dlnk P_R Δ_l²
10. **Photon dipole: -κ'*(F_1 - 4θ_b/(3k))** (scattering damps)
11. Bessel clip to [-1,1] during upward recurrence
12. **ncdm in Einstein constraints** (δρ and (ρ+p)θ)
13. a_ini=1e-7 for high-k perturbation ICs
14. **METRIC SHEAR in l=2**: 8/15*(h'+6η')/2 source → P(k) from 60% to 4%!
15. **source_E normalization**: was g*Pi/(4k²), correct is 3*g*Pi/16
16. **theta_b' extraction mismatch**: source extraction used full baryon eq,
    ODE used TCA → IBP Doppler biased. Fixed via shared helper. TT l=100: 17.5% → 8.2%.
17. **TCA single criterion → dual criteria** matching CLASS perturbations.c:6178-6179.
18. **Global TT mode leakage**: module-level global contaminated TE/compute_cls_all.

### Failed approaches (do not re-attempt)

- **Upgrading alpha_B alone** (Pequignot) without matching beta_B: x_e recombines
  10% too fast because MB95's alpha/beta/C are calibrated together in code units.
  MB95 beta is O(10) Mpc^-1; CLASS four_betaB is O(10^3) s^-1 = O(10^17) Mpc^-1.
  Must upgrade alpha, beta, and C simultaneously as a complete RECFAST ODE.
- **Smooth-blending TCA and full equations**: changes the physics, corrupts
  polarization. Use jnp.where for equation selection; sigmoid only for the
  switching criterion itself.

---

### Road to sub-percent C_l

**Error budget at l=30-200 (acoustic peaks):**

| Source | Estimated C_l impact | Fix |
|--------|---------------------|-----|
| g(τ) 2.6% low | ~5-8% (g enters squared) | Full RECFAST |
| delta_g(τ=1000) 5.7% | ~1-2% (late-time ISW) | RSA for late-time photons |
| ncdm as massless | ~0.5% (late-time constraint) | ncdm Ψ_l(q) variables |
| k-resolution | ~1% (fast_cl only) | science_cl preset |
| ODE tolerance | <0.5% | rtol=1e-6 |

**Prioritized plan (each step is an A/B with frozen baseline):**

1. **Full RECFAST recombination** — self-consistent Peebles ODE in CGS with
   Pequignot alpha_B, detailed-balance beta_B, and CLASS C factor.
   A/B: same code, only swap _ionize. Compare x_e, κ, g, C_l at l=30-200.
   Expected: g(τ) error 2.6% → <0.5%, C_l error ~8% → ~3%.

2. **Source-level verification** — after RECFAST, rerun term-level source
   checks (SW, ISW_vis, ISW_fs, Doppler) and T_l(k) at fixed (l,k) to
   confirm each source component matches CLASS before looking at C_l.

3. **RSA for late-time photons** — at τ > τ_free_streaming, set delta_g = -2h',
   theta_g = -0.5h' analytically (CLASS perturbations.c:10411-10420).
   Expected: removes 5.7% late-time delta_g error, improves ISW.

4. **ncdm perturbation variables** — add Ψ_l(q_i) to state vector (15 q-bins
   × 18 multipoles = 270 equations). Required for sub-percent at all k.

5. **Resolution sweep** — science_cl on GPU to quantify resolution floor.

6. **Comprehensive validation** — all spectra at l=30-200 for 5+ parameter points.

### Next Steps

- [ ] Implement full RECFAST (self-consistent alpha_B + beta_B + C in CGS ODE)
- [ ] A/B: compare x_e, g(τ), C_l with MB95 vs RECFAST (frozen baseline)
- [ ] After RECFAST: rerun source-level checks at matched (k,τ)
- [ ] Implement RSA for late-time photon perturbations
- [ ] Implement ncdm perturbation variables (Ψ_l(q))
- [ ] Science_cl validation sweep on GPU
- [ ] Gradient tests for C_l: d(C_l)/d(params)
- [ ] Comprehensive validation at 5+ parameter points
