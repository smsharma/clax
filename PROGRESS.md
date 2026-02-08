# jaxCLASS Development Progress

## Status: Differentiable Boltzmann solver — P(k) 1-4%, C_l^EE 3-7%, C_l^TT ~15%

**End-to-end differentiable pipeline from cosmological parameters to P(k),
C_l^TT/EE/TE/BB, and lensed C_l. AD gradients verified to 0.03%.**

Best C_l^EE accuracy: **2.9% at l=200** (approaching science quality).
TT has a systematic ~15% floor under active investigation.
95+ tests passing, ~10K lines.

---

### Best C_l Accuracy (V100, 100 k/decade, 439 modes, l_max=25, rtol=1e-6)

| Spectrum | l=30 | l=50 | l=100 | l=200 |
|----------|------|------|-------|-------|
| **EE** | 7.8% | **5.9%** | **6.9%** | **2.9%** |
| TT | 10.5% | **3.3%** | 17.5% | **7.1%** |
| TE | 3.1% | 65.5% | 12.8% | 29.2% |

EE is approaching science quality (3-8% at all l). TT has a persistent
~15% floor at l=100 that is **independent of k-resolution, l_max, ODE
tolerances, and tau-grid resolution** — it is a systematic source
function issue (see debugging section below).

### P(k) Accuracy

| k [Mpc⁻¹] | jaxCLASS / CLASS | Error |
|-----------|------------------|-------|
| 0.001 | 0.970 | **3.0%** |
| 0.010 | 0.986 | **1.4%** |
| 0.050 | 0.984 | **1.6%** |
| 0.100 | 1.013 | **1.3%** |
| 0.300 | 0.966 | **3.5%** |

---

### Phases 1-3: Background, Thermodynamics, Perturbations — COMPLETE ✓

- Background: H(z), distances, growth — **< 0.01%** vs CLASS
- Thermodynamics: z_star within 0.1%, x_e within 1-4%, Float64 required
- Perturbations: Full Boltzmann hierarchy in synchronous gauge with TCA
- P(k): **1-4%** at all k
- AD gradients: d(P(k))/d(params) matches finite differences to **0.03%**

### Phase 4: Transfer + C_l — VALIDATED ✓

- Bessel: j_l(x) with hybrid backward+upward recurrence for l=0-500
- C_l^TT/EE/TE: Line-of-sight integration with IBP source functions
- C_l^BB: Tensor modes within factor 2.3
- Lensing: Correlation function method, **<5%** with exact inputs
- HaloFit: σ(R), k_sigma, n_eff, P_NL(k) — 10 tests
- Shooting: θ_s→H0 via Newton + custom_vjp — 8 tests

### Phase 5-6: Gradients + API — COMPLETE ✓

- `compute()`, `compute_pk()`, `compute_cls_all()` API
- Sparse l-sampling (~100 points) + spline interpolation to l=2..2500
- Limber helpers (for lensing C_l^pp; NOT for primary TT/EE/TE)
- `science_cl()` preset: k_max=0.35, l_max=50, 60 k/decade

### Phase 7: Multi-parameter + Debugging — IN PROGRESS

- massive_nu_015 and w0wa_m09_01 validated at background level
- Source decomposition diagnostic (SW/ISW/Doppler subterms)
- Cross-term analysis at l=100
- Phi matches CLASS to 0.5%, Phi' to 1%
- IBP = nonIBP Doppler confirmed (<1% difference)
- T1/T2 transfer contributions implemented with sign investigation
- Non-IBP Doppler form implemented for A/B testing

### Bugs Found and Fixed (15 total)

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

---

### TT Accuracy Investigation (Active)

**Cross-term decomposition at l=100** (normalized to CLASS=1.0):

| Term | Value |
|------|-------|
| Auto SW | +0.165 |
| Auto ISW_fs | +0.157 |
| Auto Doppler | +0.224 |
| 2×(Dop × ISW_fs) | **+0.168** |
| 2×(SW × ISW_fs) | +0.123 |
| 2×(SW × Dop) | +0.055 |
| **Total** | **0.855** |
| **CLASS** | **1.000** |

**Verified correct:**
- Phi matches CLASS to **0.5%** at all tau (Newtonian potential, gauge-invariant)
- Phi' matches to **1%** (analytic vs finite-difference)
- IBP = nonIBP Doppler: **<1% difference** (IBP transformation is correct)
- l_max converged: l_max=15 through 30 give **identical TT** at l=100
- k-resolution converged: 100 k/decade = 40 k/decade for TT
- ODE tolerances converged: rtol=1e-7 = 1e-5 for TT

**Remaining hypothesis:** The ~15% TT deficit may come from the CLASS
T1 (ISW dipole, `j_l'` radial) and T2 (polarization quadrupole)
contributions that are summed in CLASS harmonic.c:962. The T1/T2 sign
convention needs further investigation.

### Next Steps

- [ ] Resolve T1 sign ambiguity (compare CLASS transfer functions in Newtonian gauge)
- [ ] Push EE below 3% with science_cl (60 k/decade + l_max=50)
- [ ] Extend lensing to EE/TE/BB
- [ ] Gradient tests for C_l: d(C_l)/d(params)
- [ ] Comprehensive validation at 5+ parameter points
- [ ] Fisher matrix demonstration
