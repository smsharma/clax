# jaxCLASS Development Progress

## Status: Working differentiable Boltzmann solver with P(k) at 1-4% accuracy

**Key achievement: End-to-end differentiable pipeline from cosmological
parameters to P(k) and C_l^TT, with P(k) matching CLASS to 1-4% at all k
and verified AD gradients matching finite differences to 0.03%.**

---

### Phase 1: Foundation -- COMPLETE ✓
- [x] `pyproject.toml`, `__init__.py` (with `compute()` and `compute_pk()` API)
- [x] `constants.py` - matching CLASS exactly
- [x] `params.py` - CosmoParams (traced), PrecisionParams (static)
- [x] `interpolation.py` - CubicSpline (pytree-registered)
- [x] `ode.py` - Diffrax wrappers
- [x] `background.py` - Friedmann ODE, distances, growth
- [x] Reference data: 3 models (LCDM, massive_nu, w0wa)
- [x] Tests: 15/15 passing (scalars, functions, gradients)

### Phase 2: Thermodynamics -- COMPLETE ✓
- [x] `thermodynamics.py` - MB95 semi-implicit recombination
  - z_star = 1089.8 (CLASS: 1090)
  - κ' matches CLASS to 0.1-4% during recombination
  - Float64 required
- [x] Tests: 6/6 passing

### Phase 3: Perturbations -- COMPLETE ✓
- [x] `perturbations.py` - Full Boltzmann hierarchy in synchronous gauge
  - h' from CONSTRAINT (00 Einstein equation)
  - η' from 0i Einstein equation
  - Metric shear source (8/15*(h'+6η')/2) in photon and neutrino l=2 equations
  - ncdm included in Einstein constraints (approximated as massless)
  - Adiabatic IC from CLASS (kτ)² expressions
  - **P(k) matches CLASS to 1-4% at ALL k values (0.001 to 0.3 Mpc⁻¹)**
- [x] `primordial.py` - power-law P_R(k)
- [x] Tests: 5/5 passing (P(k) at 2 k values, RHS finite, IC, gradient)
- [x] **GRADIENT: d(P(k))/d(omega_cdm) via AD matches FD to 0.03%**

### Phase 4: Transfer + C_l -- WORKING ✓
- [x] `bessel.py` - j_l(x) and j_l'(x), works up to l~500 (validated vs scipy)
- [x] `harmonic.py` - C_l^TT with CLASS IBP source + 4π normalization
  - Uses CLASS sync gauge IBP form (perturbations.c:7660-7678)
  - C_l = 4π ∫ dlnk P_R |T_l|² (Dodelson 2003 eq. 9.35)
  - **C_l(l=100) within 2% of CLASS** (first acoustic peak)
  - SW plateau ~30% off (IBP 1/k² sensitivity; needs TCA for improvement)
  - High l (>200) affected by hierarchy truncation at l_max=25
  - Key insight: sync gauge g*(δ_g/4+η) is NOT gauge-invariant; the IBP form IS
- [x] `lensing.py` - simple exponential damping approximation
- [x] `nonlinear.py` - placeholder
- [x] `distortions.py` - placeholder
- [x] `shooting.py` - skeleton with custom_vjp for implicit differentiation

### Phase 5-6: Gradients + API
- [x] Clean API: `jaxclass.compute(params)` and `jaxclass.compute_pk(params, k)`
- [x] **5 gradient tests passing**:
  - dH0/dh: < 1%
  - d(conf_age)/d(omega_cdm): < 1%
  - d(P(k=0.05))/d(omega_cdm): 0.03%
  - d(P(k=0.05))/d(omega_b): < 5%
  - d(P(k=0.01))/d(n_s): < 5%

### Test Summary
- **49 total tests** (45 fast + 4 slow gradient tests), **ALL PASSING**
- Test files: test_background, test_constants, test_end_to_end, test_interpolation,
  test_perturbations, test_thermodynamics

### P(k) Accuracy (flagship result)
| k [Mpc⁻¹] | jaxCLASS / CLASS | Error |
|-----------|------------------|-------|
| 0.001 | 0.970 | **3.0%** |
| 0.005 | 0.977 | **2.3%** |
| 0.010 | 0.986 | **1.4%** |
| 0.050 | 0.984 | **1.6%** |
| 0.100 | 1.013 | **1.3%** |
| 0.300 | 0.966 | **3.5%** |

### Code Size
- 16 source modules (~3000 lines)
- 6 test files (~800 lines)
- 3 doc files (~2300 lines)
- 1 reference data script (~240 lines)
- Total: ~6300 lines

### Bugs Found and Fixed (14 total)
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

### Known Limitations & Path to Full Parity
- **C_l accuracy limited by hierarchy truncation**: l_max=25 photon hierarchy limits
  k_max to ~0.08 Mpc⁻¹ before truncation artifacts appear. For accurate C_l at all l,
  need TCA (tight coupling approx) + RSA (radiation streaming approx) to efficiently
  handle large effective l_max. This is the main blocker for C_l parity.
  - IBP Doppler has 1/k² sensitivity at low k → SW plateau ~30% off
  - Need ~100 k-modes for convergent k-integration at acoustic peaks
- **No full ncdm perturbation hierarchy**: Massive neutrinos are approximated as massless
  in the Einstein constraints. Adds ~270 equations to the state vector (15 q-bins × 18 multipoles).
- **No tensor perturbations**: Needed for B-mode polarization.
- **No full lensing**: Simple exponential damping only. Need correlation function method.
- **No HaloFit/HMCode**: Linear P(k) only.
- **Float64 required**: Recombination numerics overflow in float32.
- **Shooting method**: Skeleton implemented but not yet functional.
- **C_l^EE, TE, BB**: Source functions computed but integration not yet wired up.
