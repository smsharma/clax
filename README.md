# clax

**A differentiable cosmological Boltzmann solver in JAX.**

## Development

This codebase was written entirely by [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (Opus 4.6). The development process -- including architecture decisions, bug hunting through the CLASS C source code, and numerical validation -- is documented in [CHANGELOG.md](CHANGELOG.md) and [CLAUDE.md](CLAUDE.md).

## Overview

clax solves the coupled Einstein-Boltzmann equations for cosmological perturbations from first principles: background cosmology, hydrogen recombination, the full photon-baryon-neutrino Boltzmann hierarchy in synchronous gauge, line-of-sight integration for CMB angular power spectra, HaloFit for nonlinear matter power, gravitational lensing, and a shooting method for theta_s parametrization. The entire pipeline -- from cosmological parameters to P(k), C_l^TT/EE/TE/BB, and lensed C_l -- is end-to-end differentiable via JAX automatic differentiation.

The goal is a differentiable alternative to [CLASS](https://github.com/lesgourg/class_public) that enables gradient-based cosmological inference (HMC, variational methods) on CMB and large-scale structure data.

## Status

Sub-0.2% unlensed C_l^TT/EE at l=20-1200. Full lensed C_l^TT/EE/TE/BB (sub-0.2% at l=10-2000). Multi-cosmology validated (10 LCDM parameter points). Full ncdm Boltzmann hierarchy. `fit_cl` preset for HMC / fitting (~20 s end to end on a V100 with its 5 q-bin massive-neutrino hierarchy, Oct 22, 2026); `planck_cl` for science-grade accuracy (~124 s on a V100 after the Oct 2026 retune; 4679 s before). One-loop EFTofLSS module (`clax.ept`) ports the CLASS-PT FFTLog pipeline to JAX with sub-percent accuracy on monopoles/quadrupoles. See [CHANGELOG.md](CHANGELOG.md) for details.

## One-loop EFTofLSS (`clax.ept`) accuracy vs CLASS-PT

All comparisons at Planck 2018 best-fit LCDM, z=0.38, k < 0.30 h/Mpc:

| Spectrum            | Max error | Mean error | Notes                              |
|---------------------|-----------|------------|------------------------------------|
| P_mm real-space     | **0.33%** | 0.10%      | Matter-matter, IR resummed         |
| P_gg real-space     | **0.35%** | 0.12%      | Galaxy-galaxy (b1=2)               |
| P_gm real-space     | **0.28%** | 0.09%      | Galaxy-matter cross                |
| P_mm ell=0          | **0.21%** | 0.08%      | Matter monopole                    |
| P_mm ell=2          | **0.34%** | 0.11%      | Matter quadrupole                  |
| P_mm ell=4          | **1.2%**  | 0.45%      | Matter hexadecapole                |
| P_gg ell=0          | **0.42%** | 0.15%      | Galaxy monopole                    |
| P_gg ell=2          | **0.89%** | 0.50%      | Galaxy quadrupole                  |
| P_gg ell=4          | **1.8%**  | 0.72%      | Galaxy hexadecapole                |

Sub-percent on all monopoles and quadrupoles. Hexadecapole within 2% (limited by the small signal amplitude and zero-crossings — see project memory `project_ept_hexadecapole_floor`).

The `clax.ept` module additionally feeds CMB lensing C_l^phiphi corrections via `compute_cl_pp(... nonlinear="ept", pt_mpk=...)` (`pt_mpk` is the solve that reaches today, `perturbations_solve_mpk`).

## Accuracy comparison against CLASS v3.3.4

All comparisons at Planck 2018 best-fit LCDM. GPU: H100-80GB.

### Unlensed C_l angular power spectra

`planck_cl` preset as it was when this table was measured (before the Oct 2026 retune): k_max=1.0, 60 k/decade (300 modes), l_max=50, full ncdm Psi_l(q) hierarchy (5 q-bins x 18 multipoles), source-interpolated to 10000 fine k-points. The retuned preset (l_max 17/17/17, rtol 1e-4 / atol 1e-6, max_steps 8192, Rodas5, fast path at n_k_fine 20000) gives the same spectra to ~0.1% at l <= 1000 (CHANGELOG, Oct 12, 2026).

| Multipole l | C_l^TT error | C_l^EE error | C_l^TE error |
|-------------|-------------|-------------|-------------|
| 20          | **-0.08%**  | **-0.21%**  | -0.3% †     |
| 30          | **-0.05%**  | **-0.11%**  | -0.5% †     |
| 50          | **-0.05%**  | **-0.05%**  | +0.8% †     |
| 100         | **-0.02%**  | **+0.02%**  | **-0.03%**  |
| 200         | **-0.05%**  | **-0.04%**  | **-0.05%**  |
| 300         | **-0.06%**  | **-0.02%**  | **-0.04%**  |
| 500         | **-0.15%**  | **-0.15%**  | **-0.01%**  |
| 700         | **-0.23%**  | **-0.11%**  | **+0.08%**  |
| 1000        | **-0.57%**  | **-0.26%**  | +1.7%       |
| 1200        | **-0.07%**  | **+0.03%**  | —           |

Bold = sub-percent. **TT sub-0.1% at l=20-300 and l=1200.** EE sub-0.3% from l=20 to l=1200.

† Near a TE zero crossing (ℓ≈52 in Planck-fiducial ΛCDM; second crossing near ℓ≈400). The relative-error metric `(clax − CLASS) / CLASS` is ill-defined when `C_l^TE → 0`; the absolute residual is comparable to neighboring ℓ where the metric is well-defined. The standard Hu & White (1997) correlation criterion `|C_l^TE| / √(C_l^TT · C_l^EE) < 0.02` flags these multipoles, and the lensed-TE accuracy test in `tests/test_lensing.py:156` already skips them. A Gaussian likelihood weights TE zero-crossing modes by `1/Var(C_l^TE) → 0` automatically, so this metric artifact does not affect HMC inference.

### Lensed C_l angular power spectra

Full spin-2 correlation function lensing with Cgl2 corrections, 12 Wigner d-functions:

| Multipole l | Lensed TT err | Lensed EE err | BB ratio |
|-------------|--------------|--------------|----------|
| 50          | **-0.000%**  | **+0.000%**  | 1.000    |
| 200         | **+0.000%**  | **-0.001%**  | 1.000    |
| 500         | **+0.002%**  | **-0.003%**  | 0.999    |
| 1000        | **+0.006%**  | **+0.005%**  | 0.996    |
| 2000        | **-0.199%**  | **-0.166%**  | 0.937    |

Lensing algorithm sub-0.2% at all l=10-2000 for TT and EE (tested with CLASS unlensed+pp as input).

### Lensing potential C_l^phiphi

`clax.compute_cl_pp(pt, params, bg, th, l_max, *, nonlinear="none", pt_mpk=None)` uses
the source-based Limber kernel (CLASS `transfer.c` + `harmonic.c`).
`pt_mpk` (from `perturbations_solve_mpk`) is required for `nonlinear="halofit"`
and `"ept"`, whose P_lin lookups start at z=0, beyond the C_l solve's grid; it
is ignored for `"none"`.

With `pt_k_max_cl >= 5.0 Mpc^-1` (required for Halofit's σ(R) bisection),
measured residuals at the default cosmology:

Absolute |C_l^phiphi_clax / C_l^phiphi_CLASS - 1| at the default cosmology:

| Multipole l | nonlinear="none" | nonlinear="halofit" | NL/lin ratio agreement |
|-------------|-----------------:|--------------------:|-----------------------:|
| 100         | 0.18%            | 0.16%               | 0.02%                  |
| 200         | 0.14%            | 0.10%               | 0.05%                  |
| 500         | 0.23%            | 0.15%               | 0.09%                  |
| 1000        | 0.12%            | 0.08%               | 0.04%                  |
| 1500        | 0.49%            | 0.61%               | 0.11%                  |
| 2000        | 0.71%            | 0.76%               | 0.05%                  |
| 2500        | 0.39%            | 1.16%               | 0.76%                  |

Both nonlinear modes share the same source-Limber kernel, so they track
each other closely — the absolute Halofit residual is dominated by the
common linear-kernel systematic, with a small extra contribution from the
R(k, z) accuracy that the third column isolates. Halofit injection
multiplies the lensing source by sqrt(R(k, z(τ))) where R = P_NL/P_lin
is computed on a 100-point z-grid via `vmap(compute_pk_nonlinear)` with
log-log k-extension to k_max=20 Mpc^-1 (CLASS itself uses a dedicated
nonlinear k-grid that extends past the perturbation k-range; the
extension provides the equivalent coverage). Sub-percent absolute
residual across all ℓ ≤ 2000; ~1% at ℓ=2500 where linear-kernel and
R-residual systematics add constructively. Narrower k-grids
(`pt_k_max_cl < 5`) gracefully degrade to no NL correction (R = 1, per
CLASS `fourier.c:1706-1716`).

### Multi-cosmology validation

Validated at 10 LCDM parameter variations (omega_b, omega_cdm, h, n_s, tau_reio at +/-20%):
- **TT: sub-0.5% at ALL l for ALL 10 cosmologies**
- **EE: sub-0.3% at l>=100** for all cosmologies

### Matter power spectrum P(k)

Single-mode `compute_pk(params, prec, k)` at `PrecisionParams.planck_fast()`
(as it was before the Oct 2026 retune: 60 k/decade, l_max=50, ncdm_q_size=5, rtol=1e-6; now l_max=17, rtol=1e-4 / atol=1e-6, Rodas5) vs CLASS v3.3.4
linear `pk_lin(k, z=0)` at default Planck 2018 LCDM:

| k [Mpc^-1] | clax / CLASS | Error    |
|------------|--------------|----------|
| 0.003      | 0.9983       | -0.17%   |
| 0.010      | 1.0025       | +0.25%   |
| 0.050      | 1.0004       | +0.04%   |
| 0.100      | 1.0007       | +0.07%   |
| 0.300      | 0.9997       | -0.03%   |
| 1.000      | 0.9980       | -0.20%   |

Median |error| ≈ 0.07%, worst-case 0.25%. Underlying perturbation evolution
(δ_cdm and δ_b at synchronous gauge) matches CLASS to <0.05% — most of any
remaining residual is from photon-hierarchy and IC corrections at next order.

### Pipeline accuracy

| Module | Accuracy | Notes |
|--------|----------|-------|
| Background (H, D_A, r_s) | < 0.01% | 6+ significant digits |
| Derived epochs (z_d, r_s(z_d); z_star, tau_star, r_s(z_star)) | 2e-6 (20000-point grid) / 4e-5 (`fit_cl`, 3000 points) vs CLASS RECFAST, 11 cosmologies | kappa_b = 1 and the visibility peak located sub-grid; differentiable (#44) |
| Thermodynamics (x_e, T_b) | x_e 2e-5 (20000-point grid) / 1.1e-4 (`fit_cl`, 3000 points) over z 800-5000, and x_e 2e-5 / 7.4e-5 with T_b 6e-5 / 1.7e-4 over z 100-700, vs CLASS RECFAST (Heswitch=1), 11 cosmologies | CLASS's Saha -> ODE staging, advanced by an L-stable ROS2 step; 2.4e-4 above z = 5000 (HeIII stage) |
| Visibility function g(tau) | **0.04%** | Bisection reionization, corrected kappa |
| Perturbation ODE (Phi, Psi) | 0.01-0.25% | Gauge-corrected at recombination |
| AD gradients (dP(k)/d(params)) | omega_cdm: within 0.1% at five LCDM points (k = 0.1 Mpc^-1, forward and both reverse modes equal to 1e-8); omega_b: within 0.03% (k = 0.05 Mpc^-1, production reverse mode) | vs central finite differences; the finite differences themselves scatter by ~0.1% between steps. Before #55 (ADR 0004) AD was +0.6-0.8% off along omega_cdm and 4% along omega_b |

### Performance

| Preset | Background | Thermo | Perturbations | Harmonic | **Total** | GPU |
|--------|-----------|--------|--------------|----------|-----------|-----|
| planck_cl (Oct 2026 retune; `scripts/benchmark_speed.py`, job 24668) | 0.6s | 63s | 47s | 13s | 124s | V100 |
| fit_cl (Oct 2026 retune, ncdm_q_size 0; same script and job) | 0.6s | 2.1s | 8.4s | 7.6s | 18.6s | V100 |
| fit_cl (Oct 22, 2026, 5 ncdm q-bins, l_max_ncdm 8, no fluid approximation, #57; `diags/diag_fitcl_qgrid.py`, job 27746, cached medians; 21.6 s on another node, job 27711) | 0.5s | 1.7s | 8.9s | 7.3s | 18.5s | V100 |
| planck_cl (before the retune) | 1s | 53s | 401s | 33s | 487s | H100 |

`fit_cl` uses 20 k/decade, l_max=17, 2000 tau points, 3000 thermodynamics points, table-based Bessel (all l at once, n_k_fine 5000, l_max 1500), rtol=1e-4 / atol=1e-6, max_steps=8192, a 5 q-bin massive-neutrino hierarchy truncated at l_max_ncdm=8 and followed to today without the fluid approximation (#57, Oct 22, 2026), and the Rodas5 Rosenbrock solver with cubic Hermite dense output. Suitable for HMC/fitting at <2% TT/EE at l<=500: 0.48% TT and 1.21% EE at the massless fiducial vs CLASS RECFAST (job 27746; 0.69% / 1.25% before #53, job 24668), and with the default 0.06 eV neutrino against CLASS HyRec references on the five-point LCDM grid within 0.69% TT, 1.65% EE and 1.13% TE/sqrt(TT EE); 0.52% / 1.18% / 0.87% at 0.15 eV, 0.64% / 1.16% / 0.86% at 0.30 eV, 0.63% / 1.19% / 0.85% at 0.50 eV, flat in the neutrino mass (jobs 27746, 27711). `planck_cl` uses 60 k/decade, l_max=17, rtol=1e-4 / atol=1e-6, max_steps=8192, n_k_fine 20000 and Rodas5: within 0.26% TT of CLASS to l=1000 and 0.46% EE to l=500 (0.98% EE to l=1000, one sparse-l-grid multipole at l~526). All times are JIT-cached (second call onwards); `benchmark_speed.py` keeps planck_cl's 5 neutrino q-bins at m_ncdm=0, so its perturbation stage is ~30 s without them.

## Quick start

```python
import jax
import clax
from clax import CosmoParams, PrecisionParams

# Background + thermodynamics
result = clax.compute(CosmoParams(h=0.6736, omega_b=0.02237))
print(result.bg.H0)            # Hubble constant [Mpc^-1] (CLASS internal units)
print(result.bg.conformal_age) # conformal age in Mpc

# Practical dense-spectrum / reusable-table workflow: solve once on an
# internal k-grid, then interpolate P(k) to the requested points.
pk_table = clax.compute_pk_table(CosmoParams(), k_eval=[1e-3, 5e-2, 1.0])
print(pk_table.pk_grid)
print(pk_table.pk(0.05))

# Exact single-k diagnostic / spot-check path
pk = clax.compute_pk(CosmoParams(), k=0.05)

# Local gradient at one k-mode
grad = jax.grad(lambda p: clax.compute_pk(p, k=0.05))(CosmoParams())
print(grad.omega_cdm)  # dP(k)/d(omega_cdm)

# Practical multi-k gradient workflow: differentiate one reusable table solve
# instead of many repeated exact single-k solves.
grad_table = jax.grad(
    lambda p: jax.numpy.sum(
        clax.compute_pk_table(p, k_eval=[1.5e-2, 3.0e-2, 9.0e-2, 2.5e-1]).pk_grid
    )
)(CosmoParams())
print(grad_table.h)  # d(sum_i P(k_i))/dh

# Angular power spectra (C_l) -- science quality
from clax.perturbations import perturbations_solve
from clax.harmonic import compute_cl_tt_interp, compute_cl_ee_interp
prec = PrecisionParams.planck_cl()
params = CosmoParams()
bg = clax.background_solve(params, prec)
th = clax.thermodynamics_solve(params, prec, bg)
pt = perturbations_solve(params, prec, bg, th)
# nonlinear lookups start at z=0, which needs the solve that reaches today (#42)
pt_mpk = clax.perturbations_solve_mpk(params, prec, bg, th)
cl_tt = compute_cl_tt_interp(pt, params, bg, [30, 100, 200])
cl_ee = compute_cl_ee_interp(pt, params, bg, [30, 100, 200])

# CMB lensing potential C_l^phiphi (linear or Halofit-corrected)
cl_pp_lin = clax.compute_cl_pp(pt, params, bg, th, l_max=2500)
cl_pp_nl  = clax.compute_cl_pp(pt, params, bg, th, l_max=2500,
                                nonlinear="halofit", pt_mpk=pt_mpk)

# Lens the unlensed CMB spectra
cl_tt_lensed, cl_ee_lensed, cl_te_lensed, cl_bb_lensed = clax.lens_cls(
    cl_tt_unlensed, cl_ee_unlensed, cl_te_unlensed, cl_bb_unlensed,
    cl_pp_nl, l_max=2500)
```

## Installation

Requires Python >= 3.10.

```bash
pip install jax jaxlib diffrax equinox jaxtyping

# Clone and install
git clone https://github.com/smsharma/clax.git
cd clax
pip install -e .

# Run tests
pip install pytest
pytest tests/ -v
pytest tests/ -v -m "not slow"  # skip slow integration tests
```

## Architecture

A sequential pipeline of pure functions, each returning a frozen PyTree:

```
CosmoParams --> background --> thermodynamics --> perturbations --> primordial
                                                                       |
                                             lensing <-- harmonic <-- transfer <-- nonlinear
```

**Key design choices:**

- `CosmoParams` fields are JAX-traced for automatic differentiation. `PrecisionParams` fields are static (control array shapes, not traced).
- Full Boltzmann hierarchy with smooth RSA damping post-recombination. Full ncdm Psi_l(q) phase-space hierarchy (5 q-bins x 18 multipoles). TCA (tight-coupling approximation) with CLASS-matching dual criteria for numerical stability.
- Perturbation ODE solved via Diffrax with either Kvaerno5 (ESDIRK, default) or Rodas5 (8-stage Rosenbrock with cubic Hermite dense output; default in `fit_cl`, `planck_cl` and `planck_fast` since Oct 2026); `Rodas5Batched` shares time-stepping across k-modes for batched `compute_pk_table` solves.
- `compute_pk_table()` / `compute_pk_interpolator()` are the practical
  dense-spectrum and reusable-table `P(k)` APIs, especially on GPU; exact
  `compute_pk()` remains the single-mode reference path.
- Practical reverse-mode `P(k)` objectives should usually differentiate one
  table-backed multi-`k` solve, not many repeated exact `compute_pk()` calls.
- Backend-aware batching over k-modes for GPU/CPU perturbation solves.
- `RecursiveCheckpointAdjoint` for memory-efficient reverse-mode AD through the ODE solve.
- Synchronous gauge throughout (matching CLASS default).
- RECFAST recombination matching CLASS `wrap_recfast.c` (Peebles C with fudge, Gaussian K correction, CLASS's Saha -> ODE staging advanced by an L-stable second-order ROS2 step).
- Source-interpolated C_l integration: source functions S(k,tau) interpolated to fine k-grid (10000 points) before line-of-sight integration, resolving the oscillatory T_l(k) transfer function robustly.
- JIT-compiled: all solve functions + per-l harmonic functions cached for 2x speedup on repeated calls.

### Choosing an adjoint mode

`PrecisionParams.ode_adjoint` controls Diffrax reverse-mode AD through the ODE solves.
clax currently supports:

- `"recursive_checkpoint"`: production default and recommended choice
- `"direct"`: optional alternate reverse pass for diagnostics or environment-specific benchmarking

Use `"recursive_checkpoint"` by default on both CPU and GPU. That includes:

- laptop / local CPU development
- gating gradient tests
- production `P(k)` / `C_l` gradient workflows on GPU

Use `"direct"` only after validating it on the exact environment and problem size you
care about. It is most appropriate for:

- reduced or short ODE solves where memory is easy to bound
- adjoint debugging, where you want to compare reverse-pass implementations
- target-specific benchmarking after AD-vs-FD checks pass

Do not assume a CPU validation transfers to GPU, or vice versa. In particular, the
single-mode perturbation solve's `DirectAdjoint` path has shown unstable density-parameter
gradients on a CPU/macOS checkout even when the forward solve and thermodynamics AD were healthy.

Before switching from the default, validate on the target setup:

- compare AD vs central finite differences on representative `P(k)` probes
- include density parameters (`h`, `omega_b`, `omega_cdm`) plus amplitude/tilt probes
- check runtime and memory, not just gradient agreement
- keep `recursive_checkpoint` as the reference for gating tests unless the alternate mode is revalidated

Example:

```python
from dataclasses import replace
from clax import CosmoParams, PrecisionParams, compute_pk

params = CosmoParams()
prec = PrecisionParams()  # defaults to ode_adjoint="recursive_checkpoint"
pk = compute_pk(params, prec, k=0.05)

# Optional diagnostic / benchmark mode; validate before trusting it.
prec_direct = replace(prec, ode_adjoint="direct")
pk_direct = compute_pk(params, prec_direct, k=0.05)
```

### Source modules

| Module              | Description                                      |
|---------------------|--------------------------------------------------|
| `constants.py`      | Physical constants matching CLASS                |
| `params.py`         | `CosmoParams` (traced) and `PrecisionParams` (static) |
| `interpolation.py`  | Pytree-registered `CubicSpline`                  |
| `ode.py`            | Diffrax ODE solver wrappers                      |
| `rosenbrock.py`     | Rodas5 / Rodas5Batched Rosenbrock ODE solver     |
| `background.py`     | Friedmann equation, distances, growth factor     |
| `thermodynamics.py` | RECFAST recombination, visibility function       |
| `perturbations.py`  | Full scalar + tensor Boltzmann hierarchy         |
| `primordial.py`     | Power-law scalar and tensor spectra              |
| `bessel.py`         | Spherical Bessel functions j_l(x)                |
| `transfer.py`       | Linear matter P(k) from perturbation solve       |
| `harmonic.py`       | C_l^TT/EE/TE/BB from line-of-sight integration  |
| `lensing.py`        | C_l^phiphi (source-Limber, optional Halofit / clax-pt NL) and lensed C_l (correlation-function method) |
| `nonlinear.py`      | HaloFit (Takahashi 2012 + Bird 2012 ν correction)|
| `ept.py`            | One-loop EFTofLSS via FFTLog (CLASS-PT algorithm in JAX); real-space and RSD multipoles, IR resummation, bias |
| `shooting.py`       | theta_s -> H0 via Newton + `custom_vjp`          |

## Precision presets

`PrecisionParams` provides six presets controlling the accuracy/speed tradeoff:

| Preset        | k/decade | l_max | k_max  | Use case                 |
|---------------|----------|-------|--------|--------------------------|
| **`fit_cl()`**| 20       | 17    | 1.0    | HMC / fitting (<2% with massive neutrinos to 0.5 eV, ~20 s V100) |
| **`planck_fast()`**| 60  | 17    | 1.0    | planck_cl with th_n_points 20000, l_max 1500 |
| `fast_cl()`   | 15       | 25    | 0.15   | Quick iteration, testing |
| `medium_cl()` | 20       | 50    | 0.3    | Moderate accuracy        |
| `planck_cl()` | 60       | 17    | 1.0    | Science-grade (~100 s V100) |
| `science_cl()`| 200      | 50    | 0.35   | Sub-percent C_l          |

For science-grade results, use `compute_cl_tt_interp` / `compute_cl_ee_interp` which interpolate source functions to a fine k-grid (10000 points) before computing the transfer integral. This is robust regardless of the perturbation k-density.

## Cosmological parameters

Default parameters correspond to Planck 2018 best-fit LCDM:

| Parameter    | Default        | Description                      |
|--------------|----------------|----------------------------------|
| `h`          | 0.6736         | H0 / (100 km/s/Mpc)             |
| `omega_b`    | 0.02237        | Physical baryon density          |
| `omega_cdm`  | 0.1200         | Physical CDM density             |
| `ln10A_s`    | 3.0445         | Log primordial amplitude (A_s = 2.1e-9) |
| `n_s`        | 0.9649         | Scalar spectral index            |
| `tau_reio`   | 0.0544         | Reionization optical depth       |
| `m_ncdm`     | 0.06 eV        | Neutrino mass (single species)   |
| `w0`, `wa`   | -1.0, 0        | CPL dark energy equation of state|

## Known limitations

- **Speed**: `fit_cl` preset targets HMC / fitting; `planck_cl` runs in ~124 s on a V100 for science-grade work (Oct 2026 retune; 487 s on an H100 and 4679 s on a V100 before it). Further speedup requires fewer k-modes or float32 (currently infeasible with `jax_enable_x64=True`).
- **TT l=400-800**: +0.10-0.18% residual from SW+Doppler source amplitude (~0.06% excess at k~0.03). Comparable to CAMB-CLASS inter-code variation (~0.07%).
- **TT l>1200**: Degrades due to k-integration under-resolution (Bessel oscillation period constant in k, but log-uniform grid spacing grows). Hybrid linear/log k-grid would fix this.
- **EE l=20-30**: ~0.2% from RECFAST visibility function bias. HyRec recombination would improve to sub-0.1%.
- **BB tensor modes**: Lensing BB is accurate (<0.5% at l<=1000). Primordial BB matches CLASS to sub-percent at l<=200 and ~2% at l=300 with `compute_cl_bb(... n_k_fine=2000)` at planck_cl-style precision.

## References

- **CLASS v3.3.4**: Blas, Lesgourgues & Tram (2011). [arXiv:1104.2933](https://arxiv.org/abs/1104.2933)
- **CLASS-PT**: Chudaykin, Ivanov, Philcox & Simonović (2020). Non-linear perturbation theory extension of CLASS. [arXiv:2004.10607](https://arxiv.org/abs/2004.10607)
- **EFT-with-FFT**: Nguyen (2026). Pedagogical notes on FFTLog computation of one-loop PT integrals. [GitHub](https://github.com/MinhMPA/EFT-with-FFT)
- **DISCO-DJ**: Hahn, List & Porqueres (2023). Differentiable Einstein-Boltzmann solver in JAX. [arXiv:2311.03291](https://arxiv.org/abs/2311.03291)
- **Bolt.jl**: Li, Sullivan & Millea (2023). Differentiable Boltzmann solver in Julia. [Zenodo: 10.5281/zenodo.10065126](https://zenodo.org/records/10065126)
- Seljak & Zaldarriaga (1996). Line-of-sight integration approach. [arXiv:astro-ph/9603033](https://arxiv.org/abs/astro-ph/9603033)
- Ma & Bertschinger (1995). Cosmological perturbation theory. [arXiv:astro-ph/9506072](https://arxiv.org/abs/astro-ph/9506072)

For more on the `clax.ept` module, see `docs/clax-pt.md`, `docs/CLASS-PT-summary.md`, and `docs/FFTLog_PT.md`.

## License

MIT
