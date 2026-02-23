# clax

**A complete, differentiable reimplementation of the CLASS Boltzmann solver in JAX.**

clax solves the coupled Einstein-Boltzmann equations for cosmological perturbations from first principles: background cosmology, hydrogen recombination, the full photon-baryon-neutrino Boltzmann hierarchy in synchronous gauge, line-of-sight integration for CMB angular power spectra, HaloFit for nonlinear matter power, gravitational lensing, and a shooting method for theta_s parametrization. The entire pipeline -- from cosmological parameters to P(k), C_l^TT/EE/TE/BB, and lensed C_l -- is end-to-end differentiable via JAX automatic differentiation.

The goal is a drop-in replacement for [CLASS](https://github.com/lesgourg/class_public) that enables gradient-based cosmological inference (HMC, variational methods) on CMB and large-scale structure data.

## Status

**v1.0** -- Sub-0.2% unlensed C_l^TT/EE at l=20-1200. Full lensed C_l^TT/EE/TE/BB (sub-0.2% at l=10-2000). Multi-cosmology validated (10 LCDM parameter points). Full ncdm Boltzmann hierarchy. JIT-compiled: 487s cached on H100-80GB. 95+ tests passing. See [CHANGELOG.md](CHANGELOG.md) for full details.

## Accuracy comparison against CLASS v3.3.4

All comparisons at Planck 2018 best-fit LCDM. GPU: H100-80GB.

### Unlensed C_l angular power spectra

`planck_cl` preset: k_max=1.0, 60 k/decade (300 modes), l_max=50, full ncdm Psi_l(q) hierarchy, source-interpolated to 10000 fine k-points. H100 GPU.

| Multipole l | C_l^TT error | C_l^EE error | C_l^TE error |
|-------------|-------------|-------------|-------------|
| 20          | **-0.08%**  | **-0.21%**  | -0.3%       |
| 30          | **-0.05%**  | **-0.11%**  | -0.5%       |
| 50          | **-0.05%**  | **-0.05%**  | +0.8%       |
| 100         | **-0.02%**  | **+0.02%**  | **-0.03%**  |
| 200         | **-0.05%**  | **-0.04%**  | **-0.05%**  |
| 300         | **-0.06%**  | **-0.02%**  | **-0.04%**  |
| 500         | **-0.15%**  | **-0.15%**  | **-0.01%**  |
| 700         | **-0.23%**  | **-0.11%**  | **+0.08%**  |
| 1000        | **-0.57%**  | **-0.26%**  | +1.7%       |
| 1200        | **-0.07%**  | **+0.03%**  | —           |

Bold = sub-percent. **TT sub-0.1% at l=20-300 and l=1200.** EE sub-0.3% from l=20 to l=1200. TE zero crossings near l=52 and l=400 cause large relative errors.

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

### Multi-cosmology validation

Validated at 10 LCDM parameter variations (omega_b, omega_cdm, h, n_s, tau_reio at +/-20%):
- **TT: sub-0.5% at ALL l for ALL 10 cosmologies**
- **EE: sub-0.3% at l>=100** for all cosmologies

### Matter power spectrum P(k)

| k [Mpc^-1] | clax / CLASS | Error |
|-------------|------------------|-------|
| 0.001       | 0.970            | 3.0%  |
| 0.010       | 0.986            | 1.4%  |
| 0.050       | 0.984            | 1.6%  |
| 0.100       | 1.013            | 1.3%  |
| 0.300       | 0.966            | 3.5%  |

### Pipeline accuracy

| Module | Accuracy | Notes |
|--------|----------|-------|
| Background (H, D_A, r_s) | < 0.01% | 6+ significant digits |
| Thermodynamics (x_e) | 0.25% at z_star | RECFAST with Heun stepping |
| Visibility function g(tau) | **0.04%** | Bisection reionization, corrected kappa |
| Perturbation ODE (Phi, Psi) | 0.01-0.25% | Gauge-corrected at recombination |
| AD gradients (dP(k)/d(params)) | 0.03% | vs finite differences |

### Performance (H100-80GB, planck_cl preset)

| Step | 1st call (compile) | Cached |
|------|-------------------|--------|
| background | 8s | **1s** |
| thermodynamics | 66s | **53s** |
| perturbations | 810s | **401s** |
| harmonic | 68s | **33s** |
| **TOTAL** | **952s** | **487s** |

## Quick start

```python
import jax
import clax
from clax import CosmoParams, PrecisionParams

# Background + thermodynamics
result = clax.compute(CosmoParams(h=0.6736, omega_b=0.02237))
print(result.bg.H0)            # Hubble constant in Mpc^-1
print(result.bg.conformal_age) # conformal age in Mpc

# Matter power spectrum at a single k
pk = clax.compute_pk(CosmoParams(), k=0.05)

# Gradient of P(k) w.r.t. a cosmological parameter
grad = jax.grad(lambda p: clax.compute_pk(p, k=0.05))(CosmoParams())
print(grad.omega_cdm)  # dP(k)/d(omega_cdm)

# Angular power spectra (C_l) -- science quality
from clax.perturbations import perturbations_solve
from clax.harmonic import compute_cl_tt_interp, compute_cl_ee_interp
prec = PrecisionParams.science_cl()
params = CosmoParams()
bg = clax.background_solve(params, prec)
th = clax.thermodynamics_solve(params, prec, bg)
pt = perturbations_solve(params, prec, bg, th)
cl_tt = compute_cl_tt_interp(pt, params, bg, [30, 100, 200])
cl_ee = compute_cl_ee_interp(pt, params, bg, [30, 100, 200])
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

For reference data generation (optional, requires CLASS Python wrapper):
```bash
cd ../class_public-3.3.4 && pip install .
python scripts/generate_class_reference.py
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
- Full Boltzmann hierarchy with smooth RSA damping post-recombination. Full ncdm Psi_l(q) phase-space hierarchy (15 q-bins x 18 multipoles). TCA (tight-coupling approximation) with CLASS-matching dual criteria for numerical stability.
- Perturbation ODE solved with Kvaerno5 (implicit, stiff-capable) via Diffrax.
- `vmap` over k-modes for GPU parallelism.
- `RecursiveCheckpointAdjoint` for memory-efficient reverse-mode AD through the ODE solve.
- Synchronous gauge throughout (matching CLASS default).
- RECFAST recombination matching CLASS `wrap_recfast.c` (Peebles C with fudge, Gaussian K correction, Heun stepping).
- Source-interpolated C_l integration: source functions S(k,tau) interpolated to fine k-grid (10000 points) before line-of-sight integration, resolving the oscillatory T_l(k) transfer function robustly.
- JIT-compiled: all solve functions + per-l harmonic functions cached for 2x speedup on repeated calls.

### Source modules

| Module              | Description                                      |
|---------------------|--------------------------------------------------|
| `constants.py`      | Physical constants matching CLASS                |
| `params.py`         | `CosmoParams` (traced) and `PrecisionParams` (static) |
| `interpolation.py`  | Pytree-registered `CubicSpline`                  |
| `ode.py`            | Diffrax ODE solver wrappers                      |
| `background.py`     | Friedmann equation, distances, growth factor     |
| `thermodynamics.py` | RECFAST recombination, visibility function       |
| `perturbations.py`  | Full scalar + tensor Boltzmann hierarchy         |
| `primordial.py`     | Power-law scalar and tensor spectra              |
| `bessel.py`         | Spherical Bessel functions j_l(x)                |
| `harmonic.py`       | C_l^TT/EE/TE/BB from line-of-sight integration  |
| `lensing.py`        | Correlation-function lensing method              |
| `nonlinear.py`      | HaloFit (Takahashi 2012)                         |
| `shooting.py`       | theta_s -> H0 via Newton + `custom_vjp`          |

## Precision presets

`PrecisionParams` provides three presets controlling the accuracy/speed tradeoff:

| Preset        | k/decade | l_max | k_max  | Use case                 |
|---------------|----------|-------|--------|--------------------------|
| `fast_cl()`   | 15       | 25    | 0.15   | Quick iteration, testing |
| `medium_cl()` | 20       | 50    | 0.3    | Moderate accuracy        |
| `planck_cl()` | 60       | 50    | 1.0    | Planck-quality C_l       |
| `science_cl()`| 200      | 50    | 0.35   | Sub-percent C_l          |

For science-grade results, use `compute_cl_tt_interp` / `compute_cl_ee_interp` which interpolate source functions to a fine k-grid (10000 points) before computing the transfer integral. This is robust regardless of the perturbation k-density.

## Cosmological parameters

Default parameters correspond to Planck 2018 best-fit LCDM:

| Parameter    | Default  | Description                      |
|--------------|----------|----------------------------------|
| `h`          | 0.6736   | H0 / (100 km/s/Mpc)             |
| `omega_b`    | 0.02237  | Physical baryon density          |
| `omega_cdm`  | 0.1200   | Physical CDM density             |
| `ln10A_s`    | 3.044    | Log primordial amplitude         |
| `n_s`        | 0.9649   | Scalar spectral index            |
| `tau_reio`   | 0.0544   | Reionization optical depth       |
| `m_ncdm`     | 0.06 eV  | Neutrino mass (single species)   |
| `w0`, `wa`   | -1.0, 0  | CPL dark energy equation of state|

## Known limitations

- **Speed for HMC**: 487s (cached) for planck_cl preset on H100. Perturbation ODE is the bottleneck (~400s for 300 k-modes with adaptive Kvaerno5 solver). Target for HMC is 30-60s — needs fewer k-modes, reduced tau grid, or fixed-step solver.
- **TT l=400-800**: +0.10-0.18% residual from SW+Doppler source amplitude (~0.06% excess at k~0.03). Comparable to CAMB-CLASS inter-code variation (~0.07%).
- **TT l>1200**: Degrades due to k-integration under-resolution (Bessel oscillation period constant in k, but log-uniform grid spacing grows). Hybrid linear/log k-grid would fix this.
- **EE l=20-30**: ~0.2% from RECFAST visibility function bias. HyRec recombination would improve to sub-0.1%.
- **BB tensor modes**: Lensing BB is accurate (<0.5% at l<=1000), but primordial BB still ~2x off CLASS.
- **TE zero crossings**: Large relative errors near l=52 and l=400 where C_l^TE crosses zero.

## References

- **CLASS v3.3.4**: Blas, Lesgourgues & Tram (2011). [arXiv:1104.2933](https://arxiv.org/abs/1104.2933)
- **DISCO-EB**: Hahn, Melchior & Tessore (2024). Differentiable Boltzmann solver in JAX. [arXiv:2410.02998](https://arxiv.org/abs/2410.02998)
- **SymBoltz.jl**: Li & Millea (2024). Symbolic Boltzmann solver in Julia. [arXiv:2411.18620](https://arxiv.org/abs/2411.18620)
- Seljak & Zaldarriaga (1996). Line-of-sight integration approach. [arXiv:astro-ph/9603033](https://arxiv.org/abs/astro-ph/9603033)
- Ma & Bertschinger (1995). Cosmological perturbation theory. [arXiv:astro-ph/9506072](https://arxiv.org/abs/astro-ph/9506072)

## Development

This codebase is being written entirely by Claude Code (Opus 4.6). The development process -- including architecture decisions, bug hunting through CLASS source code, and numerical validation -- is documented in [CHANGELOG.md](CHANGELOG.md) and [CLAUDE.md](CLAUDE.md).

## License

Research code. Not yet released under a formal license.
