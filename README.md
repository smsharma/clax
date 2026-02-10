# jaxCLASS

**A complete, differentiable reimplementation of the CLASS Boltzmann solver in JAX.**

jaxCLASS solves the coupled Einstein-Boltzmann equations for cosmological perturbations from first principles: background cosmology, hydrogen recombination, the full photon-baryon-neutrino Boltzmann hierarchy in synchronous gauge, line-of-sight integration for CMB angular power spectra, HaloFit for nonlinear matter power, gravitational lensing, and a shooting method for theta_s parametrization. The entire pipeline -- from cosmological parameters to P(k), C_l^TT/EE/TE/BB, and lensed C_l -- is end-to-end differentiable via JAX automatic differentiation.

The goal is a drop-in replacement for [CLASS](https://github.com/lesgourg/class_public) that enables gradient-based cosmological inference (HMC, variational methods) on CMB and large-scale structure data.

## Status

**v0.3** -- Sub-percent C_l^EE at l=12-150 and C_l^TT at l=20, 100, 150 (1-2% at l=30-50). P(k) at 1-4%. Visibility function g(tau) at 0.04%. 100 tests passing. See [PROGRESS.md](PROGRESS.md) for full details.

## Accuracy comparison against CLASS v3.3.4

All comparisons at Planck 2018 best-fit LCDM. GPU: V100-32GB.

### C_l angular power spectra (Planck-quality)

`planck_cl` preset: k_max=1.0, 60 k/decade (300 modes), l_max=50, source-interpolated to 3000 fine k-points. Convergence verified across k-densities. V100 GPU.

| Multipole l | C_l^TT error | C_l^EE error |
|-------------|-------------|-------------|
| 20          | **-0.28%**  | **-0.27%**  |
| 30          | +1.52%      | **-0.27%**  |
| 50          | +1.63%      | **-0.23%**  |
| 100         | **+0.57%**  | **-0.17%**  |
| 150         | **-0.12%**  | **-0.17%**  |
| 200         | **+0.10%**  | **-0.28%**  |
| 300         | **-0.84%**  | **-0.10%**  |
| 500         | -1.45%      | **-0.25%**  |
| 700         | -2.65%      | **-0.96%**  |
| 1000        | -9.05%      | **-0.89%**  |

Bold = sub-percent. EE is sub-percent from l=20 to l=1000. TT is sub-percent at l=20, 100-300 and 1-3% at l=30-700. High-l TT degradation (l>700) from hierarchy truncation effects.

### Matter power spectrum P(k)

| k [Mpc^-1] | jaxCLASS / CLASS | Error |
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

## Quick start

```python
import jax
import jaxclass
from jaxclass import CosmoParams, PrecisionParams

# Background + thermodynamics
result = jaxclass.compute(CosmoParams(h=0.6736, omega_b=0.02237))
print(result.bg.H0)            # Hubble constant in Mpc^-1
print(result.bg.conformal_age) # conformal age in Mpc

# Matter power spectrum at a single k
pk = jaxclass.compute_pk(CosmoParams(), k=0.05)

# Gradient of P(k) w.r.t. a cosmological parameter
grad = jax.grad(lambda p: jaxclass.compute_pk(p, k=0.05))(CosmoParams())
print(grad.omega_cdm)  # dP(k)/d(omega_cdm)

# Angular power spectra (C_l) -- science quality
from jaxclass.perturbations import perturbations_solve
from jaxclass.harmonic import compute_cl_tt_interp, compute_cl_ee_interp
prec = PrecisionParams.science_cl()
params = CosmoParams()
bg = jaxclass.background_solve(params, prec)
th = jaxclass.thermodynamics_solve(params, prec, bg)
pt = perturbations_solve(params, prec, bg, th)
cl_tt = compute_cl_tt_interp(pt, params, bg, [30, 100, 200])
cl_ee = compute_cl_ee_interp(pt, params, bg, [30, 100, 200])
```

## Installation

Requires Python >= 3.10.

```bash
pip install jax jaxlib diffrax equinox jaxtyping

# Clone and install
git clone https://github.com/smsharma/jaxclass.git
cd jaxclass
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
- Full Boltzmann hierarchy at all times -- no RSA/UFA switching. TCA (tight-coupling approximation) with CLASS-matching dual criteria for numerical stability.
- Perturbation ODE solved with Kvaerno5 (implicit, stiff-capable) via Diffrax.
- `vmap` over k-modes for GPU parallelism.
- `RecursiveCheckpointAdjoint` for memory-efficient reverse-mode AD through the ODE solve.
- Synchronous gauge throughout (matching CLASS default).
- RECFAST recombination matching CLASS `wrap_recfast.c` (Peebles C with fudge, Gaussian K correction, Heun stepping).
- Source-interpolated C_l integration: source functions S(k,tau) interpolated to fine k-grid (3000 points) before line-of-sight integration, resolving the oscillatory T_l(k) transfer function robustly.

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
| `science_cl()`| 200      | 50    | 0.35   | Sub-percent C_l          |

For science-grade results, use `compute_cl_tt_interp` / `compute_cl_ee_interp` which interpolate source functions to a fine k-grid (3000 points) before computing the transfer integral. This is robust regardless of the perturbation k-density.

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

- **TT at l=30-50**: ~1.5% residual error, converged across k-densities (physics-limited). Most likely from missing radiation streaming approximation (RSA) for post-recombination photon evolution. Implementing RSA would likely bring this to sub-percent.
- **TT Sachs-Wolfe plateau (l < 15)**: ~5% error from gauge-dependent source terms at super-horizon scales. Requires RSA + careful gauge-invariant source construction.
- **High l (l > 200)**: Accuracy degrades; needs l_max > 50 or RSA to extend the hierarchy analytically.
- **Massive neutrinos**: Approximated as massless in perturbation equations (background is correct). Full ncdm perturbation variables (Psi_l(q)) not yet implemented (~0.3% C_l effect at m=0.06 eV).
- **Single cosmology validated**: Sub-percent results demonstrated at Planck 2018 fiducial only. Multi-cosmology validation is straightforward but pending.

## References

- **CLASS v3.3.4**: Blas, Lesgourgues & Tram (2011). [arXiv:1104.2933](https://arxiv.org/abs/1104.2933)
- **DISCO-EB**: Hahn, Melchior & Tessore (2024). Differentiable Boltzmann solver in JAX. [arXiv:2410.02998](https://arxiv.org/abs/2410.02998)
- **SymBoltz.jl**: Li & Millea (2024). Symbolic Boltzmann solver in Julia. [arXiv:2411.18620](https://arxiv.org/abs/2411.18620)
- Seljak & Zaldarriaga (1996). Line-of-sight integration approach. [arXiv:astro-ph/9603033](https://arxiv.org/abs/astro-ph/9603033)
- Ma & Bertschinger (1995). Cosmological perturbation theory. [arXiv:astro-ph/9506072](https://arxiv.org/abs/astro-ph/9506072)

## Development

This codebase is being written entirely by Claude Code (Opus 4.6). The development process -- including architecture decisions, bug hunting through CLASS source code, and numerical validation -- is documented in [PROGRESS.md](PROGRESS.md) and [CLAUDE.md](CLAUDE.md).

## License

Research code. Not yet released under a formal license.
