# jaxCLASS

**A complete, differentiable reimplementation of the CLASS Boltzmann solver in JAX.**

jaxCLASS solves the coupled Einstein-Boltzmann equations for cosmological perturbations from first principles: background cosmology, hydrogen recombination, the full photon-baryon-neutrino Boltzmann hierarchy in synchronous gauge, line-of-sight integration for CMB angular power spectra, HaloFit for nonlinear matter power, gravitational lensing, and a shooting method for theta_s parametrization. The entire pipeline -- from cosmological parameters to P(k), C_l^TT/EE/TE/BB, and lensed C_l -- is end-to-end differentiable via JAX automatic differentiation.

The goal is a drop-in replacement for [CLASS](https://github.com/lesgourg/class_public) that enables gradient-based cosmological inference (HMC, variational methods) on CMB and large-scale structure data.

## Status

**v0.1** -- P(k) at science quality (1-4% vs CLASS at all k). C_l^TT within 4% at l=100 with sufficient k-resolution. 95 tests passing across 12 test files. See [PROGRESS.md](PROGRESS.md) for full details and roadmap to sub-percent accuracy.

## Key results

**Matter power spectrum** (flagship result):

| k [Mpc^-1] | jaxCLASS / CLASS | Error |
|-------------|------------------|-------|
| 0.001       | 0.970            | 3.0%  |
| 0.010       | 0.986            | 1.4%  |
| 0.050       | 0.984            | 1.6%  |
| 0.100       | 1.013            | 1.3%  |
| 0.300       | 0.966            | 3.5%  |

**Angular power spectra** (40 k/decade, 175 k-modes, l_max=25):

| Spectrum | l=100 error |
|----------|-------------|
| TT       | 4%          |
| EE       | 13%         |
| TE       | 3%          |

**Gradients**: AD derivatives match finite differences to 0.03% for d(P(k))/d(omega_cdm).

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

# Angular power spectra (C_l)
from jaxclass.perturbations import perturbations_solve
from jaxclass.harmonic import compute_cls_all
prec = PrecisionParams.fast_cl()
params = CosmoParams()
bg = jaxclass.background_solve(params, prec)
th = jaxclass.thermodynamics_solve(params, prec, bg)
pt = perturbations_solve(params, prec, bg, th)
cls = compute_cls_all(pt, params, bg, l_max=500)  # returns dict with 'tt', 'ee', 'te'
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
- Full Boltzmann hierarchy at all times -- no TCA/RSA/UFA switching. This follows the approach validated by DISCO-EB and SymBoltz.jl.
- Perturbation ODE solved with Kvaerno5 (implicit, stiff-capable) via Diffrax.
- `vmap` over k-modes for GPU parallelism.
- `RecursiveCheckpointAdjoint` for memory-efficient reverse-mode AD through the ODE solve.
- Synchronous gauge throughout (matching CLASS default).

### Source modules

| Module              | Description                                      |
|---------------------|--------------------------------------------------|
| `constants.py`      | Physical constants matching CLASS                |
| `params.py`         | `CosmoParams` (traced) and `PrecisionParams` (static) |
| `interpolation.py`  | Pytree-registered `CubicSpline`                  |
| `ode.py`            | Diffrax ODE solver wrappers                      |
| `background.py`     | Friedmann equation, distances, growth factor     |
| `thermodynamics.py` | Semi-implicit recombination (MB95), visibility   |
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
| `science_cl()`| 60       | 50    | 0.35   | Sub-percent P(k)         |

```python
prec = PrecisionParams.fast_cl()    # ~62 k-modes
prec = PrecisionParams.science_cl() # ~270 k-modes
```

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
