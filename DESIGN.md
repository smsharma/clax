# clax: Full Design Specification

A fully differentiable reimplementation of the CLASS Boltzmann solver in JAX,
targeting complete feature parity with CLASS v3.3.4.

---

## Table of Contents

1. [Goals and Non-Goals](#1-goals-and-non-goals)
2. [Landscape and Positioning](#2-landscape-and-positioning)
3. [Architecture Overview](#3-architecture-overview)
4. [Module Specifications](#4-module-specifications)
   - 4.1 [Constants and Parameters](#41-constants-and-parameters)
   - 4.2 [Interpolation Utilities](#42-interpolation-utilities)
   - 4.3 [Background](#43-background)
   - 4.4 [Thermodynamics](#44-thermodynamics)
   - 4.5 [Perturbations](#45-perturbations)
   - 4.6 [Primordial](#46-primordial)
   - 4.7 [Non-Linear Corrections](#47-non-linear-corrections)
   - 4.8 [Transfer Functions](#48-transfer-functions)
   - 4.9 [Harmonic (C_l)](#49-harmonic-c_l)
   - 4.10 [Lensing](#410-lensing)
   - 4.11 [Spectral Distortions](#411-spectral-distortions)
   - 4.12 [Shooting Method](#412-shooting-method)
   - 4.13 [Spherical Bessel Functions](#413-spherical-bessel-functions)
   - 4.14 [ODE Solver Utilities](#414-ode-solver-utilities)
5. [Differentiation Strategy](#5-differentiation-strategy)
6. [Validation and Testing](#6-validation-and-testing)
7. [Performance Strategy](#7-performance-strategy)
8. [Implementation Roadmap](#8-implementation-roadmap)
9. [Repository Layout](#9-repository-layout)
10. [Open Questions and Risks](#10-open-questions-and-risks)

---

## 1. Goals and Non-Goals

### Goals

- **Full feature parity with CLASS v3.3.4**: lensed/unlensed C_l^{TT,TE,EE,BB,PP},
  matter power spectrum P(k,z), transfer functions T(k,z), background quantities,
  number count and galaxy lensing C_l's, spectral distortions.
- **End-to-end differentiable**: exact gradients of any output with respect to any
  cosmological parameter, via JAX reverse-mode AD.
- **GPU-accelerated**: JIT-compiled with parallelism over k-modes via `jax.vmap`.
- **Accuracy**: < 0.1% agreement with CLASS for standard LCDM and single-parameter
  extensions (massive neutrinos, w0wa dark energy, curvature).
- **Gradient accuracy**: AD gradients agree with finite differences to < 1%.
- **Pythonic API**: clean interface compatible with numpyro, blackjax, flowMC for
  gradient-based sampling.

### Non-Goals (for v1)

- Non-flat geometries (K != 0). Flat FLRW only initially.
- Vector perturbation modes. Scalars and tensors only.
- Exotic species (interacting dark matter, dark radiation, scalar fields, decaying DM).
  Standard LCDM + massive neutrinos + w0wa only.
- Perturbed recombination.
- Isocurvature initial conditions (adiabatic only initially).
- Matching CLASS at the 0.01% level (0.1% is the v1 target).
- Spectral distortions beyond mu and y parameters.

### Full CLASS v3.3.4 Feature Inventory (for future parity)

The following is a complete inventory of CLASS features. Items marked [v1] are
in scope for v1; all others are deferred to later versions.

**Species:**
- [v1] Photons, baryons, CDM, cosmological constant, ultra-relativistic relics (N_eff)
- [v1] Massive neutrinos (single species with degenerate mass; CLASS supports
  arbitrary N_ncdm with independent masses, temperatures, chemical potentials,
  and custom phase-space distributions from file)
- [v1] Fluid dark energy (w0, wa CPL parameterization, with c_s², PPF for w-crossing)
- [ ] Non-flat curvature (Omega_k != 0; requires hyperspherical Bessel functions)
- [ ] Scalar field dark energy (V(φ) potentials, attractor ICs, matter coupling Q_scf)
- [ ] Early dark energy (Omega_EDE)
- [ ] Decaying CDM → dark radiation (Gamma_dcdm)
- [ ] Interacting dark matter (IDM: couples to photons, baryons, dark radiation)
- [ ] Interacting dark radiation (IDR: ETHOS framework with angular coefficients)
- [ ] Varying fundamental constants (α(z), m_e(z) with instant transition)

**Perturbation modes and initial conditions:**
- [v1] Scalars (adiabatic only)
- [v1] Tensors (gravitational waves)
- [ ] Vectors
- [ ] Isocurvature modes: CDI, BI, NID, NIV (plus all cross-correlations)
- [v1] Synchronous gauge (default)
- [ ] Newtonian gauge
- [ ] N-body gauge transfers

**Recombination and reionization:**
- [v1] Simplified RECFAST in JAX
- [ ] Full RECFAST with all options (photo-ionization modes)
- [ ] HyRec 2020
- [ ] Perturbed recombination (δT_m, δx_e)
- [v1] Tanh reionization (reio_camb)
- [ ] Binned reionization (reio_bins_tanh)
- [ ] Multi-tanh reionization (reio_many_tanh)
- [ ] Half-tanh reionization (reio_half_tanh)
- [ ] Interpolated reionization (reio_inter)

**Approximation schemes** (all 6 TCA methods, 3 RSA methods, 4 UFA methods,
4 NCDMFA methods, IDM-DR tight coupling/streaming; CLASS uses these for speed,
we bypass all of them with approximation-free integration):
- [v1] Approximation-free: full Boltzmann hierarchy at all times

**Output types:**
- [v1] C_l^{TT, TE, EE, BB, PP, TP, EP} (unlensed)
- [v1] Lensed C_l
- [v1] Linear P(k,z)
- [v1] Matter transfer functions T(k)
- [v1] Non-linear P(k) via HaloFit
- [v1] Spectral distortions (μ, y only)
- [ ] Number count C_l^{dd} (density, RSD, lensing, GR terms G1-G5)
- [ ] Galaxy lensing C_l^{ll, dl, tl}
- [ ] Cross-correlations C_l^{td, pd}
- [ ] Individual ISW decomposition (early/late split)
- [ ] Full spectral distortion PCA (residual, up to 8 components)

**Non-linear corrections:**
- [v1] HaloFit (basic + pk_eq for DE)
- [ ] HMCode 2016, 2020, 2020 with baryonic feedback (6 feedback models)

**Primordial spectrum:**
- [v1] Analytic power law (A_s, n_s, alpha_s, r, n_t)
- [ ] Inflation from V(φ) in observable window
- [ ] Inflation from H(φ)
- [ ] Inflation from V(φ) through end of inflation
- [ ] Two-scales parameterization (for isocurvature)
- [ ] External P(k) from file

**Exotic energy injection** (for spectral distortions):
- [ ] DM annihilation (σv/m, redshift dependence, halo boost)
- [ ] DM decay
- [ ] PBH evaporation
- [ ] PBH accretion (spherical and disk models)
- [ ] Multiple deposition function recipes

**Other:**
- [ ] BBN-consistent Y_He (we use fixed Y_He)
- [ ] Custom ncdm phase-space distributions from file
- [ ] Selection function (dN/dz) from file for number counts
- [ ] Full Limber scheme for CMB lensing (want_lcmb_full_limber)
- [ ] Analytic/numerical no-wiggle P(k)
- [ ] Multiple ncdm species with independent properties

---

## 2. Landscape and Positioning

### Existing codes and what they lack

| Code | Framework | Has C_l? | Has lensing? | Differentiable? | GPU? | Limitation |
|------|-----------|----------|--------------|-----------------|------|------------|
| CLASS v3.3.4 | C | Yes | Yes | No | No | Not differentiable |
| CAMB | Fortran | Yes | Yes | No | No | Not differentiable |
| DISCO-EB | JAX | **No** | **No** | Yes | Yes | P(k) only; no transfer/C_l/lensing |
| SymBoltz.jl | Julia | Yes | Partial | Yes | No | Julia ecosystem; no GPU |
| Bolt.jl | Julia | Yes | No | Partial | No | Forward-mode only; limited physics |
| jax-cosmo | JAX | Limber | No | Yes | Yes | Eisenstein-Hu, not a Boltzmann solver |

**clax fills the gap**: the first JAX code going from cosmological parameters to
lensed C_l's with full reverse-mode AD and GPU acceleration.

---

## 3. Architecture Overview

### Pipeline

The computation is a sequential chain of pure functions, mirroring CLASS:

```
CosmoParams
    │
    ▼
┌─────────────┐
│  Background  │  Friedmann ODE → H(τ), distances, growth factors
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│  Thermodynamics   │  Recombination ODE → x_e(z), visibility g(τ), c_s²(τ)
└──────┬───────────┘
       │
       ▼
┌────────────────┐
│  Perturbations  │  Einstein-Boltzmann ODE (vmapped over k) → S(k,τ)
└──────┬─────────┘
       │
       ├──────────────────┐
       ▼                  ▼
┌────────────┐    ┌──────────────┐
│ Primordial  │    │  Non-Linear   │  HMCode → P_NL(k,z)
└──────┬─────┘    └──────┬───────┘
       │                 │
       ▼                 │
┌────────────┐           │
│  Transfer   │ ◄────────┘  Bessel integrals → Δ_l(k)
└──────┬─────┘
       │
       ▼
┌────────────┐
│  Harmonic   │  ∫ dk k² P(k) |Δ_l(k)|² → C_l
└──────┬─────┘
       │
       ▼
┌────────────┐
│  Lensing    │  C_l^{unlensed} + C_l^{φφ} → C_l^{lensed}
└──────┬─────┘
       │
       ▼
  Output C_l's, P(k), etc.
```

### Top-level API

```python
import clax

# Define parameters
params = clax.CosmoParams(
    h=0.6736, omega_b=0.02237, omega_cdm=0.1200,
    tau_reio=0.0544, ln10A_s=3.044, n_s=0.9649,
)

# Compute everything
result = clax.compute(params)
# result.cls["tt"]  -> jnp.array, shape (l_max-1,)
# result.cls["ee"]  -> ...
# result.pk         -> callable P(k, z)
# result.bg         -> BackgroundResult with H(z), D_A(z), etc.

# Differentiate
dcls_dparams = jax.jacrev(lambda p: clax.compute(p).cls["tt"])(params)

# Or compute a scalar loss and get gradient
def neg_log_like(params):
    cls = clax.compute(params).cls
    return 0.5 * chi_squared(cls, data_cls, noise_cls)

grad = jax.grad(neg_log_like)(params)
```

### Data flow: PyTree-based result objects

Every module returns a frozen dataclass (registered as a JAX PyTree) containing
arrays and interpolation objects. No mutable state. No side effects.

```python
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BackgroundResult:
    tau_table: Array        # (N_bg,)
    loga_table: Array       # (N_bg,)
    bg_table: Array         # (N_bg, n_bg_quantities)
    d2bg_table: Array       # spline coefficients
    # Derived scalars
    conformal_age: float
    tau_eq: float
    z_eq: float
    ...
```

---

## 4. Module Specifications

### 4.1 Constants and Parameters

**File**: `constants.py`

Physical constants matching CLASS exactly (from `background.h`):

```python
Mpc_over_m = 3.085677581282e22
Gyr_over_Mpc = 3.06601394e2
c_SI = 2.99792458e8           # m/s
G_SI = 6.67428e-11            # m³/kg/s²
eV_SI = 1.602176487e-19       # J
k_B_SI = 1.3806504e-23        # J/K
h_P_SI = 6.62606896e-34       # J·s
sigma_B = 2 * pi**5 * k_B**4 / (15 * h_P**3 * c**2)  # Stefan-Boltzmann
T_cmb_default = 2.7255        # K (Fixsen 2009)
```

**File**: `params.py`

Two parameter containers:

```python
@dataclass(frozen=True)
class CosmoParams:
    """Cosmological parameters. All fields are JAX-traceable floats."""
    h: float              # H0 / (100 km/s/Mpc)
    omega_b: float        # Omega_b h^2
    omega_cdm: float      # Omega_cdm h^2
    T_cmb: float = 2.7255
    # Neutrinos
    N_ur: float = 2.0328  # ultra-relativistic species
    N_ncdm: int = 1       # number of massive neutrino species (static)
    m_ncdm: float = 0.06  # sum of neutrino masses in eV (single species approx)
    # Primordial
    ln10A_s: float = 3.044
    n_s: float = 0.9649
    alpha_s: float = 0.0     # running
    r_t: float = 0.0         # tensor-to-scalar ratio
    n_t: float = 0.0         # tensor tilt (default: consistency relation)
    k_pivot: float = 0.05    # Mpc^-1
    # Reionization
    tau_reio: float = 0.0544
    # Dark energy
    w0: float = -1.0
    wa: float = 0.0
    # Curvature
    Omega_k: float = 0.0

@dataclass(frozen=True)
class PrecisionParams:
    """Numerical precision parameters. NOT traced by JAX (static)."""
    # Background
    bg_n_points: int = 800
    bg_tol: float = 1e-10
    # Thermodynamics
    th_z_max: float = 5e4
    th_n_points: int = 10000
    th_tol: float = 1e-5
    # Perturbations
    pt_k_min: float = 1e-5    # Mpc^-1
    pt_k_max: float = 5.0     # Mpc^-1 (for C_l; higher for P(k))
    pt_k_per_decade: int = 30
    pt_l_max_g: int = 17      # photon Boltzmann hierarchy truncation
    pt_l_max_ur: int = 17     # massless neutrino hierarchy
    pt_l_max_ncdm: int = 17   # massive neutrino hierarchy
    pt_tau_n_points: int = 5000
    pt_ode_rtol: float = 1e-5
    pt_ode_atol: float = 1e-10
    # Transfer
    tr_l_max_scalars: int = 2500
    tr_l_max_tensors: int = 500
    tr_l_limber_switch: int = 100
    tr_limber_transition_width: float = 20.0  # for smooth blending
    # Harmonic
    hr_k_per_decade: int = 40
    # Lensing
    le_l_max: int = 2500
    # ODE solver
    ode_solver: str = "kvaerno5"  # or "tsit5" for non-stiff
    ode_max_steps: int = 16384
    ode_adjoint: str = "recursive_checkpoint"  # or "direct"
    # NCDM quadrature
    ncdm_q_size: int = 15
    ncdm_q_max: float = 15.0
```

The split between `CosmoParams` (traced) and `PrecisionParams` (static) is critical:
JAX traces through `CosmoParams` for AD, while `PrecisionParams` controls array
shapes and solver settings that must be compile-time constants.

### 4.2 Interpolation Utilities

**File**: `interpolation.py`

All interpolation must be differentiable (smooth, no branching on data values).

**Cubic spline interpolation**:
```python
def build_spline(x: Array, y: Array) -> SplineCoeffs:
    """Compute cubic spline coefficients via tridiagonal solve.

    Solves the standard tridiagonal system for natural cubic splines
    (d²y/dx² = 0 at boundaries). The tridiagonal solve is a fixed-size
    loop, differentiable in JAX.

    Args:
        x: knot positions, shape (N,), strictly increasing
        y: knot values, shape (N,) or (N, D) for D-dimensional output

    Returns:
        SplineCoeffs containing x, y, d2y (second derivatives at knots)
    """

def eval_spline(coeffs: SplineCoeffs, x_eval: Array) -> Array:
    """Evaluate cubic spline at given points.

    Uses jnp.searchsorted to find intervals, then the standard
    cubic spline formula:
        f(x) = A*y_i + B*y_{i+1} + (A³-A)*d2y_i*h²/6 + (B³-B)*d2y_{i+1}*h²/6
    where A = (x_{i+1} - x)/h, B = (x - x_i)/h, h = x_{i+1} - x_i.

    Differentiable w.r.t. both x_eval and the spline knot values y.
    """
```

**2D spline** (for source function tables S(k,τ)):
```python
def build_spline_2d(x: Array, y: Array, z: Array) -> SplineCoeffs2D:
    """Bicubic spline on a regular 2D grid.

    Args:
        x: first axis, shape (Nx,)
        y: second axis, shape (Ny,)
        z: values, shape (Nx, Ny)
    """

def eval_spline_2d(coeffs: SplineCoeffs2D, x_eval: Array, y_eval: Array) -> Array:
    """Evaluate 2D spline. Differentiable w.r.t. all arguments."""
```

**Key implementation note**: `jnp.searchsorted` is not differentiable w.r.t. the
search values (it returns integers), but we only need gradients w.r.t. the spline
knot *values*, not positions. The knot positions are fixed by the precision grid.
The spline evaluation formula *is* differentiable w.r.t. the knot values `y` and
the evaluation point `x_eval`.

### 4.3 Background

**File**: `background.py`

**Purpose**: Solve the Friedmann equation to obtain H(τ), a(τ), and derived distances.

**Input**: `CosmoParams`, `PrecisionParams`
**Output**: `BackgroundResult` (interpolation tables)

#### Physics

The Friedmann equation in conformal time:

```
H² = (8πG/3) Σ_i ρ_i(a)
```

where the energy densities are:

| Species | ρ(a) |
|---------|------|
| Photons | ρ_γ,0 a^{-4} |
| Massless neutrinos | ρ_ur,0 a^{-4} |
| Baryons | ρ_b,0 a^{-3} |
| CDM | ρ_cdm,0 a^{-3} |
| Massive neutrinos | ∫ dq q² ε(q) f_0(q) [Fermi-Dirac integral] |
| Dark energy (w0wa) | ρ_de,0 a^{-3(1+w0+wa)} exp(-3 wa (1-a)) |
| Curvature | -K a^{-2} (enters via Omega_k) |
| Cosmological constant | ρ_Λ = const (special case of w0=-1, wa=0) |

The background ODE system in log(a):

```
d(τ)/d(ln a) = 1 / (a² H(a))             [conformal time]
d(t)/d(ln a) = 1 / (a H(a))               [proper time]
d(r_s)/d(ln a) = c_s / (a² H(a))          [sound horizon]
d(D)/d(ln a) = D'                          [growth factor]
d(D')/d(ln a) = -(2 + H'/H) D' + 3/2 Ω_m(a) D   [growth equation]
```

For **massive neutrinos**, the energy density and pressure are:

```
ρ_ncdm(a) = (4π T_ncdm^4 / (2π)^3) ∫₀^∞ dq q² ε(q,a) f_0(q)
P_ncdm(a) = (4π T_ncdm^4 / (2π)^3) ∫₀^∞ dq q⁴/(3ε) f_0(q)
```

where `ε(q,a) = √(q² + (m_ncdm a / T_ncdm)²)` and `f_0(q) = 1/(e^q + 1)`.
These integrals are computed via Gauss-Laguerre quadrature with fixed nodes
(making them differentiable pure array operations).

#### Algorithm

1. Compute present-day density fractions from input parameters:
   - `Omega_g = (4 σ_B / 3c³) T_cmb⁴ / (3 H0² / 8πG)`
   - `Omega_b = omega_b / h²`, `Omega_cdm = omega_cdm / h²`
   - `Omega_ur = N_ur * (7/8) * (4/11)^{4/3} * Omega_g`
   - `Omega_ncdm` from numerical integral at a=1
   - `Omega_Lambda = 1 - Omega_k - Omega_g - Omega_ur - Omega_b - Omega_cdm - Omega_ncdm - Omega_de`

2. Set up log(a) grid from `a_ini` (deep in radiation domination, ~1e-14) to `a=1`.

3. Integrate background ODE using a non-stiff solver (Tsit5 via Diffrax).
   Save at the log(a) grid points.

4. Build cubic spline tables for all background quantities as functions of log(a).
   Also build τ(log a) and log(a)(τ) mappings.

5. Compute derived quantities:
   - `conformal_age = τ(a=1)`
   - `z_eq` from ρ_m(a_eq) = ρ_r(a_eq) (root finding or interpolation)
   - `r_s(z_drag)`, `r_s(z_star)` from the sound horizon integral
   - Angular diameter distance, luminosity distance tables

#### Quantities stored (matching CLASS `index_bg_*`)

At minimum: `a, H, H', ρ_g, ρ_b, ρ_cdm, ρ_ur, ρ_ncdm, P_ncdm, ρ_de, ρ_crit,
ρ_tot, P_tot, Omega_m, Omega_r, conf_distance, ang_distance, lum_distance,
proper_time, sound_horizon, D, f`.

### 4.4 Thermodynamics

**File**: `thermodynamics.py`

**Purpose**: Compute ionization fraction x_e(z), optical depth κ(z), visibility
function g(z), baryon sound speed c_s²(z).

**Input**: `CosmoParams`, `PrecisionParams`, `BackgroundResult`
**Output**: `ThermoResult` (interpolation tables on z or τ grid)

#### Physics: Recombination

We implement a simplified RECFAST-like solver (following DISCO-EB's approach).

**Hydrogen recombination** (Peebles 3-level atom):

```
dx_e/dz = [C_r / (H(z)(1+z))] * [α_B(T_m) x_e² n_H - β_B(T_m)(1-x_e) exp(-E_21/T_m)]
```

where:
- `C_r = (1 + K Λ_{2s} n_H (1-x_e)) / (1 + K (Λ_{2s} + β_B) n_H (1-x_e))`
  is the Peebles C factor
- `α_B(T) = F * 1.14e-13 * (T/1e4)^{-0.6166} / (1 + 0.6703*(T/1e4)^{0.5300})`
  is the case-B recombination coefficient (with fudge factor F ≈ 1.14)
- `β_B(T) = α_B(T) * (m_e T / 2π)^{3/2} exp(-E_ion/T)` is the photoionization rate
- `K = λ_{Lyα}³ / (8π H(z))` accounts for Lyman-alpha escape
- `Λ_{2s} = 8.227 s⁻¹` is the two-photon decay rate
- `E_21 = 10.2 eV`, `E_ion = 13.6 eV`

**Helium recombination**: Saha equation for HeIII→HeII (z ~ 6000) and a
similar ODE for HeII→HeI (z ~ 1800), or a simpler fitting function.

**Matter temperature evolution**:

```
dT_m/dz = (2 T_m)/(1+z) - (8 σ_T a_r T_γ⁴)/(3 m_e c H(z)(1+z)) * x_e/(1+x_e+f_He) * (T_m - T_γ)
```

where the first term is adiabatic cooling and the second is Compton heating/cooling.

#### Physics: Reionization

Tanh model (matching CLASS `reio_camb`):

```
x_e^{reio}(z) = (1 + f_He)/2 * [1 + tanh((y(z_reio) - y(z)) / Δy)]
```

where `y(z) = (1+z)^{3/2}` and `Δy` controls the width.

The reionization redshift `z_reio` is either input directly or derived from
`tau_reio` via a root-finding procedure (matching CLASS behavior).

#### Derived quantities

From x_e(z), compute:

- **Opacity**: `κ'(τ) = -x_e n_e σ_T a` (differential optical depth)
- **Optical depth**: `κ(τ) = ∫_τ^{τ_0} κ'(τ') dτ'`
- **Visibility function**: `g(τ) = -κ'(τ) exp(-κ(τ))`
- **Baryon sound speed**: `c_s²(τ) = k_B T_b / (μ m_p (1 + R))` where
  `R = 3ρ_b/(4ρ_γ) = 3 Ω_b a / (4 Ω_γ)`

#### Algorithm

1. Integrate the recombination ODE system (x_e, T_m) from z_max ~ 1e4 down to z=0
   using a stiff solver (Kvaerno5). Initial conditions: Saha equilibrium at z_max.

2. Splice in reionization: combine recombination x_e with reionization x_e
   (using the tanh profile). This is a smooth operation, no branching.

3. Compute κ'(τ) on the τ grid, then integrate to get κ(τ) (backwards integral).

4. Compute g(τ), c_s²(τ).

5. Build spline tables for all quantities.

#### Shooting for tau_reio

If the input is `tau_reio` rather than `z_reio`, we need to find `z_reio` such that
`∫ κ'(τ) dτ = tau_reio`. This is a 1D root-finding problem. In the differentiable
context: use implicit differentiation (see Section 4.12).

### 4.5 Perturbations

**File**: `perturbations.py`

This is the largest and most critical module (~40% of the total implementation effort).

**Purpose**: Integrate the linearized Einstein-Boltzmann equations for each Fourier
mode k, producing source functions S^X(k,τ) for temperature, polarization, lensing, etc.

**Input**: `CosmoParams`, `PrecisionParams`, `BackgroundResult`, `ThermoResult`
**Output**: `PerturbationResult` containing source function tables S(k,τ) for each type

#### State vector

For a single k-mode in synchronous gauge, the ODE state vector y contains:

**Metric perturbations** (2):
- `η` (synchronous gauge metric perturbation)
- `h'` (trace of metric perturbation, time derivative; or equivalently `α = (h' + 6η')/(2k²)`)

**CDM** (1):
- `δ_cdm` (CDM density contrast; θ_cdm = 0 in synchronous gauge by convention)

**Baryons** (2):
- `δ_b` (baryon density contrast)
- `θ_b` (baryon velocity divergence)

**Photons** (l_max_g + 1 quantities):
- `F_γ,0 = δ_γ` (monopole = density contrast)
- `F_γ,1 = 4θ_γ/(3k)` (dipole ~ velocity)
- `F_γ,2` (quadrupole ~ anisotropic stress / polarization source)
- `F_γ,l` for l = 3, ..., l_max_g

**Photon polarization** (l_max_g + 1 quantities):
- `G_γ,0`, `G_γ,1`, ..., `G_γ,l_max_g`
  (E-mode polarization hierarchy)

**Massless neutrinos** (l_max_ur + 1 quantities):
- `F_ur,0`, ..., `F_ur,l_max_ur`

**Massive neutrinos** (for each species: (l_max_ncdm + 1) × n_q quantities):
- `Ψ_ncdm,l(q_i)` for l = 0, ..., l_max_ncdm and q_i = 1, ..., n_q
  (distribution function perturbation at each momentum bin)

**Dark energy perturbations** (2, if w != -1):
- `δ_de`, `θ_de`

**Total state vector size** (for default settings):
- 2 (metric) + 1 (CDM) + 2 (baryons) + 18 (photons) + 18 (polarization)
  + 18 (ur neutrinos) + 18 × 15 (massive neutrinos, 15 momentum bins)
  + 2 (dark energy) = **331 equations per k-mode**

With `vmap` over ~150 k-modes, this is ~50,000 equations total, but each block
is independent.

#### Boltzmann hierarchy equations (scalar, synchronous gauge)

The full equations, matching CLASS's `perturbations.c`:

**Metric** (Einstein equations):
```
η' = (k²α - (a'/a) h'/2) / (k² - (a'/a)²)   [via algebraic constraint]
h'' + (a'/a) h' = -2k² [Σ_i (ρ_i + P_i) σ_i + ...]   [simplified]
```

In practice, use the first-order form with `h'` and the algebraic η equation from
the 00 Einstein equation.

**CDM:**
```
δ_cdm' = -h'/2
```
(θ_cdm = 0 by gauge choice)

**Baryons:**
```
δ_b' = -θ_b - h'/2
θ_b' = -(a'/a) θ_b + c_s² k² δ_b + (4ρ_γ)/(3ρ_b) a n_e σ_T (θ_γ - θ_b)
```

**Photon Boltzmann hierarchy** (l ≥ 0):
```
F_γ,0' = -k F_γ,1 - h'/6 · 2                     [monopole]
F_γ,1' = k/3 (F_γ,0 - 2F_γ,2) + ... [depends on gauge]  [dipole]
F_γ,l' = k/(2l+1) [l F_γ,l-1 - (l+1) F_γ,l+1] - κ' F_γ,l   [l ≥ 2, l < l_max]
         + κ' δ_{l,2} Π/10                         [scattering source]
F_γ,l_max' = k F_γ,l_max-1 - (l_max+1)/(τ_0-τ) F_γ,l_max - κ' F_γ,l_max  [truncation]
```

where `Π = F_γ,2 + G_γ,0 + G_γ,2` is the polarization source.

**Photon polarization hierarchy:**
```
G_γ,0' = -k G_γ,1 - κ'[G_γ,0 - Π/2]
G_γ,l' = k/(2l+1)[l G_γ,l-1 - (l+1) G_γ,l+1] - κ'[G_γ,l - δ_{l,2} Π/10]
G_γ,l_max' = k G_γ,l_max-1 - (l_max+1)/(τ_0-τ) G_γ,l_max - κ' G_γ,l_max
```

**Massless neutrinos:** Same as photons but without scattering (κ' → 0).

**Massive neutrinos** (per momentum bin q_i):
```
Ψ_l'(q_i) = k/(2l+1) [l qk/ε · Ψ_{l-1}(q_i) - (l+1) qk/ε · Ψ_{l+1}(q_i)]
              - δ_{l,0} h'/6 · dlnf0/dlnq · ε/q  [monopole source]
```

where `ε = √(q² + (m a)²)`.

**Dark energy** (w0wa, PPF scheme):
```
δ_de' = -(1+w)(θ_de + h'/2) - 3(a'/a)(c_s² - w) δ_de
θ_de' = -(1-3w)(a'/a) θ_de + c_s² k²/(1+w) δ_de
```

(For w ≠ -1. For w = -1 (cosmological constant), δ_de = θ_de = 0.)

#### Initial conditions (adiabatic)

Deep in the radiation era (τ → 0), set initial conditions following Ma & Bertschinger (1995):

```
η = 1                                     [normalization choice]
δ_γ = δ_ur = -2η [1 - k²τ²/36 + ...]     [radiation]
δ_b = δ_cdm = 3δ_γ/4                      [matter]
θ_γ = θ_ur = -k²τ η/18 + ...              [velocity]
θ_b = θ_γ                                 [tight coupling]
h = k²τ² η / 2 + ...                       [metric]
F_γ,2 = F_ur,2 = ... (higher order)
```

Initial conditions are set at sufficiently early τ_ini (when all relevant k-modes
are well outside the horizon: kτ_ini << 1).

#### Source functions

After integration, extract source functions for the transfer integral.
Following CLASS, the temperature source has three terms (avoiding integration by parts):

```
S_T,0(k,τ) = g [δ_γ/4 + Ψ]                         [Sachs-Wolfe + intrinsic]
S_T,1(k,τ) = g θ_b / k²                              [Doppler]
S_T,2(k,τ) = exp(-κ) [Ψ' - Φ'] + g Π/(4k²) + ...   [ISW + polarization]
```

Polarization source:
```
S_E(k,τ) = g Π / (4k²)
```

Lensing source:
```
S_lens(k,τ) = exp(-κ) (Ψ + Φ)
```

where Ψ and Φ are the Newtonian gauge potentials (related to synchronous gauge
quantities by a gauge transformation).

#### k-mode sampling

Logarithmic grid with refinement:
- Coarse grid: `pt_k_per_decade` points per decade from `k_min` to `k_max`
- Refinement near k_eq (matter-radiation equality) and acoustic peaks
- Total: ~100-200 k values for C_l computation, more for high-k P(k)

#### Approximation-free approach

Unlike CLASS, we do **not** use TCA, RSA, UFA, or NCDMFA. The full system is
integrated at all times with a stiff implicit solver. This works because:

1. High-order implicit methods (Kvaerno5) can take large steps through tightly-coupled eras
2. The Jacobian is sparse and structured
3. SymBoltz.jl has validated this approach gives 0.1% accuracy

The tradeoff: we integrate more equations per step, but avoid all the complexity
of approximation switching (which is ~40% of perturbations.c in CLASS).

#### Parallelization

```python
# Single k-mode solver
def _solve_single_k(k, tau_grid, bg_interp, th_interp, params, prec):
    y0 = _adiabatic_ic(k, tau_grid[0], bg_interp, params)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, y, args: _rhs(t, y, k, bg_interp, th_interp, params)),
        solver=diffrax.Kvaerno5(),
        t0=tau_grid[0], t1=tau_grid[-1], dt0=...,
        y0=y0,
        saveat=diffrax.SaveAt(ts=tau_grid),
        stepsize_controller=diffrax.PIDController(rtol=prec.pt_ode_rtol, atol=prec.pt_ode_atol),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        max_steps=prec.ode_max_steps,
    )
    return _extract_sources(sol.ys, k, bg_interp, th_interp, params)

# Vectorize over all k
_solve_all_k = jax.vmap(_solve_single_k, in_axes=(0, None, None, None, None, None))

# In perturbations_solve():
sources = _solve_all_k(k_grid, tau_grid, bg_interp, th_interp, params, prec)
# sources shape: (n_k, n_tau, n_source_types)
```

### 4.6 Primordial

**File**: `primordial.py`

**Purpose**: Compute primordial scalar and tensor power spectra.

For the standard parameterization:

```
P_R(k) = A_s (k / k_pivot)^{n_s - 1 + (1/2) α_s ln(k/k_pivot)}
P_T(k) = A_s r_t (k / k_pivot)^{n_t}
```

where `A_s = exp(ln10A_s) / 1e10`.

This is a pure array operation: evaluate the power law at each k in the grid.
Trivially differentiable.

### 4.7 Non-Linear Corrections

**File**: `nonlinear.py`

**Purpose**: Compute non-linear matter power spectrum from linear P(k) using HMCode.

**Approach**: Reimplement HMCode 2020 (Mead et al.) in JAX. The algorithm:

1. Compute σ(R) = ∫ dk k² P_lin(k) W²(kR) / (2π²) (variance in top-hat of radius R)
2. Find σ_8 (σ at R = 8 Mpc/h) and the non-linear scale R_NL where σ(R_NL) = 1
3. Compute halo model ingredients:
   - NFW profile in Fourier space
   - Halo mass function (Sheth-Tormen or Tinker)
   - Concentration-mass relation
   - One-halo and two-halo terms
4. `P_NL(k) = P_1h(k) + P_2h(k)`

All operations are smooth functions of k and the linear P(k), hence differentiable.
The σ(R) integral is a 1D quadrature (Gauss-Legendre or trapezoidal on log k).

**Simpler alternative for v1**: Implement the Halofit fitting function (Smith et al.
2003, revised by Takahashi et al. 2012), which is simpler (~200 lines) but less
accurate. Use HMCode for v2.

### 4.8 Transfer Functions

**File**: `transfer.py`

**Purpose**: Compute transfer functions Δ_l(k) from source functions via line-of-sight
integration.

**The integral**:
```
Δ_l^X(k) = ∫₀^{τ₀} dτ S^X(k,τ) j_l^{(n)}(k[τ₀ - τ])
```

where `j_l^{(n)}` is the n-th derivative of the spherical Bessel function,
and the specific radial function depends on the source type X:

| Source type | Radial function |
|-------------|----------------|
| Temperature (j=0) | j_l(x) |
| Temperature (j=1) | j_l'(x) |
| Temperature (j=2) | j_l''(x) + terms |
| E-polarization | √((l+2)!/(l-2)!) j_l(x) / x² |
| CMB lensing | l(l+1) j_l(x) / x² |
| Number counts | j_l(x) |

#### Algorithm

For each (l, k) pair:

1. **Check Limber applicability**: For `l > l_limber_switch`, use the Limber approximation.

2. **Limber approximation** (large l):
   ```
   Δ_l(k) ≈ √(π/(2l+1)) · S(k, τ₀ - (l+1/2)/k) / k
   ```
   This is just a source function evaluation -- trivially differentiable.

3. **Smooth Limber transition**: To avoid a discrete switch, blend:
   ```
   w = σ((l - l_switch) / Δl)   # sigmoid
   Δ_l = (1 - w) · Δ_l^{exact} + w · Δ_l^{limber}
   ```
   where `Δl ≈ 20` gives a smooth transition over ~40 multipoles.

4. **Exact integration** (small/moderate l):
   - Compute j_l(k(τ₀ - τ)) on the τ grid via `bessel.py`
   - Multiply by source function S(k,τ) (interpolated from perturbation output)
   - Integrate with trapezoidal rule (or Gauss-Legendre on sub-intervals)

5. **Time cut**: For each (l,k), only integrate over the τ range where the
   integrand is non-negligible. For large k(τ₀-τ), the Bessel function
   oscillates rapidly and the integral saturates.

#### Parallelization

```python
# Vectorize over l for a given k
_transfer_all_l = jax.vmap(_compute_transfer_single_l, in_axes=(0, None, ...))

# Then vectorize over k
_transfer_all = jax.vmap(_transfer_all_l, in_axes=(None, 0, ...))
```

Or, for the Limber regime, it's just an array evaluation:
```python
# All (l, k) pairs at once
tau_limber = tau_0 - (l_array[:, None] + 0.5) / k_array[None, :]
Delta_limber = jnp.sqrt(jnp.pi / (2 * l_array[:, None] + 1)) * S_interp(k_array, tau_limber) / k_array
```

#### l-sampling

Not every integer l needs to be computed. CLASS samples l sparsely and interpolates:
- l = 2, 3, 4, ..., 30 (every l)
- l = 30, 35, 40, ..., 100 (every 5)
- l = 100, 120, 140, ..., 500 (every 20)
- l = 500, 550, ..., 2500 (every 50)

Then cubic spline interpolation fills in the gaps. This reduces the number of
expensive Bessel integrals from ~2500 to ~150.

### 4.9 Harmonic (C_l)

**File**: `harmonic.py`

**Purpose**: Compute angular power spectra C_l from transfer functions and primordial P(k).

**The integral**:
```
C_l^{XY} = (4π)² ∫₀^∞ dk/k · P_R(k) · Δ_l^X(k) · Δ_l^Y(k)
```

For adiabatic initial conditions, this is a single integral over k for each l.

#### Algorithm

1. For each l in the sparse l-sampling:
   - Evaluate the integrand `P_R(k) Δ_l^X(k) Δ_l^Y(k)` on the k-grid
   - Integrate using trapezoidal rule on log(k)
   - Apply corrections from non-linear P(k) if requested (multiply integrand by
     `P_NL(k)/P_lin(k)` ratio)

2. Cubic spline interpolation from sparse l to all integer l = 2, ..., l_max.

3. Output all requested spectra: TT, TE, EE, BB, PP (lensing potential), TP, EP.

#### Cross-correlations

For number count and galaxy lensing C_l's (LSS observables):
```
C_l^{ij} = ∫ dk/k P_R(k) Δ_l^{(i)}(k) Δ_l^{(j)}(k)
```
where i,j label redshift bins. These use the density/lensing transfer functions
instead of CMB transfer functions.

### 4.10 Lensing

**File**: `lensing.py`

**Purpose**: Compute lensed CMB power spectra from unlensed spectra and the lensing potential.

**Physics**: CMB photons are deflected by the gravitational lensing potential φ.
The lensed temperature at position n̂ is T(n̂ + ∇φ). This mixes power between
multipoles.

**Algorithm** (correlation function method, following CLASS):

1. Transform unlensed C_l^{TT}, C_l^{EE}, C_l^{TE}, C_l^{BB}, C_l^{PP} to
   correlation functions via Legendre transform:
   ```
   ξ_+(θ) = Σ_l (2l+1)/(4π) [C_l^{EE} + C_l^{BB}] d^l_{22}(θ)
   ξ_-(θ) = Σ_l (2l+1)/(4π) [C_l^{EE} - C_l^{BB}] d^l_{2,-2}(θ)
   ```
   (and similarly for TT, TE), where `d^l_{mm'}` are reduced Wigner d-matrices.

2. Compute the lensing deflection variance:
   ```
   σ²(θ) = Σ_l (2l+1)/(4π) l(l+1) C_l^{φφ} [1 - d^l_{00}(θ)]
   ```

3. Compute lensed correlation functions by convolving with the lensing kernel:
   ```
   ξ^{lens}_X(θ) = exp(-l(l+1)σ²/2) ξ_X(θ) + corrections
   ```
   (Full expressions involve series expansion in C_l^{φφ}).

4. Transform back to get lensed C_l's via inverse Legendre transform.

All operations are smooth functions of the C_l arrays. The Legendre transforms
are matrix-vector multiplies (precomputed Wigner d-matrix at quadrature angles).
Fully differentiable.

**Alternative**: The flat-sky approximation (valid for l >> 1) expresses lensing
as a 2D convolution in Fourier space, which is just array multiplication.

### 4.11 Spectral Distortions

**File**: `distortions.py`

**Purpose**: Compute CMB spectral distortion parameters μ and y.

**Physics**: Energy injection into the CMB photon field at different epochs creates
characteristic spectral distortions:

- **y-distortion** (late injection, z < 5×10⁴): Compton scattering thermalizes
  partially. `y = ∫ dτ (T_e - T_γ)/(m_e c²) n_e σ_T`

- **μ-distortion** (early injection, 5×10⁴ < z < 2×10⁶): photon number is not
  conserved, leading to a chemical potential.

The distortion parameters are integrals over the heating rate Q(z):

```
μ = ∫ dz (1/ρ_γ) dQ/dz J_μ(z)
y = ∫ dz (1/ρ_γ) dQ/dz J_y(z)
```

where J_μ(z) and J_y(z) are window functions.

For the standard LCDM model, the primary distortion source is Silk damping of
acoustic waves:
```
dQ/dz = ∫ dk k² P_R(k) W(k, z)
```

This requires the primordial power spectrum and the Silk damping scale from
thermodynamics. All integrals are smooth and differentiable.

### 4.12 Shooting Method

**File**: `shooting.py`

**Purpose**: Convert user-friendly parameters (e.g., 100*θ_s, σ_8) to internal
parameters (H0, A_s) by iterative root-finding, in a way that is differentiable.

**The problem**: CLASS allows the user to specify `100*theta_s` instead of `H0`.
Finding H0 requires running `background_init()` repeatedly until the computed
θ_s matches the target. This iterative procedure is not directly differentiable
(it involves a variable number of iterations).

**Solution**: Implicit differentiation via `jax.custom_vjp`.

At convergence, the shooting condition is satisfied:
```
f(H0; θ_s_target, other_params) = θ_s(H0) - θ_s_target = 0
```

By the implicit function theorem:
```
dH0/d(θ_s_target) = -(∂f/∂H0)⁻¹ ∂f/∂(θ_s_target) = 1 / (∂θ_s/∂H0)
```

More generally, for any downstream parameter p:
```
dH0/dp = -(∂f/∂H0)⁻¹ ∂f/∂p
```

Implementation:
```python
@jax.custom_vjp
def shoot_H0(theta_s_target, other_params):
    """Find H0 such that theta_s(H0) = theta_s_target."""
    # Forward: run Newton/secant iteration (not traced by JAX)
    H0 = _newton_solve(theta_s_target, other_params)
    return H0

def shoot_H0_fwd(theta_s_target, other_params):
    H0 = shoot_H0(theta_s_target, other_params)
    return H0, (H0, theta_s_target, other_params)  # residuals for bwd

def shoot_H0_bwd(res, g):
    H0, theta_s_target, other_params = res
    # Compute ∂θ_s/∂H0 and ∂θ_s/∂(other params) via AD
    theta_s_fn = lambda H0, p: _compute_theta_s(H0, p)
    d_theta_s_d_H0 = jax.grad(theta_s_fn, argnums=0)(H0, other_params)
    d_theta_s_d_params = jax.grad(theta_s_fn, argnums=1)(H0, other_params)
    # Implicit function theorem
    dH0_d_theta_s = 1.0 / d_theta_s_d_H0
    dH0_d_params = -dH0_d_theta_s * d_theta_s_d_params
    return (g * dH0_d_theta_s, g * dH0_d_params)

shoot_H0.defvjp(shoot_H0_fwd, shoot_H0_bwd)
```

This pattern extends to any shooting parameter (sigma_8 → A_s, etc.).

### 4.13 Spherical Bessel Functions

**File**: `bessel.py`

**Purpose**: Compute j_l(x) and derivatives for the transfer function integrals.

**Algorithm**: Miller's backward recurrence.

Starting from a large l_start > l_max and initial values j_{l_start+1} = 0,
j_{l_start} = 1, recur downward:

```
j_{l-1}(x) = (2l+1)/x · j_l(x) - j_{l+1}(x)
```

Then normalize using the identity `Σ_l (2l+1) j_l²(x) = 1` or by matching
`j_0(x) = sin(x)/x`.

This is a fixed-size loop (l_start iterations) -- JIT-compiles to efficient
code and is fully differentiable.

**Derivatives**: First and second derivatives are obtained from the recurrence:
```
j_l'(x) = j_{l-1}(x) - (l+1)/x · j_l(x)
```

**For large x**: The Bessel function oscillates rapidly. For the transfer function
integral, this means the integrand becomes oscillatory and the integral saturates.
A time-cut can safely truncate the integration range.

**For small x**: `j_l(x) ≈ x^l / (2l+1)!!`, which is numerically fine for
moderate l.

**Alternative**: For very high l (> 100), use the uniform asymptotic expansion
(Olver) or WKB approximation:
```
j_l(x) ≈ (1/√(2πl)) (ex/(2l))^l for x << l
j_l(x) ≈ cos(x - (l+1)π/2) / x  for x >> l
```

These smooth approximations are differentiable.

### 4.14 ODE Solver Utilities

**File**: `ode.py`

**Purpose**: Wrappers around Diffrax that set up the solver configuration
consistently across modules.

```python
def solve_stiff(rhs_fn, t_span, y0, saveat, params, prec):
    """Solve a stiff ODE system using Kvaerno5 (implicit ESDIRK).

    Args:
        rhs_fn: callable (t, y, args) -> dy
        t_span: (t0, t1)
        y0: initial state
        saveat: SaveAt specification
        params: passed as args to rhs_fn
        prec: PrecisionParams for tolerances and adjoint choice

    Returns:
        solution states at saveat points
    """
    solver = diffrax.Kvaerno5()
    controller = diffrax.PIDController(
        rtol=prec.pt_ode_rtol, atol=prec.pt_ode_atol
    )
    adjoint = {
        "recursive_checkpoint": diffrax.RecursiveCheckpointAdjoint(),
        "direct": diffrax.DirectAdjoint(),
    }[prec.ode_adjoint]

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs_fn),
        solver=solver,
        t0=t_span[0], t1=t_span[1],
        dt0=None,  # auto
        y0=y0,
        saveat=saveat,
        stepsize_controller=controller,
        adjoint=adjoint,
        max_steps=prec.ode_max_steps,
    )
    return sol


def solve_nonstiff(rhs_fn, t_span, y0, saveat, params, prec):
    """Solve a non-stiff ODE using Tsit5 (explicit RK4/5).

    Used for background integration where the system is not stiff.
    """
    solver = diffrax.Tsit5()
    controller = diffrax.PIDController(
        rtol=prec.bg_tol, atol=prec.bg_tol * 1e-3
    )
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs_fn),
        solver=solver,
        t0=t_span[0], t1=t_span[1],
        dt0=None,
        y0=y0,
        saveat=saveat,
        stepsize_controller=controller,
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        max_steps=prec.ode_max_steps,
    )
    return sol
```

---

## 5. Differentiation Strategy

### Outer vs inner differentiation

We use a **two-level differentiation strategy**:

1. **Inner level** (ODE solver): The implicit solver (Kvaerno5) uses a Newton iteration
   at each step to solve `F(y_{n+1}) = 0`. This Newton iteration requires the
   **Jacobian of the ODE RHS**, which we compute analytically (since the
   Einstein-Boltzmann equations are known in closed form). We do NOT use AD for
   this inner Jacobian -- it would be wasteful. Instead, we provide the Jacobian
   to the solver via a custom `AbstractTerm` in Diffrax.

   Concretely, this means implementing a function `jac_fn(t, y, k, bg, th, params)`
   that returns the analytically-computed Jacobian matrix of the perturbation RHS.
   The Jacobian has a known sparsity pattern (block-tridiagonal from the Boltzmann
   hierarchy), which can be exploited.

2. **Outer level** (parameter gradients): For computing d(C_l)/d(params), we use
   JAX's reverse-mode AD (via `jax.grad` or `jax.jacrev`) applied to the entire
   pipeline `params → C_l`. This differentiates through:
   - Background ODE solve (via RecursiveCheckpointAdjoint)
   - Thermodynamics ODE solve (via RecursiveCheckpointAdjoint)
   - Perturbation ODE solve (via RecursiveCheckpointAdjoint)
   - All interpolation, integration, and post-processing steps

### Adjoint method choice

For the perturbation ODE (the most expensive):

- **RecursiveCheckpointAdjoint** (default): Splits the time interval into chunks,
  recomputes the forward pass for each chunk during the backward pass. Memory
  cost is O(√T) where T is the number of steps, vs O(T) for direct backprop.
  Compute cost is ~2x the forward pass.

- **DirectAdjoint**: Stores all intermediate states. Memory cost O(T). Faster
  but may OOM for long integrations.

- **BacksolveAdjoint** (continuous adjoint): Solves the adjoint ODE backward.
  Cheapest in memory but can be inaccurate for stiff systems. NOT recommended
  for the perturbation system.

### Handling non-differentiable operations

| Operation | Issue | Solution |
|-----------|-------|----------|
| `jnp.searchsorted` (in spline eval) | Returns integers | Only used for index lookup; gradients flow through the polynomial evaluation, not the index |
| Root finding (z_reio from tau_reio) | Iterative | Implicit differentiation via `custom_vjp` |
| Shooting (H0 from theta_s) | Iterative | Implicit differentiation via `custom_vjp` |
| k-grid construction | Static | Not traced by AD (part of PrecisionParams) |
| l-grid construction | Static | Not traced by AD |
| Truncation of Boltzmann hierarchy | Discrete l_max | Fixed at compile time (static) |
| Limber switch | Discrete l threshold | Smooth sigmoid blending |

### What we differentiate through vs. what is static

**Traced** (JAX will compute gradients w.r.t. these):
- All CosmoParams fields (h, omega_b, omega_cdm, tau_reio, ln10A_s, n_s, w0, wa, m_ncdm, ...)
- All intermediate arrays (background tables, thermo tables, source functions, transfer functions, C_l's)
- ODE initial conditions
- ODE right-hand side evaluations

**Static** (not traced, compile-time constants):
- PrecisionParams (grid sizes, tolerances, l_max, solver choice)
- k-grid, tau-grid, l-grid positions
- Number of species, hierarchy truncation
- Which C_l types to compute

---

## 6. Validation and Testing

### Test design philosophy

The test harness is the most important part of this project. Without high-quality
tests, autonomous agents will solve the wrong problem. Lessons from the Anthropic
C compiler project (16 parallel agents, 2000 sessions, 100k-line compiler):

1. **Tests must be nearly perfect.** Agents will optimize for whatever the tests
   measure. If a test is wrong or has loose tolerances, agents will produce code
   that passes the bad test but gives wrong physics. Invest more time in the test
   harness than in the code it tests.

2. **Tests must give concise, actionable feedback.** Print the max relative error
   and where it occurs, not full arrays. Pre-compute aggregate statistics.
   Log details to files, not stdout, to avoid context window pollution.

3. **Tests must be fast by default.** Every test file supports a `--fast` mode
   (~10% subsample) for rapid iteration. Full validation runs before commits.
   The perturbation tests (slowest) should test ~5 k-modes in fast mode,
   spanning the full k range.

4. **Tests must decompose monolithic tasks.** The perturbation module is one
   giant coupled system -- if we only test the final output, all agents hit
   the same bug. Instead, test sub-components independently:
   - Metric equations with mock species inputs
   - Photon hierarchy with fixed metric
   - Initial conditions in isolation
   - Source function extraction with known ODE solution
   This lets different agents work on different subsystems.

5. **Tests must enable bisection.** When C_l disagrees, we need to find the
   first module in the pipeline that diverges from CLASS. The test suite
   should make this easy by testing every intermediate quantity, not just
   the final output. This is the "oracle bisection" pattern from the
   C compiler project.

### Test hierarchy

We use a layered testing approach, from unit tests to full pipeline validation.

#### Level 1: Unit tests (fast, no CLASS dependency)

| Test | What it checks |
|------|---------------|
| `test_constants.py` | Physical constants match CLASS values exactly |
| `test_interpolation.py` | Spline accuracy on known functions; gradient correctness |
| `test_bessel.py` | j_l(x) matches scipy.special.spherical_jn to 1e-10 |
| `test_params.py` | PyTree registration; parameter defaults; derived quantities |

#### Level 2: Module tests (compare against CLASS reference data)

For each module, pre-generate reference data from CLASS (via `scripts/generate_class_reference.py`)
and check agreement:

| Test | CLASS function | Quantity | Tolerance |
|------|---------------|----------|-----------|
| `test_background.py` | `background_at_z()` | H(z), D_A(z), D_L(z), r_s(z) at 100 z-values | < 0.01% |
| `test_background.py` | derived | z_eq, z_drag, conformal_age | < 0.01% |
| `test_thermodynamics.py` | thermo table | x_e(z), g(z), κ(z) at 1000 z-values | < 0.1% |
| `test_thermodynamics.py` | derived | z_reio, z_star, r_s(z_star) | < 0.1% |
| `test_perturbations.py` | source table | S_T(k,τ) at 10 k × 100 τ values | < 0.5% |
| `test_perturbations.py` | P(k) | Matter power spectrum at 100 k-values | < 0.1% |
| `test_transfer.py` | transfer table | Δ_l^T(k) at 50 l × 50 k values | < 0.5% |
| `test_harmonic.py` | C_l | TT, TE, EE at l = 2..2500 | < 0.1% |
| `test_lensing.py` | lensed C_l | TT, TE, EE, BB lensed at l = 2..2500 | < 0.2% |
| `test_nonlinear.py` | P_NL(k,z) | Non-linear P(k) at z=0,0.5,1,2 | < 1% |
| `test_distortions.py` | μ, y | Distortion amplitudes | < 1% |

#### Level 3: Gradient tests

For each module, verify that AD gradients match finite-difference gradients:

```python
def test_background_gradients():
    """Check d(H0_class)/d(omega_b) via AD matches finite differences."""
    def H_at_z10(params):
        bg = background_solve(params, prec)
        return eval_spline(bg.H_spline, z_to_loga(10.0))

    params = CosmoParams()
    grad_ad = jax.grad(H_at_z10)(params)

    eps = 1e-5
    params_plus = params.replace(omega_b=params.omega_b + eps)
    params_minus = params.replace(omega_b=params.omega_b - eps)
    grad_fd = (H_at_z10(params_plus) - H_at_z10(params_minus)) / (2 * eps)

    assert jnp.allclose(grad_ad.omega_b, grad_fd, rtol=1e-3)
```

Repeat for:
- Every module's output w.r.t. every CosmoParam field
- Full pipeline: d(C_l^TT)/d(params) at selected l values
- d(P(k))/d(params) at selected k values
- Fisher matrix: d²(log L)/d(params)² via autodiff vs numerical Hessian

| Test | Gradient of | w.r.t. | Tolerance vs finite diff |
|------|------------|--------|--------------------------|
| `test_gradients.py::test_bg_grad` | H(z=0), r_s | All 6 LCDM params | < 0.1% |
| `test_gradients.py::test_thermo_grad` | τ_reio, z_star | All 6 LCDM params | < 0.5% |
| `test_gradients.py::test_pk_grad` | P(k) at 20 k-values | All 6 LCDM params | < 1% |
| `test_gradients.py::test_cl_grad` | C_l^TT at l=2,10,100,1000,2000 | All 6 LCDM params | < 1% |
| `test_gradients.py::test_lensed_cl_grad` | Lensed C_l^TT at selected l | All 6 LCDM params | < 2% |
| `test_gradients.py::test_fisher` | Fisher matrix (6×6) | LCDM params | eigenvalues < 5% |

#### Level 4: Extended model tests

| Test | Model extension | What changes |
|------|----------------|-------------|
| `test_massive_nu.py` | m_ncdm = 0.06, 0.15, 0.3 eV | P(k) suppression, C_l shift |
| `test_w0wa.py` | w0 = -0.9, wa = 0.1 | Background, ISW, growth |
| `test_tensors.py` | r_t = 0.01, 0.1 | C_l^BB from GW |
| `test_high_ell.py` | l_max = 5000 | Silk damping tail |
| `test_neff.py` | N_ur = 2.0, 3.046, 4.0 | Phase shift, damping |

#### Level 5: End-to-end integration tests

| Test | Description |
|------|-------------|
| `test_planck_bestfit.py` | Run at Planck 2018 best-fit; compare all C_l to CLASS |
| `test_sampling_loop.py` | Run 10 HMC steps with numpyro; verify no NaN gradients |
| `test_jit_compilation.py` | Verify `jax.jit(compute)` compiles without error |
| `test_vmap_params.py` | Verify `jax.vmap(compute)(batch_of_params)` works |
| `test_gpu.py` | Run on GPU if available; verify same results as CPU |

### Reference data generation

```python
# scripts/generate_class_reference.py
"""Generate CLASS reference data for all test cases.

Requires: pip install classy (CLASS Python wrapper)

Generates:
  reference_data/
    lcdm_fiducial/
      background.npz    # z, H, D_A, D_L, r_s, ...
      thermodynamics.npz # z, x_e, visibility, kappa, c_s2, ...
      perturbations.npz  # k, tau, S_T, S_E, S_lens, ...
      pk.npz             # k, P_lin, P_NL
      cls.npz            # l, TT, TE, EE, BB, PP (unlensed)
      cls_lensed.npz     # l, TT, TE, EE, BB (lensed)
      derived.json       # z_eq, z_star, r_s_star, tau_reio, sigma8, ...
    massive_nu_006/
      ...
    w0wa_m09_01/
      ...
    tensors_r001/
      ...
"""
```

---

## 7. Performance Strategy

### Cost model

| Module | CLASS (1 CPU core) | clax (GPU) | Parallelism |
|--------|-------------------|----------------|-------------|
| Background | 0.05s | 0.01s | None needed |
| Thermodynamics | 0.3s | 0.05s | None needed |
| Perturbations | 3-5s | 0.3-0.5s | vmap over ~150 k-modes |
| Transfer | 1-2s | 0.1-0.2s | vmap over l × k |
| Harmonic | 0.2s | 0.02s | vmap over l |
| Lensing | 0.2s | 0.02s | Array operations |
| **Total forward** | **5-8s** | **0.5-1s** | |
| **Gradient (reverse)** | N/A | **1-3s** | ~2-4x forward |

### JIT compilation

The entire `compute()` function should be JIT-compiled:

```python
@jax.jit
def compute(params: CosmoParams, prec: PrecisionParams = PrecisionParams()) -> Result:
    ...
```

Since `PrecisionParams` is static (controls array shapes), it should be passed via
`static_argnums` or as a `static_field` in an Equinox module.

First call will be slow (~30-60s for XLA compilation). Subsequent calls with the
same shapes will be fast.

### Memory considerations

The perturbation solve stores the full source function table:
- 150 k-modes × 5000 τ-points × 5 source types × 4 bytes = ~15 MB

Plus the ODE states during backward pass (with checkpointing):
- Per k-mode: 330 variables × √(5000) checkpoints × 4 bytes ≈ 100 KB
- Total: 150 × 100 KB = 15 MB

Total GPU memory: ~50-100 MB. Well within GPU limits.

### Batching over cosmologies

For MCMC chains or Fisher matrix computation:
```python
# Compute C_l for a batch of 100 parameter sets
batch_compute = jax.vmap(compute)
batch_cls = batch_compute(batch_params)  # shape: (100, l_max, n_types)
```

This gives another dimension of parallelism.

---

## 8. Implementation Roadmap

### Phase 1: Foundation (weeks 1-2)

| Task | Files | Deliverable |
|------|-------|-------------|
| Package setup | `pyproject.toml`, `__init__.py` | Installable package |
| Constants | `constants.py` | All CLASS physical constants |
| Parameters | `params.py` | CosmoParams, PrecisionParams as PyTrees |
| Interpolation | `interpolation.py` | Cubic spline (1D, 2D), eval, build |
| ODE wrappers | `ode.py` | Stiff/non-stiff solve functions |
| Background | `background.py` | H(z), distances, growth factor |
| Reference data | `scripts/generate_class_reference.py` | LCDM reference |
| Background tests | `tests/test_background.py` | < 0.01% vs CLASS |

### Phase 2: Thermodynamics (weeks 3-4)

| Task | Files | Deliverable |
|------|-------|-------------|
| RECFAST solver | `thermodynamics.py` | x_e(z), T_m(z) |
| Reionization | `thermodynamics.py` | Tanh model, tau_reio matching |
| Visibility function | `thermodynamics.py` | g(z), κ(z), c_s²(z) |
| Thermo tests | `tests/test_thermodynamics.py` | < 0.1% vs CLASS |

### Phase 3: Perturbations (weeks 5-8) -- THE CORE

| Task | Files | Deliverable |
|------|-------|-------------|
| State vector layout | `perturbations.py` | Index management for all species |
| RHS function | `perturbations.py` | Full Boltzmann hierarchy in synchronous gauge |
| Analytical Jacobian | `perturbations.py` | Sparse Jacobian function for Kvaerno5 |
| Initial conditions | `perturbations.py` | Adiabatic IC from Ma & Bertschinger |
| Source extraction | `perturbations.py` | S_T, S_E, S_lens from ODE solution |
| k-sampling | `perturbations.py` | Logarithmic grid with acoustic refinement |
| vmap integration | `perturbations.py` | All k-modes in parallel |
| Perturbation tests | `tests/test_perturbations.py` | S(k,τ) and P(k) < 0.1% vs CLASS |

### Phase 4: Transfer + C_l (weeks 9-11)

| Task | Files | Deliverable |
|------|-------|-------------|
| Bessel functions | `bessel.py` | j_l(x) via backward recurrence |
| Transfer integration | `transfer.py` | Δ_l(k) for TT, TE, EE, PP |
| Limber approximation | `transfer.py` | Smooth blended Limber for large l |
| Primordial spectrum | `primordial.py` | P_R(k), P_T(k) |
| C_l computation | `harmonic.py` | Unlensed TT, TE, EE, BB, PP |
| Lensing | `lensing.py` | Lensed spectra via correlation function method |
| Spectra tests | `tests/test_harmonic.py`, `test_lensing.py` | C_l < 0.1% vs CLASS |

### Phase 5: Extensions (weeks 12-14)

| Task | Files | Deliverable |
|------|-------|-------------|
| Shooting method | `shooting.py` | theta_s → H0 with implicit diff |
| Tensor perturbations | `perturbations.py` | GW mode integration |
| HaloFit/HMCode | `nonlinear.py` | Non-linear P(k) |
| Spectral distortions | `distortions.py` | μ, y parameters |
| Number counts | `transfer.py`, `harmonic.py` | Galaxy C_l's |
| Extended tests | `tests/test_*.py` | All extensions validated |

### Phase 6: Gradients + Production (weeks 15-16)

| Task | Files | Deliverable |
|------|-------|-------------|
| Gradient validation | `tests/test_gradients.py` | AD vs finite diff for all modules |
| Fisher matrix | `tests/test_gradients.py` | Full Fisher test |
| Memory optimization | All modules | Checkpointed adjoint tuning |
| Benchmarks | `scripts/benchmark.py` | Timing vs CLASS, vs DISCO-EB |
| API polish | `__init__.py` | Clean public API |
| Sampling demo | `scripts/` | numpyro/blackjax example |

---

## 9. Repository Layout

```
clax/
├── CLAUDE.md                         # Development instructions for AI assistants
├── DESIGN.md                         # This document
├── CHANGELOG.md                       # Development progress log
├── pyproject.toml                    # Package configuration
├── clax/
│   ├── __init__.py                   # Public API: compute(), CosmoParams, etc.
│   ├── constants.py                  # Physical constants
│   ├── params.py                     # CosmoParams, PrecisionParams
│   ├── background.py                 # Background cosmology
│   ├── thermodynamics.py             # Recombination + reionization
│   ├── perturbations.py              # Einstein-Boltzmann system
│   ├── primordial.py                 # Primordial power spectrum
│   ├── nonlinear.py                  # HaloFit / HMCode
│   ├── transfer.py                   # Transfer functions
│   ├── harmonic.py                   # C_l computation
│   ├── lensing.py                    # Lensed C_l's
│   ├── distortions.py               # Spectral distortions
│   ├── shooting.py                   # Shooting method + implicit diff
│   ├── interpolation.py              # Spline utilities
│   ├── bessel.py                     # Spherical Bessel functions
│   ├── ode.py                        # ODE solver wrappers
│   └── utils.py                      # Misc utilities
├── tests/
│   ├── conftest.py                   # Fixtures: load reference data, default params
│   ├── test_constants.py
│   ├── test_interpolation.py
│   ├── test_bessel.py
│   ├── test_background.py
│   ├── test_thermodynamics.py
│   ├── test_perturbations.py
│   ├── test_primordial.py
│   ├── test_nonlinear.py
│   ├── test_transfer.py
│   ├── test_harmonic.py
│   ├── test_lensing.py
│   ├── test_distortions.py
│   ├── test_shooting.py
│   ├── test_gradients.py            # AD gradient validation
│   ├── test_end_to_end.py           # Full pipeline tests
│   ├── test_extended_models.py      # Massive nu, w0wa, tensors
│   └── test_performance.py          # Timing and memory benchmarks
├── scripts/
│   ├── generate_class_reference.py   # Generate reference data from CLASS
│   ├── benchmark.py                  # Performance benchmarks
│   ├── validate_all.py               # Run all validations, produce report
│   └── demo_sampling.py              # HMC sampling example
└── reference_data/                   # Pre-computed CLASS outputs
    ├── lcdm_fiducial/
    ├── massive_nu_006/
    ├── massive_nu_015/
    ├── massive_nu_030/
    ├── w0wa_m09_01/
    ├── tensors_r001/
    ├── tensors_r01/
    ├── high_neff/
    └── low_neff/
```

---

## 10. Open Questions and Risks

### Risk 1: Subtle physics errors that pass tests ("overfitting to the oracle")

Unlike a C compiler (binary correctness: works or doesn't), a Boltzmann code
can be *almost right* for the wrong reasons. A sign error or missing factor of
2 in a subdominant term might produce 0.05% error at fiducial LCDM but blow up
at different parameters. An agent could introduce a fudge factor to match the
test point without fixing the actual bug.

**Mitigations:**
- **Test at MANY parameter points**, not just fiducial LCDM. The reference data
  suite includes: fiducial LCDM, 3 neutrino masses, w0wa variations, tensor
  modes, high/low N_eff. A fudge factor that works at one point will fail elsewhere.
- **Test intermediate quantities, not just final C_l.** If C_l is wrong, we need
  to know whether it's background, thermodynamics, source functions, or the
  Bessel integral. Store and compare CLASS intermediate outputs at every stage.
- **Cross-reference every equation term by term against CLASS source code.**
  The perturbation RHS must be traceable to specific lines in CLASS's
  `perturbations.c`. Document which CLASS function each code block corresponds
  to (e.g., "// cf. perturbations.c:4520-4535, perturbations_einstein()").
  This makes review possible.
- **Compare against DISCO-EB too**, not just CLASS. If our code and DISCO-EB
  agree on P(k) but differ from CLASS, it's likely a CLASS approximation effect
  rather than a bug in our code.
- **Never add a numerical fudge factor.** If a test fails, find the actual bug.
  A 0.2% error means a term is wrong, not that a coefficient needs tuning.

### Risk 2: Perturbations as a monolithic coupled system

The Einstein-Boltzmann system is deeply coupled: the metric affects all species,
all species affect the metric. Unlike the C compiler's independent test cases,
you can't easily test one piece in isolation. If all agents work on perturbations
simultaneously, they hit the same bug and overwrite each other.

**Mitigations:**
- **Decompose into testable sub-components despite the coupling:**
  - Test initial conditions at τ_ini in isolation (compare y0 against CLASS).
  - Test the metric equations (η', h') given mock species inputs from CLASS
    (freeze δ_γ, θ_γ, etc. to CLASS values, check metric output).
  - Test individual hierarchy terms: set all other species to CLASS values,
    evolve only photons for a few steps, compare. Then only neutrinos. Etc.
  - Test source function extraction given a known ODE solution (use CLASS's
    perturbation table as input, check that our source extraction matches).
  - Test the full system at a single k-mode first, then expand.
- **Serialize perturbation work.** Don't parallelize perturbations until the
  basic system works for one k-mode. Then parallelize across k-modes (each
  mode's bugs are independent).
- **Use CLASS's perturbation output as intermediate oracle.** CLASS can output
  perturbation variables at specific k values (`k_output_values` parameter).
  Generate this data and compare our integration at every τ step, not just
  the final source functions.

### Risk 3: The approximation-free approach may not work well in JAX

SymBoltz.jl validated approximation-free integration in Julia using
KLUFactorization for sparse Jacobians with ~95% sparsity. Diffrax's Kvaerno5
uses dense linear algebra. For a ~330-equation system (with massive neutrinos,
15 momentum bins × 18 multipoles), a dense 330×330 LU per Newton step could
be slow.

**Mitigations:**
- **Start without massive neutrinos.** The system is ~60 equations
  (2 metric + 1 CDM + 2 baryons + 18 photons + 18 polarization + 18 ur).
  Dense 60×60 LU is cheap. Get this working first.
- **Add neutrinos incrementally.** First 1 momentum bin (adds 18 eq → 78 total),
  then increase to 15 bins if performance allows.
- **Profile early.** After the first k-mode works, measure: how many steps does
  Kvaerno5 take? What fraction of time is in the linear solve vs RHS evaluation?
  If the linear solve dominates, consider:
  (a) Banded Jacobian (the Boltzmann hierarchy is tridiagonal in l).
  (b) Custom solver using the block structure.
  (c) Falling back to `jax.experimental.sparse` if it has matured.
- **Fallback: selective approximations.** If fully approximation-free is too
  slow, reintroduce TCA as a smooth blend (sigmoid weighting between tight-
  coupling and full equations) rather than a hard switch. This preserves
  differentiability while reducing the effective system size at early times.

### Risk 4: Gradient correctness is a second dimension of bugs

We need correct output AND correct derivatives. Gradient bugs are insidious:
NaN propagation through a single bad operation, wrong `custom_vjp` residuals
that silently give wrong gradients, checkpointing errors in the adjoint that
lose information. These often manifest as MCMC chains that don't converge
rather than obviously wrong numbers.

**Mitigations:**
- **Test gradients for EVERY module independently.** Don't just test the
  end-to-end gradient. If d(C_l)/d(omega_b) is wrong, is it because
  d(H)/d(omega_b) is wrong, or d(x_e)/d(omega_b), or d(S)/d(omega_b)?
  The gradient test suite must be as granular as the value test suite.
- **Use both forward and reverse mode.** For scalar outputs, `jax.grad`
  (reverse) must match `jax.jvp` (forward). Disagreement indicates a bug
  in one of the modes (usually the custom_vjp).
- **Finite difference at multiple step sizes.** Use eps = 1e-4, 1e-5, 1e-6
  and check that the FD estimate converges. If it doesn't, the function may
  not be smooth (indicating a non-differentiable operation sneaked in).
- **Test gradient of simple sub-functions first.** Before testing
  d(C_l)/d(params), test d(H(z=0))/d(h), d(x_e(z=1000))/d(omega_b), etc.
  Build confidence from the bottom up.
- **Watch for NaN/Inf.** Add `jax.debug.print` checks or use
  `jax.config.update("jax_debug_nans", True)` during development to catch
  NaN propagation early.

### Risk 5: CLASS oracle is imperfect (approximation differences)

CLASS uses TCA/RSA/UFA approximations that we don't. At the ~0.1% level, some
discrepancy is expected because we're computing the same physics with different
numerical methods. This means we can't blindly trust a 0.1% disagreement as
"our bug" -- it might be CLASS's approximation error.

**Mitigations:**
- **Compare against multiple references.** Use CAMB as a second oracle. If our
  code and CAMB agree but differ from CLASS, it's likely a CLASS approximation
  effect. If we differ from both, it's our bug.
- **Compare against DISCO-EB for P(k).** DISCO-EB is also approximation-free
  in JAX, so it should match us very closely for the quantities it computes.
- **Increase CLASS precision.** Run CLASS with very tight precision settings
  (`cl_ref.pre` or tighter) to minimize its approximation errors.
- **Accept ~0.1% as the floor, not 0.01%.** The approximation-free vs
  approximation-switching difference sets a floor on agreement. Don't chase
  phantom bugs below this level. Focus on getting ALL modules below 0.1%
  rather than pushing one module to 0.01%.

### Technical risks

6. **Diffrax + stiff ODE performance**: Kvaerno5 in Diffrax may not match CLASS's
   NDF15 in step efficiency for the Boltzmann system. If performance is poor,
   we may need to implement a custom BDF solver in JAX, or use Diffrax's
   `ImplicitEuler` with small steps as a fallback.

7. **Numerical stability at early times**: The tightly-coupled era has
   Δτ_step ~ (a n_e σ_T)⁻¹ << τ, which is a very stiff regime. The implicit
   solver should handle this, but may require very small initial step sizes.
   If problematic, consider starting integration at a later time and using
   analytical WKB solutions for the early radiation era.

8. **Bessel function accuracy at high l**: Miller's backward recurrence is
   stable, but for l > 1000 and large x, accuracy may degrade. Validate
   carefully against scipy. Consider using the Olver uniform asymptotic
   expansion as an alternative for l > l_cutoff.

9. **Lensing accuracy**: The correlation function method involves Legendre
   transforms that can be expensive for large l_max. Consider the flat-sky
   approximation for l > 100 and the full curved-sky only for l < 100.

10. **XLA compilation time**: The full pipeline JIT may take 30-60 seconds to
    compile. This is a one-time cost, but annoying during development. Use
    `jax.jit(lower=True).compile()` to separate tracing from compilation.

### Design questions

1. **Should we support non-flat geometries in v1?** Adding K ≠ 0 changes the
   Bessel functions (j_l → Φ_l^ν, hyperspherical), complicates the distance-
   redshift relation, and adds ~20% more code. Recommendation: defer to v2.

2. **Single massive neutrino species or arbitrary N_ncdm?** CLASS supports
   arbitrary numbers of non-cold species with different masses and temperatures.
   For v1, support 1 species with configurable mass. Use the degenerate hierarchy
   approximation (all species have the same mass m_ncdm = Σm_ν / N_ncdm).

3. **Should we provide a `classy`-compatible API?** A drop-in replacement for the
   `classy.Class` Python interface would ease adoption. But it uses mutable state
   (`set()`, `compute()`, `get_cls()`), which conflicts with our pure-function
   design. Recommendation: provide a thin compatibility wrapper that internally
   calls our pure functions.

4. **How to handle the analytical Jacobian for the implicit solver?** Options:
   (a) Hand-code the full Jacobian matrix (error-prone but fastest).
   (b) Use `jax.jacfwd` to compute it automatically (slower but correct by construction).
   (c) Use a hybrid: `jax.jacfwd` for development/validation, then optimize
   hot spots by hand.
   Recommendation: start with (b), profile, optimize to (c) if needed.

5. **RecursiveCheckpointAdjoint vs DirectAdjoint**: The memory/compute tradeoff
   depends on the problem size and GPU memory. Make this configurable via
   `PrecisionParams.ode_adjoint`.
