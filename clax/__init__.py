"""clax: Fully differentiable CLASS Boltzmann solver in JAX.

Usage:
    import clax

    # Define parameters
    params = clax.CosmoParams(h=0.6736, omega_b=0.02237, omega_cdm=0.1200)

    # Compute everything
    result = clax.compute(params)
    print(result.bg.conformal_age)  # conformal age in Mpc

    # Differentiate
    grad = jax.grad(lambda p: clax.compute(p).bg.H0)(params)
"""

import jax
jax.config.update("jax_enable_x64", True)

from clax.constants import *  # noqa: F401,F403
from clax.params import CosmoParams, PrecisionParams  # noqa: F401
from clax.background import background_solve, BackgroundResult, H_of_z, angular_diameter_distance  # noqa: F401
from clax.thermodynamics import thermodynamics_solve, ThermoResult  # noqa: F401
from clax.perturbations import MatterPerturbationResult, PerturbationResult, TensorPerturbationResult, perturbations_solve, perturbations_solve_mpk, tensor_perturbations_solve  # noqa: F401
from clax.primordial import primordial_scalar_pk, primordial_tensor_pk  # noqa: F401
from clax.harmonic import compute_cl_bb, compute_cl_tt, compute_cl_ee, compute_cl_te, compute_cls_all, compute_cls_all_fast  # noqa: F401
from clax.transfer import compute_pk_from_perturbations, compute_linear_matter_pk_from_perturbations  # noqa: F401

import functools
from dataclasses import dataclass, replace as dataclass_replace
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ComputeResult:
    """Result of the full clax computation pipeline."""
    bg: BackgroundResult
    th: ThermoResult
    # pt: PerturbationResult  # optional, expensive

    def tree_flatten(self):
        return [self.bg, self.th], None

    @classmethod
    def tree_unflatten(cls, aux, fields):
        return cls(*fields)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LinearMatterPowerResult:
    """Reusable linear matter power table backed by one PK-specific perturbation solve."""

    params: CosmoParams
    prec: PrecisionParams
    bg: BackgroundResult
    th: ThermoResult
    pt: PerturbationResult | MatterPerturbationResult
    z: float
    k_grid: jnp.ndarray
    pk_grid: jnp.ndarray

    def tree_flatten(self):
        children = [
            self.params,
            self.bg,
            self.th,
            self.pt,
            jnp.asarray(self.z),
            self.k_grid,
            self.pk_grid,
        ]
        return children, {"prec": self.prec}

    @classmethod
    def tree_unflatten(cls, aux, fields):
        params, bg, th, pt, z, k_grid, pk_grid = fields
        return cls(
            params=params,
            prec=aux["prec"],
            bg=bg,
            th=th,
            pt=pt,
            z=z,
            k_grid=k_grid,
            pk_grid=pk_grid,
        )

    @property
    def solve_k_grid(self) -> jnp.ndarray:
        """Return the internal PK perturbation solve grid in ``Mpc^-1``."""
        return self.pt.k_grid

    def delta_m(self, k, z: float | None = None):
        """Evaluate ``delta_m(k, z)`` from the stored perturbation table."""
        k_eval = _as_positive_1d("k", k)
        z_eval = self.z if z is None else z
        delta_m = compute_pk_from_perturbations(self.pt, self.bg, k_eval, z=z_eval)
        return _maybe_scalar_output(k, delta_m)

    def pk(self, k=None, z: float | None = None):
        """Evaluate linear matter ``P(k, z)`` from the stored perturbation table."""
        if k is None and z is None:
            return self.pk_grid

        k_eval = self.k_grid if k is None else _as_positive_1d("k", k)
        z_eval = self.z if z is None else z
        pk = compute_linear_matter_pk_from_perturbations(
            self.pt,
            self.bg,
            self.params,
            k_eval,
            z=z_eval,
        )
        if k is None:
            return pk
        return _maybe_scalar_output(k, pk)

    __call__ = pk


def _as_positive_1d(name: str, values) -> jnp.ndarray:
    """Validate and normalize a positive one-dimensional ``k`` array."""
    arr = jnp.atleast_1d(jnp.asarray(values, dtype=float))
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if bool(jnp.any(arr <= 0.0)):
        raise ValueError(f"{name} must contain only positive values.")
    return arr


def _maybe_scalar_output(input_values, output_values):
    """Match scalar inputs to scalar outputs for convenience."""
    if jnp.asarray(input_values).ndim == 0:
        return output_values[0]
    return output_values


def _resolve_pk_query_grid(
    *,
    k_eval,
    kmin,
    kmax,
    num,
    default_grid: jnp.ndarray,
) -> jnp.ndarray:
    """Build the user-facing ``k`` grid for table evaluation."""
    if k_eval is not None:
        return _as_positive_1d("k_eval", k_eval)

    if kmin is None and kmax is None and num is None:
        return default_grid

    kmin_val = float(default_grid[0]) if kmin is None else float(kmin)
    kmax_val = float(default_grid[-1]) if kmax is None else float(kmax)
    if kmin_val <= 0.0 or kmax_val <= 0.0:
        raise ValueError("kmin and kmax must be positive.")
    if kmax_val <= kmin_val:
        raise ValueError("kmax must be strictly larger than kmin.")

    num_val = default_grid.shape[0] if num is None else int(num)
    if num_val < 2:
        raise ValueError("num must be at least 2.")

    return jnp.geomspace(kmin_val, kmax_val, num_val)


def _resolve_pk_precision(
    prec: PrecisionParams,
    *,
    k_eval,
    kmax,
) -> PrecisionParams:
    """Extend the perturbation solve range when the requested ``k`` grid needs it."""
    requested_k = None
    if k_eval is not None:
        requested_k = float(jnp.max(_as_positive_1d("k_eval", k_eval)))
    elif kmax is not None:
        requested_k = float(kmax)

    if requested_k is None:
        return prec

    if requested_k > prec.pt_k_max_pk:
        raise ValueError(
            f"Requested k={requested_k:.6g} Mpc^-1 exceeds pt_k_max_pk={prec.pt_k_max_pk:.6g}."
        )

    solve_kmax = min(
        max(1.25 * requested_k, 1.01 * prec.pt_k_min),
        prec.pt_k_max_pk,
    )
    if solve_kmax == prec.pt_k_max_cl:
        return prec

    return dataclass_replace(prec, pt_k_max_cl=solve_kmax)


@functools.partial(jax.jit, static_argnums=(1,))
def compute(
    params: CosmoParams = CosmoParams(),
    prec: PrecisionParams = PrecisionParams(),
) -> ComputeResult:
    """Run the full clax pipeline.

    Computes background cosmology and thermodynamics.
    For perturbations/C_l, use perturbations_solve() separately
    (it's expensive and not always needed).

    Args:
        params: cosmological parameters (JAX-traced for AD)
        prec: precision parameters (static)

    Returns:
        ComputeResult with background and thermodynamics results
    """
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    return ComputeResult(bg=bg, th=th)


def compute_pk_table(
    params: CosmoParams = CosmoParams(),
    prec: PrecisionParams = PrecisionParams(),
    *,
    z: float = 0.0,
    k_eval=None,
    kmin: float | None = None,
    kmax: float | None = None,
    num: int | None = None,
    pt_pid_pcoeff: float = 0.25,
    pt_pid_icoeff: float = 0.80,
    pt_pid_dcoeff: float = 0.0,
    pt_pid_factormax: float = 20.0,
    pt_pid_factormin: float = 0.3,
) -> LinearMatterPowerResult:
    """Compute reusable linear-matter ``P(k, z)`` values from one perturbation solve.

    This follows the CLASS-style table strategy: solve a dedicated matter-power
    perturbation system on an internal logarithmic ``k`` grid once, then
    interpolate to the requested ``k`` values.

    This is the practical high-throughput API for dense spectra, reusable power
    tables, and inference workflows. It is usually the preferred path on GPU
    and whenever one perturbation solve is reused across many ``k`` queries.

    Args:
        params: cosmological parameters
        prec: precision parameters
        z: redshift at which to tabulate ``P(k, z)``
        k_eval: optional evaluation grid in ``Mpc^-1``
        kmin: minimum ``k`` when building a logarithmic evaluation grid
        kmax: maximum ``k`` when building a logarithmic evaluation grid
        num: number of points when building a logarithmic evaluation grid
        pt_pid_*: optional DISCO-EB-style scalar PID gain and step-factor tuning.
            The filtered norm variables and k-dependent weights are fixed
            internal controller policy.

    Returns:
        ``LinearMatterPowerResult`` containing the solve context plus the
        requested ``k`` grid and ``P(k, z)`` values.
    """
    solve_prec = _resolve_pk_precision(prec, k_eval=k_eval, kmax=kmax)
    bg = background_solve(params, solve_prec)
    th = thermodynamics_solve(params, solve_prec, bg)
    pt = perturbations_solve_mpk(
        params,
        solve_prec,
        bg,
        th,
        pt_pid_pcoeff=pt_pid_pcoeff,
        pt_pid_icoeff=pt_pid_icoeff,
        pt_pid_dcoeff=pt_pid_dcoeff,
        pt_pid_factormax=pt_pid_factormax,
        pt_pid_factormin=pt_pid_factormin,
    )

    k_query = _resolve_pk_query_grid(
        k_eval=k_eval,
        kmin=kmin,
        kmax=kmax,
        num=num,
        default_grid=pt.k_grid,
    )
    pk_query = compute_linear_matter_pk_from_perturbations(pt, bg, params, k_query, z=z)
    return LinearMatterPowerResult(
        params=params,
        prec=solve_prec,
        bg=bg,
        th=th,
        pt=pt,
        z=z,
        k_grid=k_query,
        pk_grid=pk_query,
    )


def compute_pk_interpolator(
    params: CosmoParams = CosmoParams(),
    prec: PrecisionParams = PrecisionParams(),
    *,
    z: float = 0.0,
    k_eval=None,
    kmin: float | None = None,
    kmax: float | None = None,
    num: int | None = None,
    pt_pid_pcoeff: float = 0.25,
    pt_pid_icoeff: float = 0.80,
    pt_pid_dcoeff: float = 0.0,
    pt_pid_factormax: float = 20.0,
    pt_pid_factormin: float = 0.3,
) -> LinearMatterPowerResult:
    """Return a reusable perturbation-table-backed linear-matter power interpolator."""
    return compute_pk_table(
        params,
        prec,
        z=z,
        k_eval=k_eval,
        kmin=kmin,
        kmax=kmax,
        num=num,
        pt_pid_pcoeff=pt_pid_pcoeff,
        pt_pid_icoeff=pt_pid_icoeff,
        pt_pid_dcoeff=pt_pid_dcoeff,
        pt_pid_factormax=pt_pid_factormax,
        pt_pid_factormin=pt_pid_factormin,
    )


def compute_pk(
    params: CosmoParams = CosmoParams(),
    prec: PrecisionParams = PrecisionParams(),
    k: float = 0.05,
    *,
    pt_pid_pcoeff: float = 0.25,
    pt_pid_icoeff: float = 0.80,
    pt_pid_dcoeff: float = 0.0,
    pt_pid_factormax: float = 20.0,
    pt_pid_factormin: float = 0.3,
) -> float:
    """Compute matter power spectrum P(k) at a single k-mode.

    Runs the full pipeline: background → thermodynamics → perturbation ODE.
    Differentiable w.r.t. params.

    This is the exact single-mode reference path. Use it for diagnostics,
    spot checks, and isolated local gradients. For practical multi-``k``
    workflows, prefer ``compute_pk_table()`` / ``compute_pk_interpolator()``.

    Args:
        params: cosmological parameters
        prec: precision parameters
        k: wavenumber in Mpc^-1
        pt_pid_*: optional DISCO-EB-style scalar PID gain and step-factor tuning.
            The filtered norm variables and k-dependent weights are fixed
            internal controller policy.

    Returns:
        P(k) in Mpc^3
    """
    from clax.perturbations import (
        _matter_delta_m_single_k_impl,
        _resolve_scalar_pid_config,
    )

    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)

    pid_config = _resolve_scalar_pid_config(
        pt_pid_pcoeff=pt_pid_pcoeff,
        pt_pid_icoeff=pt_pid_icoeff,
        pt_pid_dcoeff=pt_pid_dcoeff,
        pt_pid_factormax=pt_pid_factormax,
        pt_pid_factormin=pt_pid_factormin,
    )
    delta_m = _matter_delta_m_single_k_impl(params, prec, bg, th, pid_config, k)
    primordial = primordial_scalar_pk(jnp.asarray([k]), params)[0]
    return 2.0 * jnp.pi**2 / k**3 * primordial * delta_m**2
