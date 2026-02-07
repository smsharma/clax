"""jaxCLASS: Fully differentiable CLASS Boltzmann solver in JAX.

Usage:
    import jaxclass

    # Define parameters
    params = jaxclass.CosmoParams(h=0.6736, omega_b=0.02237, omega_cdm=0.1200)

    # Compute everything
    result = jaxclass.compute(params)
    print(result.bg.conformal_age)  # conformal age in Mpc

    # Differentiate
    grad = jax.grad(lambda p: jaxclass.compute(p).bg.H0)(params)
"""

import jax
jax.config.update("jax_enable_x64", True)

from jaxclass.constants import *  # noqa: F401,F403
from jaxclass.params import CosmoParams, PrecisionParams  # noqa: F401
from jaxclass.background import background_solve, BackgroundResult, H_of_z, angular_diameter_distance  # noqa: F401
from jaxclass.thermodynamics import thermodynamics_solve, ThermoResult  # noqa: F401
from jaxclass.perturbations import PerturbationResult, TensorPerturbationResult, tensor_perturbations_solve  # noqa: F401
from jaxclass.primordial import primordial_scalar_pk, primordial_tensor_pk  # noqa: F401
from jaxclass.harmonic import compute_cl_bb  # noqa: F401

from dataclasses import dataclass
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ComputeResult:
    """Result of the full jaxCLASS computation pipeline."""
    bg: BackgroundResult
    th: ThermoResult
    # pt: PerturbationResult  # optional, expensive

    def tree_flatten(self):
        return [self.bg, self.th], None

    @classmethod
    def tree_unflatten(cls, aux, fields):
        return cls(*fields)


def compute(
    params: CosmoParams = CosmoParams(),
    prec: PrecisionParams = PrecisionParams(),
) -> ComputeResult:
    """Run the full jaxCLASS pipeline.

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


def compute_pk(
    params: CosmoParams = CosmoParams(),
    prec: PrecisionParams = PrecisionParams(),
    k: float = 0.05,
) -> float:
    """Compute matter power spectrum P(k) at a single k-mode.

    Runs the full pipeline: background → thermodynamics → perturbation ODE.
    Differentiable w.r.t. params.

    Args:
        params: cosmological parameters
        prec: precision parameters
        k: wavenumber in Mpc^-1

    Returns:
        P(k) in Mpc^3
    """
    import diffrax
    from jaxclass.perturbations import _build_indices, _adiabatic_ic, _perturbation_rhs

    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)

    l_max = prec.pt_l_max_g
    idx = _build_indices(l_max, prec.pt_l_max_pol_g, prec.pt_l_max_ur)

    tau_ini = 0.5
    tau_end = bg.conformal_age * 0.999

    y0 = _adiabatic_ic(k, jnp.array(tau_ini), bg, params, idx, idx['n_eq'])
    args = (k, bg, th, params, idx, l_max, prec.pt_l_max_pol_g, prec.pt_l_max_ur)

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(_perturbation_rhs),
        solver=diffrax.Kvaerno5(),
        t0=tau_ini, t1=tau_end, dt0=tau_ini * 0.1,
        y0=y0, saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=diffrax.PIDController(
            rtol=prec.pt_ode_rtol, atol=prec.pt_ode_atol
        ),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        max_steps=prec.ode_max_steps,
        args=args,
    )

    y_f = sol.ys[-1]
    rho_b = bg.rho_b_of_loga.evaluate(jnp.array(0.0))
    rho_cdm = bg.rho_cdm_of_loga.evaluate(jnp.array(0.0))
    delta_m = (rho_b * y_f[idx['delta_b']] + rho_cdm * y_f[idx['delta_cdm']]) / (rho_b + rho_cdm)

    A_s = jnp.exp(params.ln10A_s) / 1e10
    pk = 2.0 * jnp.pi**2 / k**3 * A_s * (k / params.k_pivot)**(params.n_s - 1) * delta_m**2
    return pk
