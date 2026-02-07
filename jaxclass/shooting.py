"""Shooting method for jaxCLASS.

Converts user-friendly parameters (e.g., 100*theta_s) to internal
parameters (H0) by iterative root-finding, in a differentiable way
via jax.custom_vjp and the implicit function theorem.

Key function:
    shoot_H0_from_theta_s(theta_s_target, other_params) -> H0

References:
    DESIGN.md Section 4.12
    CLASS input.c: shooting method
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxclass.background import background_solve, BackgroundResult
from jaxclass.params import CosmoParams, PrecisionParams


def _compute_theta_s(h: float, params_template: CosmoParams, prec: PrecisionParams) -> float:
    """Compute 100*theta_s for a given h value.

    theta_s = r_s(z_star) / D_A(z_star) where z_star is the recombination redshift.
    We approximate z_star from the sound horizon and angular diameter distance.
    """
    params = params_template.replace(h=h)
    bg = background_solve(params, prec)

    # Approximate z_star ~ 1090 (could use thermodynamics for more accuracy)
    z_star = 1090.0
    loga_star = jnp.log(1.0 / (1.0 + z_star))

    rs = bg.rs_of_loga.evaluate(loga_star)
    tau_star = bg.tau_of_loga.evaluate(loga_star)
    chi_star = bg.conformal_age - tau_star
    da_star = chi_star / (1.0 + z_star)

    theta_s = rs / da_star
    return 100.0 * theta_s


@jax.custom_vjp
def shoot_h_from_theta_s(
    theta_s_100_target: float,
    params_template: CosmoParams,
    prec: PrecisionParams,
) -> float:
    """Find h such that 100*theta_s(h) = theta_s_100_target.

    Uses Newton's method for the forward solve, and implicit differentiation
    (via custom_vjp) for the backward pass.

    Args:
        theta_s_100_target: target value of 100*theta_s (e.g., 1.04110)
        params_template: template CosmoParams (h will be overwritten)
        prec: precision parameters

    Returns:
        h value such that 100*theta_s = target
    """
    # Newton iteration
    h = 0.6736  # initial guess
    for _ in range(10):
        theta_s = _compute_theta_s(h, params_template, prec)
        # Finite difference derivative
        eps = 1e-4
        dtheta_dh = (_compute_theta_s(h + eps, params_template, prec) - theta_s) / eps
        h = h - (theta_s - theta_s_100_target) / dtheta_dh
    return h


def _shoot_fwd(theta_s_100_target, params_template, prec):
    h = shoot_h_from_theta_s(theta_s_100_target, params_template, prec)
    return h, (h, theta_s_100_target, params_template, prec)


def _shoot_bwd(res, g):
    h, theta_s_target, params_template, prec = res
    # Implicit function theorem:
    # d(h)/d(theta_s_target) = 1 / (d(theta_s)/d(h))
    dtheta_dh = jax.grad(lambda h_: _compute_theta_s(h_, params_template, prec))(h)
    dh_dtheta = 1.0 / dtheta_dh
    return (g * dh_dtheta, None, None)


shoot_h_from_theta_s.defvjp(_shoot_fwd, _shoot_bwd)
