"""Shooting method for jaxCLASS.

Converts user-friendly parameters (e.g., 100*theta_s) to internal
parameters (H0) by iterative root-finding, in a differentiable way
via jax.custom_vjp and the implicit function theorem.

Key function:
    shoot_h_from_theta_s(theta_s_target, other_params, prec) -> h

The shooting method finds h such that 100*theta_s(h) = target, where
theta_s = r_s(z_rec) / r_a(z_rec) following CLASS convention.
Here r_a = D_A * (1+z) = chi for flat universe (comoving angular diameter distance).

References:
    DESIGN.md Section 4.12
    CLASS input.c: shooting method (lines 1457-1460)
    CLASS thermodynamics.c:1809: 100*theta_s = 100*rs_rec/ra_rec
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxclass.background import background_solve, BackgroundResult
from jaxclass.thermodynamics import thermodynamics_solve, ThermoResult
from jaxclass.params import CosmoParams, PrecisionParams


def _compute_theta_s(h: float, params_template: CosmoParams, prec: PrecisionParams) -> float:
    """Compute 100*theta_s for a given h value.

    Follows CLASS convention (input.c:1459):
        100*theta_s = 100 * rs_rec / ra_rec
    where:
        rs_rec = comoving sound horizon at z_rec (max of visibility function)
        ra_rec = da_rec * (1 + z_rec) = comoving angular diameter distance at z_rec
               = chi(z_rec) = tau_0 - tau_rec  (for flat universe)

    Uses the full thermodynamics solver to find z_rec (visibility peak).
    """
    params = params_template.replace(h=h)
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)

    # rs_rec: comoving sound horizon at z_rec (visibility peak)
    # CLASS uses z_rec (visibility max), not z_star (optical depth = 1)
    # Our ThermoResult stores both; use z_rec for CLASS compatibility
    # th.rs_star is r_s at z_star -- we need r_s at z_rec
    # Since we store z_rec, compute rs at z_rec from background
    loga_rec = jnp.log(1.0 / (1.0 + th.z_rec))
    rs_rec = bg.rs_of_loga.evaluate(loga_rec)

    # ra_rec = comoving angular diameter distance = comoving distance (flat)
    tau_rec = bg.tau_of_loga.evaluate(loga_rec)
    ra_rec = bg.conformal_age - tau_rec

    # cf. CLASS input.c:1459: output[i] = 100.*th.rs_rec/th.ra_rec
    theta_s_100 = 100.0 * rs_rec / ra_rec
    return theta_s_100


def make_shoot_h_from_theta_s(prec: PrecisionParams):
    """Create a shooting function with prec as a closure variable.

    PrecisionParams is not a valid JAX type (not registered as a pytree),
    so it cannot be passed through custom_vjp. Instead, we close over it.

    Args:
        prec: precision parameters (static, becomes closure)

    Returns:
        shoot_fn: function(theta_s_100_target, params_template) -> h
    """

    @jax.custom_vjp
    def shoot_fn(theta_s_100_target: float, params_template: CosmoParams) -> float:
        """Find h such that 100*theta_s(h) = theta_s_100_target.

        Uses Newton's method with a fixed number of iterations.
        Initial guess from CLASS input.c:1190:
            h_guess = 3.54*theta_s^2 - 5.455*theta_s + 2.548
        """
        # CLASS's initial guess formula (input.c:1190)
        h0 = 3.54 * theta_s_100_target**2 - 5.455 * theta_s_100_target + 2.548

        def newton_step(i, h):
            theta_s = _compute_theta_s(h, params_template, prec)
            # Finite difference derivative for the forward Newton solve
            eps = 1e-4
            theta_s_plus = _compute_theta_s(h + eps, params_template, prec)
            dtheta_dh = (theta_s_plus - theta_s) / eps
            # Damped Newton update to prevent oscillation
            # (the discrete z_rec grid can cause 2-cycles in undamped Newton)
            update = (theta_s - theta_s_100_target) / dtheta_dh
            h = h - 0.5 * update
            return h

        h_final = jax.lax.fori_loop(0, 25, newton_step, h0)
        return h_final

    def shoot_fwd(theta_s_100_target, params_template):
        h = shoot_fn(theta_s_100_target, params_template)
        return h, (h, theta_s_100_target, params_template)

    def shoot_bwd(res, g):
        h, theta_s_target, params_template = res
        # Implicit function theorem:
        # F(h, theta_s_target) = theta_s(h) - theta_s_target = 0
        # d(h)/d(theta_s_target) = 1 / (d(theta_s)/d(h))
        # by the implicit function theorem.
        dtheta_dh = jax.grad(lambda h_: _compute_theta_s(h_, params_template, prec))(h)
        dh_dtheta = 1.0 / dtheta_dh
        # Return gradient w.r.t. (theta_s_100_target, params_template)
        return (g * dh_dtheta, None)

    shoot_fn.defvjp(shoot_fwd, shoot_bwd)
    return shoot_fn
