"""ODE solver wrappers around Diffrax for jaxCLASS.

Provides consistent interfaces for non-stiff (background) and stiff
(perturbation) ODE integration, with configurable adjoint methods
for reverse-mode AD.

References:
    Diffrax docs: https://docs.kidger.site/diffrax/
    DISCO-EB: uses custom Rosenbrock solvers; we start with Diffrax builtins.
"""

import diffrax
import jax.numpy as jnp
from jaxtyping import Array, Float


def solve_nonstiff(
    rhs_fn,
    t0: float,
    t1: float,
    y0: Float[Array, "D"],
    saveat: diffrax.SaveAt,
    args=None,
    rtol: float = 1e-10,
    atol: float = 1e-13,
    max_steps: int = 16384,
    adjoint: str = "recursive_checkpoint",
):
    """Solve a non-stiff ODE system using Tsit5 (explicit RK4/5).

    Used for background integration (Friedmann equation is not stiff).

    Args:
        rhs_fn: callable (t, y, args) -> dy, the ODE right-hand side
        t0: initial time
        t1: final time
        y0: initial state vector
        saveat: Diffrax SaveAt specification (e.g., SaveAt(ts=time_grid))
        args: additional arguments passed to rhs_fn
        rtol: relative tolerance
        atol: absolute tolerance
        max_steps: maximum number of solver steps
        adjoint: "recursive_checkpoint" or "direct"

    Returns:
        Diffrax solution object with .ys (saved states) and .ts (saved times)
    """
    solver = diffrax.Tsit5()
    controller = diffrax.PIDController(rtol=rtol, atol=atol)
    adjoint_obj = _get_adjoint(adjoint)

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs_fn),
        solver=solver,
        t0=t0,
        t1=t1,
        dt0=None,
        y0=y0,
        saveat=saveat,
        stepsize_controller=controller,
        adjoint=adjoint_obj,
        max_steps=max_steps,
        args=args,
    )
    return sol


def solve_stiff(
    rhs_fn,
    t0: float,
    t1: float,
    y0: Float[Array, "D"],
    saveat: diffrax.SaveAt,
    args=None,
    rtol: float = 1e-5,
    atol: float = 1e-10,
    max_steps: int = 16384,
    adjoint: str = "recursive_checkpoint",
    dt0=None,
):
    """Solve a stiff ODE system using Kvaerno5 (implicit ESDIRK).

    Used for perturbation integration (Einstein-Boltzmann system is stiff)
    and recombination (thermodynamics).

    Args:
        rhs_fn: callable (t, y, args) -> dy
        t0: initial time
        t1: final time
        y0: initial state vector
        saveat: Diffrax SaveAt specification
        args: additional arguments passed to rhs_fn
        rtol: relative tolerance
        atol: absolute tolerance
        max_steps: maximum number of solver steps
        adjoint: "recursive_checkpoint" or "direct"
        dt0: initial step size (None for auto)

    Returns:
        Diffrax solution object
    """
    solver = diffrax.Kvaerno5()
    controller = diffrax.PIDController(rtol=rtol, atol=atol)
    adjoint_obj = _get_adjoint(adjoint)

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs_fn),
        solver=solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        saveat=saveat,
        stepsize_controller=controller,
        adjoint=adjoint_obj,
        max_steps=max_steps,
        args=args,
    )
    return sol


def _get_adjoint(adjoint: str):
    """Return the Diffrax adjoint method from string name."""
    if adjoint == "recursive_checkpoint":
        return diffrax.RecursiveCheckpointAdjoint()
    elif adjoint == "direct":
        return diffrax.DirectAdjoint()
    else:
        raise ValueError(f"Unknown adjoint method: {adjoint}")
