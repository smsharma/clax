"""Rosenbrock-Wanner ODE solvers for stiff systems.

Implements the Rodas5 (8-stage, order 5/4) Rosenbrock method as a
Diffrax-compatible adaptive solver. This avoids Newton iteration: each
step requires only one Jacobian evaluation and one LU factorization,
followed by linear back-substitution per stage.

For the Einstein-Boltzmann system (~60-150 equations), this is ~3-5x
faster per step than implicit ESDIRK methods (Kvaerno5) which need
iterative Newton convergence.

Mathematical formulation (transformed W-form):
    W = I/(h*gamma) - J,  where J = df/dy
    For each stage i:
        W * k_i = f(t + c_i*h, y + sum_j a_{ij}*k_j) + h*d_i*dT
                  + sum_j (C_{ij}/h)*k_j
    y_{n+1} = y_n + sum_i b_i * k_i

Reverse-mode AD and the h = 0 step (found while making C_l differentiable,
smsharma/clax#54; V100 job 26946).  Under ``jax.vmap`` (the perturbation solve
vmaps the k-modes) diffrax's step loop runs until the slowest member finishes; a
member that has already reached t1 executes masked steps with h = 0 exactly.  With
the divisions by h above, W and the C_ij/h couplings were inf on those steps; the
forward pass discarded them, but the backward pass multiplied their zero cotangent
by inf, and 0*inf = NaN made jax.grad of every C_l and source functional NaN at
fit_cl in every parameter.  Every division by h therefore uses
``dt_div = where(h == 0, 1, h)``: equal to h on every real step, so the arithmetic
and the accepted-step sequence are those of the unguarded solver to the bit, and
finite on a masked h = 0 step, whose result is discarded and whose cotangent is
then 0 * finite = 0.  Edge case: on a masked step the substituted
W = I/gamma - J is singular only if J has the real eigenvalue +1/gamma ~ +5.26
(growth on a 0.19 time-unit scale; 1/Mpc for the perturbations).  The masked step
would then be inf and the NaN would return, in reverse mode only.  At fit_cl the
end-time Jacobians of all 100 k-modes have Re(lambda) <= 4.5e-4/Mpc and
|lambda| <= 1.4/Mpc, so |lambda - 1/gamma| >= 5.26 (job 27662).

An equivalent alternative removes the divisions algebraically: multiply each
stage equation by h*gamma, W' = h*gamma*W = I - h*gamma*J and
    W' * k_i = h*gamma*(f(...) + h*d_i*dT) + gamma * sum_j C_{ij}*k_j,
the same k_i in exact arithmetic, with W' = I and all k_i = 0 at h = 0.  It
also gives finite gradients, but its different rounding flipped step
accept/reject decisions in 31 of 100 fit_cl k-modes and moved source_T0 by
1.9e-4 of its peak, a re-sampling of the rtol 1e-4 integration error (V100
job 27641, Oct 22, 2026).  The guard was chosen because it keeps the
accepted-step sequence and arithmetic to the bit (job 27656).

Dense output (state between accepted steps): both solvers return
``dense_info = dict(y0, y1, k0=dt*f(t0,y0), k1=dt*f(t1,y1))`` and use
diffrax's ``ThirdOrderHermitePolynomialInterpolation`` (cubic Hermite, the
same dense output as diffrax's Kvaerno5), at the cost of one extra RHS
evaluation per step.  The perturbation solve saves sources on a 5000-point
conformal-time grid (``SaveAt(ts=tau_grid)``) while Rodas5's late-time steps
are hundreds of Mpc, so the interpolant sets the late-ISW source.  With the
previous ``LocalLinearInterpolation`` unlensed TT was 46% low at l=2 relative
to Kvaerno5 at rtol 1e-4 / atol 1e-6 and 4.1% at 1e-5 / 1e-10 (precision
sweep of Oct 10-11, 2026, finding 5; error scaling as the square of the
late-time step).  Cubic Hermite error scales as the fourth power.
The extra evaluation is recovered FSAL-style: f(t1, y1) is carried in
``solver_state`` and reused as the next step's f(t0, y0), recomputed only on
the first step or after a jump (``made_jump``), with the same predicate as
diffrax's Runge-Kutta solvers.  A rejected step keeps the previous ``solver_state``
so the carried value stays f(t0, y0) of the retried step.
Measured on a V100 (Oct 12, 2026; 300 k-modes, planck_cl with ncdm_q_size 0,
hierarchy 12/10/17, rtol 1e-4 / atol 1e-6, 5000 tau points): rodas5 vs
kvaerno5 unlensed TT within 0.09% at l=2 and 0.013% at l=31-2500 (was 46%
and 1.1%); perturbation stage 22.8 s (linear) -> 25.5 s (Hermite) -> 24.9 s
(Hermite + carry), against 88 s for Kvaerno5.

References:
    Rodas5: Di Marzo (1993), "RODAS5(4) - Méthodes de Rosenbrock d'ordre 5(4)"
    Hairer & Wanner (1996), "Solving ODEs II", Section IV.7
    Transformed formulation: cf. DISCO-EB (Hahn, List & Porqueres, arXiv:2311.03291)
"""

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy.linalg as jla
from diffrax import AbstractAdaptiveSolver, AbstractTerm, ODETerm, RESULTS
from diffrax import ThirdOrderHermitePolynomialInterpolation


# ===========================================================================
# Rodas5 tableau (Di Marzo 1993, Type 1 — transformed W-formulation)
# ===========================================================================
_R5_GAMMA = 0.19

# Stage coupling coefficients a_{ij}: u_i = y0 + sum_j a_{ij} * k_j
_R5_A21 = 2.0
_R5_A31 = 3.040894194418781
_R5_A32 = 1.041747909077569
_R5_A41 = 2.576417536461461
_R5_A42 = 1.622083060776640
_R5_A43 = -0.9089668560264532
_R5_A51 = 2.760842080225597
_R5_A52 = 1.446624659844071
_R5_A53 = -0.3036980084553738
_R5_A54 = 0.2877498600325443
_R5_A61 = -14.09640773051259
_R5_A62 = 6.925207756232704
_R5_A63 = -41.47510893210728
_R5_A64 = 2.343771018586405
_R5_A65 = 24.13215229196062
# Stages 7, 8 accumulate: u7 = u6 + k6, u8 = u7 + k7

# Linear coupling coefficients C_{ij} (divided by dt in RHS)
_R5_C21 = -10.31323885133993
_R5_C31 = -21.04823117650003
_R5_C32 = -7.234992135176716
_R5_C41 = 32.22751541853323
_R5_C42 = -4.943732386540191
_R5_C43 = 19.44922031041879
_R5_C51 = -20.69865579590063
_R5_C52 = -8.816374604402768
_R5_C53 = 1.260436877740897
_R5_C54 = -0.7495647613787146
_R5_C61 = -46.22004352711257
_R5_C62 = -17.49534862857472
_R5_C63 = -289.6389582892057
_R5_C64 = 93.60855400400906
_R5_C65 = 318.3822534212147
_R5_C71 = 34.20013733472935
_R5_C72 = -14.15535402717690
_R5_C73 = 57.82335640988400
_R5_C74 = 25.83362985412365
_R5_C75 = 1.408950972071624
_R5_C76 = -6.551835421242162
_R5_C81 = 42.57076742291101
_R5_C82 = -13.80770672017997
_R5_C83 = 93.98938432427124
_R5_C84 = 18.77919633714503
_R5_C85 = -31.58359187223370
_R5_C86 = -6.685968952921985
_R5_C87 = -5.810979938412932

# Time node fractions: t_i = t0 + c_i * dt
_R5_C2 = 0.38
_R5_C3 = 0.3878509998321533
_R5_C4 = 0.4839718937873840
_R5_C5 = 0.4570477008819580
# c6 = c7 = c8 = 1.0 (implicit)

# Time derivative coefficients: rhs += dt * d_i * dT
_R5_D1 = _R5_GAMMA
_R5_D2 = -0.1823079225333714636
_R5_D3 = -0.319231832186874912
_R5_D4 = 0.3449828624725343
_R5_D5 = -0.377417564392089818
# d6 = d7 = d8 = 0.0 (implicit)


def _lu_solve(lu_piv, b):
    """Wrapper for LU back-substitution."""
    return jla.lu_solve(lu_piv, b)


class Rodas5(AbstractAdaptiveSolver):
    """8-stage Rosenbrock method of order 5(4) (Di Marzo 1993).

    Uses the transformed W-formulation where:
        W = I/(h*gamma) - J
    with every division by h guarded against the masked h = 0 steps of a
    vmapped solve (see the module docstring), and the error estimate is
    simply k_8 (the last stage).

    This is the method used by DISCO-EB for the Einstein-Boltzmann system.
    It is L-stable and stiffly accurate, making it suitable for the stiff
    photon-baryon tight-coupling regime.

    Compared to Kvaerno5 (ESDIRK), this avoids Newton iteration entirely:
    each step needs one Jacobian + one LU factorization + 8 back-substitutions.
    The RHS evaluation at (t1, y1) needed by the cubic Hermite dense output is
    carried in ``solver_state`` into the next step as its f(t0, y0) (see the
    module docstring).
    """

    term_structure = AbstractTerm
    interpolation_cls = ThirdOrderHermitePolynomialInterpolation

    def order(self, terms):
        return 5

    def error_order(self, terms):
        return 4

    def init(self, terms, t0, t1, y0, args):
        # (first_step, f carried from the previous step's end); see module
        # docstring.  The zeros are a placeholder of the right shape.
        return jnp.array(True), jnp.zeros_like(y0)

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        dt = t1 - t0
        # Divisor for every 1/dt below: equal to dt on every real step (bit for
        # bit), 1 on the masked dt = 0 steps vmap runs after a member reaches
        # t1, so their discarded result and its cotangent stay finite (module
        # docstring).  dt itself still sets the stage times and dense output.
        dt_div = jnp.where(dt == 0, 1.0, dt)
        f = lambda t, y: terms.vf(t, y, args)

        # -- f(t0, y0): carried from the previous step unless it is invalid --
        # unvmap_any keeps the predicate unbatched under vmap over k-modes, so
        # lax.cond stays a real branch instead of evaluating both sides.
        first_step, f_carry = solver_state
        eval_f0 = eqxi.unvmap_any(first_step | made_jump)
        f0 = lax.cond(eval_f0, lambda: f(t0, y0), lambda: f_carry)

        # -- Jacobian and time derivative via forward-mode AD --
        n = y0.shape[0]
        J = jax.jacfwd(lambda y: f(t0, y))(y0)
        dT = jax.jacfwd(lambda t: f(t, y0))(t0)

        # -- Form W = I/(dt*gamma) - J and LU factorize (guarded divisor) --
        dtgamma = dt_div * _R5_GAMMA
        W = jnp.eye(n) / dtgamma - J
        lu_piv = jla.lu_factor(W)

        # -- Stage 1 --
        rhs1 = f0 + dt * _R5_D1 * dT
        k1 = _lu_solve(lu_piv, rhs1)

        # -- Stage 2 --
        u2 = y0 + _R5_A21 * k1
        rhs2 = f(t0 + _R5_C2 * dt, u2) + dt * _R5_D2 * dT + _R5_C21 / dt_div * k1
        k2 = _lu_solve(lu_piv, rhs2)

        # -- Stage 3 --
        u3 = y0 + _R5_A31 * k1 + _R5_A32 * k2
        rhs3 = (f(t0 + _R5_C3 * dt, u3) + dt * _R5_D3 * dT
                + _R5_C31 / dt_div * k1 + _R5_C32 / dt_div * k2)
        k3 = _lu_solve(lu_piv, rhs3)

        # -- Stage 4 --
        u4 = y0 + _R5_A41 * k1 + _R5_A42 * k2 + _R5_A43 * k3
        rhs4 = (f(t0 + _R5_C4 * dt, u4) + dt * _R5_D4 * dT
                + _R5_C41 / dt_div * k1 + _R5_C42 / dt_div * k2 + _R5_C43 / dt_div * k3)
        k4 = _lu_solve(lu_piv, rhs4)

        # -- Stage 5 --
        u5 = y0 + _R5_A51 * k1 + _R5_A52 * k2 + _R5_A53 * k3 + _R5_A54 * k4
        rhs5 = (f(t0 + _R5_C5 * dt, u5) + dt * _R5_D5 * dT
                + _R5_C51 / dt_div * k1 + _R5_C52 / dt_div * k2
                + _R5_C53 / dt_div * k3 + _R5_C54 / dt_div * k4)
        k5 = _lu_solve(lu_piv, rhs5)

        # -- Stage 6 (at t0 + dt) --
        u6 = (y0 + _R5_A61 * k1 + _R5_A62 * k2 + _R5_A63 * k3
              + _R5_A64 * k4 + _R5_A65 * k5)
        rhs6 = (f(t0 + dt, u6)
                + _R5_C61 / dt_div * k1 + _R5_C62 / dt_div * k2
                + _R5_C63 / dt_div * k3 + _R5_C64 / dt_div * k4
                + _R5_C65 / dt_div * k5)
        k6 = _lu_solve(lu_piv, rhs6)

        # -- Stage 7 (at t0 + dt, accumulating) --
        u7 = u6 + k6
        rhs7 = (f(t0 + dt, u7)
                + _R5_C71 / dt_div * k1 + _R5_C72 / dt_div * k2
                + _R5_C73 / dt_div * k3 + _R5_C74 / dt_div * k4
                + _R5_C75 / dt_div * k5 + _R5_C76 / dt_div * k6)
        k7 = _lu_solve(lu_piv, rhs7)

        # -- Stage 8 (at t0 + dt, accumulating — error stage) --
        u8 = u7 + k7
        rhs8 = (f(t0 + dt, u8)
                + _R5_C81 / dt_div * k1 + _R5_C82 / dt_div * k2
                + _R5_C83 / dt_div * k3 + _R5_C84 / dt_div * k4
                + _R5_C85 / dt_div * k5 + _R5_C86 / dt_div * k6
                + _R5_C87 / dt_div * k7)
        k8 = _lu_solve(lu_piv, rhs8)

        # -- Solution and error --
        y1 = u8 + k8
        y_error = k8

        # -- Dense output: cubic Hermite from (y, dt*y') at both step ends --
        # f(t1, y1) is not among the stage evaluations (stage 8 is at u8 != y1).
        f1 = f(t1, y1)
        dense_info = dict(y0=y0, y1=y1, k0=dt * f0, k1=dt * f1)
        new_solver_state = (jnp.array(False), f1)
        return y1, y_error, dense_info, new_solver_state, RESULTS.successful


class Rodas5Batched(AbstractAdaptiveSolver):
    """Batched Rodas5 for solving multiple k-modes with shared time-stepping.

    y0 has shape ``(batch_size, n_eq)``.  All modes in the batch share the
    same adaptive step size, controlled by a single scalar error norm across
    the batch.  Internally vmaps the Jacobian, LU factorisation and
    back-substitution over the batch dimension.

    Args convention for ``diffeqsolve``::

        args = (f_single, batched_per_mode_data)

    where ``f_single(t, y, per_mode_datum)`` is the single-mode RHS used
    for ``jax.jacfwd``, and ``batched_per_mode_data`` (e.g. an array of
    k-values with shape ``(batch_size,)``) is forwarded to ``terms.vf``
    for batched function evaluations.

    cf. DISCO-EB ``Rodas5Batched`` (ode_integrators_stiff.py:846-1011)

    Dense output is the cubic Hermite interpolant from (y, dt*y') at both
    step ends, as for ``Rodas5``; the batched f(t1, y1) is carried in
    ``solver_state`` into the next step as its f(t0, y0).
    """

    term_structure = ODETerm
    interpolation_cls = ThirdOrderHermitePolynomialInterpolation

    def order(self, terms):
        return 5

    def error_order(self, terms):
        return 4

    def init(self, terms, t0, t1, y0, args):
        # (first_step, f carried from the previous step's end); see Rodas5.
        return jnp.array(True), jnp.zeros_like(y0)

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        f, _args = args

        n = y0[0].shape[0]  # state dimension from first batch element
        dt = terms.contr(t0, t1)
        # Guarded divisor, as in Rodas5.step: dt on every real step, 1 on a
        # masked dt = 0 step (a vmap over batched solves).
        dt_div = jnp.where(dt == 0, 1.0, dt)

        # Pre-compute scaled coupling coefficients C_ij/dt (guarded divisor)
        dtC21 = _R5_C21 / dt_div
        dtC31 = _R5_C31 / dt_div
        dtC32 = _R5_C32 / dt_div
        dtC41 = _R5_C41 / dt_div
        dtC42 = _R5_C42 / dt_div
        dtC43 = _R5_C43 / dt_div
        dtC51 = _R5_C51 / dt_div
        dtC52 = _R5_C52 / dt_div
        dtC53 = _R5_C53 / dt_div
        dtC54 = _R5_C54 / dt_div
        dtC61 = _R5_C61 / dt_div
        dtC62 = _R5_C62 / dt_div
        dtC63 = _R5_C63 / dt_div
        dtC64 = _R5_C64 / dt_div
        dtC65 = _R5_C65 / dt_div
        dtC71 = _R5_C71 / dt_div
        dtC72 = _R5_C72 / dt_div
        dtC73 = _R5_C73 / dt_div
        dtC74 = _R5_C74 / dt_div
        dtC75 = _R5_C75 / dt_div
        dtC76 = _R5_C76 / dt_div
        dtC81 = _R5_C81 / dt_div
        dtC82 = _R5_C82 / dt_div
        dtC83 = _R5_C83 / dt_div
        dtC84 = _R5_C84 / dt_div
        dtC85 = _R5_C85 / dt_div
        dtC86 = _R5_C86 / dt_div
        dtC87 = _R5_C87 / dt_div

        dtd1 = dt * _R5_D1
        dtd2 = dt * _R5_D2
        dtd3 = dt * _R5_D3
        dtd4 = dt * _R5_D4
        dtd5 = dt * _R5_D5
        dtgamma = dt_div * _R5_GAMMA

        I = jnp.eye(n)

        # Batched Jacobian and time derivative via forward-mode AD
        dt_f = jax.jacfwd(f, 0)
        jac_f = jax.jacfwd(f, 1)

        dt_f_batched = jax.vmap(dt_f, in_axes=(None, 0, 0))
        jac_f_batched = jax.vmap(jac_f, in_axes=(None, 0, 0))

        lu_batched = jax.vmap(
            lambda a: jla.lu_factor(I / dtgamma - a))

        dT = dt_f_batched(t0, y0, _args)           # (batch, n)
        jac_blocks = jac_f_batched(t0, y0, _args)   # (batch, n, n)

        lu_and_piv = lu_batched(jac_blocks)

        lu_solve_batched = jax.vmap(jla.lu_solve, (0, 0))

        # -- 8 Rosenbrock stages (transformed W-formulation) --
        # Stage 1: f(t0, y0) carried from the previous step unless invalid
        first_step, f_carry = solver_state
        eval_f0 = eqxi.unvmap_any(first_step | made_jump)
        dy1 = lax.cond(eval_f0, lambda: terms.vf(t=t0, y=y0, args=_args),
                       lambda: f_carry)
        rhs = dy1 + dtd1 * dT
        k1 = lu_solve_batched(lu_and_piv, rhs)

        # Stage 2
        u = y0 + _R5_A21 * k1
        du = terms.vf(t=t0 + _R5_C2 * dt, y=u, args=_args)
        rhs = du + dtd2 * dT + dtC21 * k1
        k2 = lu_solve_batched(lu_and_piv, rhs)

        # Stage 3
        u = y0 + _R5_A31 * k1 + _R5_A32 * k2
        du = terms.vf(t=t0 + _R5_C3 * dt, y=u, args=_args)
        rhs = du + dtd3 * dT + (dtC31 * k1 + dtC32 * k2)
        k3 = lu_solve_batched(lu_and_piv, rhs)

        # Stage 4
        u = y0 + _R5_A41 * k1 + _R5_A42 * k2 + _R5_A43 * k3
        du = terms.vf(t=t0 + _R5_C4 * dt, y=u, args=_args)
        rhs = du + dtd4 * dT + (dtC41 * k1 + dtC42 * k2 + dtC43 * k3)
        k4 = lu_solve_batched(lu_and_piv, rhs)

        # Stage 5
        u = y0 + _R5_A51 * k1 + _R5_A52 * k2 + _R5_A53 * k3 + _R5_A54 * k4
        du = terms.vf(t=t0 + _R5_C5 * dt, y=u, args=_args)
        rhs = du + dtd5 * dT + (dtC51 * k1 + dtC52 * k2
                                + dtC53 * k3 + dtC54 * k4)
        k5 = lu_solve_batched(lu_and_piv, rhs)

        # Stage 6 (at t0 + dt)
        u = (y0 + _R5_A61 * k1 + _R5_A62 * k2 + _R5_A63 * k3
             + _R5_A64 * k4 + _R5_A65 * k5)
        du = terms.vf(t=t0 + dt, y=u, args=_args)
        rhs = du + (dtC61 * k1 + dtC62 * k2 + dtC63 * k3
                    + dtC64 * k4 + dtC65 * k5)
        k6 = lu_solve_batched(lu_and_piv, rhs)

        # Stage 7 (accumulating)
        u = u + k6
        du = terms.vf(t=t0 + dt, y=u, args=_args)
        rhs = du + (dtC71 * k1 + dtC72 * k2 + dtC73 * k3
                    + dtC74 * k4 + dtC75 * k5 + dtC76 * k6)
        k7 = lu_solve_batched(lu_and_piv, rhs)

        # Stage 8 (error estimate stage)
        u = u + k7
        du = terms.vf(t=t0 + dt, y=u, args=_args)
        rhs = du + (dtC81 * k1 + dtC82 * k2 + dtC83 * k3
                    + dtC84 * k4 + dtC85 * k5 + dtC86 * k6
                    + dtC87 * k7)
        k8 = lu_solve_batched(lu_and_piv, rhs)

        # Solution and error estimate
        y1 = u + k8
        y_error = k8  # embedded error: (batch_size, n_eq)

        # Dense output: cubic Hermite from (y, dt*y') at both step ends
        f1 = terms.vf(t=t1, y=y1, args=_args)
        dense_info = dict(y0=y0, y1=y1, k0=dt * dy1, k1=dt * f1)
        new_solver_state = (jnp.array(False), f1)
        return y1, y_error, dense_info, new_solver_state, RESULTS.successful
