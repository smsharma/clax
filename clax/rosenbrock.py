"""Rosenbrock-Wanner ODE solvers for stiff systems.

Implements Rodas5 (8-stage, order 5/4) and GRKT4 (4-stage, order 4/3)
Rosenbrock methods as Diffrax-compatible adaptive solvers. These avoid
Newton iteration: each step requires only one Jacobian evaluation and
one LU factorization, followed by linear back-substitution per stage.

For the Einstein-Boltzmann system (~60-150 equations), this is ~3-5x
faster per step than implicit ESDIRK methods (Kvaerno5) which need
iterative Newton convergence.

Mathematical formulation (transformed W-form):
    W = I/(h*gamma) - J,  where J = df/dy
    For each stage i:
        W * k_i = f(t + c_i*h, y + sum_j a_{ij}*k_j) + h*d_i*dT
                  + sum_j (C_{ij}/h)*k_j
    y_{n+1} = y_n + sum_i b_i * k_i

References:
    Rodas5: Di Marzo (1993), "RODAS5(4) - Méthodes de Rosenbrock d'ordre 5(4)"
    GRKT4: Kaps & Rentrop (1979), "Generalized Runge-Kutta methods of order 4"
    Hairer & Wanner (1996), "Solving ODEs II", Section IV.7
    Transformed formulation: cf. DISCO-EB (Hahn, List & Porqueres, arXiv:2311.03291)
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
from diffrax import AbstractAdaptiveSolver, AbstractTerm, ODETerm, RESULTS
from diffrax._local_interpolation import LocalLinearInterpolation


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


# ===========================================================================
# GRKT4 tableau (Kaps & Rentrop 1979, GRK4A variant)
# ===========================================================================
_G4_GAMMA = 0.395

_G4_ALPHA21 = 0.438
_G4_ALPHA31 = 0.796920457938
_G4_ALPHA32 = 0.730795420615e-1

_G4_GAMMA21 = -0.767672395484
_G4_GAMMA31 = -0.851675323742
_G4_GAMMA32 = 0.522967289188
_G4_GAMMA41 = 0.288463109545
_G4_GAMMA42 = 0.880214273381e-1
_G4_GAMMA43 = -0.337389840627

# Solution weights (4th order)
_G4_C1 = 0.199293275701
_G4_C2 = 0.482645235674
_G4_C3 = 0.680614886256e-1
_G4_C4 = 0.25

# Embedded weights (3rd order)
_G4_CHAT1 = 0.346325833758
_G4_CHAT2 = 0.285693175712
_G4_CHAT3 = 0.367980990530


def _lu_solve(lu_piv, b):
    """Wrapper for LU back-substitution."""
    return jla.lu_solve(lu_piv, b)


class Rodas5(AbstractAdaptiveSolver):
    """8-stage Rosenbrock method of order 5(4) (Di Marzo 1993).

    Uses the transformed W-formulation where:
        W = I/(h*gamma) - J
    and the error estimate is simply k_8 (the last stage).

    This is the method used by DISCO-EB for the Einstein-Boltzmann system.
    It is L-stable and stiffly accurate, making it suitable for the stiff
    photon-baryon tight-coupling regime.

    Compared to Kvaerno5 (ESDIRK), this avoids Newton iteration entirely:
    each step needs one Jacobian + one LU factorization + 8 back-substitutions.
    """

    term_structure = AbstractTerm
    interpolation_cls = LocalLinearInterpolation

    def order(self, terms):
        return 5

    def error_order(self, terms):
        return 4

    def init(self, terms, t0, t1, y0, args):
        return None

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        dt = t1 - t0
        f = lambda t, y: terms.vf(t, y, args)

        # -- Jacobian and time derivative via forward-mode AD --
        f0 = f(t0, y0)
        n = y0.shape[0]
        J = jax.jacfwd(lambda y: f(t0, y))(y0)
        dT = jax.jacfwd(lambda t: f(t, y0))(t0)

        # -- Form W = I/(dt*gamma) - J and LU factorize --
        dtgamma = dt * _R5_GAMMA
        W = jnp.eye(n) / dtgamma - J
        lu_piv = jla.lu_factor(W)

        # -- Stage 1 --
        rhs1 = f0 + dt * _R5_D1 * dT
        k1 = _lu_solve(lu_piv, rhs1)

        # -- Stage 2 --
        u2 = y0 + _R5_A21 * k1
        rhs2 = f(t0 + _R5_C2 * dt, u2) + dt * _R5_D2 * dT + _R5_C21 / dt * k1
        k2 = _lu_solve(lu_piv, rhs2)

        # -- Stage 3 --
        u3 = y0 + _R5_A31 * k1 + _R5_A32 * k2
        rhs3 = (f(t0 + _R5_C3 * dt, u3) + dt * _R5_D3 * dT
                + _R5_C31 / dt * k1 + _R5_C32 / dt * k2)
        k3 = _lu_solve(lu_piv, rhs3)

        # -- Stage 4 --
        u4 = y0 + _R5_A41 * k1 + _R5_A42 * k2 + _R5_A43 * k3
        rhs4 = (f(t0 + _R5_C4 * dt, u4) + dt * _R5_D4 * dT
                + _R5_C41 / dt * k1 + _R5_C42 / dt * k2 + _R5_C43 / dt * k3)
        k4 = _lu_solve(lu_piv, rhs4)

        # -- Stage 5 --
        u5 = y0 + _R5_A51 * k1 + _R5_A52 * k2 + _R5_A53 * k3 + _R5_A54 * k4
        rhs5 = (f(t0 + _R5_C5 * dt, u5) + dt * _R5_D5 * dT
                + _R5_C51 / dt * k1 + _R5_C52 / dt * k2
                + _R5_C53 / dt * k3 + _R5_C54 / dt * k4)
        k5 = _lu_solve(lu_piv, rhs5)

        # -- Stage 6 (at t0 + dt) --
        u6 = (y0 + _R5_A61 * k1 + _R5_A62 * k2 + _R5_A63 * k3
              + _R5_A64 * k4 + _R5_A65 * k5)
        rhs6 = (f(t0 + dt, u6)
                + _R5_C61 / dt * k1 + _R5_C62 / dt * k2
                + _R5_C63 / dt * k3 + _R5_C64 / dt * k4
                + _R5_C65 / dt * k5)
        k6 = _lu_solve(lu_piv, rhs6)

        # -- Stage 7 (at t0 + dt, accumulating) --
        u7 = u6 + k6
        rhs7 = (f(t0 + dt, u7)
                + _R5_C71 / dt * k1 + _R5_C72 / dt * k2
                + _R5_C73 / dt * k3 + _R5_C74 / dt * k4
                + _R5_C75 / dt * k5 + _R5_C76 / dt * k6)
        k7 = _lu_solve(lu_piv, rhs7)

        # -- Stage 8 (at t0 + dt, accumulating — error stage) --
        u8 = u7 + k7
        rhs8 = (f(t0 + dt, u8)
                + _R5_C81 / dt * k1 + _R5_C82 / dt * k2
                + _R5_C83 / dt * k3 + _R5_C84 / dt * k4
                + _R5_C85 / dt * k5 + _R5_C86 / dt * k6
                + _R5_C87 / dt * k7)
        k8 = _lu_solve(lu_piv, rhs8)

        # -- Solution and error --
        y1 = u8 + k8
        y_error = k8

        dense_info = dict(y0=y0, y1=y1)
        return y1, y_error, dense_info, solver_state, RESULTS.successful


class GRKT4(AbstractAdaptiveSolver):
    """4-stage Rosenbrock method of order 4(3) (Kaps & Rentrop 1979).

    A simpler and faster Rosenbrock method than Rodas5. Uses only 4 stages
    (vs 8) and 4 function evaluations per step (vs 8). Order 4 with
    embedded 3rd-order error estimation.

    Suitable for moderately stiff systems or when lower accuracy is
    acceptable (e.g., fit_cl preset).
    """

    term_structure = AbstractTerm
    interpolation_cls = LocalLinearInterpolation

    def order(self, terms):
        return 4

    def error_order(self, terms):
        return 3

    def init(self, terms, t0, t1, y0, args):
        return None

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        dt = t1 - t0
        f = lambda t, y: terms.vf(t, y, args)

        # -- Jacobian and time derivative --
        f0 = f(t0, y0)
        n = y0.shape[0]
        J = jax.jacfwd(lambda y: f(t0, y))(y0)
        dT = jax.jacfwd(lambda t: f(t, y0))(t0)

        # -- Standard Rosenbrock: W = I/h - gamma*J (Hairer & Wanner IV.7) --
        # Coupling uses J @ (gamma_ij * k_j) — NOT gamma_ij/h * k_j.
        W = jnp.eye(n) / dt - _G4_GAMMA * J
        lu_piv = jla.lu_factor(W)

        # -- Stage 1 --
        rhs1 = f0 + _G4_GAMMA * dt * dT
        k1 = _lu_solve(lu_piv, rhs1)

        # -- Stage 2 --
        u2 = y0 + _G4_ALPHA21 * k1
        rhs2 = (f(t0 + _G4_ALPHA21 * dt, u2)
                + _G4_GAMMA * dt * dT
                + J @ (_G4_GAMMA21 * k1))
        k2 = _lu_solve(lu_piv, rhs2)

        # -- Stage 3 --
        u3 = y0 + _G4_ALPHA31 * k1 + _G4_ALPHA32 * k2
        rhs3 = (f(t0 + (_G4_ALPHA31 + _G4_ALPHA32) * dt, u3)
                + _G4_GAMMA * dt * dT
                + J @ (_G4_GAMMA31 * k1 + _G4_GAMMA32 * k2))
        k3 = _lu_solve(lu_piv, rhs3)

        # -- Stage 4 (stiffly accurate: coupling uses solution weights) --
        u4 = y0 + _G4_C1 * k1 + _G4_C2 * k2 + _G4_C3 * k3
        rhs4 = (f(t0 + dt, u4)
                + _G4_GAMMA * dt * dT
                + J @ (_G4_GAMMA41 * k1 + _G4_GAMMA42 * k2 + _G4_GAMMA43 * k3))
        k4 = _lu_solve(lu_piv, rhs4)

        # -- 4th-order solution --
        y1 = y0 + _G4_C1 * k1 + _G4_C2 * k2 + _G4_C3 * k3 + _G4_C4 * k4

        # -- Error estimate: difference between 4th and 3rd order --
        y_error = ((_G4_C1 - _G4_CHAT1) * k1
                   + (_G4_C2 - _G4_CHAT2) * k2
                   + (_G4_C3 - _G4_CHAT3) * k3
                   + _G4_C4 * k4)

        dense_info = dict(y0=y0, y1=y1)
        return y1, y_error, dense_info, solver_state, RESULTS.successful


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
    """

    term_structure = ODETerm
    interpolation_cls = LocalLinearInterpolation

    def order(self, terms):
        return 5

    def error_order(self, terms):
        return 4

    def init(self, terms, t0, t1, y0, args):
        return None

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump

        f, _args = args

        n = y0[0].shape[0]  # state dimension from first batch element
        dt = terms.contr(t0, t1)

        # Pre-compute scaled coupling coefficients
        dtC21 = _R5_C21 / dt
        dtC31 = _R5_C31 / dt
        dtC32 = _R5_C32 / dt
        dtC41 = _R5_C41 / dt
        dtC42 = _R5_C42 / dt
        dtC43 = _R5_C43 / dt
        dtC51 = _R5_C51 / dt
        dtC52 = _R5_C52 / dt
        dtC53 = _R5_C53 / dt
        dtC54 = _R5_C54 / dt
        dtC61 = _R5_C61 / dt
        dtC62 = _R5_C62 / dt
        dtC63 = _R5_C63 / dt
        dtC64 = _R5_C64 / dt
        dtC65 = _R5_C65 / dt
        dtC71 = _R5_C71 / dt
        dtC72 = _R5_C72 / dt
        dtC73 = _R5_C73 / dt
        dtC74 = _R5_C74 / dt
        dtC75 = _R5_C75 / dt
        dtC76 = _R5_C76 / dt
        dtC81 = _R5_C81 / dt
        dtC82 = _R5_C82 / dt
        dtC83 = _R5_C83 / dt
        dtC84 = _R5_C84 / dt
        dtC85 = _R5_C85 / dt
        dtC86 = _R5_C86 / dt
        dtC87 = _R5_C87 / dt

        dtd1 = dt * _R5_D1
        dtd2 = dt * _R5_D2
        dtd3 = dt * _R5_D3
        dtd4 = dt * _R5_D4
        dtd5 = dt * _R5_D5
        dtgamma = dt * _R5_GAMMA

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
        # Stage 1
        dy1 = terms.vf(t=t0, y=y0, args=_args)
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

        dense_info = dict(y0=y0, y1=y1)
        return y1, y_error, dense_info, None, RESULTS.successful
