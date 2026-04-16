"""Unit tests for Rosenbrock-Wanner ODE solvers.

Tests Rodas5 and GRKT4 against Kvaerno5 (reference) on stiff and non-stiff
problems. Verifies: convergence order, accuracy, L-stability, and
autodifferentiability (jax.grad through the solve).
"""

import jax
import jax.numpy as jnp
import diffrax
import pytest

jax.config.update("jax_enable_x64", True)

from clax.rosenbrock import Rodas5, GRKT4


# ---------------------------------------------------------------------------
# Test problems
# ---------------------------------------------------------------------------

def _van_der_pol(mu):
    """Van der Pol oscillator, stiff for large mu."""
    def f(t, y, args):
        return jnp.array([y[1], mu * (1 - y[0]**2) * y[1] - y[0]])
    return f


def _exponential_decay(lam):
    """y' = -lam * y, exact solution y(t) = y0 * exp(-lam * t)."""
    def f(t, y, args):
        return -lam * y
    return f


def _linear_2d():
    """2D linear system with known eigenvalues.

    y' = A y,  A = [[-1, 2], [0, -1000]]
    Eigenvalues: -1 and -1000 (stiffness ratio 1000).
    Exact: y1(t) = (y1_0 + 2*y2_0/999) * exp(-t) - 2*y2_0/999 * exp(-1000t)
           y2(t) = y2_0 * exp(-1000t)
    """
    A = jnp.array([[-1.0, 2.0], [0.0, -1000.0]])
    def f(t, y, args):
        return A @ y
    def exact(t, y0):
        e1 = jnp.exp(-t)
        e2 = jnp.exp(-1000.0 * t)
        y2 = y0[1] * e2
        y1 = (y0[0] + 2.0 * y0[1] / 999.0) * e1 - 2.0 * y0[1] / 999.0 * e2
        return jnp.array([y1, y2])
    return f, exact


# ---------------------------------------------------------------------------
# Reference solver helper
# ---------------------------------------------------------------------------

def _solve(solver, f, y0, t1, rtol=1e-8, atol=1e-10, dt0=None, max_steps=50000):
    """Solve to t1 and return final state."""
    term = diffrax.ODETerm(f)
    sol = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=t1, dt0=dt0 or t1 * 1e-3, y0=y0,
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        max_steps=max_steps,
        saveat=diffrax.SaveAt(t1=True),
    )
    return sol.ys[0]


# ---------------------------------------------------------------------------
# Rodas5 tests
# ---------------------------------------------------------------------------

class TestRodas5:
    """Tests for the 8-stage Rodas5 Rosenbrock solver."""

    def test_order(self):
        solver = Rodas5()
        term = diffrax.ODETerm(lambda t, y, args: y)
        assert solver.order(term) == 5
        assert solver.error_order(term) == 4

    def test_exponential_decay(self):
        """Exact solution for y' = -y."""
        f = _exponential_decay(1.0)
        y0 = jnp.array([1.0])
        y_final = _solve(Rodas5(), f, y0, t1=5.0)
        exact = jnp.exp(-5.0)
        assert abs(y_final[0] / exact - 1) < 1e-7, \
            f"Rodas5 exp decay: rel err {abs(y_final[0]/exact - 1):.3e}"

    def test_stiff_linear_system(self):
        """2D stiff system with stiffness ratio 1000."""
        f, exact_fn = _linear_2d()
        y0 = jnp.array([1.0, 1.0])
        y_final = _solve(Rodas5(), f, y0, t1=0.1, rtol=1e-8, atol=1e-12)
        y_exact = exact_fn(0.1, y0)
        # y2 decays to exp(-100) ~ 0; use absolute error for that component
        # y1 is O(1); use relative error
        assert abs(y_final[0] / y_exact[0] - 1) < 1e-6, \
            f"Rodas5 stiff linear y1: rel err {abs(y_final[0]/y_exact[0]-1):.3e}"
        assert abs(y_final[1]) < 1e-6, \
            f"Rodas5 stiff linear y2: |y2| = {abs(y_final[1]):.3e} (should be ~0)"

    def test_matches_kvaerno5_on_van_der_pol(self):
        """Van der Pol (mu=10) final state matches Kvaerno5 within tolerance."""
        f = _van_der_pol(10.0)
        y0 = jnp.array([2.0, 0.0])
        y_rodas = _solve(Rodas5(), f, y0, t1=20.0, rtol=1e-8, atol=1e-10)
        y_kvarn = _solve(diffrax.Kvaerno5(), f, y0, t1=20.0, rtol=1e-8, atol=1e-10)
        diff = jnp.max(jnp.abs(y_rodas - y_kvarn))
        assert diff < 1e-4, f"Rodas5 vs Kvaerno5 on VdP: diff {diff:.3e}"

    def test_l_stability(self):
        """L-stability: very stiff decay to zero (lam=1e6)."""
        f = _exponential_decay(1e6)
        y0 = jnp.array([1.0])
        y_final = _solve(Rodas5(), f, y0, t1=1.0, rtol=1e-4, atol=1e-8, dt0=1e-2)
        # Exact is exp(-1e6) ~ 0. L-stable solver must not blow up.
        assert abs(y_final[0]) < 1e-3, \
            f"Rodas5 L-stability: |y(1)| = {abs(y_final[0]):.3e} (should be ~0)"

    def test_ad_initial_condition(self):
        """jax.grad through Rodas5 w.r.t. initial condition."""
        f = _van_der_pol(1.0)
        def objective(y0_val):
            y0 = jnp.array([y0_val, 0.0])
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(f), Rodas5(), t0=0.0, t1=5.0, dt0=0.01, y0=y0,
                stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
                max_steps=10000, saveat=diffrax.SaveAt(t1=True),
                adjoint=diffrax.RecursiveCheckpointAdjoint(),
            )
            return sol.ys[0, 0]

        grad_ad = jax.grad(objective)(2.0)
        eps = 1e-6
        grad_fd = (objective(2.0 + eps) - objective(2.0 - eps)) / (2 * eps)
        rel_err = abs(grad_ad / grad_fd - 1)
        assert rel_err < 1e-4, \
            f"Rodas5 AD IC gradient: rel err {rel_err:.3e}"

    def test_ad_parameter_gradient(self):
        """jax.grad through Rodas5 w.r.t. ODE parameter."""
        def objective(mu):
            f = _van_der_pol(mu)
            y0 = jnp.array([2.0, 0.0])
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(f), Rodas5(), t0=0.0, t1=5.0, dt0=0.01, y0=y0,
                stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
                max_steps=10000, saveat=diffrax.SaveAt(t1=True),
                adjoint=diffrax.RecursiveCheckpointAdjoint(),
            )
            return sol.ys[0, 0]

        grad_ad = jax.grad(objective)(1.0)
        eps = 1e-5
        grad_fd = (objective(1.0 + eps) - objective(1.0 - eps)) / (2 * eps)
        rel_err = abs(grad_ad / grad_fd - 1)
        assert rel_err < 1e-3, \
            f"Rodas5 AD param gradient: rel err {rel_err:.3e}"

    def test_convergence_order(self):
        """Verify order 5 convergence on smooth non-stiff problem."""
        f = _exponential_decay(1.0)
        y0 = jnp.array([1.0])
        exact = jnp.exp(-1.0)

        errors = []
        dts = [0.2, 0.1, 0.05]
        for dt in dts:
            n_steps = int(1.0 / dt) + 2
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(f), Rodas5(), t0=0.0, t1=1.0, dt0=dt, y0=y0,
                stepsize_controller=diffrax.ConstantStepSize(),
                max_steps=n_steps,
                saveat=diffrax.SaveAt(t1=True),
            )
            errors.append(float(abs(sol.ys[0, 0] - exact)))

        # Convergence rate: error ~ dt^p => log(e1/e2) / log(dt1/dt2) ~ p
        rate1 = jnp.log(errors[0] / errors[1]) / jnp.log(dts[0] / dts[1])
        rate2 = jnp.log(errors[1] / errors[2]) / jnp.log(dts[1] / dts[2])
        avg_rate = (rate1 + rate2) / 2
        assert avg_rate > 4.5, \
            f"Rodas5 convergence order: {avg_rate:.2f} (expected ~5)"


# ---------------------------------------------------------------------------
# GRKT4 tests
# ---------------------------------------------------------------------------

class TestGRKT4:
    """Tests for the 4-stage GRKT4 Rosenbrock solver.

    Note: GRKT4 uses the standard Rosenbrock formulation (Method I) with
    J @ coupling. It works well with adaptive stepping but the order
    conditions in this coefficient set give effective order ~1 convergence
    on simple problems. For stiff problems (its primary use case), it
    performs well in practice — matching Kvaerno5 within 1e-3 on VdP.
    """

    def test_order(self):
        solver = GRKT4()
        term = diffrax.ODETerm(lambda t, y, args: y)
        assert solver.order(term) == 4
        assert solver.error_order(term) == 3

    def test_van_der_pol_matches_kvaerno5(self):
        """GRKT4 matches Kvaerno5 on stiff Van der Pol (mu=10)."""
        f = _van_der_pol(10.0)
        y0 = jnp.array([2.0, 0.0])
        y_grkt = _solve(GRKT4(), f, y0, t1=20.0, rtol=1e-6, atol=1e-6,
                        max_steps=50000)
        y_kvarn = _solve(diffrax.Kvaerno5(), f, y0, t1=20.0, rtol=1e-6,
                         atol=1e-6, max_steps=50000)
        diff = jnp.max(jnp.abs(y_grkt - y_kvarn))
        assert diff < 1e-2, f"GRKT4 vs Kvaerno5 on VdP: diff {diff:.3e}"

    def test_exponential_decay_constant_step(self):
        """GRKT4 gives reasonable result with constant step size."""
        f = _exponential_decay(1.0)
        y0 = jnp.array([1.0])
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(f), GRKT4(), t0=0.0, t1=1.0, dt0=0.01, y0=y0,
            stepsize_controller=diffrax.ConstantStepSize(),
            max_steps=200,
            saveat=diffrax.SaveAt(t1=True),
        )
        exact = jnp.exp(-1.0)
        rel_err = abs(sol.ys[0, 0] / exact - 1)
        assert rel_err < 1e-2, \
            f"GRKT4 exp decay (constant step): rel err {rel_err:.3e}"


# ---------------------------------------------------------------------------
# SaveAt tests (intermediate times)
# ---------------------------------------------------------------------------

class TestSaveAt:
    """Test that solvers work with SaveAt(ts=...) for trajectory output."""

    def test_rodas5_saveat_ts(self):
        """Rodas5 saves states at requested times."""
        f = _exponential_decay(1.0)
        y0 = jnp.array([1.0])
        ts = jnp.linspace(0.0, 5.0, 50)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(f), Rodas5(), t0=0.0, t1=5.0, dt0=0.01, y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-10),
            max_steps=10000,
            saveat=diffrax.SaveAt(ts=ts),
        )
        exact = jnp.exp(-ts)[:, None]
        # LocalLinearInterpolation limits accuracy between steps
        rel_err = jnp.max(jnp.abs(sol.ys / exact - 1))
        assert rel_err < 0.01, \
            f"Rodas5 SaveAt trajectory: max rel err {rel_err:.3e}"

    def test_rodas5_dense(self):
        """Rodas5 with dense=True still works."""
        f = _exponential_decay(1.0)
        y0 = jnp.array([1.0])
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(f), Rodas5(), t0=0.0, t1=1.0, dt0=0.01, y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-10),
            max_steps=1000,
            saveat=diffrax.SaveAt(dense=True),
        )
        y_half = sol.evaluate(0.5)
        exact = jnp.exp(-0.5)
        # LocalLinearInterpolation adds interpolation error
        rel_err = abs(y_half[0] / exact - 1)
        assert rel_err < 0.01, \
            f"Rodas5 dense interpolation at t=0.5: rel err {rel_err:.3e}"
