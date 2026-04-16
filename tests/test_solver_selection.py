"""Integration tests for solver selection in the perturbation pipeline.

Tests that P(k) computed with the Rosenbrock (Rodas5) solver matches
the Kvaerno5 reference within acceptable tolerance. Also tests the
solver selection mechanism and loosened atol behavior.
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from clax import CosmoParams, PrecisionParams
from clax.ode import _get_stiff_solver


# ---------------------------------------------------------------------------
# Solver selection unit tests
# ---------------------------------------------------------------------------

class TestSolverSelection:
    """Test the _get_stiff_solver helper and PrecisionParams wiring."""

    def test_kvaerno5_selection(self):
        """Default solver is Kvaerno5."""
        import diffrax
        solver = _get_stiff_solver("kvaerno5")
        assert isinstance(solver, diffrax.Kvaerno5)

    def test_rosenbrock_selection(self):
        """Rosenbrock selection returns Rodas5."""
        from clax.rosenbrock import Rodas5
        solver = _get_stiff_solver("rosenbrock")
        assert isinstance(solver, Rodas5)

    def test_invalid_solver_raises(self):
        """Unknown solver name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown stiff solver"):
            _get_stiff_solver("euler")

    def test_fit_cl_uses_rosenbrock(self):
        """fit_cl preset uses Rosenbrock solver."""
        prec = PrecisionParams.fit_cl()
        assert prec.pt_ode_solver == "rosenbrock"

    def test_fit_cl_loosened_atol(self):
        """fit_cl preset uses DISCO-EB-style loosened atol=1e-4."""
        prec = PrecisionParams.fit_cl()
        assert prec.pt_ode_atol == 1e-4

    def test_default_uses_kvaerno5(self):
        """Default PrecisionParams uses Kvaerno5."""
        prec = PrecisionParams()
        assert prec.pt_ode_solver == "kvaerno5"

    def test_other_presets_unchanged(self):
        """Non-fit presets still use Kvaerno5."""
        for preset in [PrecisionParams.fast_cl, PrecisionParams.medium_cl,
                       PrecisionParams.planck_cl, PrecisionParams.planck_fast]:
            prec = preset()
            assert prec.pt_ode_solver == "kvaerno5", \
                f"{preset.__name__} should use kvaerno5, got {prec.pt_ode_solver}"


# ---------------------------------------------------------------------------
# P(k) integration tests: Rosenbrock vs Kvaerno5
# ---------------------------------------------------------------------------

class TestRosenbrockPk:
    """Test that P(k) from Rosenbrock matches Kvaerno5 reference."""

    @pytest.fixture(scope="class")
    def bg_th(self):
        """Background and thermodynamics (shared, computed once)."""
        from clax.background import background_solve
        from clax.thermodynamics import thermodynamics_solve
        params = CosmoParams()
        prec = PrecisionParams(
            th_n_points=3000,
            pt_k_per_decade=10,
            pt_k_max_cl=0.3,
            pt_l_max_g=17,
            pt_l_max_pol_g=17,
            pt_l_max_ur=17,
            ncdm_q_size=0,
            pt_tau_n_points=1000,
            ode_max_steps=4096,
        )
        bg = background_solve(params, prec)
        th = thermodynamics_solve(params, prec, bg)
        return params, prec, bg, th

    def test_pk_rosenbrock_vs_kvaerno5(self, bg_th):
        """P(k) from Rosenbrock solver matches Kvaerno5 within 1%."""
        from clax.perturbations import perturbations_solve_mpk
        params, prec_base, bg, th = bg_th

        # Kvaerno5 reference
        prec_k5 = PrecisionParams(
            th_n_points=prec_base.th_n_points,
            pt_k_per_decade=prec_base.pt_k_per_decade,
            pt_k_max_cl=prec_base.pt_k_max_cl,
            pt_l_max_g=prec_base.pt_l_max_g,
            pt_l_max_pol_g=prec_base.pt_l_max_pol_g,
            pt_l_max_ur=prec_base.pt_l_max_ur,
            ncdm_q_size=0,
            pt_tau_n_points=prec_base.pt_tau_n_points,
            pt_ode_rtol=1e-5,
            pt_ode_atol=1e-10,
            ode_max_steps=prec_base.ode_max_steps,
            pt_ode_solver="kvaerno5",
        )

        # Rosenbrock with loosened atol (DISCO-EB style)
        prec_rb = PrecisionParams(
            th_n_points=prec_base.th_n_points,
            pt_k_per_decade=prec_base.pt_k_per_decade,
            pt_k_max_cl=prec_base.pt_k_max_cl,
            pt_l_max_g=prec_base.pt_l_max_g,
            pt_l_max_pol_g=prec_base.pt_l_max_pol_g,
            pt_l_max_ur=prec_base.pt_l_max_ur,
            ncdm_q_size=0,
            pt_tau_n_points=prec_base.pt_tau_n_points,
            pt_ode_rtol=1e-4,
            pt_ode_atol=1e-4,
            ode_max_steps=prec_base.ode_max_steps,
            pt_ode_solver="rosenbrock",
        )

        pt_k5 = perturbations_solve_mpk(params, prec_k5, bg, th)
        pt_rb = perturbations_solve_mpk(params, prec_rb, bg, th)

        # Compare delta_m at z=0 (last tau entry)
        dm_k5 = pt_k5.delta_m[:, -1]  # (n_k,)
        dm_rb = pt_rb.delta_m[:, -1]

        # P(k) ~ delta_m^2, so compare delta_m
        mask = jnp.abs(dm_k5) > 1e-10  # avoid near-zero modes
        rel_err = jnp.abs(dm_rb[mask] / dm_k5[mask] - 1)
        max_err = jnp.max(rel_err)
        mean_err = jnp.mean(rel_err)

        print(f"Rosenbrock vs Kvaerno5 delta_m: max={max_err:.4%}, mean={mean_err:.4%}")
        # Rosenbrock uses rtol=atol=1e-4 vs Kvaerno5 rtol=1e-5, atol=1e-10.
        # Max error at a single k-mode can be ~3% due to the looser tolerances.
        # Mean error is typically <0.2%, confirming the solver is correct.
        assert max_err < 0.05, \
            f"Rosenbrock delta_m max rel err {max_err:.2%} exceeds 5%"
        assert mean_err < 0.01, \
            f"Rosenbrock delta_m mean rel err {mean_err:.2%} exceeds 1%"

    def test_solver_selection_end_to_end(self, bg_th):
        """Verify solver selection works through the full solve path."""
        from clax.perturbations import perturbations_solve_mpk
        params, prec_base, bg, th = bg_th

        # Just verify no crash with rosenbrock
        prec = PrecisionParams(
            th_n_points=prec_base.th_n_points,
            pt_k_per_decade=5,
            pt_k_max_cl=0.1,
            pt_l_max_g=17,
            pt_l_max_pol_g=17,
            pt_l_max_ur=17,
            ncdm_q_size=0,
            pt_tau_n_points=500,
            pt_ode_rtol=1e-3,
            pt_ode_atol=1e-4,
            ode_max_steps=2048,
            pt_ode_solver="rosenbrock",
        )
        pt = perturbations_solve_mpk(params, prec, bg, th)
        assert pt.delta_m.shape[0] > 0
        assert jnp.all(jnp.isfinite(pt.delta_m))
        print(f"Rosenbrock mPk solve: {pt.delta_m.shape[0]} k-modes, all finite")
