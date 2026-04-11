"""Tests thermodynamics-layer forward behavior and targeted gradient contracts.

Contract:
- Thermodynamics quantities and recombination-era functions match the documented CLASS-derived references.
- The repaired reionization and opacity-derivative AD paths remain consistent with finite differences.

Scope:
- Covers ``z_star``, ``z_rec``, ionization history, visibility behavior, and the repaired thermodynamics gradient subcontracts.
- Excludes background and perturbation-layer contracts owned elsewhere.

Notes:
- These tests use CLASS-generated reference data for forward assertions plus narrow FD spot checks for the repaired thermodynamics AD paths.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve
from clax.params import CosmoParams, PrecisionParams
from tests.pk_test_utils import PK_GRAD_PARAM_STEPS


PREC = PrecisionParams(
    bg_n_points=400, ncdm_bg_n_points=200, bg_tol=1e-8,
    th_n_points=10000, th_z_max=5e3,
)


@pytest.fixture(scope="module")
def bg():
    """Compute the fiducial background state once for this module."""
    return background_solve(CosmoParams(), PREC)


@pytest.fixture(scope="module")
def th(bg):
    """Compute the fiducial thermodynamics state once for this module."""
    return thermodynamics_solve(CosmoParams(), PREC, bg)


class TestThermoScalars:
    """Tests thermodynamics scalar quantities."""

    def test_z_star(self, th, lcdm_derived):
        """``z_star`` matches CLASS; expects <1% relative error."""
        ref = lcdm_derived['z_star']
        val = float(th.z_star)
        rel_err = abs(val - ref) / ref
        assert rel_err < 0.01, f"z_star: {val:.1f} vs CLASS {ref:.1f}, err={rel_err:.2%}"

    def test_z_rec(self, th, lcdm_derived):
        """``z_rec`` matches CLASS; expects <2% relative error."""
        ref = lcdm_derived['z_rec']
        val = float(th.z_rec)
        rel_err = abs(val - ref) / ref
        assert rel_err < 0.02, f"z_rec: {val:.1f} vs CLASS {ref:.1f}, err={rel_err:.2%}"


class TestIonizationFraction:
    """Tests ionization-fraction behavior."""

    def test_xe_high_z(self, th):
        """``x_e(z=3000)`` is near full ionization; expects a value near 1.08."""
        xe = float(th.xe_of_loga.evaluate(jnp.log(jnp.array(1.0/3001.0))))
        assert abs(xe - 1.08) < 0.05, f"xe(z=3000) = {xe:.4f}, expected ~1.08"

    def test_xe_recombination(self, th, lcdm_thermo_ref):
        """``x_e`` during recombination matches CLASS; expects <30% relative error at the probe redshifts."""
        ref_z = lcdm_thermo_ref['z']
        ref_xe = lcdm_thermo_ref['x_e']

        # Test at specific recombination redshifts
        for z_test in [1200.0, 1100.0, 1000.0, 800.0]:
            idx = np.argmin(np.abs(ref_z - z_test))
            xe_class = ref_xe[idx]
            la = float(jnp.log(1.0 / (1.0 + z_test)))
            xe_us = float(th.xe_of_loga.evaluate(jnp.array(la)))

            if xe_class > 0.001:
                rel_err = abs(xe_us - xe_class) / xe_class
                assert rel_err < 0.30, (
                    f"xe(z={z_test:.0f}): us={xe_us:.6f} CLASS={xe_class:.6f} "
                    f"err={rel_err:.1%}"
                )

    def test_xe_reionization(self, th):
        """``x_e(z=0)`` is fully reionized; expects a value near 1.16."""
        xe = float(th.xe_of_loga.evaluate(jnp.array(0.0)))
        assert abs(xe - 1.16) < 0.02, f"xe(z=0) = {xe:.4f}, expected ~1.16"


class TestVisibility:
    """Tests visibility-function behavior."""

    def test_visibility_peaks_at_recombination(self, th):
        """The visibility function peaks near recombination; expects ``z_star`` close to 1090."""
        assert abs(float(th.z_star) - 1090) < 30, f"z_star = {float(th.z_star):.1f}"


def _thermo_ad_fd_pair(param_name, quantity_fn):
    """Return ``(ad, fd)`` for one scalar thermodynamics quantity."""
    params = CosmoParams()
    step = PK_GRAD_PARAM_STEPS[param_name]
    x0 = getattr(params, param_name)

    def wrapped(x):
        varied = params.replace(**{param_name: x})
        bg = background_solve(varied, PREC)
        th = thermodynamics_solve(varied, PREC, bg)
        return quantity_fn(th)

    ad = float(jax.grad(wrapped)(x0))
    fd = float((wrapped(x0 + step) - wrapped(x0 - step)) / (2.0 * step))
    return ad, fd


class TestThermoGradients:
    """Tests thermodynamics gradient behavior at the repaired opacity branches."""

    @pytest.mark.parametrize("param_name", ["h", "omega_b"])
    def test_reionization_gradients_match_fd(self, param_name):
        """``z_reio`` and late-time ``x_e`` gradients match finite differences."""
        z_ad, z_fd = _thermo_ad_fd_pair(param_name, lambda th: th.z_reio)
        z_rel = abs(z_ad - z_fd) / (abs(z_fd) + 1e-30)
        assert z_rel < 0.01, (
            f"z_reio grad {param_name}: AD={z_ad:.6e} FD={z_fd:.6e} rel={z_rel:.2%}"
        )

        xe_ad, xe_fd = _thermo_ad_fd_pair(
            param_name, lambda th: th.xe_of_loga.evaluate(jnp.array(-2.0))
        )
        xe_rel = abs(xe_ad - xe_fd) / (abs(xe_fd) + 1e-30)
        assert xe_rel < 0.01, (
            f"xe(loga=-2) grad {param_name}: AD={xe_ad:.6e} FD={xe_fd:.6e} rel={xe_rel:.2%}"
        )

    def test_opacity_logderivative_gradient_matches_fd_for_omega_b(self):
        """Stored ``dκ̇/dloga`` table remains differentiable through recombination."""
        ad, fd = _thermo_ad_fd_pair(
            "omega_b", lambda th: th.dkappa_dot_dloga_of_loga.evaluate(jnp.array(-8.0))
        )
        rel = abs(ad - fd) / (abs(fd) + 1e-30)
        assert rel < 0.01, (
            f"dkappa_dot_dloga(loga=-8) grad omega_b: AD={ad:.6e} FD={fd:.6e} rel={rel:.2%}"
        )
