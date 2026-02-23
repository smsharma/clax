"""Test thermodynamics module against CLASS reference data."""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve, xe_of_z
from clax.params import CosmoParams, PrecisionParams


PREC = PrecisionParams(
    bg_n_points=400, ncdm_bg_n_points=200, bg_tol=1e-8,
    th_n_points=10000, th_z_max=5e3,
)


@pytest.fixture(scope="module")
def bg():
    return background_solve(CosmoParams(), PREC)


@pytest.fixture(scope="module")
def th(bg):
    return thermodynamics_solve(CosmoParams(), PREC, bg)


class TestThermoScalars:
    def test_z_star(self, th, lcdm_derived):
        """z_star (max visibility) should match CLASS to < 1%."""
        ref = lcdm_derived['z_star']
        val = float(th.z_star)
        rel_err = abs(val - ref) / ref
        assert rel_err < 0.01, f"z_star: {val:.1f} vs CLASS {ref:.1f}, err={rel_err:.2%}"

    def test_z_rec(self, th, lcdm_derived):
        """z_rec (optical depth = 1) should match CLASS to < 2%."""
        ref = lcdm_derived['z_rec']
        val = float(th.z_rec)
        rel_err = abs(val - ref) / ref
        assert rel_err < 0.02, f"z_rec: {val:.1f} vs CLASS {ref:.1f}, err={rel_err:.2%}"


class TestIonizationFraction:
    def test_xe_high_z(self, th):
        """At z=3000, x_e should be close to fully ionized (~1.08)."""
        xe = float(th.xe_of_loga.evaluate(jnp.log(jnp.array(1.0/3001.0))))
        assert abs(xe - 1.08) < 0.05, f"xe(z=3000) = {xe:.4f}, expected ~1.08"

    def test_xe_recombination(self, th, lcdm_thermo_ref):
        """x_e during recombination (z=800-1500) should match CLASS to < 30%.

        The MB95 simplified recombination model is less accurate than full RECFAST,
        so we use a looser tolerance here.
        """
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
        """At z=0, x_e should be fully reionized (~1.16)."""
        xe = float(th.xe_of_loga.evaluate(jnp.array(0.0)))
        assert abs(xe - 1.16) < 0.02, f"xe(z=0) = {xe:.4f}, expected ~1.16"


class TestVisibility:
    def test_visibility_peaks_at_recombination(self, th):
        """Visibility function should peak near z~1090."""
        # z_star should be within 2% of 1090
        assert abs(float(th.z_star) - 1090) < 30, f"z_star = {float(th.z_star):.1f}"
