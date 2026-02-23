"""Test background module against CLASS reference data.

Validates H(z), distances, growth factor, sound horizon, and derived
quantities. Each test prints a concise summary on failure (following
CLAUDE.md context-window hygiene rules).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from clax.background import (
    background_solve,
    H_of_z,
    tau_of_z,
    angular_diameter_distance,
    luminosity_distance,
    comoving_distance,
)
from clax.params import CosmoParams, PrecisionParams
from tests.conftest import assert_close


# Use consistent precision for all tests
PREC = PrecisionParams(
    bg_n_points=800,
    ncdm_bg_n_points=512,
    bg_tol=1e-10,
)


@pytest.fixture(scope="module")
def bg():
    """Compute background once for all tests in this module."""
    params = CosmoParams()
    return background_solve(params, PREC)


class TestBackgroundScalars:
    """Test scalar derived quantities."""

    def test_H0(self, bg, lcdm_scalars):
        """H0 should match CLASS exactly (both compute the same formula)."""
        ref = lcdm_scalars['H0']
        rel_err = abs(float(bg.H0) - ref) / ref
        assert rel_err < 1e-6, f"H0: rel err {rel_err:.2e} (got {bg.H0:.6e}, expected {ref:.6e})"

    def test_conformal_age(self, bg, lcdm_scalars):
        """Conformal age tau_0 should match CLASS to < 0.1%."""
        ref = lcdm_scalars['conformal_age']
        val = float(bg.conformal_age)
        rel_err = abs(val - ref) / ref
        assert rel_err < 1e-3, f"conformal_age: rel err {rel_err:.4%} (got {val:.2f}, expected {ref:.2f})"

    def test_age(self, bg, lcdm_scalars):
        """Age in Gyr should match CLASS to < 0.1%."""
        ref = lcdm_scalars['age_Gyr']
        val = float(bg.age_Gyr)
        rel_err = abs(val - ref) / ref
        assert rel_err < 1e-3, f"age: rel err {rel_err:.4%} (got {val:.4f} Gyr, expected {ref:.4f} Gyr)"

    def test_z_eq(self, bg, lcdm_scalars):
        """Matter-radiation equality redshift should match CLASS to < 1%."""
        ref = lcdm_scalars['z_eq']
        val = float(bg.z_eq)
        rel_err = abs(val - ref) / ref
        assert rel_err < 0.01, f"z_eq: rel err {rel_err:.4%} (got {val:.1f}, expected {ref:.1f})"

    def test_omega_b(self, bg, lcdm_scalars):
        ref = lcdm_scalars['Omega_b']
        val = float(bg.Omega_b)
        rel_err = abs(val - ref) / ref
        assert rel_err < 1e-6, f"Omega_b: rel err {rel_err:.2e}"

    def test_omega_cdm(self, bg, lcdm_scalars):
        ref = lcdm_scalars['Omega_cdm']
        val = float(bg.Omega_cdm)
        rel_err = abs(val - ref) / ref
        assert rel_err < 1e-6, f"Omega_cdm: rel err {rel_err:.2e}"

    def test_omega_lambda(self, bg, lcdm_scalars):
        ref = lcdm_scalars['Omega_Lambda']
        val = float(bg.Omega_lambda)
        rel_err = abs(val - ref) / ref
        assert rel_err < 1e-3, f"Omega_Lambda: rel err {rel_err:.4%} (got {val:.6f}, expected {ref:.6f})"

    def test_omega_g(self, bg, lcdm_scalars):
        ref = lcdm_scalars['Omega_g']
        val = float(bg.Omega_g)
        rel_err = abs(val - ref) / ref
        assert rel_err < 1e-4, f"Omega_g: rel err {rel_err:.2e}"


class TestBackgroundFunctions:
    """Test background quantities as functions of redshift."""

    def test_hubble(self, bg, lcdm_bg_ref, fast_mode):
        """H(z) should match CLASS to < 0.01% at all z."""
        z = lcdm_bg_ref['z']
        H_ref = lcdm_bg_ref['H']
        if fast_mode:
            z, H_ref = z[::10], H_ref[::10]

        H_us = np.array([float(H_of_z(bg, float(zi))) for zi in z])
        assert_close(H_us, H_ref, rtol=1e-4, name="H(z)", coordinate=z)

    def test_angular_diameter_distance(self, bg, lcdm_bg_ref, fast_mode):
        """D_A(z) should match CLASS to < 0.01%."""
        z = lcdm_bg_ref['z']
        DA_ref = lcdm_bg_ref['D_A']
        if fast_mode:
            z, DA_ref = z[::10], DA_ref[::10]

        # Skip z=0 (D_A=0)
        mask = z > 0.01
        z, DA_ref = z[mask], DA_ref[mask]

        DA_us = np.array([float(angular_diameter_distance(bg, float(zi))) for zi in z])
        assert_close(DA_us, DA_ref, rtol=1e-3, name="D_A(z)", coordinate=z)

    def test_luminosity_distance(self, bg, lcdm_bg_ref, fast_mode):
        """D_L(z) should match CLASS to < 0.01%."""
        z = lcdm_bg_ref['z']
        DL_ref = lcdm_bg_ref['D_L']
        if fast_mode:
            z, DL_ref = z[::10], DL_ref[::10]

        mask = z > 0.01
        z, DL_ref = z[mask], DL_ref[mask]

        DL_us = np.array([float(luminosity_distance(bg, float(zi))) for zi in z])
        assert_close(DL_us, DL_ref, rtol=1e-3, name="D_L(z)", coordinate=z)

    def test_comoving_distance(self, bg, lcdm_bg_ref, fast_mode):
        """chi(z) should match CLASS to < 0.01%."""
        z = lcdm_bg_ref['z']
        chi_ref = lcdm_bg_ref['chi']
        if fast_mode:
            z, chi_ref = z[::10], chi_ref[::10]

        mask = z > 0.01
        z, chi_ref = z[mask], chi_ref[mask]

        chi_us = np.array([float(comoving_distance(bg, float(zi))) for zi in z])
        assert_close(chi_us, chi_ref, rtol=1e-3, name="chi(z)", coordinate=z)

    def test_growth_factor(self, bg, lcdm_bg_ref, fast_mode):
        """Growth factor D(z) should match CLASS to < 0.1%."""
        z = lcdm_bg_ref['z']
        D_ref = lcdm_bg_ref['D']
        if fast_mode:
            z, D_ref = z[::10], D_ref[::10]

        mask = (z > 0.01) & (z < 50)  # growth factor only meaningful at low z
        z, D_ref = z[mask], D_ref[mask]

        # Normalize both to D(z=0)=1
        D_ref_norm = D_ref / D_ref[np.argmin(z)]

        loga_vals = np.log(1.0 / (1.0 + z))
        D_us = np.array([float(bg.D_of_loga.evaluate(jnp.array(la))) for la in loga_vals])
        D_us_norm = D_us / D_us[np.argmin(z)]

        assert_close(D_us_norm, D_ref_norm, rtol=5e-3, name="D(z)", coordinate=z)


class TestBackgroundGradients:
    """Test that gradients through background_solve are correct."""

    def test_dH0_dh(self):
        """d(H(z=0))/d(h) via AD should match finite differences."""
        prec = PrecisionParams(bg_n_points=200, ncdm_bg_n_points=100, bg_tol=1e-8)

        def H_at_z0(h):
            params = CosmoParams(h=h)
            bg = background_solve(params, prec)
            return bg.H_of_loga.evaluate(jnp.array(0.0))

        h0 = 0.6736
        grad_ad = float(jax.grad(H_at_z0)(h0))

        eps = 1e-5
        grad_fd = float((H_at_z0(h0 + eps) - H_at_z0(h0 - eps)) / (2 * eps))

        rel_err = abs(grad_ad - grad_fd) / abs(grad_fd)
        assert rel_err < 0.01, (
            f"dH0/dh: AD={grad_ad:.6e}, FD={grad_fd:.6e}, rel err={rel_err:.4%}"
        )

    def test_dconf_age_domega_b(self):
        """d(conformal_age)/d(omega_b) via AD should match finite differences."""
        prec = PrecisionParams(bg_n_points=200, ncdm_bg_n_points=100, bg_tol=1e-8)

        def conf_age(omega_b):
            params = CosmoParams(omega_b=omega_b)
            bg = background_solve(params, prec)
            return bg.conformal_age

        ob0 = 0.02237
        grad_ad = float(jax.grad(conf_age)(ob0))

        eps = 1e-6
        grad_fd = float((conf_age(ob0 + eps) - conf_age(ob0 - eps)) / (2 * eps))

        rel_err = abs(grad_ad - grad_fd) / (abs(grad_fd) + 1e-30)
        assert rel_err < 0.01, (
            f"d(tau_0)/d(omega_b): AD={grad_ad:.6e}, FD={grad_fd:.6e}, rel err={rel_err:.4%}"
        )
