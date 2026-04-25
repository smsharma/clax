"""Tests for Halofit z-cutoff behavior.

Verifies that compute_pk_nonlinear:
1. At z=0, gives NL ratio > 1 at k=1 Mpc^-1
2. At high z (z=50), returns pk_lin unchanged (sigma guard triggers)
3. No crashes or NaN at any z

Uses CLASS reference data: reference_data/lcdm_fiducial/pk_nonlinear.npz
"""
import numpy as np
import pytest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from clax.nonlinear import compute_pk_nonlinear


@pytest.fixture
def setup():
    data = np.load("reference_data/lcdm_fiducial/pk_nonlinear.npz")
    k = data['k']
    pk_lin_z0 = data['pk_lin_z0']
    cosmo = dict(
        Omega_m_0=0.31519,
        Omega_lambda_0=0.68473,
        Omega_r_0=9.1e-5,
        fnu=0.0045,
        h=0.6736,
    )
    return k, pk_lin_z0, cosmo


def _scale_pk_to_z(pk_z0, z, Omega_m_0=0.31519):
    """Approximate P_lin(k,z) = P_lin(k,0) * (D(z)/D(0))^2.

    Uses the growth suppression factor g(a) ~ Omega_m(a)^0.55.
    """
    a = 1.0 / (1.0 + z)
    # Approximate growth factor ratio (adequate for this test)
    D_ratio = a * (Omega_m_0 / (Omega_m_0 + (1 - Omega_m_0) * a**3)) ** 0.55
    return pk_z0 * D_ratio ** 2


class TestZCutoff:
    """Test Halofit z-cutoff behavior."""

    def test_z0_has_nl_correction(self, setup):
        """At z=0, P_NL/P_lin > 1 at k=1 Mpc^-1."""
        k, pk_lin, cosmo = setup
        pk_nl = compute_pk_nonlinear(
            jnp.array(k), jnp.array(pk_lin), z=0.0, **cosmo)
        ratio = np.array(pk_nl) / pk_lin

        idx = np.argmin(np.abs(k - 1.0))
        assert ratio[idx] > 1.5, f"At z=0, k=1: ratio={ratio[idx]:.2f}, expected > 1.5"

    def test_z2_has_smaller_correction(self, setup):
        """At z=2, NL ratio at k=1 should be smaller than at z=0."""
        k, pk_lin, cosmo = setup
        pk_lin_z2 = _scale_pk_to_z(pk_lin, z=2.0)
        pk_nl_z2 = compute_pk_nonlinear(
            jnp.array(k), jnp.array(pk_lin_z2), z=2.0, **cosmo)
        ratio_z2 = np.array(pk_nl_z2) / pk_lin_z2

        pk_nl_z0 = compute_pk_nonlinear(
            jnp.array(k), jnp.array(pk_lin), z=0.0, **cosmo)
        ratio_z0 = np.array(pk_nl_z0) / pk_lin

        idx = np.argmin(np.abs(k - 1.0))
        assert ratio_z2[idx] < ratio_z0[idx], (
            f"z=2 ratio ({ratio_z2[idx]:.2f}) should be < z=0 ratio ({ratio_z0[idx]:.2f})")

    def test_z50_returns_plin(self, setup):
        """At z=50, sigma(R) < 1 on any grid, so should return pk_lin."""
        k, pk_lin, cosmo = setup
        pk_lin_z50 = _scale_pk_to_z(pk_lin, z=50.0)
        pk_nl_z50 = compute_pk_nonlinear(
            jnp.array(k), jnp.array(pk_lin_z50), z=50.0, **cosmo)

        np.testing.assert_array_equal(
            np.array(pk_nl_z50), np.array(pk_lin_z50),
            err_msg="z=50: should return pk_lin unchanged (sigma guard)")

    def test_no_nan_at_any_z(self, setup):
        """No NaN in P_NL at z = 0, 1, 5, 20."""
        k, pk_lin, cosmo = setup
        for z in [0.0, 1.0, 5.0, 20.0]:
            pk_z = _scale_pk_to_z(pk_lin, z) if z > 0 else pk_lin
            pk_nl = compute_pk_nonlinear(
                jnp.array(k), jnp.array(pk_z), z=z, **cosmo)
            assert not np.any(np.isnan(np.array(pk_nl))), f"NaN at z={z}"
            assert not np.any(np.isinf(np.array(pk_nl))), f"Inf at z={z}"
