"""Tests for Halofit sigma(R) convergence guard and P_NL/P_lin accuracy.

Verifies that:
1. With a wide k-grid (k_max=50), Halofit gives correct P_NL/P_lin ratios
2. With a narrow k-grid (k_max=0.35), convergence guard returns P_lin unchanged
3. With k_max=5.0, convergence guard passes and halofit_parameters match

Uses CLASS reference data: reference_data/lcdm_fiducial/pk_nonlinear.npz
"""
import numpy as np
import pytest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from clax.nonlinear import (
    compute_pk_nonlinear,
    halofit_parameters,
    _sigma_convergence_check,
)


@pytest.fixture
def class_reference():
    data = np.load("reference_data/lcdm_fiducial/pk_nonlinear.npz")
    return {
        'k': data['k'],
        'pk_lin_z0': data['pk_lin_z0'],
        'pk_nl_z0': data['pk_nl_z0'],
    }


@pytest.fixture
def cosmo_params():
    """Background parameters for Planck 2018 LCDM."""
    return dict(
        Omega_m_0=0.31519,
        Omega_lambda_0=0.68473,
        Omega_r_0=9.1e-5,
        fnu=0.0045,
        h=0.6736,
    )


class TestSigmaConvergenceCheck:
    """Test _sigma_convergence_check function."""

    def test_wide_grid_passes(self, class_reference):
        """k_max=50 should pass convergence check."""
        k = class_reference['k']
        pk = class_reference['pk_lin_z0']
        lnk = jnp.log(jnp.array(k))
        assert _sigma_convergence_check(lnk, jnp.array(pk)) is True

    def test_narrow_grid_fails(self, class_reference):
        """k_max=0.35 should fail convergence check."""
        k = class_reference['k']
        pk = class_reference['pk_lin_z0']
        mask = k <= 0.35
        lnk = jnp.log(jnp.array(k[mask]))
        assert _sigma_convergence_check(lnk, jnp.array(pk[mask])) is False

    def test_kmax5_passes(self, class_reference):
        """k_max=5.0 should pass convergence check."""
        k = class_reference['k']
        pk = class_reference['pk_lin_z0']
        mask = k <= 5.0
        lnk = jnp.log(jnp.array(k[mask]))
        assert _sigma_convergence_check(lnk, jnp.array(pk[mask])) is True


class TestHalofitAccuracy:
    """Test Halofit P_NL/P_lin accuracy vs CLASS reference."""

    def test_pnl_ratio_wide_grid(self, class_reference, cosmo_params):
        """P_NL/P_lin on full grid should match CLASS within 5%."""
        k = jnp.array(class_reference['k'])
        pk_lin = jnp.array(class_reference['pk_lin_z0'])
        pk_nl_ref = class_reference['pk_nl_z0']

        pk_nl = compute_pk_nonlinear(k, pk_lin, z=0.0, **cosmo_params)
        ratio_clax = np.array(pk_nl) / np.array(pk_lin)
        ratio_class = pk_nl_ref / class_reference['pk_lin_z0']

        # Check at specific k values
        for k_val, tol in [(0.1, 0.05), (0.5, 0.05), (1.0, 0.05), (5.0, 0.10)]:
            idx = np.argmin(np.abs(class_reference['k'] - k_val))
            rel_err = abs(ratio_clax[idx] - ratio_class[idx]) / ratio_class[idx]
            assert rel_err < tol, (
                f"k={class_reference['k'][idx]:.3f}: "
                f"clax={ratio_clax[idx]:.4f} CLASS={ratio_class[idx]:.4f} err={rel_err:.4f}"
            )


class TestConvergenceGuard:
    """Test that narrow grids return P_lin unchanged."""

    def test_narrow_grid_returns_plin(self, class_reference, cosmo_params):
        """k_max=0.35: compute_pk_nonlinear should return pk_lin (no NL)."""
        k = class_reference['k']
        pk_lin = class_reference['pk_lin_z0']
        mask = k <= 0.35
        k_narrow = jnp.array(k[mask])
        pk_narrow = jnp.array(pk_lin[mask])

        pk_nl = compute_pk_nonlinear(k_narrow, pk_narrow, z=0.0, **cosmo_params)
        np.testing.assert_array_equal(
            np.array(pk_nl), np.array(pk_narrow),
            err_msg="Narrow grid should return pk_lin unchanged"
        )

    def test_kmax5_halofit_params_match(self, class_reference):
        """k_max=5 halofit_parameters should match k_max=50 within 2%."""
        k = class_reference['k']
        pk = class_reference['pk_lin_z0']

        lnk_full = jnp.log(jnp.array(k))
        pk_full = jnp.array(pk)
        k_sigma_full, n_eff_full, C_full = halofit_parameters(lnk_full, pk_full)

        mask = k <= 5.0
        lnk_5 = jnp.log(jnp.array(k[mask]))
        pk_5 = jnp.array(pk[mask])
        k_sigma_5, n_eff_5, C_5 = halofit_parameters(lnk_5, pk_5)

        rel_k = abs(float(k_sigma_5) - float(k_sigma_full)) / float(k_sigma_full)
        rel_n = abs(float(n_eff_5) - float(n_eff_full)) / abs(float(n_eff_full))
        rel_C = abs(float(C_5) - float(C_full)) / abs(float(C_full))

        assert rel_k < 0.02, f"k_sigma: {float(k_sigma_5):.4f} vs {float(k_sigma_full):.4f}, err={rel_k:.4f}"
        assert rel_n < 0.02, f"n_eff: {float(n_eff_5):.4f} vs {float(n_eff_full):.4f}, err={rel_n:.4f}"
        assert rel_C < 0.05, f"C: {float(C_5):.4f} vs {float(C_full):.4f}, err={rel_C:.4f}"
