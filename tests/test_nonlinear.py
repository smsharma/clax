"""Test non-linear P(k) (HaloFit) against CLASS reference data."""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import os
import pytest

from jaxclass.nonlinear import (
    sigma_R,
    halofit_parameters,
    halofit_nl_pk,
    compute_pk_nonlinear,
)

REFERENCE_DIR = os.path.join(os.path.dirname(__file__), '..', 'reference_data')


@pytest.fixture(scope="module")
def pk_nl_ref():
    """Load nonlinear P(k) reference data from CLASS."""
    path = os.path.join(REFERENCE_DIR, 'lcdm_fiducial', 'pk_nonlinear.npz')
    return dict(np.load(path))


# Fiducial cosmology parameters (Planck 2018)
OMEGA_B = 0.02237
OMEGA_CDM = 0.1200
H = 0.6736
M_NCDM = 0.06

# Derived
OMEGA_B_FRAC = OMEGA_B / H**2
OMEGA_CDM_FRAC = OMEGA_CDM / H**2
OMEGA_M_0 = OMEGA_B_FRAC + OMEGA_CDM_FRAC + 0.00138  # approximate ncdm contribution
OMEGA_LAMBDA_0 = 1.0 - OMEGA_M_0 - 5.4e-5 - 3.7e-5  # 1 - Om - Og - Our (approx)
OMEGA_R_0 = 5.4e-5 + 3.7e-5  # Og + Our (approx)
FNU = 0.00138 / OMEGA_M_0  # Omega_ncdm / Omega_m


class TestSigmaR:
    """Test sigma(R) integral computation."""

    def test_sigma_decreasing(self, pk_nl_ref):
        """sigma(R) should decrease with increasing R."""
        k = pk_nl_ref['k']
        pk_lin = pk_nl_ref['pk_lin_z0']
        lnk = jnp.log(k)
        pk = jnp.array(pk_lin)

        s1 = float(sigma_R(1.0, lnk, pk))
        s5 = float(sigma_R(5.0, lnk, pk))
        s10 = float(sigma_R(10.0, lnk, pk))

        assert s1 > s5 > s10, f"sigma not decreasing: {s1:.3f}, {s5:.3f}, {s10:.3f}"

    def test_sigma_positive(self, pk_nl_ref):
        """sigma(R) should be positive for any R."""
        k = pk_nl_ref['k']
        pk_lin = pk_nl_ref['pk_lin_z0']
        lnk = jnp.log(k)
        pk = jnp.array(pk_lin)

        for R in [0.5, 1.0, 5.0, 10.0, 50.0]:
            s = float(sigma_R(R, lnk, pk))
            assert s > 0, f"sigma(R={R}) = {s}, expected > 0"


class TestHaloFitParameters:
    """Test the non-linear scale and spectral parameters."""

    def test_k_sigma_range(self, pk_nl_ref):
        """k_sigma should be in a physically reasonable range (~0.1-0.5 Mpc^-1)."""
        k = pk_nl_ref['k']
        pk_lin = pk_nl_ref['pk_lin_z0']
        lnk = jnp.log(k)
        pk = jnp.array(pk_lin)

        k_sigma, n_eff, C = halofit_parameters(lnk, pk)
        k_sigma = float(k_sigma)
        n_eff = float(n_eff)

        assert 0.05 < k_sigma < 1.0, f"k_sigma = {k_sigma:.4f} out of range"
        assert -3.0 < n_eff < 0.0, f"n_eff = {n_eff:.4f} out of range"

    def test_sigma_at_k_sigma_is_one(self, pk_nl_ref):
        """sigma(1/k_sigma) should be 1 by construction."""
        k = pk_nl_ref['k']
        pk_lin = pk_nl_ref['pk_lin_z0']
        lnk = jnp.log(k)
        pk = jnp.array(pk_lin)

        k_sigma, _, _ = halofit_parameters(lnk, pk)
        R_nl = 1.0 / k_sigma
        s = float(sigma_R(R_nl, lnk, pk))

        assert abs(s - 1.0) < 1e-4, f"sigma(1/k_sigma) = {s:.6f}, expected 1.0"


class TestHaloFitPk:
    """Test non-linear P(k) against CLASS HaloFit reference."""

    def test_pk_nl_z0_low_k(self, pk_nl_ref):
        """At low k, non-linear P(k) should equal linear P(k)."""
        k = pk_nl_ref['k']
        pk_lin = pk_nl_ref['pk_lin_z0']

        pk_nl = compute_pk_nonlinear(
            jnp.array(k), jnp.array(pk_lin),
            Omega_m_0=OMEGA_M_0, Omega_lambda_0=OMEGA_LAMBDA_0,
            Omega_r_0=OMEGA_R_0, fnu=FNU, h=H, z=0.0,
        )

        # At k < 0.001, should be identical to linear
        mask = k < 0.001
        np.testing.assert_allclose(
            np.array(pk_nl[mask]), pk_lin[mask], rtol=1e-10,
            err_msg="Non-linear P(k) should equal linear P(k) at very low k"
        )

    def test_pk_nl_z0_enhancement(self, pk_nl_ref):
        """At high k, non-linear P(k) should exceed linear P(k)."""
        k = pk_nl_ref['k']
        pk_lin = pk_nl_ref['pk_lin_z0']

        pk_nl = compute_pk_nonlinear(
            jnp.array(k), jnp.array(pk_lin),
            Omega_m_0=OMEGA_M_0, Omega_lambda_0=OMEGA_LAMBDA_0,
            Omega_r_0=OMEGA_R_0, fnu=FNU, h=H, z=0.0,
        )

        # At k ~ 1 Mpc^-1, should have significant enhancement
        idx_k1 = np.argmin(np.abs(k - 1.0))
        ratio = float(pk_nl[idx_k1]) / pk_lin[idx_k1]
        assert ratio > 2.0, f"P_NL/P_lin at k=1: {ratio:.2f}, expected > 2"

    def test_pk_nl_z0_vs_class(self, pk_nl_ref):
        """Non-linear P(k) at z=0 should match CLASS to < 10% for k < 10 Mpc^-1."""
        k = pk_nl_ref['k']
        pk_lin = pk_nl_ref['pk_lin_z0']
        pk_class_nl = pk_nl_ref['pk_nl_z0']

        pk_nl = compute_pk_nonlinear(
            jnp.array(k), jnp.array(pk_lin),
            Omega_m_0=OMEGA_M_0, Omega_lambda_0=OMEGA_LAMBDA_0,
            Omega_r_0=OMEGA_R_0, fnu=FNU, h=H, z=0.0,
        )

        # Compare where non-linear effects matter: k > 0.01 and k < 10
        mask = (k > 0.01) & (k < 10.0)
        ratio = np.array(pk_nl[mask]) / pk_class_nl[mask]
        max_err = np.max(np.abs(ratio - 1.0))
        mean_err = np.mean(np.abs(ratio - 1.0))

        # Find worst k
        worst_idx = np.argmax(np.abs(ratio - 1.0))
        k_worst = k[mask][worst_idx]

        assert max_err < 0.10, (
            f"P_NL(k) max error {max_err:.2%} at k={k_worst:.3f} Mpc^-1 "
            f"(mean {mean_err:.2%}), expected < 10%"
        )

    def test_pk_nl_z0_vs_class_quasi_linear(self, pk_nl_ref):
        """Non-linear P(k) in the quasi-linear regime should match CLASS well."""
        k = pk_nl_ref['k']
        pk_lin = pk_nl_ref['pk_lin_z0']
        pk_class_nl = pk_nl_ref['pk_nl_z0']

        pk_nl = compute_pk_nonlinear(
            jnp.array(k), jnp.array(pk_lin),
            Omega_m_0=OMEGA_M_0, Omega_lambda_0=OMEGA_LAMBDA_0,
            Omega_r_0=OMEGA_R_0, fnu=FNU, h=H, z=0.0,
        )

        # Quasi-linear regime: 0.01 < k < 0.3
        mask = (k > 0.01) & (k < 0.3)
        ratio = np.array(pk_nl[mask]) / pk_class_nl[mask]
        max_err = np.max(np.abs(ratio - 1.0))

        assert max_err < 0.05, (
            f"Quasi-linear P_NL(k) max error {max_err:.2%}, expected < 5%"
        )


class TestHaloFitDifferentiable:
    """Test that HaloFit is differentiable."""

    def test_grad_pk_nl_wrt_pk_lin(self, pk_nl_ref):
        """Gradient of P_NL w.r.t. P_lin should be computable."""
        k_test = jnp.array([0.01, 0.1, 1.0, 10.0])
        # Create a simple power-law linear P(k)
        A_s = 2.1e-9
        n_s = 0.9649
        pk_test = A_s * (k_test / 0.05) ** (n_s - 1) * 2 * jnp.pi**2 / k_test**3 * 1e3

        def pk_nl_sum(pk_lin):
            return jnp.sum(compute_pk_nonlinear(
                k_test, pk_lin,
                Omega_m_0=0.31, Omega_lambda_0=0.69,
                Omega_r_0=9e-5, fnu=0.004, h=0.6736, z=0.0,
            ))

        grad = jax.grad(pk_nl_sum)(pk_test)
        assert jnp.all(jnp.isfinite(grad)), "Gradient contains NaN/Inf"
        assert jnp.any(grad != 0.0), "Gradient is all zeros"

    def test_jit_compatible(self, pk_nl_ref):
        """HaloFit should be JIT-compilable."""
        k = jnp.array(pk_nl_ref['k'])
        pk_lin = jnp.array(pk_nl_ref['pk_lin_z0'])

        @jax.jit
        def compute_nl(pk_lin):
            return compute_pk_nonlinear(
                k, pk_lin,
                Omega_m_0=OMEGA_M_0, Omega_lambda_0=OMEGA_LAMBDA_0,
                Omega_r_0=OMEGA_R_0, fnu=FNU, h=H, z=0.0,
            )

        pk_nl = compute_nl(pk_lin)
        assert jnp.all(jnp.isfinite(pk_nl)), "JIT result contains NaN/Inf"
