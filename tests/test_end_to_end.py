"""End-to-end tests for the full jaxCLASS pipeline.

Tests the compute() and compute_pk() functions, including gradient tests
that differentiate through background → thermodynamics → perturbations.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import jaxclass
from jaxclass import CosmoParams, PrecisionParams


# Low-res precision for speed
PREC_FAST = PrecisionParams(
    bg_n_points=200, ncdm_bg_n_points=100, bg_tol=1e-8,
    th_n_points=5000, th_z_max=5e3,
    pt_l_max_g=6, pt_l_max_pol_g=6, pt_l_max_ur=6,
    pt_ode_rtol=1e-3, pt_ode_atol=1e-6,
    ode_max_steps=16384,
)


class TestCompute:
    """Test the top-level compute() function."""

    def test_compute_returns_result(self):
        """compute() should return a ComputeResult with bg and th."""
        result = jaxclass.compute(CosmoParams(), PREC_FAST)
        assert hasattr(result, 'bg')
        assert hasattr(result, 'th')
        assert float(result.bg.H0) > 0

    def test_compute_H0(self, lcdm_scalars):
        """H0 from compute() should match CLASS."""
        result = jaxclass.compute(CosmoParams(), PREC_FAST)
        ref = lcdm_scalars['H0']
        rel = abs(float(result.bg.H0) - ref) / ref
        assert rel < 1e-4, f"H0 rel err: {rel:.2e}"

    def test_compute_z_star(self, lcdm_derived):
        """z_star from compute() should match CLASS to < 1%."""
        result = jaxclass.compute(CosmoParams(), PREC_FAST)
        ref = lcdm_derived['z_star']
        rel = abs(float(result.th.z_star) - ref) / ref
        assert rel < 0.01, f"z_star: got {float(result.th.z_star):.1f}, CLASS {ref:.1f}"


class TestComputePk:
    """Test the compute_pk() function."""

    def test_pk_positive(self):
        """P(k) should be positive."""
        pk = jaxclass.compute_pk(CosmoParams(), PREC_FAST, k=0.05)
        assert float(pk) > 0, f"P(k=0.05) = {float(pk)}"

    def test_pk_low_k(self, lcdm_pk_ref):
        """P(k=0.001) should match CLASS to < 5%."""
        pk = float(jaxclass.compute_pk(CosmoParams(), PREC_FAST, k=0.001))
        idx_ref = np.argmin(np.abs(lcdm_pk_ref['k'] - 0.001))
        pk_class = lcdm_pk_ref['pk_lin_z0'][idx_ref]
        ratio = pk / pk_class
        assert abs(ratio - 1) < 0.05, f"P(k=0.001): ratio={ratio:.4f}"


class TestGradients:
    """Test AD gradients through the full pipeline."""

    def test_grad_H0_wrt_h(self):
        """d(H0)/d(h) should be non-zero and match FD."""
        def H0_fn(h):
            result = jaxclass.compute(CosmoParams(h=h), PREC_FAST)
            return result.bg.H0

        h0 = 0.6736
        grad_ad = float(jax.grad(H0_fn)(h0))
        eps = 1e-5
        grad_fd = float((H0_fn(h0 + eps) - H0_fn(h0 - eps)) / (2 * eps))
        rel = abs(grad_ad - grad_fd) / abs(grad_fd)
        assert rel < 0.01, f"dH0/dh: AD={grad_ad:.4e} FD={grad_fd:.4e} err={rel:.2%}"

    def test_grad_conf_age_wrt_omega_cdm(self):
        """d(conf_age)/d(omega_cdm) should match FD."""
        def age_fn(omega_cdm):
            result = jaxclass.compute(CosmoParams(omega_cdm=omega_cdm), PREC_FAST)
            return result.bg.conformal_age

        oc0 = 0.12
        grad_ad = float(jax.grad(age_fn)(oc0))
        eps = 1e-5
        grad_fd = float((age_fn(oc0 + eps) - age_fn(oc0 - eps)) / (2 * eps))
        rel = abs(grad_ad - grad_fd) / abs(grad_fd)
        assert rel < 0.01, f"d(age)/d(ocdm): AD={grad_ad:.4e} FD={grad_fd:.4e} err={rel:.2%}"

    @pytest.mark.slow
    def test_grad_pk_wrt_omega_cdm(self):
        """d(P(k))/d(omega_cdm) through full Boltzmann pipeline should match FD."""
        def pk_fn(omega_cdm):
            return jaxclass.compute_pk(CosmoParams(omega_cdm=omega_cdm), PREC_FAST, k=0.05)

        oc0 = 0.12
        grad_ad = float(jax.grad(pk_fn)(oc0))
        eps = 1e-4
        grad_fd = float((pk_fn(oc0 + eps) - pk_fn(oc0 - eps)) / (2 * eps))
        rel = abs(grad_ad - grad_fd) / abs(grad_fd)
        assert rel < 0.05, (
            f"d(P(k))/d(omega_cdm): AD={grad_ad:.4e} FD={grad_fd:.4e} err={rel:.2%}"
        )

    @pytest.mark.slow
    def test_grad_pk_wrt_omega_b(self):
        """d(P(k))/d(omega_b) through full Boltzmann pipeline should match FD."""
        def pk_fn(omega_b):
            return jaxclass.compute_pk(CosmoParams(omega_b=omega_b), PREC_FAST, k=0.05)

        ob0 = 0.02237
        grad_ad = float(jax.grad(pk_fn)(ob0))
        eps = 1e-5
        grad_fd = float((pk_fn(ob0 + eps) - pk_fn(ob0 - eps)) / (2 * eps))
        rel = abs(grad_ad - grad_fd) / abs(grad_fd)
        assert rel < 0.05, (
            f"d(P(k))/d(omega_b): AD={grad_ad:.4e} FD={grad_fd:.4e} err={rel:.2%}"
        )

    @pytest.mark.slow
    def test_grad_pk_wrt_n_s(self):
        """d(P(k))/d(n_s) through full Boltzmann pipeline should match FD."""
        # Use k != k_pivot so n_s has a non-zero effect through the primordial spectrum
        def pk_fn(n_s):
            return jaxclass.compute_pk(CosmoParams(n_s=n_s), PREC_FAST, k=0.01)

        ns0 = 0.9649
        grad_ad = float(jax.grad(pk_fn)(ns0))
        eps = 1e-4
        grad_fd = float((pk_fn(ns0 + eps) - pk_fn(ns0 - eps)) / (2 * eps))
        rel = abs(grad_ad - grad_fd) / (abs(grad_fd) + 1e-30)
        assert rel < 0.05, (
            f"d(P(k=0.01))/d(n_s): AD={grad_ad:.4e} FD={grad_fd:.4e} err={rel:.2%}"
        )
