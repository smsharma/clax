"""Tests for compute_cl_pp_limber and compute_cl_pp_transfer.

Validates the corrected C_l^phiphi implementations against CLASS reference
values (Planck 2018 LCDM, linear).

The original compute_cl_pp had three compounding bugs:
  1. Wrong source: exp(-kappa)*2*phi instead of (phi+psi)*W_lcmb
  2. Spurious [2/(l(l+1))]^2 prefactor
  3. Included pre-recombination times

Both new implementations fix all three and are validated here.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from clax import CosmoParams, PrecisionParams
from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve
from clax.perturbations import perturbations_solve
from clax.lensing import compute_cl_pp_limber, compute_cl_pp_transfer

from dataclasses import replace as _dc_replace

# CLASS reference C_l^pp (linear, Planck 2018 LCDM, from classy)
CLASS_PP = {
    2: 8.540281e-09,
    10: 5.612864e-11,
    50: 1.288976e-13,
    100: 6.202699e-15,
    500: 1.793808e-18,
    1000: 3.471028e-20,
    2500: 1.173568e-22,
}


@pytest.fixture(scope="module")
def pipeline():
    """Run the perturbation solver once for all tests."""
    params = CosmoParams(
        h=0.6736, omega_b=0.02237, omega_cdm=0.12,
        ln10A_s=np.log(2.089e-9 * 1e10), n_s=0.9649, tau_reio=0.052,
    )
    prec = PrecisionParams.medium_cl()
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    pt = perturbations_solve(params, prec, bg, th)
    return params, prec, bg, th, pt


class TestLimber:
    """Tests for compute_cl_pp_limber (Limber + Poisson, ABCMB approach)."""

    def test_matches_class_l50_l100(self, pipeline):
        """Limber matches CLASS to < 1% at l >= 50."""
        params, _, bg, th, pt = pipeline
        cl = compute_cl_pp_limber(pt, params, bg, th, l_max=100, n_chi=500)
        for l in [50, 100]:
            ratio = float(cl[l]) / CLASS_PP[l]
            assert abs(ratio - 1) < 0.01, (
                f"l={l}: Limber/CLASS = {ratio:.4f}, expected < 1% error")

    def test_matches_class_l10(self, pipeline):
        """Limber matches CLASS to < 5% at l=10 (Limber approximation error)."""
        params, _, bg, th, pt = pipeline
        cl = compute_cl_pp_limber(pt, params, bg, th, l_max=10, n_chi=500)
        ratio = float(cl[10]) / CLASS_PP[10]
        assert abs(ratio - 1) < 0.05, (
            f"l=10: Limber/CLASS = {ratio:.4f}, expected < 5% error")

    def test_positive(self, pipeline):
        """C_l^pp must be positive for all l >= 2."""
        params, _, bg, th, pt = pipeline
        cl = compute_cl_pp_limber(pt, params, bg, th, l_max=100, n_chi=300)
        assert jnp.all(cl[2:] > 0), "C_l^pp must be positive for l >= 2"

    def test_decreasing(self, pipeline):
        """C_l^pp should decrease from l=2 to l=100."""
        params, _, bg, th, pt = pipeline
        cl = compute_cl_pp_limber(pt, params, bg, th, l_max=100, n_chi=300)
        assert float(cl[2]) > float(cl[10]) > float(cl[100])


class TestTransfer:
    """Tests for compute_cl_pp_transfer (full Bessel, CLASS approach)."""

    def test_matches_class_l100(self, pipeline):
        """Transfer matches CLASS to < 1% at l=100."""
        params, _, bg, th, pt = pipeline
        cl = compute_cl_pp_transfer(pt, params, bg, th, [100])
        ratio = float(cl[0]) / CLASS_PP[100]
        assert abs(ratio - 1) < 0.01, (
            f"l=100: Transfer/CLASS = {ratio:.4f}, expected < 1% error")

    def test_matches_class_l50(self, pipeline):
        """Transfer matches CLASS to < 1% at l=50."""
        params, _, bg, th, pt = pipeline
        cl = compute_cl_pp_transfer(pt, params, bg, th, [50])
        ratio = float(cl[0]) / CLASS_PP[50]
        assert abs(ratio - 1) < 0.01, (
            f"l=50: Transfer/CLASS = {ratio:.4f}, expected < 1% error")

    def test_matches_class_l10(self, pipeline):
        """Transfer matches CLASS to < 10% at l=10 (k-grid limited)."""
        params, _, bg, th, pt = pipeline
        cl = compute_cl_pp_transfer(pt, params, bg, th, [10])
        ratio = float(cl[0]) / CLASS_PP[10]
        assert abs(ratio - 1) < 0.10, (
            f"l=10: Transfer/CLASS = {ratio:.4f}, expected < 10% error")

    def test_positive(self, pipeline):
        """C_l^pp must be positive for all tested l."""
        params, _, bg, th, pt = pipeline
        cl = compute_cl_pp_transfer(pt, params, bg, th, [2, 10, 50, 100])
        assert jnp.all(cl > 0), "C_l^pp must be positive"


class TestConsistency:
    """Cross-check both implementations agree where they should."""

    def test_limber_vs_transfer_l50_l100(self, pipeline):
        """Both methods agree to < 2% at l=50 and l=100."""
        params, _, bg, th, pt = pipeline
        cl_l = compute_cl_pp_limber(pt, params, bg, th, l_max=100, n_chi=500)
        cl_t = compute_cl_pp_transfer(pt, params, bg, th, [50, 100])
        for i, l in enumerate([50, 100]):
            ratio = float(cl_l[l]) / float(cl_t[i])
            assert abs(ratio - 1) < 0.02, (
                f"l={l}: Limber/Transfer = {ratio:.4f}, expected < 2% agreement")
