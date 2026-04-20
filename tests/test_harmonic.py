"""Tests scalar harmonic-layer forward behavior.

Contract:
- Scalar ``C_l`` computations are finite, sign-correct, and approximately consistent with CLASS.

Scope:
- Covers low and mid-``l`` TT/EE/TE forward checks.
- Excludes high-``l`` helper/API behavior owned by ``test_high_l.py``.

Notes:
- These tests use the ``fast_cl`` preset and therefore own approximate, not science-grade, tolerances.
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
from clax.harmonic import compute_cl_tt, compute_cl_ee, compute_cl_te


PREC = PrecisionParams.fast_cl()


@pytest.fixture(scope="module")
def pipeline():
    """Run the full pipeline once for all tests in this module."""
    params = CosmoParams()
    bg = background_solve(params, PREC)
    th = thermodynamics_solve(params, PREC, bg)
    pt = perturbations_solve(params, PREC, bg, th)
    return params, bg, th, pt


class TestClTT:
    """Tests scalar TT-spectrum behavior."""

    def test_cl_tt_l100(self, pipeline, lcdm_cls_ref):
        """``C_l^TT`` at ``l=100`` matches CLASS; expects <30% relative error."""
        params, bg, _, pt = pipeline
        cl = compute_cl_tt(pt, params, bg, [100])
        cl_us = float(cl[0])
        cl_class = float(lcdm_cls_ref['tt'][100])
        ratio = cl_us / cl_class
        print(f"C_l^TT(l=100): clax={cl_us:.4e}, CLASS={cl_class:.4e}, ratio={ratio:.4f}")
        assert abs(ratio - 1) < 0.30, f"C_l^TT(l=100): ratio={ratio:.4f}, expected within 30%"

    def test_cl_tt_l50(self, pipeline, lcdm_cls_ref):
        """``C_l^TT`` at ``l=50`` matches CLASS; expects <50% relative error."""
        params, bg, _, pt = pipeline
        cl = compute_cl_tt(pt, params, bg, [50])
        cl_us = float(cl[0])
        cl_class = float(lcdm_cls_ref['tt'][50])
        ratio = cl_us / cl_class
        print(f"C_l^TT(l=50): clax={cl_us:.4e}, CLASS={cl_class:.4e}, ratio={ratio:.4f}")
        assert abs(ratio - 1) < 0.50, f"C_l^TT(l=50): ratio={ratio:.4f}, expected within 50%"

    def test_cl_tt_l10(self, pipeline, lcdm_cls_ref):
        """``C_l^TT`` at ``l=10`` matches CLASS; expects <50% relative error."""
        params, bg, _, pt = pipeline
        cl = compute_cl_tt(pt, params, bg, [10])
        cl_us = float(cl[0])
        cl_class = float(lcdm_cls_ref['tt'][10])
        ratio = cl_us / cl_class
        print(f"C_l^TT(l=10): clax={cl_us:.4e}, CLASS={cl_class:.4e}, ratio={ratio:.4f}")
        assert abs(ratio - 1) < 0.50, f"C_l^TT(l=10): ratio={ratio:.4f}, expected within 50%"

    def test_cl_tt_positive(self, pipeline):
        """``C_l^TT`` is positive on the probe grid; expects positive values."""
        params, bg, _, pt = pipeline
        cl = compute_cl_tt(pt, params, bg, [10, 50, 100])
        for i, l in enumerate([10, 50, 100]):
            assert float(cl[i]) > 0, f"C_l^TT(l={l}) = {float(cl[i]):.4e} is not positive"


class TestClEE:
    """Tests scalar EE-spectrum behavior."""

    def test_cl_ee_positive(self, pipeline):
        """``C_l^EE`` is positive on the probe grid; expects positive values."""
        params, bg, _, pt = pipeline
        cl = compute_cl_ee(pt, params, bg, [100, 200])
        for i, l in enumerate([100, 200]):
            assert float(cl[i]) > 0, f"C_l^EE(l={l}) = {float(cl[i]):.4e} is not positive"

    def test_cl_ee_finite(self, pipeline):
        """``C_l^EE`` is finite on the probe grid; expects no NaN or Inf values."""
        params, bg, _, pt = pipeline
        cl = compute_cl_ee(pt, params, bg, [100, 200])
        for i, l in enumerate([100, 200]):
            assert np.isfinite(float(cl[i])), f"C_l^EE(l={l}) is not finite"

    def test_cl_ee_l100_accuracy(self, pipeline, lcdm_cls_ref):
        """``C_l^EE`` at ``l=100`` matches CLASS; expects <60% relative error."""
        params, bg, _, pt = pipeline
        cl = compute_cl_ee(pt, params, bg, [100])
        cl_us = float(cl[0])
        cl_class = float(lcdm_cls_ref['ee'][100])
        ratio = cl_us / cl_class
        print(f"C_l^EE(l=100): clax={cl_us:.4e}, CLASS={cl_class:.4e}, ratio={ratio:.4f}")
        assert abs(ratio - 1) < 0.60, f"C_l^EE(l=100): ratio={ratio:.4f}"

    def test_cl_ee_l200_accuracy(self, pipeline, lcdm_cls_ref):
        """``C_l^EE`` at ``l=200`` matches CLASS; expects <60% relative error."""
        params, bg, _, pt = pipeline
        cl = compute_cl_ee(pt, params, bg, [200])
        cl_us = float(cl[0])
        cl_class = float(lcdm_cls_ref['ee'][200])
        ratio = cl_us / cl_class
        print(f"C_l^EE(l=200): clax={cl_us:.4e}, CLASS={cl_class:.4e}, ratio={ratio:.4f}")
        assert abs(ratio - 1) < 0.60, f"C_l^EE(l=200): ratio={ratio:.4f}"

class TestClTE:
    """Tests scalar TE-spectrum behavior."""

    def test_cl_te_sign(self, pipeline, lcdm_cls_ref):
        """``C_l^TE`` sign matches CLASS; expects matching signs on the probe grid."""
        params, bg, _, pt = pipeline
        cl = compute_cl_te(pt, params, bg, [100, 200])
        for i, l in enumerate([100, 200]):
            cl_us = float(cl[i])
            cl_class = float(lcdm_cls_ref['te'][l])
            sign_match = (cl_us * cl_class) > 0
            print(f"C_l^TE(l={l}): clax={cl_us:.4e}, CLASS={cl_class:.4e}, sign_match={sign_match}")
            assert sign_match, f"C_l^TE(l={l}): sign mismatch"

    def test_cl_te_finite(self, pipeline):
        """``C_l^TE`` is finite on the probe grid; expects no NaN or Inf values."""
        params, bg, _, pt = pipeline
        cl = compute_cl_te(pt, params, bg, [100, 200])
        for i, l in enumerate([100, 200]):
            assert np.isfinite(float(cl[i])), f"C_l^TE(l={l}) is not finite"

    def test_cl_te_l100_accuracy(self, pipeline, lcdm_cls_ref):
        """``C_l^TE`` at ``l=100`` matches CLASS; expects <60% relative error."""
        params, bg, _, pt = pipeline
        cl = compute_cl_te(pt, params, bg, [100])
        cl_us = float(cl[0])
        cl_class = float(lcdm_cls_ref['te'][100])
        ratio = cl_us / cl_class
        print(f"C_l^TE(l=100): clax={cl_us:.4e}, CLASS={cl_class:.4e}, ratio={ratio:.4f}")
        assert abs(ratio - 1) < 0.60, f"C_l^TE(l=100): ratio={ratio:.4f}"
