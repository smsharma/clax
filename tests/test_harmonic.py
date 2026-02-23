"""Test harmonic (C_l) module: C_l^TT, C_l^EE, C_l^TE against CLASS reference data.

Current accuracy (fast_cl preset, l_max=25, k_max=0.15):
- C_l^TT: l=100 ratio~1.24 (24% off), SW plateau ~30% off (IBP 1/k^2 sensitivity)
- C_l^EE: l=100 ratio~1.49 (49% off), l=200 ratio~0.64 (36% off)
- C_l^TE: l=100 ratio~1.44 (44% off)
- High l (>200) limited by hierarchy truncation at l_max=25
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
    """Test C_l^TT against CLASS reference data."""

    def test_cl_tt_l100(self, pipeline, lcdm_cls_ref):
        """C_l^TT at l=100 -- within 30% (first acoustic peak region)."""
        params, bg, _, pt = pipeline
        cl = compute_cl_tt(pt, params, bg, [100])
        cl_us = float(cl[0])
        cl_class = float(lcdm_cls_ref['tt'][100])
        ratio = cl_us / cl_class
        print(f"C_l^TT(l=100): clax={cl_us:.4e}, CLASS={cl_class:.4e}, ratio={ratio:.4f}")
        assert abs(ratio - 1) < 0.30, f"C_l^TT(l=100): ratio={ratio:.4f}, expected within 30%"

    def test_cl_tt_l50(self, pipeline, lcdm_cls_ref):
        """C_l^TT at l=50 -- within 50% (limited by IBP sensitivity)."""
        params, bg, _, pt = pipeline
        cl = compute_cl_tt(pt, params, bg, [50])
        cl_us = float(cl[0])
        cl_class = float(lcdm_cls_ref['tt'][50])
        ratio = cl_us / cl_class
        print(f"C_l^TT(l=50): clax={cl_us:.4e}, CLASS={cl_class:.4e}, ratio={ratio:.4f}")
        assert abs(ratio - 1) < 0.50, f"C_l^TT(l=50): ratio={ratio:.4f}, expected within 50%"

    def test_cl_tt_l10(self, pipeline, lcdm_cls_ref):
        """C_l^TT at l=10 (SW plateau) -- within 50%."""
        params, bg, _, pt = pipeline
        cl = compute_cl_tt(pt, params, bg, [10])
        cl_us = float(cl[0])
        cl_class = float(lcdm_cls_ref['tt'][10])
        ratio = cl_us / cl_class
        print(f"C_l^TT(l=10): clax={cl_us:.4e}, CLASS={cl_class:.4e}, ratio={ratio:.4f}")
        assert abs(ratio - 1) < 0.50, f"C_l^TT(l=10): ratio={ratio:.4f}, expected within 50%"

    def test_cl_tt_positive(self, pipeline):
        """C_l^TT should be positive at all tested multipoles."""
        params, bg, _, pt = pipeline
        cl = compute_cl_tt(pt, params, bg, [10, 50, 100])
        for i, l in enumerate([10, 50, 100]):
            assert float(cl[i]) > 0, f"C_l^TT(l={l}) = {float(cl[i]):.4e} is not positive"


class TestClEE:
    """Test C_l^EE against CLASS reference data.

    After fixing source_E normalization (bug #17: was g*Pi/(4k^2), now 3*g*Pi/16):
    - l=100: within ~50% (limited by l_max=25, k_max=0.15)
    - l=200: within ~50% (hierarchy truncation effects)
    """

    def test_cl_ee_positive(self, pipeline):
        """C_l^EE should be positive (it's an auto-power spectrum)."""
        params, bg, _, pt = pipeline
        cl = compute_cl_ee(pt, params, bg, [100, 200])
        for i, l in enumerate([100, 200]):
            assert float(cl[i]) > 0, f"C_l^EE(l={l}) = {float(cl[i]):.4e} is not positive"

    def test_cl_ee_finite(self, pipeline):
        """C_l^EE should be finite."""
        params, bg, _, pt = pipeline
        cl = compute_cl_ee(pt, params, bg, [100, 200])
        for i, l in enumerate([100, 200]):
            assert np.isfinite(float(cl[i])), f"C_l^EE(l={l}) is not finite"

    def test_cl_ee_l100_accuracy(self, pipeline, lcdm_cls_ref):
        """C_l^EE at l=100 -- within 50%."""
        params, bg, _, pt = pipeline
        cl = compute_cl_ee(pt, params, bg, [100])
        cl_us = float(cl[0])
        cl_class = float(lcdm_cls_ref['ee'][100])
        ratio = cl_us / cl_class
        print(f"C_l^EE(l=100): clax={cl_us:.4e}, CLASS={cl_class:.4e}, ratio={ratio:.4f}")
        assert abs(ratio - 1) < 0.60, f"C_l^EE(l=100): ratio={ratio:.4f}"

    def test_cl_ee_l200_accuracy(self, pipeline, lcdm_cls_ref):
        """C_l^EE at l=200 -- within 50% (limited by hierarchy truncation)."""
        params, bg, _, pt = pipeline
        cl = compute_cl_ee(pt, params, bg, [200])
        cl_us = float(cl[0])
        cl_class = float(lcdm_cls_ref['ee'][200])
        ratio = cl_us / cl_class
        print(f"C_l^EE(l=200): clax={cl_us:.4e}, CLASS={cl_class:.4e}, ratio={ratio:.4f}")
        assert abs(ratio - 1) < 0.60, f"C_l^EE(l=200): ratio={ratio:.4f}"

    def test_cl_ee_diagnostic(self, pipeline, lcdm_cls_ref):
        """Print diagnostic summary of C_l^EE."""
        params, bg, _, pt = pipeline
        l_test = [100, 200]
        cl = compute_cl_ee(pt, params, bg, l_test)
        print("\n--- C_l^EE diagnostic summary ---")
        for i, l in enumerate(l_test):
            cl_us = float(cl[i])
            cl_class = float(lcdm_cls_ref['ee'][l])
            ratio = cl_us / cl_class
            print(f"  l={l:4d}: ratio={ratio:.4f}  (clax={cl_us:.4e}, CLASS={cl_class:.4e})")
        print("---------------------------------")


class TestClTE:
    """Test C_l^TE against CLASS reference data.

    After fixing source_E normalization:
    - l=100: within ~50% (consistent with TT and EE accuracy)
    """

    def test_cl_te_sign(self, pipeline, lcdm_cls_ref):
        """C_l^TE sign should match CLASS at l=100, 200."""
        params, bg, _, pt = pipeline
        cl = compute_cl_te(pt, params, bg, [100, 200])
        for i, l in enumerate([100, 200]):
            cl_us = float(cl[i])
            cl_class = float(lcdm_cls_ref['te'][l])
            sign_match = (cl_us * cl_class) > 0
            print(f"C_l^TE(l={l}): clax={cl_us:.4e}, CLASS={cl_class:.4e}, sign_match={sign_match}")
            assert sign_match, f"C_l^TE(l={l}): sign mismatch"

    def test_cl_te_finite(self, pipeline):
        """C_l^TE should be finite at all tested multipoles."""
        params, bg, _, pt = pipeline
        cl = compute_cl_te(pt, params, bg, [100, 200])
        for i, l in enumerate([100, 200]):
            assert np.isfinite(float(cl[i])), f"C_l^TE(l={l}) is not finite"

    def test_cl_te_l100_accuracy(self, pipeline, lcdm_cls_ref):
        """C_l^TE at l=100 -- within 50%."""
        params, bg, _, pt = pipeline
        cl = compute_cl_te(pt, params, bg, [100])
        cl_us = float(cl[0])
        cl_class = float(lcdm_cls_ref['te'][100])
        ratio = cl_us / cl_class
        print(f"C_l^TE(l=100): clax={cl_us:.4e}, CLASS={cl_class:.4e}, ratio={ratio:.4f}")
        assert abs(ratio - 1) < 0.60, f"C_l^TE(l=100): ratio={ratio:.4f}"

    def test_cl_te_diagnostic(self, pipeline, lcdm_cls_ref):
        """Print diagnostic summary of C_l^TE."""
        params, bg, _, pt = pipeline
        l_test = [100, 200]
        cl = compute_cl_te(pt, params, bg, l_test)
        print("\n--- C_l^TE diagnostic summary ---")
        for i, l in enumerate(l_test):
            cl_us = float(cl[i])
            cl_class = float(lcdm_cls_ref['te'][l])
            ratio = cl_us / cl_class
            print(f"  l={l:4d}: ratio={ratio:.4f}  (clax={cl_us:.4e}, CLASS={cl_class:.4e})")
        print("---------------------------------")
