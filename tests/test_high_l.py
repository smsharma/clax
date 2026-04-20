"""Tests high-``l`` harmonic helpers and consistency checks.

Contract:
- High-``l`` helper APIs and Limber-related paths behave consistently and return sane outputs.

Scope:
- Covers Limber/exact consistency, ``sparse_l_grid``, and ``compute_cls_all`` API behavior.
- Excludes low/mid-``l`` scalar-spectrum checks owned by ``test_harmonic.py``.

Notes:
- These tests use the ``fast_cl`` preset and therefore enforce consistency, not precision-grade accuracy.
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
from clax.harmonic import compute_cl_tt, sparse_l_grid


PREC = PrecisionParams.fast_cl()


@pytest.fixture(scope="module")
def pipeline():
    """Run the full pipeline once for all tests in this module."""
    params = CosmoParams()
    bg = background_solve(params, PREC)
    th = thermodynamics_solve(params, PREC, bg)
    pt = perturbations_solve(params, PREC, bg, th)
    return params, bg, th, pt


# ---------------------------------------------------------------------------
# TestLimberConsistency
# ---------------------------------------------------------------------------

class TestLimberConsistency:
    """Tests Limber and exact-transfer consistency."""

    def test_limber_vs_exact_l200(self, pipeline):
        """Limber and exact TT paths stay in the same order of magnitude; expects positive finite outputs and ratio within 1e-2 to 1e2."""
        params, bg, _, pt = pipeline

        # Force exact at l=200 (l_switch=1000 means Limber only kicks in at l~900+)
        cl_exact = compute_cl_tt(pt, params, bg, [200], l_switch=1000, delta_l=50)
        cl_exact_val = float(cl_exact[0])

        # Force Limber at l=200 (l_switch=100 means pure Limber by l~200)
        cl_limber = compute_cl_tt(pt, params, bg, [200], l_switch=100, delta_l=50)
        cl_limber_val = float(cl_limber[0])

        # Both should be positive and finite
        assert cl_exact_val > 0, f"Exact C_l(l=200) = {cl_exact_val:.4e} not positive"
        assert cl_limber_val > 0, f"Limber C_l(l=200) = {cl_limber_val:.4e} not positive"
        assert np.isfinite(cl_exact_val), "Exact C_l(l=200) is not finite"
        assert np.isfinite(cl_limber_val), "Limber C_l(l=200) is not finite"

        ratio = cl_limber_val / cl_exact_val
        print(f"Limber vs exact at l=200: exact={cl_exact_val:.4e}, "
              f"limber={cl_limber_val:.4e}, ratio={ratio:.4f}")

        # With fast_cl, Limber at l=200 can be off by a large factor due to
        # coarse k-grid. Just verify they are in the same ballpark (factor 100).
        assert 0.01 < ratio < 100.0, (
            f"Limber/exact ratio at l=200 = {ratio:.4f}; expected between 1e-2 and 1e2"
        )

    def test_limber_positive(self, pipeline):
        """Limber TT values are positive; expects positive outputs on the probe grid."""
        params, bg, _, pt = pipeline

        # l_switch=100 forces Limber at these multipoles
        cl = compute_cl_tt(pt, params, bg, [500, 1000], l_switch=100, delta_l=50)
        for i, l in enumerate([500, 1000]):
            val = float(cl[i])
            print(f"Limber C_l^TT(l={l}) = {val:.4e}")
            assert val > 0, f"Limber C_l^TT(l={l}) = {val:.4e} is not positive"


# ---------------------------------------------------------------------------
# TestHighL
# ---------------------------------------------------------------------------

class TestSparseLGrid:
    """Tests sparse-``l`` grid construction."""

    def test_sparse_l_grid_coverage(self):
        """``sparse_l_grid(2500)`` covers the full range; expects 2 first and 2500 last."""
        l_grid = sparse_l_grid(2500)

        assert l_grid[0] == 2, f"First l = {l_grid[0]}, expected 2"
        assert l_grid[-1] == 2500, f"Last l = {l_grid[-1]}, expected 2500"

        # Should be sorted and unique
        assert np.all(np.diff(l_grid) > 0), "l_grid is not strictly increasing"

    def test_sparse_l_grid_count(self):
        """``sparse_l_grid(2500)`` stays compact; expects between 50 and 200 samples."""
        l_grid = sparse_l_grid(2500)
        n = len(l_grid)
        print(f"sparse_l_grid(2500): {n} values")

        # Should be roughly 100 (allow 50-200 range)
        assert 50 < n < 200, f"sparse_l_grid(2500) has {n} values, expected ~100"


# ---------------------------------------------------------------------------
# TestComputeClsAll
# ---------------------------------------------------------------------------

class TestComputeClsAll:
    """Tests ``compute_cls_all`` helper behavior."""

    def test_cls_all_shape(self, pipeline):
        """``compute_cls_all`` returns the documented keys and shapes; expects length ``l_max + 1`` arrays."""
        from clax.harmonic import compute_cls_all

        params, bg, _, pt = pipeline
        l_max = 500
        result = compute_cls_all(pt, params, bg, l_max=l_max)

        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        for key in ['ell', 'tt', 'ee', 'te']:
            assert key in result, f"Missing key '{key}' in result"
            arr = result[key]
            assert len(arr) == l_max + 1, (
                f"result['{key}'] has length {len(arr)}, expected {l_max + 1}"
            )

    def test_cls_all_matches_individual(self, pipeline):
        """``compute_cls_all`` matches the individual TT path; expects <1% relative difference at ``l=100``."""
        from clax.harmonic import compute_cls_all

        params, bg, _, pt = pipeline

        # compute_cls_all uses pure exact Bessel (no Limber)
        result = compute_cls_all(pt, params, bg, l_max=500)
        cl_all_tt_100 = float(result['tt'][100])

        # compute_cl_tt with l_switch=1000 forces pure exact Bessel at l=100,
        # matching compute_cls_all's behavior
        cl_individual = compute_cl_tt(pt, params, bg, [100], l_switch=1000)
        cl_ind_tt_100 = float(cl_individual[0])

        ratio = cl_all_tt_100 / cl_ind_tt_100
        print(f"compute_cls_all TT(l=100)={cl_all_tt_100:.4e}, "
              f"compute_cl_tt(l=100)={cl_ind_tt_100:.4e}, ratio={ratio:.4f}")

        assert abs(ratio - 1) < 0.01, (
            f"cls_all vs individual at l=100: ratio={ratio:.4f}, expected <1% difference"
        )
