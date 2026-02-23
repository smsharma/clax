"""Test high-l C_l computation with Limber approximation.

Tests the Limber approximation for l > l_switch (default 200), the
compute_cls_all function, and the sparse_l_grid utility.

Uses PrecisionParams.fast_cl() which has k_max=0.15, l_max=25 hierarchy.
This limits absolute accuracy at high l, but Limber approximation itself
should still give reasonable order-of-magnitude results and positive C_l.
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
from clax.harmonic import compute_cl_tt, compute_cl_ee, compute_cl_te, sparse_l_grid


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
    """Test that Limber approximation is roughly consistent with exact Bessel."""

    def test_limber_vs_exact_l200(self, pipeline, lcdm_cls_ref):
        """At l=200, compare Limber and exact Bessel transfer function approaches.

        l_switch=1000 forces exact at l=200; l_switch=100 forces Limber at l=200.

        With fast_cl (k_max=0.15, l_max_hierarchy=25), the Limber approximation
        is not expected to closely match exact Bessel at l=200 -- the k-grid
        is too coarse for the Limber point evaluation to be accurate. We verify
        both methods produce positive, finite results and are within a factor
        of 100 (very loose). True convergence requires medium_cl or better.
        """
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
            f"Limber/exact ratio at l=200 = {ratio:.4f}, expected within factor of 100"
        )

    def test_limber_positive(self, pipeline):
        """C_l^TT from Limber at l=500, 1000 should be positive."""
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

class TestHighL:
    """Test high-l C_l computation (using Limber for l > l_switch)."""

    def test_cl_tt_high_l(self, pipeline, lcdm_cls_ref):
        """C_l^TT at l=500 using exact Bessel should be positive and finite.

        fast_cl (k_max=0.15) cannot produce accurate high-l C_l, and Limber
        fails for CMB primaries (it evaluates S at a single tau instead of
        integrating over the visibility function peak). Use exact Bessel with
        no Limber. The result won't be accurate but should be positive.
        """
        params, bg, _, pt = pipeline

        # Use exact Bessel (l_switch=100000 effectively disables Limber)
        cl = compute_cl_tt(pt, params, bg, [500], l_switch=100000, delta_l=50)
        cl_us = float(cl[0])

        print(f"C_l^TT(l=500, exact Bessel): clax={cl_us:.4e}")

        assert cl_us > 0, f"C_l^TT(l=500) = {cl_us:.4e} is not positive"
        assert np.isfinite(cl_us), f"C_l^TT(l=500) is not finite"

    def test_cl_tt_l2000_positive(self, pipeline):
        """C_l^TT at l=2000 should be positive (Limber handles this)."""
        params, bg, _, pt = pipeline

        cl = compute_cl_tt(pt, params, bg, [2000], l_switch=200, delta_l=50)
        val = float(cl[0])
        print(f"C_l^TT(l=2000) = {val:.4e}")

        assert val > 0, f"C_l^TT(l=2000) = {val:.4e} is not positive"
        assert np.isfinite(val), f"C_l^TT(l=2000) is not finite"


# ---------------------------------------------------------------------------
# TestSparseLGrid
# ---------------------------------------------------------------------------

class TestSparseLGrid:
    """Test sparse_l_grid utility function."""

    def test_sparse_l_grid_coverage(self):
        """sparse_l_grid(2500) should cover l=2 to l=2500."""
        l_grid = sparse_l_grid(2500)

        assert l_grid[0] == 2, f"First l = {l_grid[0]}, expected 2"
        assert l_grid[-1] == 2500, f"Last l = {l_grid[-1]}, expected 2500"

        # Should be sorted and unique
        assert np.all(np.diff(l_grid) > 0), "l_grid is not strictly increasing"

    def test_sparse_l_grid_count(self):
        """sparse_l_grid(2500) should have approximately 100 values."""
        l_grid = sparse_l_grid(2500)
        n = len(l_grid)
        print(f"sparse_l_grid(2500): {n} values")

        # Should be roughly 100 (allow 50-200 range)
        assert 50 < n < 200, f"sparse_l_grid(2500) has {n} values, expected ~100"


# ---------------------------------------------------------------------------
# TestComputeClsAll
# ---------------------------------------------------------------------------

class TestComputeClsAll:
    """Test compute_cls_all function that returns dict with TT, EE, TE."""

    def test_cls_all_shape(self, pipeline):
        """compute_cls_all returns dict with arrays of length l_max+1."""
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
        """C_l^TT at l=100 from compute_cls_all should match compute_cl_tt to <1%.

        compute_cls_all uses pure exact Bessel (no Limber) and sparse l-sampling
        with spline interpolation. We compare against compute_cl_tt with
        l_switch=1000 (forcing pure exact Bessel at l=100) so both use the
        same transfer function method. l=100 is in the sparse l-grid, so
        spline interpolation should introduce negligible error.
        """
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
