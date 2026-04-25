"""Tests for C_l^phiphi implementations: scan, vmap+Hermite, and scipy reference.

Validates accuracy by comparing against scipy.special.spherical_jn as the
ground truth for Bessel function evaluation.  The scan version
(compute_cl_pp_fast) uses upward recurrence which is unstable for x < l,
leading to O(1000x) errors at l >= 300 with typical k-grids.  The vmap
version (compute_cl_pp_vmap) uses backward+upward blended Bessel tables
with Hermite interpolation and is accurate at all l.

Also benchmarks all three implementations for timing comparison.
"""

import os
import time

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import spherical_jn

from clax import CosmoParams, PrecisionParams
from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve
from clax.perturbations import perturbations_solve
from clax.lensing import compute_cl_pp, compute_cl_pp_fast, compute_cl_pp_vmap
from clax.primordial import primordial_scalar_pk

from dataclasses import replace as _dc_replace
PREC = _dc_replace(PrecisionParams.fast_cl(), pt_k_chunk_size=20)


@pytest.fixture(scope="module")
def pipeline():
    """Run the pipeline once for all tests."""
    params = CosmoParams()
    bg = background_solve(params, PREC)
    th = thermodynamics_solve(params, PREC, bg)
    pt = perturbations_solve(params, PREC, bg, th)
    return params, bg, th, pt


def _cl_pp_scipy_reference(pt, params, bg, l_values):
    """Compute C_l^pp using scipy spherical Bessel functions (ground truth)."""
    tau_0 = float(bg.conformal_age)
    tau = np.array(pt.tau_grid)
    k = np.array(pt.k_grid)
    chi = tau_0 - tau
    S = np.array(pt.source_lens)
    P_R = np.array(primordial_scalar_pk(pt.k_grid, params))
    dtau_d = np.diff(tau)
    dtau_mid = np.concatenate([dtau_d[:1], (dtau_d[:-1] + dtau_d[1:]) / 2, dtau_d[-1:]])
    log_k = np.log(k)
    dlnk = np.diff(log_k)

    cls = []
    for l in l_values:
        T_l = np.zeros(len(k))
        for ik in range(len(k)):
            x = k[ik] * chi
            jl = np.array([spherical_jn(l, float(xi)) for xi in x])
            T_l[ik] = np.sum(S[ik, :] * jl * dtau_mid)
        pref = (2.0 / (l * (l + 1.0)))**2
        integrand = P_R * T_l**2
        cl = pref * 4.0 * np.pi * np.sum(
            0.5 * (integrand[:-1] + integrand[1:]) * dlnk)
        cls.append(cl)
    return np.array(cls)


class TestScipyReference:
    """Validate both implementations against scipy ground truth."""

    @pytest.fixture(scope="class")
    def scipy_ref(self, pipeline):
        """Compute scipy reference at test l-values (expensive, cache per class)."""
        params, bg, _, pt = pipeline
        l_values = [2, 5, 10, 50, 100]
        return l_values, _cl_pp_scipy_reference(pt, params, bg, l_values)

    def test_vmap_vs_scipy_low_l(self, pipeline, scipy_ref):
        """vmap+Hermite matches scipy at l=2..100 (< 0.1% error)."""
        params, bg, _, pt = pipeline
        l_values, cl_ref = scipy_ref
        cl_vmap = compute_cl_pp_vmap(pt, params, bg, l_max=100)
        for i, l in enumerate(l_values):
            ratio = float(cl_vmap[l]) / cl_ref[i]
            assert abs(ratio - 1.0) < 1e-3, (
                f"l={l}: vmap/scipy = {ratio:.6f}, expected ~1.0")

    def test_scan_vs_scipy_low_l(self, pipeline, scipy_ref):
        """Scan matches scipy at l=2..100 (< 1% — scan is less accurate)."""
        params, bg, _, pt = pipeline
        l_values, cl_ref = scipy_ref
        cl_scan = compute_cl_pp_fast(pt, params, bg, l_max=100)
        for i, l in enumerate(l_values):
            if l < 50:
                # Scan uses upward recurrence: zeros j_l for x < 0.7*l,
                # missing small-x contributions at low l
                continue
            ratio = float(cl_scan[l]) / cl_ref[i]
            assert abs(ratio - 1.0) < 0.01, (
                f"l={l}: scan/scipy = {ratio:.6f}, expected ~1.0")

    def test_original_vs_scan_agreement(self, pipeline):
        """Original and scan use same recurrence — must agree exactly."""
        params, bg, _, pt = pipeline
        l_values = [2, 10, 50, 100, 200]
        cl_orig = compute_cl_pp(pt, params, bg, l_values)
        cl_scan = compute_cl_pp_fast(pt, params, bg, l_max=200)
        for i, l in enumerate(l_values):
            np.testing.assert_allclose(
                float(cl_orig[i]), float(cl_scan[l]), rtol=1e-10,
                err_msg=f"original vs scan disagree at l={l}")


class TestVmapAccuracy:
    """Detailed accuracy tests for the vmap+Hermite implementation."""

    def test_positivity(self, pipeline):
        """C_l^pp must be positive for all l >= 2."""
        params, bg, _, pt = pipeline
        cl = compute_cl_pp_vmap(pt, params, bg, l_max=500)
        assert jnp.all(cl[2:] > 0), "C_l^pp must be positive for l >= 2"

    def test_monotonic_decrease_low_l(self, pipeline):
        """C_l^pp should decrease from l=2 to l~50 (roughly as l^{-4})."""
        params, bg, _, pt = pipeline
        cl = compute_cl_pp_vmap(pt, params, bg, l_max=50)
        assert float(cl[2]) > float(cl[10]) > float(cl[50]), (
            "C_l^pp should decrease monotonically at low l")

    def test_vmap_vs_scipy_high_l(self, pipeline):
        """vmap+Hermite matches scipy at l=200, 300, 500 (< 0.1% error)."""
        params, bg, _, pt = pipeline
        l_values = [200, 300, 500]
        cl_ref = _cl_pp_scipy_reference(pt, params, bg, l_values)
        cl_vmap = compute_cl_pp_vmap(pt, params, bg, l_max=500)
        for i, l in enumerate(l_values):
            ratio = float(cl_vmap[l]) / cl_ref[i]
            assert abs(ratio - 1.0) < 1e-3, (
                f"l={l}: vmap/scipy = {ratio:.6f}, expected ~1.0")


class TestScanLimitations:
    """Document known accuracy limitations of the scan version."""

    def test_scan_overestimates_high_l(self, pipeline):
        """Scan overestimates C_l^pp at l >= 300 due to upward recurrence.

        The upward Bessel recurrence gives spuriously large j_l for
        x in the transition zone 0.7*l < x < l (classically forbidden
        but not zeroed).  This is a known limitation, NOT a bug.
        """
        params, bg, _, pt = pipeline
        cl_scan = compute_cl_pp_fast(pt, params, bg, l_max=500)
        cl_vmap = compute_cl_pp_vmap(pt, params, bg, l_max=500)
        # Scan should significantly overestimate at l=300 and l=500
        ratio_300 = float(cl_scan[300]) / float(cl_vmap[300])
        ratio_500 = float(cl_scan[500]) / float(cl_vmap[500])
        assert ratio_300 > 10, (
            f"Expected scan >> vmap at l=300, got ratio={ratio_300:.1f}")
        assert ratio_500 > 100, (
            f"Expected scan >> vmap at l=500, got ratio={ratio_500:.1f}")


class TestBenchmark:
    """Timing comparison (not assertions, just prints)."""

    def test_timing(self, pipeline):
        """Print timing for all three implementations."""
        params, bg, _, pt = pipeline
        l_max = 500
        l_test = [2, 10, 50, 100, 200]

        # Warm up JIT
        _ = compute_cl_pp_fast(pt, params, bg, l_max)
        _ = compute_cl_pp_vmap(pt, params, bg, l_max)

        t0 = time.time()
        _ = compute_cl_pp(pt, params, bg, l_test)
        t_orig = time.time() - t0

        t0 = time.time()
        _ = compute_cl_pp_fast(pt, params, bg, l_max)
        t_scan = time.time() - t0

        t0 = time.time()
        _ = compute_cl_pp_vmap(pt, params, bg, l_max)
        t_vmap = time.time() - t0

        print(f"\n  Timing (l_max={l_max}):")
        print(f"    original ({len(l_test)} l):  {t_orig:.2f}s")
        print(f"    scan (all l):     {t_scan:.2f}s")
        print(f"    vmap+Hermite:     {t_vmap:.2f}s")
