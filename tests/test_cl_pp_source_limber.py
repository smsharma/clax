"""Tests for source-based Limber C_l^pp (CLASS-accurate at all l).

``compute_cl_pp_source_limber`` mirrors CLASS's exact algorithm:
    source(k, tau) = (phi+psi)(k, tau) * W_lcmb(tau)
    T_l(k) = [S*chi]_interp * IPhiFlat / (l+0.5)       (Limber)
    C_l^pp = 4pi * integral d(lnk) * P_R(k) * T_l(k)^2

Target accuracy: <3% vs CLASS for l <= 2500 (linear case).
"""

import os
import pytest
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from dataclasses import replace as _replace


@pytest.fixture(scope="module")
def pipeline():
    """Run pipeline once for all tests."""
    from clax import CosmoParams, PrecisionParams
    from clax.background import background_solve
    from clax.thermodynamics import thermodynamics_solve
    from clax.perturbations import perturbations_solve

    prec = _replace(PrecisionParams.fast_cl(),
                    pt_k_max_cl=5.0,
                    pt_k_chunk_size=20)
    params = CosmoParams()
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    pt = perturbations_solve(params, prec, bg, th)
    return params, bg, th, pt


@pytest.fixture(scope="module")
def class_reference():
    """Generate CLASS linear C_l^pp with matching parameters."""
    try:
        from classy import Class
    except ImportError:
        pytest.skip("CLASS Python wrapper not available")

    cosmo = Class()
    cosmo.set({
        'A_s': 2.1e-9, 'n_s': 0.9649, 'tau_reio': 0.052,
        'omega_b': 0.02237, 'omega_cdm': 0.12, 'h': 0.6736,
        'YHe': 0.2425, 'N_ur': 2.0328, 'N_ncdm': 1, 'm_ncdm': 0.06,
        'output': 'lCl,tCl', 'lensing': 'Yes',
        'l_switch_limber': 9, 'non linear': 'none',
    })
    cosmo.compute()
    pp = cosmo.raw_cl(2500)['pp']
    cosmo.struct_cleanup()
    return pp


class TestSourceLimberExists:
    """Function exists and has the expected signature."""

    def test_import(self):
        """compute_cl_pp_source_limber is importable from clax.lensing."""
        from clax.lensing import compute_cl_pp_source_limber  # noqa: F401

    def test_returns_correct_shape(self, pipeline):
        """Returns array of shape (l_max+1,) with l=0,1 zeroed."""
        from clax.lensing import compute_cl_pp_source_limber
        params, bg, th, pt = pipeline
        cl = compute_cl_pp_source_limber(pt, params, bg, th, l_max=100)
        assert cl.shape == (101,), f"Expected (101,), got {cl.shape}"
        assert float(cl[0]) == 0.0
        assert float(cl[1]) == 0.0
        assert float(cl[2]) > 0.0


class TestSourceLimberAccuracy:
    """Accuracy vs CLASS reference."""

    def test_matches_class_at_low_l(self, pipeline, class_reference):
        """Matches CLASS to <3% for l = 100, 200, 500."""
        from clax.lensing import compute_cl_pp_source_limber
        params, bg, th, pt = pipeline
        cl = np.array(compute_cl_pp_source_limber(
            pt, params, bg, th, l_max=500))
        pp_class = class_reference

        for l_val in [100, 200, 500]:
            ratio = cl[l_val] / pp_class[l_val]
            err = abs(ratio - 1.0)
            print(f"  l={l_val}: source_limber/CLASS = {ratio:.4f} ({err:.1%})")
            assert err < 0.03, (
                f"l={l_val}: {err:.1%} error exceeds 3% "
                f"(ours={cl[l_val]:.4e}, CLASS={pp_class[l_val]:.4e})")

    def test_matches_class_at_high_l(self, pipeline, class_reference):
        """Matches CLASS to <5% for l = 1000, 1500, 2000, 2500."""
        from clax.lensing import compute_cl_pp_source_limber
        params, bg, th, pt = pipeline
        cl = np.array(compute_cl_pp_source_limber(
            pt, params, bg, th, l_max=2500))
        pp_class = class_reference

        for l_val in [1000, 1500, 2000, 2500]:
            ratio = cl[l_val] / pp_class[l_val]
            err = abs(ratio - 1.0)
            print(f"  l={l_val}: source_limber/CLASS = {ratio:.4f} ({err:.1%})")
            assert err < 0.05, (
                f"l={l_val}: {err:.1%} error exceeds 5% "
                f"(ours={cl[l_val]:.4e}, CLASS={pp_class[l_val]:.4e})")

    def test_beats_poisson_limber_at_high_l(self, pipeline, class_reference):
        """Source-based Limber is more accurate than Poisson-based at l > 1000."""
        from clax.lensing import compute_cl_pp_source_limber, compute_cl_pp_limber
        params, bg, th, pt = pipeline

        cl_source = np.array(compute_cl_pp_source_limber(
            pt, params, bg, th, l_max=2500))
        cl_poisson = np.array(compute_cl_pp_limber(
            pt, params, bg, th, l_max=2500, n_chi=500, nonlinear=False))
        pp_class = class_reference

        for l_val in [1500, 2000, 2500]:
            err_source = abs(cl_source[l_val] / pp_class[l_val] - 1.0)
            err_poisson = abs(cl_poisson[l_val] / pp_class[l_val] - 1.0)
            print(f"  l={l_val}: source err={err_source:.1%}, "
                  f"poisson err={err_poisson:.1%}")
            assert err_source < err_poisson, (
                f"l={l_val}: source ({err_source:.1%}) should beat "
                f"poisson ({err_poisson:.1%})")


class TestSourceLimberPositivity:
    """Basic physical sanity checks."""

    def test_positive_for_l_ge_2(self, pipeline):
        """C_l^pp is positive for all l >= 2."""
        from clax.lensing import compute_cl_pp_source_limber
        params, bg, th, pt = pipeline
        cl = np.array(compute_cl_pp_source_limber(
            pt, params, bg, th, l_max=100))
        for l in range(2, 101):
            assert cl[l] > 0, f"C_l^pp(l={l}) = {cl[l]:.4e} is not positive"

    def test_decreasing_with_l(self, pipeline):
        """C_l^pp decreases monotonically for l >= 10."""
        from clax.lensing import compute_cl_pp_source_limber
        params, bg, th, pt = pipeline
        cl = np.array(compute_cl_pp_source_limber(
            pt, params, bg, th, l_max=500))
        for l in range(10, 500):
            assert cl[l] >= cl[l + 1], (
                f"Not decreasing: C_l({l})={cl[l]:.4e} < C_l({l+1})={cl[l+1]:.4e}")
