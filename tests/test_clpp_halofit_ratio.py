"""End-to-end test: Halofit C_l^pp NL/linear ratio vs CLASS reference.

Runs the full pipeline (background + thermo + perturbations + Limber C_l^pp)
with nonlinear=True and compares against CLASS v3.3.4 Halofit data.

Reference: reference_data/classpt_clpp_halofit.npz
  Generated with CLASS v3.3.4 (full Limber scheme) and CosmoParams defaults.

Runtime: ~90-120s (perturbation solve at k_max=5.0 with lean hierarchy).
"""
import numpy as np
import pytest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import clax
from clax.perturbations import perturbations_solve
from clax.lensing import compute_cl_pp_limber
from dataclasses import replace as dc_replace


@pytest.fixture(scope="module")
def pipeline_results():
    """Run the perturbation solve once for all tests in this module."""
    params = clax.CosmoParams()  # defaults match reference data exactly
    prec = dc_replace(clax.PrecisionParams(),
        pt_k_max_cl=5.0, pt_k_per_decade=15, pt_tau_n_points=1500,
        pt_l_max_g=10, pt_l_max_pol_g=6, pt_l_max_ur=10,
        pt_ode_rtol=1e-4, pt_ode_atol=1e-7,
        ode_max_steps=16384, pt_ode_solver="rodas5", pt_k_chunk_size=20,
    )
    bg = clax.background_solve(params, prec)
    th = clax.thermodynamics_solve(params, prec, bg)
    pt = perturbations_solve(params, prec, bg, th)
    return params, prec, bg, th, pt


@pytest.fixture(scope="module")
def cl_pp_results(pipeline_results):
    """Compute linear and Halofit C_l^pp via Limber."""
    params, prec, bg, th, pt = pipeline_results
    l_max = 2500

    cl_pp_lin = np.array(compute_cl_pp_limber(
        pt, params, bg, th, l_max=l_max, n_chi=300, nonlinear=False))
    cl_pp_hf = np.array(compute_cl_pp_limber(
        pt, params, bg, th, l_max=l_max, n_chi=300, nonlinear=True))

    return cl_pp_lin, cl_pp_hf


@pytest.fixture(scope="module")
def class_reference():
    return np.load("reference_data/classpt_clpp_halofit.npz")


class TestClppLinear:
    """Linear C_l^pp should match CLASS v3.3.4 to <1% at all l."""

    def test_linear_clpp_all_l(self, cl_pp_results, class_reference):
        """Linear C_l^pp matches CLASS to <1% for l = 100, 500, 1000, 2500."""
        cl_pp_lin, _ = cl_pp_results
        ref = class_reference

        print("\nLinear C_l^pp vs CLASS v3.3.4:")
        print(f"  {'l':>5s}  {'clax':>12s}  {'CLASS':>12s}  {'err':>8s}")
        for l_val in [100, 500, 1000, 2000, 2500]:
            idx = l_val - 2
            rel_err = abs(cl_pp_lin[l_val] - ref['pp_lin'][idx]) / ref['pp_lin'][idx]
            print(f"  {l_val:5d}  {cl_pp_lin[l_val]:12.4e}  {ref['pp_lin'][idx]:12.4e}  {rel_err:8.2%}")
            assert rel_err < 0.01, (
                f"Linear C_l^pp at l={l_val}: {rel_err:.2%} error exceeds 1%")


class TestClppHalofitRatio:
    """Compare C_l^pp Halofit/linear ratio against CLASS reference."""

    def test_ratio_at_low_l(self, cl_pp_results, class_reference):
        """NL/linear ratio matches CLASS within 10% for l <= 500.

        Our Limber chi-integral applies the NL correction to P(k) in the
        integrand, while CLASS applies sqrt(P_NL/P_lin) to the source at
        each (k,tau) before the Limber evaluation. This leads to ~5%
        differences in the NL correction weighting at high l.
        """
        cl_pp_lin, cl_pp_hf = cl_pp_results
        ref = class_reference

        test_ells = [100, 200, 500]
        print("\nNL/linear ratio comparison (l <= 500):")
        print(f"  {'l':>5s}  {'clax':>8s}  {'CLASS':>8s}  {'err':>8s}")
        for l_val in test_ells:
            idx = l_val - 2
            ref_ratio = ref['pp_halofit'][idx] / ref['pp_lin'][idx]
            our_ratio = cl_pp_hf[l_val] / cl_pp_lin[l_val]

            ref_corr = ref_ratio - 1.0
            our_corr = our_ratio - 1.0
            rel_err = abs(our_corr - ref_corr) / abs(ref_corr) if abs(ref_corr) > 0.005 else abs(our_corr - ref_corr)
            print(f"  {l_val:5d}  {our_ratio:8.4f}  {ref_ratio:8.4f}  {rel_err:8.2%}")
            assert rel_err < 0.10, (
                f"l={l_val}: NL correction err={rel_err:.1%} exceeds 10%")

    def test_ratio_at_high_l(self, cl_pp_results, class_reference):
        """NL/linear ratio at high l: document residual from NL weighting.

        At l > 1000, our P(k)-based NL correction gives ~5% less NL
        boost than CLASS's source-based correction. This is not a bug
        in the linear C_l^pp (which matches to <0.1%) but a difference
        in how the Halofit ratio is applied.
        """
        cl_pp_lin, cl_pp_hf = cl_pp_results
        ref = class_reference

        print("\nNL/linear ratio comparison (high l):")
        print(f"  {'l':>5s}  {'clax':>8s}  {'CLASS':>8s}  {'clax/CL':>8s}")
        for l_val in [1000, 1500, 2000, 2500]:
            idx = l_val - 2
            ref_ratio = ref['pp_halofit'][idx] / ref['pp_lin'][idx]
            our_ratio = cl_pp_hf[l_val] / cl_pp_lin[l_val]
            print(f"  {l_val:5d}  {our_ratio:8.4f}  {ref_ratio:8.4f}  {our_ratio/ref_ratio:8.4f}")
            # Our ratio should be within 10% of CLASS at all l
            assert abs(our_ratio / ref_ratio - 1) < 0.10, (
                f"l={l_val}: ratio discrepancy exceeds 10%")

    def test_ratio_monotonic_increase(self, cl_pp_results):
        """PP ratio should increase from l=100 to l~2000."""
        cl_pp_lin, cl_pp_hf = cl_pp_results

        r100 = cl_pp_hf[100] / cl_pp_lin[100]
        r500 = cl_pp_hf[500] / cl_pp_lin[500]
        r1000 = cl_pp_hf[1000] / cl_pp_lin[1000]

        assert r500 > r100, f"ratio should increase: r500={r500:.4f} < r100={r100:.4f}"
        assert r1000 > r500, f"ratio should increase: r1000={r1000:.4f} < r500={r500:.4f}"

    def test_kmax_validation(self, pipeline_results):
        """nonlinear=True with k_max < 5 should raise ValueError."""
        params, prec, bg, th, pt = pipeline_results
        import copy
        pt_narrow = copy.copy(pt)
        mask = pt.k_grid <= 0.35
        object.__setattr__(pt_narrow, 'k_grid', pt.k_grid[mask])
        object.__setattr__(pt_narrow, 'delta_m', pt.delta_m[mask, :])

        with pytest.raises(ValueError, match="pt_k_max_cl >= 5.0"):
            compute_cl_pp_limber(pt_narrow, params, bg, th,
                                 l_max=100, nonlinear=True)
