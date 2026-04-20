"""Tests tensor-mode forward behavior.

Contract:
- Tensor perturbations and tensor ``C_l^BB`` outputs are finite, positive where required, and reference-consistent to the documented tolerance.

Scope:
- Covers tensor source shapes/finiteness and a coarse ``C_l^BB`` comparison.
- Excludes scalar-spectrum and lensing contracts owned elsewhere.

Notes:
- These tests use a reduced-precision tensor preset and therefore enforce coarse order-of-magnitude tolerances.
"""

import os

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from clax import CosmoParams, PrecisionParams
from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve
from clax.perturbations import tensor_perturbations_solve
from clax.harmonic import compute_cl_bb

# Reference data
REFERENCE_DIR = os.path.join(os.path.dirname(__file__), '..', 'reference_data')

# Reduced precision for tensor solve speed
TENSOR_PREC = PrecisionParams(
    pt_l_max_g=10,
    pt_l_max_pol_g=10,
    pt_l_max_ur=10,
    pt_k_max_cl=0.1,
    pt_k_per_decade=10,
    pt_tau_n_points=1000,
    pt_ode_rtol=1e-3,
    pt_ode_atol=1e-6,
    ode_max_steps=32768,
)

TENSOR_PARAMS = CosmoParams(r_t=0.1)


@pytest.fixture(scope="module")
def bg():
    """Compute the tensor background state once for this module."""
    return background_solve(TENSOR_PARAMS, TENSOR_PREC)


@pytest.fixture(scope="module")
def th(bg):
    """Compute the tensor thermodynamics state once for this module."""
    return thermodynamics_solve(TENSOR_PARAMS, TENSOR_PREC, bg)


@pytest.fixture(scope="module")
def tpt(bg, th):
    """Compute the tensor perturbation result once for this module."""
    return tensor_perturbations_solve(TENSOR_PARAMS, TENSOR_PREC, bg, th)


@pytest.fixture(scope="module")
def tensor_ref():
    """Load tensor CLASS reference data once for this module."""
    return dict(np.load(os.path.join(REFERENCE_DIR, 'tensor_r01', 'cls_tensor.npz')))


@pytest.mark.slow
class TestTensorPerturbations:
    """Tests tensor perturbation outputs."""

    def test_source_finite(self, tpt):
        """Tensor source functions are finite; expects no NaN or Inf entries."""
        assert jnp.all(jnp.isfinite(tpt.source_t)), "source_t has NaN/Inf"
        assert jnp.all(jnp.isfinite(tpt.source_p)), "source_p has NaN/Inf"
        print(f"source_t range: [{float(jnp.min(tpt.source_t)):.4e}, {float(jnp.max(tpt.source_t)):.4e}]")
        print(f"source_p range: [{float(jnp.min(tpt.source_p)):.4e}, {float(jnp.max(tpt.source_p)):.4e}]")

    def test_source_shapes(self, tpt):
        """Tensor source shapes match ``(n_k, n_tau)``; expects the documented shapes."""
        n_k = len(tpt.k_grid)
        n_tau = len(tpt.tau_grid)
        assert tpt.source_t.shape == (n_k, n_tau), (
            f"source_t shape {tpt.source_t.shape} != ({n_k}, {n_tau})"
        )
        assert tpt.source_p.shape == (n_k, n_tau), (
            f"source_p shape {tpt.source_p.shape} != ({n_k}, {n_tau})"
        )
        print(f"Tensor grid: {n_k} k-modes x {n_tau} tau points")


@pytest.mark.slow
class TestClBB:
    """Tests tensor ``C_l^BB`` behavior."""

    def test_cl_bb_positive(self, tpt, bg):
        """Tensor ``C_l^BB`` is positive on the probe grid; expects positive values for ``l >= 2``."""
        l_values = jnp.array([2, 10, 50, 100], dtype=jnp.float64)
        cl_bb = compute_cl_bb(tpt, TENSOR_PARAMS, bg, l_values)
        for i, l in enumerate([2, 10, 50, 100]):
            val = float(cl_bb[i])
            print(f"  C_l^BB(l={l}) = {val:.4e}")
            assert val > 0, f"C_l^BB(l={l}) = {val:.4e} is not positive"

    def test_cl_bb_vs_class(self, tpt, bg, tensor_ref):
        """Tensor ``C_l^BB`` matches CLASS coarsely at low and mid ``l``; expects a ratio in [0.05, 20.0] on the probe grid."""
        l_test = [2, 10, 50]
        l_values = jnp.array(l_test, dtype=jnp.float64)
        cl_bb = compute_cl_bb(tpt, TENSOR_PARAMS, bg, l_values)

        ref_ell = tensor_ref['ell']
        ref_bb = tensor_ref['bb']

        print("\n  l    | computed      | CLASS ref     | ratio")
        print("  -----+--------------+--------------+--------")
        for i, l in enumerate(l_test):
            computed = float(cl_bb[i])
            idx = int(np.argmin(np.abs(ref_ell - l)))
            reference = float(ref_bb[idx])
            ratio = computed / reference if reference != 0 else float('inf')
            print(f"  {l:4d} | {computed:13.4e} | {reference:13.4e} | {ratio:.3f}")

            # Very loose tolerance: within factor of 20 for the reduced tensor preset.
            # The l=100 point is intentionally excluded here; with pt_k_max_cl=0.1
            # and l_max=10 the file only owns a low/mid-l coarse-validation contract.
            assert 0.05 < ratio < 20.0, (
                f"C_l^BB(l={l}): ratio={ratio:.3f} outside [0.05, 20.0] "
                f"(computed={computed:.4e}, CLASS={reference:.4e})"
            )
