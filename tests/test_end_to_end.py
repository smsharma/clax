"""Tests public API smoke behavior.

Contract:
- Top-level public entrypoints execute and return structurally sane outputs.

Scope:
- Covers lightweight smoke checks for ``clax.compute()`` and ``clax.compute_pk()``.
- Excludes reference-data accuracy and gradient contracts owned by dedicated files.

Notes:
- These tests are intentionally cheap and do not own physics-accuracy guarantees.
"""

import jax
jax.config.update("jax_enable_x64", True)

from dataclasses import replace as dataclass_replace

import clax
import pytest
from clax import CosmoParams, PrecisionParams
from clax.perturbations import MatterPerturbationResult


# Low-res precision for speed
PREC_FAST = PrecisionParams(
    bg_n_points=100,
    ncdm_bg_n_points=64,
    bg_tol=1e-8,
    th_n_points=1200,
    th_z_max=5e3,
    pt_k_min=1e-3,
    pt_k_max_cl=0.1,
    pt_k_max_pk=0.2,
    pt_k_per_decade=8,
    pt_l_max_g=6,
    pt_l_max_pol_g=6,
    pt_l_max_ur=6,
    pt_l_max_ncdm=6,
    pt_tau_n_points=96,
    pt_ode_rtol=1e-2,
    pt_ode_atol=1e-5,
    ode_max_steps=8192,
)


class TestCompute:
    """Tests ``clax.compute()`` smoke behavior."""

    def test_compute_returns_result(self):
        """``compute()`` returns the expected result object; expects bg and th fields."""
        result = clax.compute(CosmoParams(), PREC_FAST)
        assert hasattr(result, 'bg')
        assert hasattr(result, 'th')
        assert float(result.bg.H0) > 0

    def test_compute_returns_finite_scalars(self):
        """``compute()`` returns finite scalar outputs; expects finite H0 and z_star."""
        result = clax.compute(CosmoParams(), PREC_FAST)
        assert jax.numpy.isfinite(result.bg.H0), "H0: found non-finite value; expected finite output"
        assert jax.numpy.isfinite(result.th.z_star), "z_star: found non-finite value; expected finite output"


class TestComputePk:
    """Tests ``clax.compute_pk()`` smoke behavior."""

    def test_pk_positive(self):
        """``compute_pk()`` returns a positive value; expects ``P(k=0.05) > 0``."""
        pk = clax.compute_pk(CosmoParams(), PREC_FAST, k=0.05)
        assert float(pk) > 0, f"P(k=0.05) = {float(pk)}"

    def test_pk_positive_with_pid_overrides(self):
        """``compute_pk()`` accepts scalar PID gain overrides and stays finite/positive."""
        pk = clax.compute_pk(
            CosmoParams(),
            PREC_FAST,
            k=0.05,
            pt_pid_pcoeff=0.2,
            pt_pid_icoeff=0.7,
            pt_pid_factormax=10.0,
            pt_pid_factormin=0.5,
        )
        assert jax.numpy.isfinite(pk), "P(k=0.05): found non-finite value with PID overrides"
        assert float(pk) > 0, f"P(k=0.05) = {float(pk)}"

    def test_pk_table_returns_positive_grid(self):
        """``compute_pk_table()`` returns a positive reusable ``P(k)`` table."""
        result = clax.compute_pk_table(CosmoParams(), PREC_FAST, k_eval=jax.numpy.geomspace(1e-3, 0.1, 4))
        assert result.k_grid.shape == result.pk_grid.shape
        assert jax.numpy.all(result.pk_grid > 0), f"Non-positive values found in {result.pk_grid}"

    def test_pk_table_uses_reduced_mpk_perturbation_payload(self):
        """Public PK tables should keep only the reduced matter-power perturbation payload."""
        result = clax.compute_pk_table(CosmoParams(), PREC_FAST, k_eval=jax.numpy.geomspace(1e-3, 0.1, 4))
        assert isinstance(result.pt, MatterPerturbationResult)
        assert hasattr(result.pt, "delta_m")
        assert not hasattr(result.pt, "source_T0")

    def test_pk_table_accepts_pid_overrides(self):
        """``compute_pk_table()`` accepts scalar PID gain overrides and returns finite positive values."""
        result = clax.compute_pk_table(
            CosmoParams(),
            PREC_FAST,
            k_eval=jax.numpy.geomspace(1e-3, 0.1, 4),
            pt_pid_pcoeff=0.2,
            pt_pid_icoeff=0.7,
            pt_pid_factormax=10.0,
            pt_pid_factormin=0.5,
        )
        assert jax.numpy.all(jax.numpy.isfinite(result.pk_grid)), "P(k) table: found non-finite values with PID overrides"
        assert jax.numpy.all(result.pk_grid > 0), f"Non-positive values found in {result.pk_grid}"

    def test_pk_table_auto_batch_matches_full_vmap(self):
        """Auto-batched and explicit full-``vmap`` table solves should agree on the returned ``P(k)`` grid."""
        k_eval = jax.numpy.geomspace(1e-3, 0.1, 4)
        auto_result = clax.compute_pk_table(CosmoParams(), dataclass_replace(PREC_FAST, pt_k_chunk_size=0), k_eval=k_eval)
        full_result = clax.compute_pk_table(CosmoParams(), dataclass_replace(PREC_FAST, pt_k_chunk_size=-1), k_eval=k_eval)
        rel_diff = jax.numpy.max(jax.numpy.abs(auto_result.pk_grid / full_result.pk_grid - 1.0))
        assert float(rel_diff) < 1e-3, f"Auto/full-vmap table mismatch {float(rel_diff):.3e}"

    def test_pk_table_multi_k_gradient_smoke(self):
        """One public-table solve should support finite non-zero multi-``k`` reverse-mode gradients."""
        k_eval = jax.numpy.array([1.5e-2, 3.0e-2, 9.0e-2, 1.8e-1])
        weights = jax.numpy.array([0.1, 0.2, 0.3, 0.4])

        def objective(params):
            result = clax.compute_pk_table(params, PREC_FAST, k_eval=k_eval)
            return jax.numpy.sum(weights * result.pk_grid)

        grad_tree = jax.grad(objective)(CosmoParams())
        assert jax.numpy.isfinite(grad_tree.h), f"Expected finite d(sum P)/dh, got {grad_tree.h}"
        assert float(jax.numpy.abs(grad_tree.h)) > 0.0, "Expected non-zero d(sum P)/dh"

    def test_pid_filter_selection_kwargs_are_removed_from_public_api(self):
        """Removed PID filter-selection kwargs should raise ``TypeError`` on public PK entrypoints."""
        with pytest.raises(TypeError):
            clax.compute_pk(CosmoParams(), PREC_FAST, k=0.05, pt_pid_filter_indices=("eta",))
        with pytest.raises(TypeError):
            clax.compute_pk_table(
                CosmoParams(),
                PREC_FAST,
                k_eval=jax.numpy.geomspace(1e-3, 0.1, 4),
                pt_pid_filter_weights_mode="unity",
            )
        with pytest.raises(TypeError):
            clax.compute_pk_interpolator(CosmoParams(), PREC_FAST, pt_pid_filter_indices=("eta",))

    def test_pk_interpolator_scalar_query(self):
        """``compute_pk_interpolator()`` supports scalar re-evaluation on the stored table."""
        result = clax.compute_pk_interpolator(CosmoParams(), PREC_FAST)
        pk = result.pk(0.05)
        assert float(pk) > 0, f"P(k=0.05) = {float(pk)}"

    def test_pk_interpolator_rejects_out_of_range_query(self):
        """Table-backed ``P(k)`` queries should not silently extrapolate beyond the solved k-grid."""
        result = clax.compute_pk_interpolator(CosmoParams(), PREC_FAST)
        with pytest.raises(ValueError, match="solved perturbation grid"):
            result.pk(0.2)

    def test_pk_positive_across_ncdm_fluid_modes(self):
        """``compute_pk()`` stays finite and positive for all CLASS-style ``ncdmfa`` modes."""
        for mode in ("mb", "hu", "class", "none"):
            prec = dataclass_replace(PREC_FAST, ncdm_fluid_approximation=mode)
            pk = clax.compute_pk(CosmoParams(), prec, k=0.05)
            assert jax.numpy.isfinite(pk), f"P(k=0.05) non-finite for mode={mode!r}"
            assert float(pk) > 0, f"P(k=0.05) = {float(pk)} for mode={mode!r}"
