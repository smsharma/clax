"""Tests linear-matter power-spectrum forward accuracy.

Contract:
- Public table-backed scalar ``P(k, z=0)`` predictions match the CLASS reference
  to ``<=1%`` for ``k <= 1 Mpc^-1``.
- Non-fiducial cosmologies (parameter variations) match CLASS to ``<=3%``.

Scope:
- Covers one cached public ``compute_pk_table()`` solve per precision mode plus
  reuse of the stored perturbation table across redshifts.
- Multi-cosmology parametrized tests exercise omega_b, omega_cdm, massive
  neutrinos, and w0-wa dark energy variations.
- Excludes direct single-mode spot checks and ``P(k)`` gradient contracts owned
  by dedicated files.

Notes:
- These tests compare against CLASS on an explicit log-spaced probe grid and
  rely on the default memory-managed perturbation batching policy.
"""

import functools
import os

import jax
jax.config.update("jax_enable_x64", True)

import clax
import numpy as np
import pytest

from clax import CosmoParams

from tests.pk_test_utils import (
    PK_CONTRACT_PREC,
    PK_FAST_PREC,
    pk_reference_grid,
    pk_reference_values,
)

REFERENCE_DIR = os.path.join(os.path.dirname(__file__), '..', 'reference_data')

# Mapping from reference directory name to CosmoParams overrides.
# Values must match scripts/generate_multipoint_reference.py exactly.
MULTIPOINT_COSMOLOGIES = {
    "omega_b_high": {"omega_b": 0.02237 * 1.20},
    "omega_cdm_high": {"omega_cdm": 0.1200 * 1.20},
    "omega_cdm_low": {"omega_cdm": 0.1200 * 0.80},
    "massive_nu_015": {"m_ncdm": 0.15},
    "w0wa_m09_01": {"w0_fld": -0.9, "wa_fld": 0.1},
}

# Low-resolution precision for multi-cosmology smoke tests (fast).
_MULTIPOINT_PREC = PK_FAST_PREC


@functools.lru_cache(maxsize=2)
def _pk_accuracy_result(fast_mode: bool):
    """Build one cached table solve for the forward-accuracy tests."""
    k_eval = pk_reference_grid(bool(fast_mode))
    prec = PK_FAST_PREC if fast_mode else PK_CONTRACT_PREC
    result = clax.compute_pk_table(CosmoParams(), prec, z=0.0, k_eval=k_eval)
    return k_eval, result


class TestPkAccuracy:
    """Tests public table-backed scalar ``P(k)`` accuracy against CLASS arrays."""

    @pytest.mark.slow
    def test_pk_matches_class(self, lcdm_pk_ref, fast_mode):
        """Interpolated ``P(k, z=0)`` matches CLASS; expects <1% max relative error up to ``1 Mpc^-1``."""
        k_eval, result = _pk_accuracy_result(bool(fast_mode))
        pk_clax = np.asarray(result.pk_grid)
        pk_class = pk_reference_values(lcdm_pk_ref, k_eval, key="pk_m_lin_z0")
        rel_err = np.abs(pk_clax / pk_class - 1.0)
        max_err = float(np.max(rel_err))
        worst_idx = int(np.argmax(rel_err))
        worst_k = float(k_eval[worst_idx])

        assert max_err < 0.01, (
            f"P(k): relative error {max_err:.2%} at k={worst_k:.6g} Mpc^-1; "
            f"clax={pk_clax[worst_idx]:.6e}, CLASS={pk_class[worst_idx]:.6e}, expected <1%"
        )

    @pytest.mark.slow
    def test_pk_matches_class_at_z_half(self, lcdm_pk_ref, fast_mode):
        """The cached perturbation table reproduces ``P(k, z=0.5)`` against CLASS."""
        k_eval, result = _pk_accuracy_result(bool(fast_mode))
        pk_clax = np.asarray(result.pk(k_eval, z=0.5))
        pk_class = pk_reference_values(lcdm_pk_ref, k_eval, key="pk_m_z0.5")
        rel_err = np.abs(pk_clax / pk_class - 1.0)
        max_err = float(np.max(rel_err))
        worst_idx = int(np.argmax(rel_err))
        worst_k = float(k_eval[worst_idx])

        assert max_err < 0.015, (
            f"P(k, z=0.5): relative error {max_err:.2%} at k={worst_k:.6g} Mpc^-1; "
            f"clax={pk_clax[worst_idx]:.6e}, CLASS={pk_class[worst_idx]:.6e}, expected <1.5%"
        )


class TestPkMultiCosmology:
    """P(k) accuracy at non-fiducial cosmologies against stored CLASS reference."""

    @pytest.mark.slow
    @pytest.mark.parametrize("cosmo_key", list(MULTIPOINT_COSMOLOGIES.keys()))
    def test_pk_multi_cosmology(self, cosmo_key, fast_mode):
        """P(k, z=0) at non-fiducial cosmology matches CLASS within 3%."""
        pk_path = os.path.join(REFERENCE_DIR, cosmo_key, "pk.npz")
        if not os.path.exists(pk_path):
            pytest.skip(f"No pk.npz for {cosmo_key}")

        overrides = MULTIPOINT_COSMOLOGIES[cosmo_key]
        params = CosmoParams(**overrides)
        prec = _MULTIPOINT_PREC

        ref = dict(np.load(pk_path, allow_pickle=True))
        k_ref = np.asarray(ref["k"])
        pk_key = "pk_m_lin_z0" if "pk_m_lin_z0" in ref else "pk_lin_z0"
        pk_ref_all = np.asarray(ref[pk_key])

        # Probe at 3 representative k values within the solved range
        k_probe = np.array([0.01, 0.05, 0.2])
        pk_class = np.exp(np.interp(np.log(k_probe), np.log(k_ref), np.log(pk_ref_all)))

        pk_clax = np.array([
            float(clax.compute_pk(params, prec, float(kk))) for kk in k_probe
        ])

        rel_err = np.abs(pk_clax / pk_class - 1.0)
        max_err = float(np.max(rel_err))
        worst_idx = int(np.argmax(rel_err))
        worst_k = float(k_probe[worst_idx])

        assert max_err < 0.03, (
            f"P(k) {cosmo_key}: max relative error {max_err:.2%} at k={worst_k:.4g} Mpc^-1; "
            f"clax={pk_clax[worst_idx]:.4e}, CLASS={pk_class[worst_idx]:.4e}, expected <3%"
        )
