"""Tests lensing-layer forward behavior.

Contract:
- Lensing helpers and lensed spectra are finite, positive where required, and consistent with CLASS-derived references.

Scope:
- Covers ``C_l^pp`` positivity plus lensed TT/EE/TE/BB behavior using CLASS unlensed inputs.
- Excludes unlensed scalar-spectrum contracts owned by ``test_harmonic.py``.

Notes:
- The lensed-spectrum checks isolate the lensing algorithm by feeding CLASS unlensed spectra.
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
from clax.perturbations import perturbations_solve
from clax.lensing import compute_cl_pp, lens_cl_tt, lens_cls

REFERENCE_DIR = os.path.join(os.path.dirname(__file__), '..', 'reference_data')

from dataclasses import replace as _dc_replace
PREC = _dc_replace(PrecisionParams.fast_cl(), pt_k_chunk_size=20)


@pytest.fixture(scope="module")
def pipeline():
    """Run the full pipeline once for all tests in this module."""
    params = CosmoParams()
    bg = background_solve(params, PREC)
    th = thermodynamics_solve(params, PREC, bg)
    pt = perturbations_solve(params, PREC, bg, th)
    return params, bg, th, pt


def _load_cls_ref():
    """Load unlensed C_l reference data directly (avoids fixture scope issues)."""
    path = os.path.join(REFERENCE_DIR, 'lcdm_fiducial', 'cls.npz')
    return dict(np.load(path, allow_pickle=True))


class TestCLpp:
    """Tests lensing-potential spectrum behavior."""

    def test_cl_pp_positive(self, pipeline):
        """``C_l^pp`` is positive on the probe grid; expects positive values for ``l >= 2``."""
        params, bg, _, pt = pipeline
        l_values = [2, 10, 50, 100, 200]
        cl_pp = compute_cl_pp(pt, params, bg, l_values)
        for i, l in enumerate(l_values):
            val = float(cl_pp[i])
            print(f"C_l^pp(l={l}) = {val:.4e}")
            assert val > 0, f"C_l^pp(l={l}) = {val:.4e} is not positive"


class TestLensedTT:
    """Tests lensed TT-spectrum behavior."""

    @pytest.fixture(scope="class")
    def lensed_tt(self):
        """Compute lensed TT from CLASS reference unlensed inputs."""
        cls_ref = _load_cls_ref()
        cl_tt_unlensed = jnp.array(cls_ref['tt'])
        cl_pp = jnp.array(cls_ref['pp'])
        l_max = 2500
        cl_lensed = lens_cl_tt(cl_tt_unlensed, cl_pp, l_max=l_max)
        return np.array(cl_lensed)

    def test_lens_cl_tt_shape(self, lensed_tt):
        """Lensed TT has the expected shape; expects ``(2501,)``."""
        assert lensed_tt.shape == (2501,), f"Expected shape (2501,), got {lensed_tt.shape}"

    def test_lens_cl_tt_vs_class(self, lensed_tt, lcdm_cls_lensed_ref):
        """Lensed TT matches CLASS; expects <0.5% relative error on the probe grid."""
        tt_ref = lcdm_cls_lensed_ref['tt']
        test_ells = [10, 50, 100, 200, 500, 1000, 1500, 2000]

        for l in test_ells:
            cl_us = float(lensed_tt[l])
            cl_class = float(tt_ref[l])
            if abs(cl_class) < 1e-30:
                print(f"C_l^TT_lensed(l={l}): CLASS={cl_class:.4e}, skipping (too small)")
                continue
            ratio = cl_us / cl_class
            print(f"C_l^TT_lensed(l={l}): clax={cl_us:.4e}, CLASS={cl_class:.4e}, ratio={ratio:.4f}")
            # Correlation function method should be accurate when given exact inputs
            assert abs(ratio - 1) < 0.005, (
                f"C_l^TT_lensed(l={l}): ratio={ratio:.4f}, expected within 0.5%"
            )

    def test_lens_cl_tt_positive(self, lensed_tt):
        """Lensed TT is positive on the probe grid; expects positive values for ``l >= 2``."""
        for l in [10, 100, 500, 1000, 2000]:
            val = float(lensed_tt[l])
            assert val > 0, f"Lensed C_l^TT(l={l}) = {val:.4e} is not positive"

    def test_lens_cl_tt_lensing_effect(self, lensed_tt, lcdm_cls_lensed_ref):
        """Lensing changes TT relative to the unlensed input; expects a non-zero effect."""
        cls_ref = _load_cls_ref()
        tt_unlensed = cls_ref['tt']
        for l in [100, 500, 1000]:
            cl_lensed = float(lensed_tt[l])
            cl_unlensed = float(tt_unlensed[l])
            diff = abs(cl_lensed - cl_unlensed) / abs(cl_unlensed)
            print(f"l={l}: lensing effect = {diff:.4%}")
            assert diff > 1e-6, f"Lensing had no effect at l={l}"


class TestLensCls:
    """Tests full lensed-spectrum behavior."""

    @pytest.fixture(scope="class")
    def lensed_all(self):
        """Compute all lensed spectra from CLASS reference unlensed inputs."""
        cls_ref = _load_cls_ref()
        cl_tt = jnp.array(cls_ref['tt'])
        cl_ee = jnp.array(cls_ref['ee'])
        cl_te = jnp.array(cls_ref['te'])
        cl_bb = jnp.array(cls_ref['bb'])
        cl_pp = jnp.array(cls_ref['pp'])
        tt, ee, te, bb = lens_cls(cl_tt, cl_ee, cl_te, cl_bb, cl_pp, l_max=2500)
        return {
            'tt': np.array(tt), 'ee': np.array(ee),
            'te': np.array(te), 'bb': np.array(bb),
        }

    def test_lens_cls_shapes(self, lensed_all):
        """All lensed spectra have the expected shape; expects ``(2501,)`` arrays."""
        for key in ['tt', 'ee', 'te', 'bb']:
            assert lensed_all[key].shape == (2501,), \
                f"Lensed {key} shape {lensed_all[key].shape}, expected (2501,)"

    def test_lensed_tt_accuracy(self, lensed_all, lcdm_cls_lensed_ref):
        """Lensed TT matches CLASS; expects <0.2% relative error on the probe grid."""
        tt_ref = lcdm_cls_lensed_ref['tt']
        for l in [10, 100, 500, 1000, 1500]:
            err = abs(lensed_all['tt'][l] - tt_ref[l]) / abs(tt_ref[l])
            print(f"Lensed TT l={l}: err={err:.4%}")
            assert err < 0.002, f"Lensed TT l={l}: err={err:.4%} exceeds 0.2%"

    def test_lensed_ee_accuracy(self, lensed_all, lcdm_cls_lensed_ref):
        """Lensed EE matches CLASS; expects <0.2% relative error on the probe grid."""
        ee_ref = lcdm_cls_lensed_ref['ee']
        for l in [10, 100, 500, 1000, 1500]:
            err = abs(lensed_all['ee'][l] - ee_ref[l]) / abs(ee_ref[l])
            print(f"Lensed EE l={l}: err={err:.4%}")
            assert err < 0.002, f"Lensed EE l={l}: err={err:.4%} exceeds 0.2%"

    def test_lensed_te_accuracy(self, lensed_all, lcdm_cls_lensed_ref):
        """Lensed TE matches CLASS; expects <1% relative error on probe grid.

        Skip l values near TE zero crossings (where |TE|/sqrt(TT*EE) < 0.02)
        since relative error is meaningless near zeros. At l=1500 in LCDM,
        TE ≈ -2e-19 (a zero crossing), so we skip that and probe l=1300 instead.
        """
        import numpy as np
        te_ref = lcdm_cls_lensed_ref['te']
        tt_ref = lcdm_cls_lensed_ref['tt']
        ee_ref = lcdm_cls_lensed_ref['ee']
        for l in [10, 100, 500, 1000, 1300]:
            if abs(te_ref[l]) < 1e-30:
                continue
            # Skip near-zero crossings where relative error is ill-defined
            corr = abs(te_ref[l]) / np.sqrt(tt_ref[l] * ee_ref[l])
            if corr < 0.02:
                continue
            err = abs(lensed_all['te'][l] - te_ref[l]) / abs(te_ref[l])
            print(f"Lensed TE l={l}: err={err:.4%} (|TE|/sqrt(TT*EE)={corr:.3f})")
            assert err < 0.01, f"Lensed TE l={l}: err={err:.2%} exceeds 1%"

    def test_lensed_bb_positive(self, lensed_all):
        """Lensed BB is positive on the probe grid; expects positive values."""
        for l in [10, 100, 500, 1000]:
            val = float(lensed_all['bb'][l])
            print(f"Lensed BB l={l}: {val:.4e}")
            assert val > 0, f"Lensed BB(l={l}) = {val:.4e} not positive"

    def test_lensed_bb_vs_class(self, lensed_all, lcdm_cls_lensed_ref):
        """Lensed BB matches CLASS; expects <5% relative error on the probe grid."""
        bb_ref = lcdm_cls_lensed_ref['bb']
        for l in [50, 100, 200, 500]:
            ratio = lensed_all['bb'][l] / bb_ref[l]
            print(f"Lensed BB l={l}: ratio={ratio:.4f}")
            assert abs(ratio - 1) < 0.05, \
                f"Lensed BB l={l}: ratio={ratio:.4f} exceeds 5% error"
