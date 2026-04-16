"""Tests non-fiducial regression points.

Contract:
- Selected alternate cosmologies continue to reproduce their stored CLASS-derived regressions.

Scope:
- Covers one massive-neutrino point and one ``w0wa`` dark-energy point.
- Excludes fiducial contracts owned by the dedicated module test files.

Notes:
- These are regression tests for alternate cosmologies, not primary ownership tests for any module.
"""

import json
import os

import jax.numpy as jnp
import numpy as np
import pytest

import clax
from clax import CosmoParams, PrecisionParams
from clax.background import background_solve

# Reference data paths
REFERENCE_DIR = os.path.join(os.path.dirname(__file__), '..', 'reference_data')

# Shared precision settings (low-res for speed).
#
# ncdm_fluid_approximation="none" keeps the full Boltzmann hierarchy
# throughout rather than switching to a fluid closure at late times.
# The fluid switch (modes "class", "mb", "hu") has numerical convergence
# issues for massive neutrino cosmologies at mid-range k (~0.05 Mpc^-1)
# with the current Kvaerno5 solver: the tangent at the switch point
# causes the step controller to shrink the step indefinitely.
# Using "none" is slower but numerically robust.
PREC = PrecisionParams(
    bg_n_points=200, ncdm_bg_n_points=100, bg_tol=1e-8,
    th_n_points=5000, th_z_max=5e3,
    pt_l_max_g=17, pt_l_max_pol_g=17, pt_l_max_ur=17,
    pt_k_max_cl=0.3,
    pt_ode_rtol=1e-3, pt_ode_atol=1e-6,
    ode_max_steps=262144,
    ncdm_fluid_approximation="none",
)


def _load_scalars(model_name):
    path = os.path.join(REFERENCE_DIR, model_name, 'scalars.json')
    with open(path) as f:
        return json.load(f)


def _load_pk(model_name):
    path = os.path.join(REFERENCE_DIR, model_name, 'pk.npz')
    return dict(np.load(path, allow_pickle=True))


# -----------------------------------------------------------------------
# Massive neutrino tests (m_ncdm = 0.15 eV)
# -----------------------------------------------------------------------

class TestMassiveNu:
    """Tests the massive-neutrino regression point."""

    @pytest.fixture(scope="class")
    def ref(self):
        return _load_scalars('massive_nu_015')

    @pytest.fixture(scope="class")
    def bg(self):
        params = CosmoParams(m_ncdm=0.15)
        return background_solve(params, PREC)

    def test_H0(self, bg, ref):
        """H0 matches the stored regression value; expects <0.1% relative error."""
        computed = float(bg.H0)
        expected = ref['H0']
        rel_err = abs(computed - expected) / expected
        assert rel_err < 1e-3, (
            f"H0: rel err {rel_err:.4%} (got {computed:.6e}, expected {expected:.6e})"
        )

    def test_conformal_age(self, bg, ref):
        """Conformal age matches the stored regression value; expects <0.1% relative error."""
        computed = float(bg.conformal_age)
        expected = ref['conformal_age']
        rel_err = abs(computed - expected) / expected
        assert rel_err < 1e-3, (
            f"conformal_age: rel err {rel_err:.4%} (got {computed:.2f}, expected {expected:.2f})"
        )

    @pytest.mark.slow
    def test_pk_at_k005(self, ref):
        """``P(k=0.05)`` matches the stored regression value; expects <10% relative error."""
        params = CosmoParams(m_ncdm=0.15)
        pk_computed = float(clax.compute_pk(params, PREC, k=0.05))

        pk_ref_data = _load_pk('massive_nu_015')
        k_ref = pk_ref_data['k']
        pk_ref = pk_ref_data['pk_lin_z0']
        idx = np.argmin(np.abs(k_ref - 0.05))
        pk_expected = pk_ref[idx]

        ratio = pk_computed / pk_expected
        assert abs(ratio - 1) < 0.10, (
            f"P(k=0.05) massive_nu: ratio={ratio:.4f} "
            f"(got {pk_computed:.4e}, expected {pk_expected:.4e})"
        )


# -----------------------------------------------------------------------
# w0-wa dark energy tests (w0=-0.9, wa=0.1)
# -----------------------------------------------------------------------

class TestW0wa:
    """Tests the ``w0wa`` dark-energy regression point."""

    @pytest.fixture(scope="class")
    def ref(self):
        return _load_scalars('w0wa_m09_01')

    @pytest.fixture(scope="class")
    def bg(self):
        params = CosmoParams(w0=-0.9, wa=0.1)
        return background_solve(params, PREC)

    def test_H0(self, bg, ref):
        """H0 matches the stored regression value; expects <0.1% relative error."""
        computed = float(bg.H0)
        expected = ref['H0']
        rel_err = abs(computed - expected) / expected
        assert rel_err < 1e-3, (
            f"H0: rel err {rel_err:.4%} (got {computed:.6e}, expected {expected:.6e})"
        )

    def test_conformal_age(self, bg, ref):
        """Conformal age matches the stored regression value; expects <1% relative error."""
        computed = float(bg.conformal_age)
        expected = ref['conformal_age']
        rel_err = abs(computed - expected) / expected
        assert rel_err < 0.01, (
            f"conformal_age: rel err {rel_err:.4%} (got {computed:.2f}, expected {expected:.2f})"
        )
