"""Test fixtures for clax test suite.

Provides:
- Session-scoped pipeline fixtures (run expensive solves ONCE)
- Reference data loading from CLASS-generated .npz files
- Default CosmoParams and PrecisionParams
- --fast flag for quick regression checks

Pipeline fixtures (session-scoped, shared across test files):
    pipeline_fast_cl      — CosmoParams() + fast_cl preset
    pipeline_fast_cl_k5   — CosmoParams() + fast_cl + pt_k_max_cl=5.0
    pipeline_fast_cl_k5_mpk — the k5 pipeline, last element the tau0-complete
                              matter solve pt_mpk (smsharma/clax#42)
"""

import json
import os

# Enable 64-bit JAX (required for recombination numerics)
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest
from dataclasses import replace as _dc_replace

from clax import CosmoParams, PrecisionParams
from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve
from clax.perturbations import perturbations_solve

# Path to reference data
REFERENCE_DIR = os.path.join(os.path.dirname(__file__), '..', 'reference_data')


# ---------------------------------------------------------------------------
# Session-scoped pipeline fixtures
# ---------------------------------------------------------------------------
# Each perturbation solve takes 2-5 min on CPU.  By sharing results across
# all test files that use the same precision settings, the full test suite
# runs in ~10 min instead of 60+.

@pytest.fixture(scope="session")
def pipeline_fast_cl():
    """Background + thermo + perturbations with fast_cl preset.

    Used by: test_harmonic, test_high_l, test_lensing, test_cl_pp_implementations.
    Returns (params, prec, bg, th, pt).
    """
    params = CosmoParams()
    prec = _dc_replace(PrecisionParams.fast_cl(), pt_k_chunk_size=20)
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    pt = perturbations_solve(params, prec, bg, th)
    return params, prec, bg, th, pt


@pytest.fixture(scope="session")
def pipeline_fast_cl_k5():
    """Background + thermo + perturbations with fast_cl + k_max=5.

    Extends the k-grid for source-based Limber and Halofit sigma(R).
    Used by: test_cl_pp_source_limber, test_clpp_limber_accuracy,
             test_lensing_nonlinear, test_clpp_halofit_ratio.
    Returns (params, prec, bg, th, pt).
    """
    params = CosmoParams()
    prec = _dc_replace(PrecisionParams.fast_cl(),
                       pt_k_max_cl=5.0, pt_k_chunk_size=20)
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    pt = perturbations_solve(params, prec, bg, th)
    return params, prec, bg, th, pt


@pytest.fixture(scope="session")
def pipeline_fast_cl_k5_mpk(pipeline_fast_cl_k5):
    """The tau0-complete matter solve for the SAME params/prec/bg/th as
    pipeline_fast_cl_k5. Use it for any P(k)/EPT lookup near z=0: the C_l
    solve's grid stops at 0.999*tau0 (smsharma/clax#42).
    Returns (params, prec, bg, th, pt_mpk)."""
    from clax.perturbations import perturbations_solve_mpk
    params, prec, bg, th, _ = pipeline_fast_cl_k5
    return params, prec, bg, th, perturbations_solve_mpk(params, prec, bg, th)


def pytest_addoption(parser):
    parser.addoption(
        "--fast", action="store_true", default=False,
        help="Run the fast subset: subsample grids within tests AND skip "
             "tests marked @pytest.mark.slow"
    )


def pytest_collection_modifyitems(config, items):
    """Make ``--fast`` actually skip ``@pytest.mark.slow`` tests.

    ``--fast`` used to do only half of what its name promises: it fed the
    ``fast_mode`` fixture, which subsamples grids *inside* a test, but nothing
    deselected the tests marked ``slow``. ``pyproject.toml`` declares the marker
    and even documents ``-m "not slow"``, but ``addopts`` never applies it, so
    the command ``CLAUDE.md`` prescribes before every commit --
    ``pytest tests/ --fast -x -q`` -- ran all 21 slow tests and could not
    finish. Measured on bare ``main``: terminated at 3:00:01 by its harness
    timeout, with an identical timeout on a branch, so every "full suite"
    run in that state was silently truncated rather than green.

    Skipping (rather than deselecting) keeps the skipped tests visible as ``s``
    in the summary, so it is obvious that a fast run is not a full run.
    Run without ``--fast`` for the complete suite, or ``-m slow`` for only the
    slow ones.
    """
    if not config.getoption("--fast"):
        return
    skip_slow = pytest.mark.skip(
        reason="slow test skipped by --fast (omit --fast for the full suite)"
    )
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def fast_mode(request):
    return request.config.getoption("--fast")


# ---------------------------------------------------------------------------
# Multi-cosmology test grids (PROJECT RULE, see CLAUDE.md "Test at many
# parameter points": physics-facing value/gradient tests run at 3-5
# cosmologies, not just fiducial).
#
# Motivation (issue #30): the 2.34% h-gradient discrepancy was a SIGNED
# CANCELLATION of two independent errors that happened to nearly cancel at
# fiducial LCDM on one functional. Errors that cancel at one point in
# parameter space do not cancel elsewhere; fiducial-only tests structurally
# cannot see them.
#
# Values MUST stay in lockstep with scripts/generate_multipoint_reference.py
# (same names, same offsets) so matching reference data exists under
# reference_data/<name>/ for value tests. Consistency tests (grad-vs-jvp,
# finiteness, parity) need no reference data and may use any grid point.
# ---------------------------------------------------------------------------
COSMOLOGY_GRID_LCDM = {
    # name -> CosmoParams overrides (empty = fiducial)
    "lcdm_fiducial": {},
    "h_high": {"h": 0.6736 * 1.10},
    "omega_b_high": {"omega_b": 0.02237 * 1.20},
    "omega_cdm_low": {"omega_cdm": 0.1200 * 0.80},
    "ns_high": {"n_s": 0.9649 * 1.05},
}

# For changes touching massive neutrinos / the ncdm sector, the rule uses
# this grid instead (CLAUDE.md's reference suite: 0.06, 0.15, 0.3 eV).
# Only massive_nu_015 has CLASS reference data today; the other masses are
# for consistency tests. Massive-nu solves conventionally set
# ncdm_fluid_approximation="none" (cf. tests/test_multipoint.py PREC).
COSMOLOGY_GRID_NULCDM = {
    "lcdm_fiducial": {},
    "massive_nu_006": {"m_ncdm": 0.06},
    "massive_nu_015": {"m_ncdm": 0.15},
    "massive_nu_030": {"m_ncdm": 0.30},
}


def cosmology_reference_dir(name):
    """Path to reference_data/<name> if CLASS reference data exists, else None."""
    path = os.path.join(REFERENCE_DIR, name)
    return path if os.path.isdir(path) else None


def _make_cosmology_fixture(grid):
    from clax import CosmoParams

    @pytest.fixture(params=list(grid.items()), ids=list(grid.keys()))
    def _fixture(request):
        """(name, CosmoParams) per grid point. Under --fast only fiducial
        runs (the rule's full sweep belongs to full-mode/GPU runs)."""
        name, overrides = request.param
        if request.config.getoption("--fast") and overrides:
            pytest.skip(f"--fast runs fiducial only (skipping {name})")
        return name, CosmoParams(**overrides)

    return _fixture


lcdm_cosmology = _make_cosmology_fixture(COSMOLOGY_GRID_LCDM)
nulcdm_cosmology = _make_cosmology_fixture(COSMOLOGY_GRID_NULCDM)

# Recombination staging (smsharma/clax#48): the 5 LCDM points plus low omega_b,
# helium, N_eff and massive-neutrino extremes. Names match reference_data/<name>/
# thermo_recfast_hesw1.npz (CLASS RECFAST, recfast_Heswitch=1). omega_b_low is the
# one point whose Saha x_H at the hydrogen ODE start (z = 1600) is above 0.995, so
# the ODE starts with the Peebles C = 1 and crosses that switch (thermodynamics.py
# `_recfast_dxH_dlna`, wrap_recfast.c:164).
NEFF_NCDM = (0.71611 / (4.0 / 11.0) ** (1.0 / 3.0)) ** 4  # 1.0132: one ncdm species' early N_eff
COSMOLOGY_GRID_RECOMB = {
    **COSMOLOGY_GRID_LCDM,
    "omega_b_low": {"omega_b": 0.02237 * 0.80},
    "YHe_022": {"Y_He": 0.22},
    "YHe_028": {"Y_He": 0.28},
    "Neff_2": {"N_ur": 2.0 - NEFF_NCDM},
    "Neff_4": {"N_ur": 4.0 - NEFF_NCDM},
    "massive_nu_015": {"m_ncdm": 0.15},
}
recomb_cosmology = _make_cosmology_fixture(COSMOLOGY_GRID_RECOMB)
recomb_downstream_cosmology = _make_cosmology_fixture(
    {k: COSMOLOGY_GRID_RECOMB[k] for k in ("lcdm_fiducial", "omega_b_high", "YHe_028")})

# C_l value gates (smsharma/clax#54): cosmologies with CLASS cls.npz and
# cls_lensed.npz references (scripts/generate_multipoint_reference.py VARIATIONS;
# massive_nu_015 from the m_ncdm grid). Names match reference_data/<name>/.
CL_VALUE_GRID = {
    "lcdm_fiducial": {},
    "omega_b_high": {"omega_b": 0.02237 * 1.20},
    "h_high": {"h": 0.6736 * 1.10},
    "h_low": {"h": 0.6736 * 0.90},
    "massive_nu_015": {"m_ncdm": 0.15},
}
cl_value_cosmology = _make_cosmology_fixture(CL_VALUE_GRID)


@pytest.fixture
def lcdm_bg_ref():
    """Load LCDM fiducial background reference data."""
    path = os.path.join(REFERENCE_DIR, 'lcdm_fiducial', 'background.npz')
    return dict(np.load(path, allow_pickle=True))


@pytest.fixture
def lcdm_scalars():
    """Load LCDM fiducial scalar quantities."""
    path = os.path.join(REFERENCE_DIR, 'lcdm_fiducial', 'scalars.json')
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def lcdm_derived():
    """Load LCDM fiducial derived parameters."""
    path = os.path.join(REFERENCE_DIR, 'lcdm_fiducial', 'derived.json')
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def lcdm_thermo_ref():
    """Load LCDM fiducial thermodynamics reference data."""
    path = os.path.join(REFERENCE_DIR, 'lcdm_fiducial', 'thermodynamics.npz')
    return dict(np.load(path, allow_pickle=True))


@pytest.fixture
def lcdm_cls_ref():
    """Load LCDM fiducial C_l reference data (unlensed)."""
    path = os.path.join(REFERENCE_DIR, 'lcdm_fiducial', 'cls.npz')
    return dict(np.load(path, allow_pickle=True))


@pytest.fixture
def lcdm_cls_lensed_ref():
    """Load LCDM fiducial lensed C_l reference data."""
    path = os.path.join(REFERENCE_DIR, 'lcdm_fiducial', 'cls_lensed.npz')
    return dict(np.load(path, allow_pickle=True))


@pytest.fixture
def lcdm_pk_ref():
    """Load LCDM fiducial P(k) reference data."""
    path = os.path.join(REFERENCE_DIR, 'lcdm_fiducial', 'pk.npz')
    return dict(np.load(path, allow_pickle=True))


def relative_error(computed, reference, eps=1e-30):
    """Compute relative error, avoiding division by zero."""
    return np.abs(computed - reference) / (np.abs(reference) + eps)


def max_relative_error(computed, reference, eps=1e-30):
    """Return (max_rel_err, index_of_max)."""
    rel = relative_error(computed, reference, eps)
    idx = np.argmax(rel)
    return float(rel[idx]), int(idx)


def assert_close(computed, reference, rtol, name="quantity", coordinate=None):
    """Assert computed matches reference within rtol, with clear error message.

    Follows CLAUDE.md principle: concise, actionable error messages.
    """
    max_err, idx = max_relative_error(computed, reference)
    if max_err > rtol:
        coord_str = f" at index {idx}"
        if coordinate is not None:
            coord_str = f" at {coordinate[idx]:.6g}"
        msg = (
            f"{name}: max rel error {max_err:.4%}{coord_str}"
            f" (expected {reference[idx]:.6e}, got {computed[idx]:.6e})"
            f" -- tolerance {rtol:.4%}"
        )
        raise AssertionError(msg)
