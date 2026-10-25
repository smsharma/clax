"""Gradient tests for clax/ept.py: AD vs finite-difference check for P_mm.

Tests that jax.grad(compute_ept w.r.t. pk_lin_h) matches finite differences
when IR resummation is precomputed via _ir_precomputed parameter.

Usage:
    pytest tests/test_ept_gradients.py -v
    pytest tests/test_ept_gradients.py -v --fast   # only 16 k-modes
"""

# Force CPU backend BEFORE importing JAX (Metal does not support float64)
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"

import pytest
import numpy as np

# Configure JAX for float64 precision before any JAX import
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
from scipy.interpolate import CubicSpline

from clax.ept import (
    compute_ept, ept_kgrid, EPTPrecisionParams, pk_mm_real,
    pk_gg_real, pk_mm_l0, pk_gg_l2,
    _ir_resummation_numpy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    """Add --fast flag for quick subsampled tests."""
    try:
        parser.addoption("--fast", action="store_true", default=False,
                         help="Run fast subset of gradient tests (every 16th k-mode)")
    except ValueError:
        pass  # option already added by conftest


@pytest.fixture(scope="module")
def ept_setup():
    """Load fiducial linear pk and precompute IR decomposition once."""
    # Load fiducial linear P(k) in 1/Mpc units from reference data
    pk_data = np.load(
        os.path.join(os.path.dirname(__file__), "..", "reference_data",
                     "lcdm_fiducial", "pk.npz")
    )
    k_Mpc = pk_data["k"]          # 1/Mpc
    pk_lin_Mpc = pk_data["pk_lin_z0"]  # (Mpc)^3 at z=0

    h = 0.6736
    fz = 0.47   # growth rate at z=0 for fiducial LCDM (approximate)

    # Convert to h/Mpc units
    k_h_ref = k_Mpc / h           # h/Mpc
    pk_h_ref = pk_lin_Mpc * h**3  # (Mpc/h)^3

    # Interpolate to EPT grid (log-log)
    prec = EPTPrecisionParams()
    k_ept = ept_kgrid(prec)
    lcs = CubicSpline(np.log(k_h_ref), np.log(pk_h_ref), extrapolate=True)
    pk_lin_ept_np = np.exp(lcs(np.log(k_ept)))

    assert np.all(np.isfinite(pk_lin_ept_np)), "pk_lin_ept has non-finite values"

    pk_lin_ept = jnp.array(pk_lin_ept_np)
    k_ept_jax = jnp.array(k_ept)

    # Precompute IR decomposition (NumPy, outside JAX trace)
    pk_nw_np, pk_w_np, sigma2_bao, delta_sigma2_bao = _ir_resummation_numpy(pk_lin_ept_np, k_ept)
    assert np.all(np.isfinite(pk_nw_np)), "pk_nw_np has non-finite values"
    assert np.all(np.isfinite(pk_w_np)), "pk_w_np has non-finite values"
    assert np.isfinite(sigma2_bao), "sigma2_bao is not finite"

    return {
        "pk_lin_ept": pk_lin_ept,
        "k_ept_jax": k_ept_jax,
        "k_ept_np": k_ept,
        "prec": prec,
        "h": h,
        "fz": fz,
        "ir_precomputed": (pk_nw_np, pk_w_np, sigma2_bao, delta_sigma2_bao),
    }


# ---------------------------------------------------------------------------
# Helper: scalar objective function
# ---------------------------------------------------------------------------

def _make_f(k_ept_jax, h, fz, ir_precomputed, prec, cs0=0.0):
    """Return a scalar function f(pk_lin) = sum(pk_mm_real(...))."""
    def f(pk_lin):
        ept = compute_ept(
            pk_lin, k_ept_jax, h=h, f=fz,
            prec=prec, _ir_precomputed=ir_precomputed,
        )
        return jnp.sum(pk_mm_real(ept, cs0=cs0))
    return f


# ---------------------------------------------------------------------------
# Test 1: AD gradient is computable and finite
# ---------------------------------------------------------------------------

def test_grad_computable(ept_setup):
    """jax.grad of P_mm sum w.r.t. pk_lin must run without error and be finite."""
    setup = ept_setup
    f = _make_f(
        setup["k_ept_jax"], setup["h"], setup["fz"],
        setup["ir_precomputed"], setup["prec"],
    )
    pk_lin = setup["pk_lin_ept"]

    g = jax.grad(f)(pk_lin)

    assert g.shape == pk_lin.shape, f"grad shape mismatch: {g.shape} vs {pk_lin.shape}"
    n_finite = int(jnp.sum(jnp.isfinite(g)))
    n_total = g.size
    finite_frac = n_finite / n_total
    print(f"\nGrad finite: {n_finite}/{n_total} = {finite_frac:.1%}")
    print(f"Max |grad|: {float(jnp.max(jnp.abs(g[jnp.isfinite(g)]))):.3e}")

    assert finite_frac > 0.9, (
        f"Only {finite_frac:.1%} of gradient entries are finite; "
        "expected >90% (boundary k-modes may be NaN due to UV/IR cutoff)"
    )


# ---------------------------------------------------------------------------
# Test 2: AD gradient matches finite differences
# ---------------------------------------------------------------------------

def test_grad_vs_finite_diff(ept_setup, request):
    """AD gradient must match finite-difference gradient to <1% relative error.

    Uses central differences: g_fd[i] = (f(pk+eps*e_i) - f(pk-eps*e_i)) / (2*eps).
    With --fast flag: checks only every 16th k-mode (16 out of 256 points).
    """
    setup = ept_setup
    f = _make_f(
        setup["k_ept_jax"], setup["h"], setup["fz"],
        setup["ir_precomputed"], setup["prec"],
    )
    pk_lin = setup["pk_lin_ept"]
    nk = pk_lin.shape[0]

    # Determine which indices to test
    fast_mode = request.config.getoption("--fast", default=False)
    if fast_mode:
        # Every 16th mode: 16 points spanning the full k range
        indices = np.arange(0, nk, 16)
        print(f"\n--fast mode: testing {len(indices)}/{nk} k-modes")
    else:
        # Every 4th mode for reasonable speed (64 points)
        indices = np.arange(0, nk, 4)
        print(f"\nFull mode: testing {len(indices)}/{nk} k-modes")

    # AD gradient (full)
    g_ad = jax.grad(f)(pk_lin)

    # Finite-difference gradient at selected indices
    eps = 1e-4 * float(jnp.mean(pk_lin))  # ~0.01% of mean pk

    g_fd = np.zeros(len(indices))
    for j, i in enumerate(indices):
        e_i = jnp.zeros(nk).at[i].set(1.0)
        fp = f(pk_lin + eps * e_i)
        fm = f(pk_lin - eps * e_i)
        g_fd[j] = float((fp - fm) / (2.0 * eps))

    g_ad_sel = np.array(g_ad[indices])

    # Relative error: |g_ad - g_fd| / (|g_fd| + small)
    abs_err = np.abs(g_ad_sel - g_fd)
    rel_err = abs_err / (np.abs(g_fd) + 1e-10 * np.max(np.abs(g_fd)))

    # Only count modes where FD gradient is not negligibly small
    significant = np.abs(g_fd) > 1e-12 * np.max(np.abs(g_fd))
    n_significant = significant.sum()

    if n_significant > 0:
        rel_err_sig = rel_err[significant]
        max_rel_err = rel_err_sig.max()
        mean_rel_err = rel_err_sig.mean()
        pass_rate = (rel_err_sig < 0.01).mean()

        print(f"Tested {len(indices)} k-modes, {n_significant} significant")
        print(f"Max relative error: {max_rel_err:.4f} ({max_rel_err*100:.2f}%)")
        print(f"Mean relative error: {mean_rel_err:.4f} ({mean_rel_err*100:.2f}%)")
        print(f"Pass rate (<1% rel err): {pass_rate:.1%}")

        assert pass_rate > 0.90, (
            f"Only {pass_rate:.1%} of k-modes pass <1% AD vs FD relative error "
            f"(max rel err = {max_rel_err:.4f}). "
            "Expected >90% of significant modes to agree."
        )
    else:
        pytest.skip("No significant gradient values found (all near zero)")


# ---------------------------------------------------------------------------
# Test 3: JVP == VJP consistency (forward mode matches reverse mode)
# ---------------------------------------------------------------------------

def test_jvp_equals_vjp(ept_setup):
    """Forward-mode (jvp) result must equal dot(v, g_vjp).

    This checks that there is no custom_vjp bug: for a scalar function f,
    jvp(f, (x,), (v,)) tangent should equal dot(grad(f)(x), v).
    """
    setup = ept_setup
    f = _make_f(
        setup["k_ept_jax"], setup["h"], setup["fz"],
        setup["ir_precomputed"], setup["prec"],
    )
    pk_lin = setup["pk_lin_ept"]

    # Tangent vector: ones (so jvp gives sum of gradient)
    v = jnp.ones_like(pk_lin)

    # Forward mode: jvp tangent = sum(grad)
    _, jvp_tangent = jax.jvp(f, (pk_lin,), (v,))

    # Reverse mode: grad, then dot with v
    g_vjp = jax.grad(f)(pk_lin)
    vjp_dot = jnp.dot(g_vjp, v)

    rel_diff = float(jnp.abs(jvp_tangent - vjp_dot) / (jnp.abs(vjp_dot) + 1e-30))
    print(f"\nJVP tangent: {float(jvp_tangent):.6e}")
    print(f"VJP dot:     {float(vjp_dot):.6e}")
    print(f"Relative diff: {rel_diff:.2e}")

    assert rel_diff < 1e-6, (
        f"JVP ({float(jvp_tangent):.6e}) and VJP dot ({float(vjp_dot):.6e}) "
        f"disagree by {rel_diff:.2e} relative (expected <1e-6 for pure JAX ops)"
    )


# ---------------------------------------------------------------------------
# Test 4: Gradient is nonzero at BAO scales (physical sanity check)
# ---------------------------------------------------------------------------

def test_grad_nonzero_at_bao(ept_setup):
    """Gradient of P_mm sum w.r.t. pk_lin must be nonzero at BAO scales k~0.05-0.15.

    If the IR precomputed path is broken, grad would be zero everywhere.
    BAO scales: k ~ 0.05 to 0.15 h/Mpc.
    """
    setup = ept_setup
    f = _make_f(
        setup["k_ept_jax"], setup["h"], setup["fz"],
        setup["ir_precomputed"], setup["prec"],
    )
    pk_lin = setup["pk_lin_ept"]
    k_ept_np = setup["k_ept_np"]

    g = jax.grad(f)(pk_lin)
    g_np = np.array(g)

    # BAO k range
    bao_mask = (k_ept_np >= 0.05) & (k_ept_np <= 0.15)
    g_bao = g_np[bao_mask]
    g_bao_finite = g_bao[np.isfinite(g_bao)]

    print(f"\nBAO k-modes ({bao_mask.sum()} modes):")
    if len(g_bao_finite) > 0:
        print(f"  |grad| range: [{np.abs(g_bao_finite).min():.3e}, {np.abs(g_bao_finite).max():.3e}]")
        print(f"  Any nonzero: {np.any(g_bao_finite != 0)}")

    assert len(g_bao_finite) > 0, "No finite gradient values in BAO k range"
    assert np.any(np.abs(g_bao_finite) > 0), (
        "Gradient is zero at all BAO scales — IR precomputed path may not be flowing gradients"
    )


# ---------------------------------------------------------------------------
# Test 5: Gradient with cs0 != 0 (counterterm contributes)
# ---------------------------------------------------------------------------

def test_grad_with_counterterm(ept_setup, request):
    """Gradient is nonzero and finite with nonzero EFT counterterm cs0."""
    setup = ept_setup
    cs0 = 10.0  # typical EFT sound speed in (Mpc/h)^2

    f = _make_f(
        setup["k_ept_jax"], setup["h"], setup["fz"],
        setup["ir_precomputed"], setup["prec"], cs0=cs0,
    )
    pk_lin = setup["pk_lin_ept"]

    g = jax.grad(f)(pk_lin)
    g_np = np.array(g)
    g_finite = g_np[np.isfinite(g_np)]

    print(f"\nWith cs0={cs0}:")
    print(f"  Finite entries: {len(g_finite)}/{len(g_np)}")
    if len(g_finite) > 0:
        print(f"  Max |grad|: {np.abs(g_finite).max():.3e}")

    assert len(g_finite) > 0.9 * len(g_np), (
        f"Too many non-finite gradient entries with cs0={cs0}"
    )
    assert np.any(np.abs(g_finite) > 0), "Gradient is all zeros with cs0 != 0"


# ---------------------------------------------------------------------------
# Helper: scalar objective for galaxy spectra
# ---------------------------------------------------------------------------

def _make_f_gg_real(k_ept_jax, h, fz, ir_precomputed, prec):
    """Return f(b1) = sum(pk_gg_real(b1, b2=0, bG2=0, bGamma3=0, Pshot=0))."""
    def f(b1, pk_lin):
        ept_out = compute_ept(
            pk_lin, k_ept_jax, h=h, f=fz,
            prec=prec, _ir_precomputed=ir_precomputed,
        )
        return jnp.sum(pk_gg_real(ept_out, b1, b2=0.0, bG2=0.0, bGamma3=0.0,
                                  cs=0.0, cs0=0.0, Pshot=0.0))
    return f


def _make_f_mm_l0(k_ept_jax, h, fz, ir_precomputed, prec):
    """Return f(pk_lin) = sum(pk_mm_l0(...)) -- tests the RSD path."""
    def f(pk_lin):
        ept_out = compute_ept(
            pk_lin, k_ept_jax, h=h, f=fz,
            prec=prec, _ir_precomputed=ir_precomputed,
        )
        return jnp.sum(pk_mm_l0(ept_out, cs0=0.0))
    return f


def _make_f_gg_l2(k_ept_jax, h, fz, ir_precomputed, prec):
    """Return f(b1, pk_lin) = sum(pk_gg_l2(...)) -- galaxy RSD path."""
    def f(b1, pk_lin):
        ept_out = compute_ept(
            pk_lin, k_ept_jax, h=h, f=fz,
            prec=prec, _ir_precomputed=ir_precomputed,
        )
        return jnp.sum(pk_gg_l2(ept_out, b1, b2=0.0, bG2=0.0, bGamma3=0.0,
                                cs2=0.0, b4=0.0))
    return f


# ---------------------------------------------------------------------------
# Test 6: AD vs FD for d(sum(pk_gg_real))/d(b1)
# ---------------------------------------------------------------------------

def test_grad_pk_gg_real_wrt_b1(ept_setup):
    """AD gradient of pk_gg_real w.r.t. b1 must match finite differences.

    At b1=2.0 (b2=bG2=bGamma3=0, Pshot=0), pk_gg_real = b1^2 (Ptree+Ploop).
    Gradient w.r.t. b1 = 2*b1*(Ptree+Ploop), so well-defined.
    """
    setup = ept_setup
    f = _make_f_gg_real(
        setup["k_ept_jax"], setup["h"], setup["fz"],
        setup["ir_precomputed"], setup["prec"],
    )
    pk_lin = setup["pk_lin_ept"]
    b1_val = 2.0

    # AD gradient w.r.t. b1 (first positional argument)
    g_ad = float(jax.grad(f, argnums=0)(b1_val, pk_lin))

    # Central finite difference
    eps = 1e-4
    fp = float(f(b1_val + eps, pk_lin))
    fm = float(f(b1_val - eps, pk_lin))
    g_fd = (fp - fm) / (2.0 * eps)

    rel_err = abs(g_ad - g_fd) / (abs(g_fd) + 1e-30)
    print(f"\nd(sum(pk_gg_real))/d(b1): AD={g_ad:.6e}, FD={g_fd:.6e}, "
          f"rel_err={rel_err:.4e}")

    assert rel_err < 0.01, (
        f"AD vs FD disagree for d(pk_gg_real)/d(b1): "
        f"AD={g_ad:.6e}, FD={g_fd:.6e}, rel_err={rel_err:.2%}"
    )


# ---------------------------------------------------------------------------
# Test 7: AD vs FD for d(sum(pk_mm_l0))/d(pk_lin) -- RSD path
# ---------------------------------------------------------------------------

def test_grad_pk_mm_l0_wrt_pk_lin(ept_setup, request):
    """AD gradient of pk_mm_l0 w.r.t. pk_lin must match finite differences.

    This tests the RSD multipole path (tree + 1-loop vv/vd/dd decomposition).
    With --fast: checks every 16th k-mode.
    """
    setup = ept_setup
    f = _make_f_mm_l0(
        setup["k_ept_jax"], setup["h"], setup["fz"],
        setup["ir_precomputed"], setup["prec"],
    )
    pk_lin = setup["pk_lin_ept"]
    nk = pk_lin.shape[0]

    fast_mode = request.config.getoption("--fast", default=False)
    if fast_mode:
        indices = np.arange(0, nk, 16)
    else:
        indices = np.arange(0, nk, 4)
    print(f"\npk_mm_l0 gradient: testing {len(indices)}/{nk} k-modes")

    # AD gradient (full)
    g_ad = jax.grad(f)(pk_lin)

    # Finite-difference at selected indices
    eps = 1e-4 * float(jnp.mean(pk_lin))
    g_fd = np.zeros(len(indices))
    for j, i in enumerate(indices):
        e_i = jnp.zeros(nk).at[i].set(1.0)
        fp = f(pk_lin + eps * e_i)
        fm = f(pk_lin - eps * e_i)
        g_fd[j] = float((fp - fm) / (2.0 * eps))

    g_ad_sel = np.array(g_ad[indices])

    # Relative error on significant modes
    abs_err = np.abs(g_ad_sel - g_fd)
    rel_err = abs_err / (np.abs(g_fd) + 1e-10 * np.max(np.abs(g_fd)))
    significant = np.abs(g_fd) > 1e-12 * np.max(np.abs(g_fd))
    n_significant = significant.sum()

    if n_significant == 0:
        pytest.skip("No significant gradient values found")

    rel_err_sig = rel_err[significant]
    max_rel_err = rel_err_sig.max()
    pass_rate = (rel_err_sig < 0.05).mean()

    print(f"  {n_significant} significant modes, max rel err={max_rel_err:.4f}, "
          f"pass rate (<5%)={pass_rate:.1%}")

    assert pass_rate > 0.80, (
        f"pk_mm_l0 gradient: only {pass_rate:.1%} of modes pass <5% rel err "
        f"(max={max_rel_err:.4f}). Expected >80%."
    )


# ---------------------------------------------------------------------------
# Test 8: AD vs FD for d(sum(pk_gg_l2))/d(b1) -- galaxy RSD path
# ---------------------------------------------------------------------------

def test_grad_pk_gg_l2_wrt_b1(ept_setup):
    """AD gradient of pk_gg_l2 w.r.t. b1 must match finite differences.

    Tests the galaxy bias + RSD quadrupole path. Uses central FD at b1=2.0.
    """
    setup = ept_setup
    f = _make_f_gg_l2(
        setup["k_ept_jax"], setup["h"], setup["fz"],
        setup["ir_precomputed"], setup["prec"],
    )
    pk_lin = setup["pk_lin_ept"]
    b1_val = 2.0

    # AD gradient w.r.t. b1 (first argument)
    g_ad = float(jax.grad(f, argnums=0)(b1_val, pk_lin))

    # Central finite difference
    eps = 1e-4
    fp = float(f(b1_val + eps, pk_lin))
    fm = float(f(b1_val - eps, pk_lin))
    g_fd = (fp - fm) / (2.0 * eps)

    rel_err = abs(g_ad - g_fd) / (abs(g_fd) + 1e-30)
    print(f"\nd(sum(pk_gg_l2))/d(b1): AD={g_ad:.6e}, FD={g_fd:.6e}, "
          f"rel_err={rel_err:.4e}")

    assert rel_err < 0.05, (
        f"AD vs FD disagree for d(pk_gg_l2)/d(b1): "
        f"AD={g_ad:.6e}, FD={g_fd:.6e}, rel_err={rel_err:.2%}"
    )


# ---------------------------------------------------------------------------
# End-to-end: CosmoParams -> ... -> EPT (coverage gap)
#
# Tests 1-8 above all differentiate w.r.t. ``pk_lin`` (or ``b1``) directly.
# The tests below start the differentiation at an actual ``CosmoParams``
# field and flow through ``clax.ept.compute_ept_from_clax``, which is the
# public entry point that couples clax's own cosmology objects to EPT (see
# ``clax/lensing.py``'s nonlinear="ept" path and ``clax/ept.py``'s own
# comment: "h is JAX-traced here for d(pk_h)/d(h) flows through h^3 factor").
#
# Test 9/10 hold background+perturbations FIXED and vary only ln10A_s, which
# only rescales the primordial amplitude downstream of the (already-solved)
# perturbation ODE -- cheap once the solve exists, and isolates the
# CosmoParams -> primordial P_R(k) -> compute_ept chain. They use the
# session-scoped ``pipeline_fast_cl_k5_mpk`` fixture (tests/conftest.py): the
# tau0-complete matter solve for the same params/prec/bg/th as
# ``pipeline_fast_cl_k5``. The C_l solve's tau grid stops at 0.999*tau0, so a
# z=0 lookup on it reads z~0.003 (smsharma/clax#42). The fixture costs one
# matter solve per session, shared with test_ept_h_channels.py and
# test_ir_resummation_jax.py.
#
# Test 11 re-solves background -> thermodynamics -> perturbations for every
# probed ``h`` -- the genuinely full CosmoParams-to-EPT chain, including the
# part of the gradient that flows through delta_m(k) itself (not just the
# explicit h^3 unit-conversion factor). This is heavy (3 full perturbation
# solves) so it is marked slow and skipped under --fast.
# ---------------------------------------------------------------------------


def _make_f_from_cosmoparams(bg, pt, param_name):
    """Return f(value) = sum(pk_mm_real(compute_ept_from_clax(...))).

    ``bg``/``pt`` are held fixed (from one pre-solved fiducial cosmology);
    only the ``CosmoParams`` fields that ``compute_ept_from_clax`` re-reads
    at call time (e.g. ``ln10A_s``, ``n_s`` via the primordial spectrum, or
    ``h`` via the explicit h^3 conversion) affect the output through this
    closure.
    """
    from clax import CosmoParams
    from clax.ept import compute_ept_from_clax, pk_mm_real

    base_params = CosmoParams()

    def f(value):
        p = base_params.replace(**{param_name: value})
        ept = compute_ept_from_clax(p, bg, pt, z=0.0)
        return jnp.sum(pk_mm_real(ept))

    return f


def test_grad_ln10A_s_end_to_end_from_cosmoparams_matches_fd(fast_mode, request):
    """d(sum(pk_mm_real))/d(ln10A_s), starting from ``CosmoParams`` (not
    ``pk_lin`` directly) through ``compute_ept_from_clax``, matches central
    finite differences to within a documented, measured, structural bound
    (NOT the project's usual <1% -- see FINDING below).

    Skips under --fast: the session-scoped ``pipeline_fast_cl_k5_mpk``
    fixture (k_max=5.0, ~85 k-modes) is built on first use, and building it
    runs two perturbation solves (the C_l solve ``pipeline_fast_cl_k5`` it is
    derived from, then the matter solve), which would be added to every
    --fast run. Fetched lazily via
    ``request.getfixturevalue`` (after the skip check) rather than as a
    normal fixture parameter, since pytest resolves fixture parameters
    before the test body -- and before the skip -- runs.

    FINDING (CLOSED -- job 13132's 1.39% frozen-pk_nw finding superseded):
    ``clax/ept.py::compute_ept_from_clax`` used to compute the no-wiggle
    (smooth/broadband) component ``pk_nw`` via plain NumPy on a
    ``stop_gradient``-frozen snapshot of ``pk_h``, structurally dropping
    ``d(pk_nw)/d(pk_lin_h)`` from the AD graph. This branch (commit 01b5162:
    traced ``_ir_resummation_jax`` splitter; commit 322a6ab: wired into
    ``compute_ept_from_clax``) replaces that NumPy splitter with a
    differentiable JAX one, so ``pk_nw`` now carries a gradient too and the
    dropped term is closed by construction -- not by a fudge factor.

    Measured THEN (GPU-allocated job 14146 -- actual JAX platform
    unverified: this file's import-time CPU pin -- C_l solve, full validation
    suite on fix/ir-resummation-traced @ 322a6ab): AD=1.322286e+06, FD=1.322286e+06,
    rel_err=1.8231e-07 -- ~76,000x smaller than the pre-closure 1.39%
    (0.0139 / 1.8231e-07 ~= 76,244), and
    consistent with plain central-FD truncation noise (eps=1e-3) rather than
    a remaining structural gap. Re-measured NOW on the tau0-complete matter
    solve (CPU job 21739): AD=1.333357e+06, FD=1.333358e+06,
    rel_err=1.8255e-07 -- the same relative figure to 0.13%. This test's
    frozen-bg/pt setup (``pipeline_fast_cl_k5_mpk`` fixture, no perturbation
    re-solve) also means
    ``pk_mm_real`` -- the real-space, non-RSD matter power spectrum --
    never touches the RSD-basis freeze still deferred elsewhere in
    ``compute_ept_from_clax`` (that freeze only matters for redshift-space
    multipoles), so this particular test path has no other residual channel
    left to show. Contrast the ``h`` end-to-end test below, which re-solves
    the full background/thermodynamics/perturbations pipeline per probed
    ``h``; its job-14146 residual (1.38%) did not collapse with the traced
    splitter, and is of the size of the CPU/GPU spread of the AD gradient
    measured on identical code (see OBSERVED AD SPREAD in that test's
    docstring). The bound below (4e-7) is 2x the job-14146 1.8231e-07
    (=3.6462e-07), rounded up to one significant figure -- never tighter
    than 2x measured, per this branch's ratchet rule, and confirmed green
    on a second independent GPU-allocated run (see CHANGELOG for the confirm
    job ID). 2x the matter-solve 1.8255e-07 (=3.651e-07) rounds up to the
    same 4e-7.

    Headroom note: 4e-7 is a THIN margin in absolute terms -- only ~2.2x
    the measured 1.8231e-07 and ~2.2x the matter-solve 1.8255e-07 (2x
    exactly would be 3.6462e-07 / 3.651e-07; 4e-7 is
    the next value with one significant figure at or above that, per the
    ratchet rule's mechanical rounding, not a deliberately generous
    margin). Accepted by the controller because the job-14146 measurement
    reproduced bit-for-bit identically across two independent GPU-allocated
    runs (jobs 14146 and 14147) -- this residual is FD-truncation noise from a fixed
    eps=1e-3, not floating-point summation-order variance, so it is expected
    to be stable run-to-run rather than a source of flakiness.
    """
    if fast_mode:
        pytest.skip("full k_max=5.0 perturbation solve fixture -- full mode only")
    # z=0 EPT needs the tau0-complete matter solve (smsharma/clax#42).
    params, _prec, bg, _th, pt = request.getfixturevalue("pipeline_fast_cl_k5_mpk")
    f = _make_f_from_cosmoparams(bg, pt, "ln10A_s")
    x0 = float(params.ln10A_s)

    g_ad = float(jax.grad(f)(jnp.asarray(x0)))

    eps = 1e-3
    fp = float(f(x0 + eps))
    fm = float(f(x0 - eps))
    g_fd = (fp - fm) / (2.0 * eps)

    rel_err = abs(g_ad - g_fd) / (abs(g_fd) + 1e-30)
    print(f"\nd(sum(pk_mm_real))/d(ln10A_s) [from CosmoParams]: "
          f"AD={g_ad:.6e}, FD={g_fd:.6e}, rel_err={rel_err:.4e}")

    # 4e-7 = 2x the measured 1.8231e-07 (job 14146), rounded up to one
    # significant figure -- never tighter than 2x measured (2x the matter-solve
    # 1.8255e-07 of CPU job 21739 rounds to the same 4e-7). See FINDING in
    # the docstring: the traced IR-resummation splitter closes the
    # frozen-pk_nw gap that produced the old 1.39% (job 13132); the residual
    # here is FD-truncation noise, not a structural gap.
    assert rel_err < 4e-7, (
        f"AD vs FD disagree for d(sum(pk_mm_real))/d(ln10A_s) starting from "
        f"CosmoParams: AD={g_ad:.6e}, FD={g_fd:.6e}, rel_err={rel_err:.4e} "
        f"(expected <4e-7; see FINDING in this test's docstring -- the "
        f"traced IR-resummation splitter should close the frozen-pk_nw gap "
        f"in this frozen-bg/pt test path)"
    )


def test_jvp_equals_vjp_from_cosmoparams_ln10A_s(fast_mode, request):
    """Forward-mode (jvp) equals reverse-mode (grad) for the
    ``CosmoParams.ln10A_s -> compute_ept_from_clax`` chain.

    No diffrax ODE solve lives inside this closure (``bg``/``pt`` are fixed
    inputs), so ``RecursiveCheckpointAdjoint``'s ``custom_vjp`` boundary is
    not a factor here -- unlike jvp through the perturbation ODE itself
    (see ``tests/test_pk_forward_mode.py``), this jvp needs no
    ``ode_adjoint="direct"`` escape hatch.

    Skips under --fast for the same reason as the AD-vs-FD test above (it
    would build the session-scoped ``pipeline_fast_cl_k5_mpk`` fixture).
    """
    if fast_mode:
        pytest.skip("full k_max=5.0 perturbation solve fixture -- full mode only")
    # z=0 EPT needs the tau0-complete matter solve (smsharma/clax#42).
    params, _prec, bg, _th, pt = request.getfixturevalue("pipeline_fast_cl_k5_mpk")
    f = _make_f_from_cosmoparams(bg, pt, "ln10A_s")
    x0 = float(params.ln10A_s)

    _, jvp_tangent = jax.jvp(f, (jnp.asarray(x0),), (jnp.asarray(1.0),))
    g_vjp = jax.grad(f)(jnp.asarray(x0))

    rel_diff = float(jnp.abs(jvp_tangent - g_vjp) / (jnp.abs(g_vjp) + 1e-30))
    print(f"\nJVP={float(jvp_tangent):.6e} VJP={float(g_vjp):.6e} "
          f"rel_diff={rel_diff:.2e}")

    assert rel_diff < 1e-6, (
        f"JVP ({float(jvp_tangent):.6e}) and VJP ({float(g_vjp):.6e}) "
        f"disagree by {rel_diff:.2e} relative (expected <1e-6 for pure JAX ops)"
    )


# NOTE: this test carried a strict xfail(raises=UnexpectedTracerError) recording a
# real tracer leak in the scalar PID controller: the filtered-norm weights were
# captured in a lambda closure instead of being pytree leaves, so they escaped the
# vmap trace via _solve_k_modes_batched -> lax.map -> vmap. Fixed in this branch by
# _ScalarPidFilteredNorm (clax/perturbations.py); the marker is removed because the
# test now passes with JAX_CHECK_TRACER_LEAKS=1 armed (GPU job 13207).
def test_grad_h_end_to_end_from_cosmoparams_matches_fd(fast_mode):
    """d(sum(pk_mm_real))/dh, fully re-solved (background -> thermodynamics
    -> matter perturbations -> compute_ept_from_clax) for every probed ``h``
    -- the genuine CosmoParams-to-EPT coverage gap this module closes.

    Heavy: 3 full perturbation solves (AD + FD+ + FD-) at the same
    fast_cl(k_max=5.0) precision as the session-scoped
    ``pipeline_fast_cl_k5`` / ``pipeline_fast_cl_k5_mpk`` fixtures, so full
    mode only. The solve is ``perturbations_solve_mpk`` (it reaches
    tau0 exactly), not the C_l solve ``perturbations_solve`` (a z=0 lookup on
    its grid reads z~0.003; smsharma/clax#42). See OBSERVED AD SPREAD below
    for the measured spread of this gradient.

    HISTORY (traced IR-resummation splitter closure; figures in this
    paragraph and the next two are as measured THEN, on the C_l solve, in
    GPU-allocated jobs 14140/14146 -- actual JAX platform unverified: this
    file's import-time CPU pin): the
    k_mpc-channel fix (commit 8bd9cdb, pre-this-branch) traced
    ``k_mpc = k_h * h`` through ``h``, closing the resampling channel that
    job 13313 attributed -9.48e4 of the stage gradient to. This branch
    additionally closes the frozen-pk_nw IR-split gap (commit 01b5162:
    traced ``_ir_resummation_jax`` splitter; commit 322a6ab: wired into
    ``compute_ept_from_clax`` -- same closure documented in the ln10A_s
    FINDING above, which collapsed 1.39% -> 1.8231e-07 for that parameter).

    For ``h``, though, closing that same channel did NOT collapse the
    residual the same way: GPU-allocated job 14146 (full validation suite on
    fix/ir-resummation-traced @ 322a6ab) measured AD=4.046783e6 vs
    FD=3.991575e6, rel_err=1.3831e-02 (1.38%) -- essentially unchanged from,
    and marginally above, the pre-closure job-14140 measurement of 1.1924e-02
    (1.19%) that set the then-current 0.03 bound. At the time this
    non-closure was read as falsifying the "same frozen-pk_nw class as
    ln10A_s" attribution: ln10A_s -- same observable (``pk_mm_real``), same
    commit, bg/pt frozen -- collapsed to 1.8231e-07. OBSERVED AD SPREAD below
    shows a spread of this size between CPU and GPU on identical code, so the
    1.38% alone does not discriminate between channels. The RSD-basis
    freeze does not touch this test: ``compute_ept_from_clax``'s own in-line
    comment states plainly that ``pk_mm_real`` "is unaffected -- it never reads
    those FFTLog bases" (that freeze only matters for redshift-space
    multipoles, which this test's real-space observable never touches).

    At the time, the leading suspect for the surviving 1.38% was the DST
    grid endpoints inside ``_ir_resummation_jax`` -- ``k_min2 = 7e-5/h_conc``,
    ``k_max2 = 7.0/h_conc`` -- and the static ``in_range`` mask built from
    them, all constructed from a concrete ``h_conc = stop_gradient(h)``.
    That function's own in-line comment justifies freezing these endpoints
    on the grounds that "their h-derivative is a boundary term with
    negligible content (P*k weight ~0 at both cuts)". Under AD at a fixed
    ``h``, this snapshot is a constant with zero gradient contribution by
    construction. Under central FD, ``h_conc`` is a different concrete float
    at ``h0+eps`` and ``h0-eps``, so the whole DST grid shifts between the
    FD+ and FD- evaluations. This freeze is in the code, but the 1.38% is
    not evidence for it (see OBSERVED AD SPREAD below); the channel stays a
    candidate only within that spread and is not ranked. Closing it would
    need a clax/ source change.

    OBSERVED AD SPREAD (supersedes the readings above): the exact functional
    of this test (matter solve, frozen export of commit 45be3cd, run by a
    standalone discriminator script, not in the repo) was
    differentiated three ways on CPU and on GPU, on IDENTICAL code (CPU job
    22526 on igpu06, GPU job 22527 on igpu04):

                  FD (eps=1e-3)   reverse grad          forward jvp (direct)
        CPU       4.021530e6      4.031146e6 (+0.24%)   4.149531e6 (+3.18%)
        GPU       4.022818e6      4.075966e6 (+1.32%)   4.022098e6 (-0.018%)

    (relative to the same platform's FD; ``ode_adjoint="direct"`` for the
    forward mode.) FD agrees across the two platforms to 0.03%, and GPU
    forward mode reproduces FD to 1.8e-4, so ~4.0228e6 is the best available
    value of the true derivative. Both AD modes move by 1-3% between CPU and
    GPU on identical code. Reverse mode's spread is of the size
    smsharma/clax#30 reported (~2% on h-like parameters). The functional
    differentiates the WHOLE chain -- background, thermodynamics, the
    perturbation ODE (whose adaptive step choices can differ by platform) and
    the EPT stage -- and the discriminator did NOT localise where in that
    chain the spread arises. The CPU forward-mode +3.18% is unexplained: the
    repo's own #30 note (clax/thermodynamics.py:1010-1016) records forward
    mode through thermodynamics as exact. The bar below bounds this observed
    platform-dependent spread of the reverse-mode gradient. This file pins JAX
    to CPU at import (the ``os.environ`` lines at the top), so CI measures the
    CPU reverse-mode value, +0.24%. A GPU run of an unpinned copy of this test
    itself (job 22431) gave rel_err=1.3211e-02, under the 0.03 bar.

    UPDATE (Oct 11, 2026, smsharma/clax#48): every reading above was taken on
    the unfixed recombination (clipped explicit helium/hydrogen steps whose
    chatter made derivatives depend on step placement). With the Saha -> ODE
    staging fix (ADR 0003) this test on CPU (job 24318) reads AD=4.022097e6,
    FD=4.022357e6, rel_err=6.4640e-05, against +0.24% before; the FD value moves
    by only 0.02% (4.021530e6 -> 4.022357e6). The GPU reverse and the CPU/GPU
    forward readings were not repeated, so this test alone does not show whether
    the +1.32% and +3.18% spreads were that chatter. The bar (0.03) is
    unchanged.

    UPDATE (Oct 25, 2026, smsharma/clax#54): the frozen DST grid described above
    is gone. ``k_min2 = 7e-5/h``, ``k_max2 = 7/h`` and the ``in_range`` mask of
    ``_ir_resummation_jax`` now follow the traced h (static shape), so there is
    no concrete ``h_conc`` and AD differentiates the grid's motion, which central
    FD always saw. EPT P(k) itself moves by <= 1.6e-12 (reordered arithmetic, job
    28536). This test on CPU (job 28628) reads AD=4.022200e6, FD=4.022595e6,
    rel_err=9.8022e-05 (job 24318 above was on the #48 branch, before the later
    changes on main). The stage-level companion
    ``test_stage_grad_h_matches_fd_per_k`` (``tests/test_ept_h_channels.py``)
    drops from a median 1.879e-03 with the frozen grid (main 5ac3df3, CPU job
    30245) to 1.423e-05 (job 28628): the frozen grid was that residual.

    History on the C_l solve (``perturbations_solve``, not the matter solve):
    GPU-allocated job 14146 (actual JAX platform unverified: this file's
    import-time CPU pin) gave rel_err=1.3831e-02 (AD=4.046783e6,
    FD=3.991575e6) and CPU job 21812 on unmodified upstream e894567 gave
    1.3775e-02 (AD=4.066306e6, FD=4.011052e6). The change from that 1.3775e-02
    to the CPU matter-solve 2.3911e-03 (job 21739) lies within the measured
    spread in the table, which says nothing about what caused the gap; an
    earlier attribution of ~83% of it to the C_l solve's tau-grid end is
    withdrawn. (Code facts only: the C_l solve's z~0 lookup clamps at
    0.999*tau0, see smsharma/clax#42; the matter solve reaches tau0
    (``tau_max_factor=1.0``), and its saved values follow tau0 through the
    traced save times.) The frozen-bg/pt per-k companion
    ``test_stage_grad_h_matches_fd_per_k`` in ``tests/test_ept_h_channels.py``
    differentiates only the EPT stage with bg/pt frozen, so neither
    thermodynamics nor the perturbation solve is in its differentiated path:
    1.926e-03 on CPU, 1.924e-03 on GPU (job 22431).

    Bar arithmetic: 0.03 is main's value, restored by user decision; it is
    2x job 14146's 1.3831e-02 (=0.0277) rounded up to one significant figure.
    This test differentiates in reverse mode, whose observed deviations are
    +0.24% (CPU) and +1.32% (GPU). The +3.18% CPU forward-mode value is not
    exercised here.
    """
    if fast_mode:
        pytest.skip("3 full perturbation solves -- full mode only")

    from dataclasses import replace as _dc_replace

    from clax import CosmoParams, PrecisionParams
    from clax.background import background_solve
    from clax.thermodynamics import thermodynamics_solve
    from clax.perturbations import perturbations_solve_mpk
    from clax.ept import compute_ept_from_clax, pk_mm_real

    prec = _dc_replace(PrecisionParams.fast_cl(), pt_k_max_cl=5.0, pt_k_chunk_size=20)
    base_params = CosmoParams()

    def f(h_val):
        p = base_params.replace(h=h_val)
        bg = background_solve(p, prec)
        th = thermodynamics_solve(p, prec, bg)
        pt = perturbations_solve_mpk(p, prec, bg, th)   # reaches today (#42)
        ept = compute_ept_from_clax(p, bg, pt, z=0.0)
        return jnp.sum(pk_mm_real(ept))

    h0 = float(base_params.h)
    g_ad = float(jax.grad(f)(jnp.asarray(h0)))

    eps = 1e-3
    fp = float(f(h0 + eps))
    fm = float(f(h0 - eps))
    g_fd = (fp - fm) / (2.0 * eps)

    rel_err = abs(g_ad - g_fd) / (abs(g_fd) + 1e-30)
    print(f"\nd(sum(pk_mm_real))/dh [full CosmoParams pipeline]: "
          f"AD={g_ad:.6e}, FD={g_fd:.6e}, rel_err={rel_err:.4e}")

    # 0.03 is main's bar, restored by user decision: 2x job 14146's 1.3831e-02
    # (=0.0277), rounded up to one significant figure. It bounds the observed
    # platform-dependent spread of the AD gradient: on identical code the
    # reverse-mode gradient reads +0.24% (CPU, job 22526) and +1.32% (GPU, job
    # 22527) against FD, forward mode +3.18% (CPU) and -0.018% (GPU); FD agrees
    # across platforms to 0.03%. Where in the chain (background,
    # thermodynamics, perturbation ODE, EPT stage) the spread arises is not
    # localised. See OBSERVED AD SPREAD in the docstring. This file pins JAX
    # to CPU, so CI sees the CPU reverse-mode value (+0.24%).
    assert rel_err < 0.03, (
        f"AD vs FD disagree for d(sum(pk_mm_real))/dh (full CosmoParams "
        f"pipeline): AD={g_ad:.6e}, FD={g_fd:.6e}, rel_err={rel_err:.2%} "
        f"(expected <3%; see OBSERVED AD SPREAD in this test's docstring -- "
        f"on identical code the reverse-mode gradient read +0.24% (CPU) and "
        f"+1.32% (GPU) against FD, and forward mode +3.18% (CPU); where in "
        f"the chain that spread arises was not localised)"
    )
