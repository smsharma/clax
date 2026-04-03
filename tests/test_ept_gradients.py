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
    pk_nw_np, pk_w_np, sigma2_bao = _ir_resummation_numpy(pk_lin_ept_np, k_ept)
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
        "ir_precomputed": (pk_nw_np, pk_w_np, sigma2_bao),
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
