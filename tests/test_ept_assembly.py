"""Assembly function sanity tests for clax.ept.

No reference data required -- these test algebraic consistency of the
bias combination functions (pk_gg_real, pk_mm_real, etc.).

Tests:
  1. pk_gg_real(b1=1, b2=0, ...) ~ pk_mm_real(cs0) (galaxy reduces to matter)
  2. Counterterm sign: pk_mm_real(cs0=10) < pk_mm_real(cs0=0) (subtracts power)
  3. Shot noise additive: pk_gg_real(..., Pshot=X) - pk_gg_real(..., Pshot=0) ~ X

Usage:
    pytest tests/test_ept_assembly.py -v
    pytest tests/test_ept_assembly.py -v --fast
"""

# Force CPU backend BEFORE importing JAX
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"

import pytest
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp

from clax.ept import (
    compute_ept, EPTPrecisionParams,
    pk_mm_real, pk_gg_real,
)


# ---------------------------------------------------------------------------
# Fixture: compute EPT from fiducial pk_lin
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def assembly_setup():
    """Load pk_lin from CLASS-PT reference, run compute_ept, return components."""
    ref_path = os.path.join(
        os.path.dirname(__file__), "..", "reference_data",
        "classpt_z0.38_fullrange.npz"
    )
    if not os.path.isfile(ref_path):
        pytest.skip(f"Reference data not found: {ref_path}")

    ref = np.load(ref_path, allow_pickle=True)
    k_ept = ref["k_h"]
    pk_lin_ept_np = ref["pk_lin"]
    h = float(ref["h"])
    fz = float(ref["fz"])

    pk_lin_ept = jnp.array(pk_lin_ept_np)
    k_ept_jax = jnp.array(k_ept)

    prec = EPTPrecisionParams()
    ept_out = compute_ept(
        pk_lin_ept, k_ept_jax, h=h, f=fz, prec=prec,
    )

    return {
        "ept_out": ept_out,
        "k_ept": k_ept,
    }


# ---------------------------------------------------------------------------
# Test 1: pk_gg_real(b1=1, b2=0, bG2=0, bGamma3=0, Pshot=0) ~ pk_mm_real
# ---------------------------------------------------------------------------

def test_pk_gg_real_reduces_to_pk_mm(assembly_setup):
    """pk_gg_real with b1=1 and all other biases zero should equal pk_mm_real.

    When b1=1, b2=bG2=bGamma3=0, cs=0, Pshot=0:
      pk_gg_real = 1^2*(Pk_tree+Pk_loop) + 2*(0+cs0)*Pk_ctr = pk_mm_real(cs0)
    """
    ept_out = assembly_setup["ept_out"]
    k_ept = assembly_setup["k_ept"]
    cs0 = 5.0

    p_mm = np.array(pk_mm_real(ept_out, cs0=cs0))
    p_gg = np.array(pk_gg_real(
        ept_out, b1=1.0, b2=0.0, bG2=0.0, bGamma3=0.0,
        cs=0.0, cs0=cs0, Pshot=0.0,
    ))

    # Compare on k < 0.5 h/Mpc where both are well-defined
    mask = k_ept < 0.5
    abs_ref = np.abs(p_mm[mask])
    valid = abs_ref > 1e-3 * abs_ref.max()

    if valid.sum() < 5:
        pytest.skip("Too few valid points for comparison")

    rel_err = np.abs(p_gg[mask][valid] - p_mm[mask][valid]) / abs_ref[valid]
    max_err = float(rel_err.max())
    mean_err = float(rel_err.mean())

    print(f"\npk_gg(b1=1,b2=0,...) vs pk_mm: max_err={max_err:.6e}, "
          f"mean_err={mean_err:.6e}")

    assert max_err < 1e-10, (
        f"pk_gg_real(b1=1, others=0) should exactly equal pk_mm_real, "
        f"but max rel err = {max_err:.2e}"
    )


# ---------------------------------------------------------------------------
# Test 2: Counterterm sign -- cs0 > 0 subtracts power
# ---------------------------------------------------------------------------

def test_counterterm_sign(assembly_setup):
    """pk_mm_real(cs0=10) - pk_mm_real(cs0=0) should be negative.

    The counterterm is 2*cs0*Pk_ctr where Pk_ctr = -k^2 * P_lin.
    Since cs0 > 0 and Pk_ctr < 0, the correction is negative (subtracts power).
    """
    ept_out = assembly_setup["ept_out"]
    k_ept = assembly_setup["k_ept"]

    p_cs0 = np.array(pk_mm_real(ept_out, cs0=0.0))
    p_cs10 = np.array(pk_mm_real(ept_out, cs0=10.0))

    diff = p_cs10 - p_cs0

    # Check on k > 0.01 where the counterterm is non-negligible
    mask = k_ept > 0.01
    diff_masked = diff[mask]

    n_negative = int((diff_masked < 0).sum())
    n_total = len(diff_masked)
    frac_negative = n_negative / n_total

    print(f"\nCounterterm sign: {n_negative}/{n_total} = {frac_negative:.1%} "
          f"of modes have P(cs0=10) < P(cs0=0)")
    print(f"  diff range: [{diff_masked.min():.3e}, {diff_masked.max():.3e}]")

    assert frac_negative > 0.95, (
        f"Expected >95% of modes to have negative counterterm correction, "
        f"but only {frac_negative:.1%}. Counterterm sign may be wrong."
    )


# ---------------------------------------------------------------------------
# Test 3: Shot noise is additive
# ---------------------------------------------------------------------------

def test_pshot_additive(assembly_setup):
    """pk_gg_real(..., Pshot=X) - pk_gg_real(..., Pshot=0) should be X everywhere.

    Shot noise enters as a constant additive term in pk_gg_real.
    """
    ept_out = assembly_setup["ept_out"]
    Pshot_val = 1000.0

    p_no_shot = np.array(pk_gg_real(
        ept_out, b1=2.0, b2=0.5, bG2=-0.1, bGamma3=0.0,
        cs=0.0, cs0=0.0, Pshot=0.0,
    ))
    p_with_shot = np.array(pk_gg_real(
        ept_out, b1=2.0, b2=0.5, bG2=-0.1, bGamma3=0.0,
        cs=0.0, cs0=0.0, Pshot=Pshot_val,
    ))

    diff = p_with_shot - p_no_shot
    abs_err = np.abs(diff - Pshot_val)
    max_abs_err = float(abs_err.max())

    print(f"\nPshot additive: max |diff - {Pshot_val}| = {max_abs_err:.6e}")

    assert max_abs_err < 1e-8, (
        f"Shot noise should be additive, but max deviation = {max_abs_err:.2e}. "
        f"Expected P(Pshot={Pshot_val}) - P(Pshot=0) = {Pshot_val} everywhere."
    )
