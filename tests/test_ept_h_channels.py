"""Channel tests for the h-dependence of compute_ept_from_clax.

GPU job 13313 attributed the stage-level AD-vs-FD h-gradient gap to the
frozen k_mpc = k_h * stop_gradient(h) resampling channel (-9.48e4 of the
stage gradient), with the frozen-pk_nw IR split (+3.27e4) as the
documented residual and the rs_h/f/h-arg channels negligible (-1.0e2).
These tests pin the fix.

CLOSURE UPDATE (job 14146, fix/ir-resummation-traced @ 322a6ab): the
frozen-pk_nw IR split is now closed too (commit 01b5162 traced
``_ir_resummation_jax``; commit 322a6ab wired it into
``compute_ept_from_clax``), on top of the pre-existing k_mpc fix. See
``test_stage_grad_h_matches_fd_per_k`` below for the measured effect.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from clax import CosmoParams
from clax.ept import compute_ept_from_clax, pk_mm_real


@pytest.fixture(scope="module")
def stage_setup(request):
    # z=0 EPT needs the tau0-complete matter solve (smsharma/clax#42).
    params, _prec, bg, _th, pt = request.getfixturevalue("pipeline_fast_cl_k5_mpk")
    return params, bg, pt


def _pk_of_h(bg, pt):
    base = CosmoParams()

    def f(h_val):
        p = base.replace(h=h_val)
        return pk_mm_real(compute_ept_from_clax(p, bg, pt, z=0.0))

    return f


def test_stage_grad_h_matches_fd_per_k(fast_mode, request):
    """Per-k d(pk_mm_real)/dh through the EPT stage: AD vs central FD.

    RED before the k_mpc channel is traced: FD carries the resampling
    term dP/dlnk * (1/h) which AD drops entirely, giving order-unity
    per-k relative errors at BAO scales. GREEN after the k_mpc fix
    (job 14140): residual was the frozen-pk_nw share, median 3.294e-02.

    FURTHER CLOSED (GPU-allocated job 14146, actual JAX platform unverified:
    test_ept_gradients.py's import-time CPU pin; C_l solve,
    fix/ir-resummation-traced @ 322a6ab; figures here are as measured THEN): with
    the frozen-pk_nw IR split also traced through JAX now (commit 01b5162,
    wired in by 322a6ab), this frozen-bg/pt stage test (no full pipeline
    re-solve, so no discretization noise -- unlike the end-to-end h test in
    ``tests/test_ept_gradients.py``) measured median rel 9.825e-03
    (max 3.619e-02) over 31 modes in [0.05,0.3] -- ~3.35x lower than the
    prior 3.294e-02.

    CURRENT (frozen bg/pt): this branch (tau0-complete matter solve via
    ``pipeline_fast_cl_k5_mpk``, job 21738) measures median 1.926e-03 (max
    1.526e-02); UNMODIFIED upstream e894567 (C_l solve via
    ``pipeline_fast_cl_k5``, job 21814) measures median 1.930e-03 (max
    1.520e-02); a GPU run of an unpinned copy of this test (job 22431) gives
    median 1.924e-03 on the matter-solve fixture. So this test is UNAFFECTED
    by the tau-grid-end change (smsharma/clax#42): it freezes bg/pt, so the
    tau grid end never moves with h. It differentiates only the EPT stage
    (``jacfwd``, bg/pt frozen), so neither ``thermodynamics_solve`` nor the
    perturbation solve is in its differentiated path; it does not show the
    CPU/GPU AD spread observed for the end-to-end h test (see OBSERVED AD
    SPREAD in ``tests/test_ept_gradients.py``): its own CPU and GPU values
    agree to 0.1%. Its documented 9.825e-03 was already out of date on
    upstream: the upstream figure above differs from GPU-allocated job
    14146's 9.825e-03 (322a6ab; actual JAX platform unverified). This branch
    measures 1.926e-03 on CPU and 1.924e-03 on GPU, so the platform does not
    explain the difference; the cause was not investigated.

    The surviving median is NOT a further-openable share of the now-closed
    frozen-pk_nw channel (that channel's own d(pk_nw)/dh contribution flows
    exactly through the traced splitter). Its cause is not isolated. The DST
    grid endpoints/``in_range`` mask inside ``_ir_resummation_jax`` (built
    from a concrete ``h_conc = stop_gradient(h)`` that moves under central FD
    but is pinned under AD; see the HISTORY section of the docstring of
    ``test_grad_h_end_to_end_from_cosmoparams_matches_fd`` in
    ``tests/test_ept_gradients.py`` for the mechanism) remain a candidate, as
    for that test's residual, but nothing measured so far singles it out.

    UPDATE (Oct 25, 2026, smsharma/clax#54): that candidate was the residual.
    The DST grid endpoints and the ``in_range`` mask now follow the traced h
    (static shape; no concrete ``h_conc``), so AD differentiates the grid's
    motion as central FD sees it. On CPU this test reads median 1.879e-03 (max
    1.324e-02) on main 5ac3df3 with the frozen grid (job 30245; #48's
    recombination change is already in that tree) and median 1.423e-05 (max
    2.559e-04) with the traced grid (job 28628). The 0.004 bar below is left as
    it was; the paragraphs above and below describe the frozen grid.

    Ratchet arithmetic: 2x the largest of the frozen-grid measurements
    (1.930e-03 CPU upstream; 1.926e-03 CPU and 1.924e-03 GPU on this
    branch) is 3.86e-03, rounded up to one significant figure is 4e-3. The
    bound is tightened from 0.02 (2x the stale 9.825e-03) to 0.004. The GPU
    figure (job 22431) confirms the 0.004 bar on a second platform."""
    if fast_mode:
        pytest.skip("uses the shared full-mode pipeline fixture")
    params, bg, pt = request.getfixturevalue("stage_setup")
    f = _pk_of_h(bg, pt)
    h0 = float(params.h)

    g_ad = jax.jacfwd(f)(jnp.asarray(h0))  # fwd == rev for this stage; cheap
    eps = 1e-3
    g_fd = (f(h0 + eps) - f(h0 - eps)) / (2.0 * eps)

    k_h = np.asarray(compute_ept_from_clax(params, bg, pt, z=0.0).kh)
    sel = (k_h > 0.05) & (k_h < 0.3)
    rel = np.abs(np.asarray(g_ad - g_fd))[sel] / (
        np.abs(np.asarray(g_fd))[sel] + 1e-30)
    med = float(np.median(rel))
    print(f"\nper-k d(pk_mm)/dh AD-vs-FD: median rel {med:.3e} "
          f"(max {float(rel.max()):.3e}) over {int(sel.sum())} modes in [0.05,0.3]")
    # 0.004 = 2x the largest current measurement (1.930e-03, upstream e894567
    # C_l solve, CPU job 21814; this branch's matter solve gave 1.926e-03 on
    # CPU, job 21738, and 1.924e-03 on GPU, job 22431) = 3.86e-03, rounded up
    # to one significant figure -- never
    # tighter than 2x measured. Tightened from 0.02, which was 2x the 9.825e-03
    # of job 14146; upstream e894567 already measures 1.930e-03 (see docstring).
    # The traced IR-resummation splitter closed most of the frozen-pk_nw
    # share on top of the pre-existing k_mpc fix; the surviving median is
    # not attributed to a specific channel.
    assert med < 0.004, (
        f"median per-k AD-vs-FD rel err {med:.3e} >= 0.004: either the "
        f"k_mpc resampling channel (job 13313: -9.48e4) or the frozen-pk_nw "
        f"IR split (closed by commit 01b5162/322a6ab) has regressed, or the "
        f"unattributed residual has grown (candidate: the DST-grid-endpoint/"
        f"in_range channel, see the HISTORY section in "
        f"tests/test_ept_gradients.py's h end-to-end test docstring; "
        f"measured 1.926e-03 (CPU, job 21738) and 1.924e-03 (GPU, job "
        f"22431) on the matter solve, 1.930e-03 on upstream e894567, job "
        f"21814)")


def test_growth_rate_is_not_hardcoded(request, fast_mode):
    """compute_ept_from_clax must use the background growth rate, not 0.8.

    bg.Omega_m_of_z does not exist, so the hasattr fallback at
    ept.py:2061 silently yields the literal 0.8 for every cosmology and
    redshift. LCDM at z=0 has f ~ 0.52-0.53. RED until Task 4 routes
    f through bg.f_of_loga."""
    if fast_mode:
        pytest.skip("uses the shared full-mode pipeline fixture")
    params, bg, pt = request.getfixturevalue("stage_setup")
    ept = compute_ept_from_clax(params, bg, pt, z=0.0)
    f_val = float(jax.lax.stop_gradient(jnp.asarray(ept.f)))
    ref = float(bg.f_of_loga.evaluate(jnp.log(jnp.asarray(1.0))))
    assert abs(f_val - ref) < 0.01, (
        f"EPT growth rate {f_val} != background f(z=0) {ref:.4f} "
        f"(the hardcoded-0.8 fallback is still active)")
    # Physical oracle bound, independent of the f_of_loga implementation:
    # LCDM z=0 has f ~ Omega_m**0.55 = 0.315**0.55 ~ 0.53 (measured 0.5258,
    # GPU-allocated job 14140, actual JAX platform unverified). Catches a
    # broken f_grid/spline that the
    # self-referential check above cannot.
    assert 0.45 < f_val < 0.60, (
        f"EPT growth rate {f_val} outside the physical LCDM z=0 range")


def test_eptcomponents_pytree_roundtrip(request, fast_mode):
    """h/f/sigma2 are leaves: tree_map touches them, jit caching is safe."""
    if fast_mode:
        pytest.skip("uses the shared full-mode pipeline fixture")
    params, bg, pt = request.getfixturevalue("stage_setup")
    ept = compute_ept_from_clax(params, bg, pt, z=0.0)
    leaves, treedef = jax.tree_util.tree_flatten(ept)
    ept2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert float(ept2.f) == float(ept.f)
    assert float(ept2.h) == float(ept.h)
    n_scalar_leaves = sum(1 for l in leaves if jnp.ndim(l) == 0)
    assert n_scalar_leaves >= 4, "h/f/sigma2/delta_sigma2 must be leaves"
