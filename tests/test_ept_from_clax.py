"""Wiring tests for clax.ept.ept_inputs_from_clax / compute_ept_from_clax(omfid, field)
(clax-pt validation Part 2, Task C0; spec §6.2, §7 Phase 3).

Multi-cosmology rule: the wiring tests sweep `lcdm_cosmology` with a real
background_solve and a synthetic perturbation table (no ODE solve, login-node
cheap); the delta_cb physics tests sweep `nulcdm_cosmology` and are `slow`
(one perturbation solve per neutrino mass, cached at module scope).
"""
import inspect
import types
from dataclasses import replace as _dc_replace

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from clax import CosmoParams, PrecisionParams
from clax.background import background_solve, sound_horizon_drag
from clax.thermodynamics import thermodynamics_solve
from clax.perturbations import perturbations_solve_mpk
import clax.ept as ept_mod
from clax.ept import (
    EPTPrecisionParams, ept_kgrid, ept_inputs_from_clax, compute_ept_from_clax,
)
from clax.ap import ap_ratios

Z = 0.38
OMFID = 0.31
BG_PREC = PrecisionParams.fast_cl()          # background only: ~5 s on 2 CPU threads
NMAX = EPTPrecisionParams().nmax


def _synthetic_pt(bg, cb_over_m=1.05, n_k=64, n_tau=48):
    """Stand-in for PerturbationResult: ept_inputs_from_clax reads only
    k_grid, tau_grid, delta_m, delta_cb. delta_m = (k/k0)^-1 (tau/tau0)^2 is
    smooth, positive and tau-dependent (the tau interpolation is exercised);
    delta_cb = cb_over_m * delta_m so the field ratio is known exactly."""
    k = np.logspace(-4, np.log10(3.0), n_k)                     # Mpc^-1
    tau0 = float(bg.tau_of_loga.evaluate(0.0))
    tau = np.linspace(0.05 * tau0, tau0, n_tau)
    dm = (k[:, None] / k[0]) ** -1.0 * (tau[None, :] / tau0) ** 2
    return types.SimpleNamespace(k_grid=jnp.asarray(k), tau_grid=jnp.asarray(tau),
                                 delta_m=jnp.asarray(dm), delta_cb=jnp.asarray(cb_over_m * dm))


def _record_compute_ept(monkeypatch):
    """Replace clax.ept.compute_ept with a recorder so the wiring is tested
    without the 20 s loop; returns the list of recorded kwargs."""
    calls = []

    def fake(pk_lin_h, k_h, h, f, prec=None, _ir_precomputed=None, rs_h=99.0,
             hratio=1.0, Dratio=1.0):
        calls.append(dict(pk_lin_h=pk_lin_h, k_h=k_h, h=h, f=f, rs_h=rs_h,
                          hratio=hratio, Dratio=Dratio, ir=_ir_precomputed))
        return "sentinel"

    monkeypatch.setattr(ept_mod, "compute_ept", fake)
    return calls


# ---------------------------------------------------------------------------
# field selection
# ---------------------------------------------------------------------------

def test_field_selects_delta_cb_or_delta_m(lcdm_cosmology):
    name, params = lcdm_cosmology
    bg = background_solve(params, BG_PREC)
    pt = _synthetic_pt(bg, cb_over_m=1.05)
    pk_cb, f_cb = ept_inputs_from_clax(params, bg, pt, Z, field="cb")
    pk_m, f_m = ept_inputs_from_clax(params, bg, pt, Z, field="m")
    assert pk_cb.shape == (NMAX,) and pk_m.shape == (NMAX,)
    assert np.all(np.isfinite(np.asarray(pk_cb))) and np.all(np.asarray(pk_cb) > 0), name
    np.testing.assert_allclose(np.asarray(pk_cb / pk_m), 1.05 ** 2, rtol=1e-12)
    f_want = float(bg.f_of_loga.evaluate(jnp.log(1.0 / (1.0 + Z))))
    assert float(f_cb) == f_want and float(f_m) == f_want, (name, float(f_cb), f_want)


def test_field_validation():
    params = CosmoParams()
    bg = background_solve(params, BG_PREC)
    pt = _synthetic_pt(bg)
    with pytest.raises(ValueError, match="field"):
        ept_inputs_from_clax(params, bg, pt, Z, field="matter")
    # An object with no delta_cb (real MatterPerturbationResults now carry one)
    no_cb = types.SimpleNamespace(k_grid=pt.k_grid, tau_grid=pt.tau_grid, delta_m=pt.delta_m)
    with pytest.raises(ValueError, match="delta_cb"):
        ept_inputs_from_clax(params, bg, no_cb, Z, field="cb")
    pk_m, _ = ept_inputs_from_clax(params, bg, no_cb, Z, field="m")
    assert pk_m.shape == (NMAX,)


def test_inputs_match_bf8ac18_extraction(lcdm_cosmology):
    """field="m" reproduces the pre-C0 extraction (delta_m spline, primordial
    normalisation, h^3) to round-off: the refactor must not move numbers."""
    name, params = lcdm_cosmology
    bg = background_solve(params, BG_PREC)
    pt = _synthetic_pt(bg)
    from clax.primordial import primordial_scalar_pk
    from clax.interpolation import CubicSpline as CS
    h = params.h
    k_h = ept_kgrid()
    k_mpc = jnp.asarray(k_h) * h
    tau_z = bg.tau_of_loga.evaluate(jnp.log(1.0 / (1.0 + Z)))
    dm_z = jax.vmap(lambda dm_k: CS(pt.tau_grid, dm_k).evaluate(tau_z))(pt.delta_m)
    dm_ept = CS(jnp.log(pt.k_grid), dm_z).evaluate(jnp.log(k_mpc))
    want = 2.0 * jnp.pi ** 2 / k_mpc ** 3 * primordial_scalar_pk(k_mpc, params) * dm_ept ** 2 * h ** 3
    got, _ = ept_inputs_from_clax(params, bg, pt, Z, field="m")
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-12, err_msg=name)


# ---------------------------------------------------------------------------
# omfid wiring
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("z", [0.0, Z])
def test_omfid_wiring(lcdm_cosmology, monkeypatch, z):
    name, params = lcdm_cosmology
    bg = background_solve(params, BG_PREC)
    pt = _synthetic_pt(bg)
    calls = _record_compute_ept(monkeypatch)
    assert compute_ept_from_clax(params, bg, pt, z=z) == "sentinel"
    compute_ept_from_clax(params, bg, pt, z=z, omfid=OMFID)
    none_call, ap_call = calls
    assert (float(none_call["hratio"]), float(none_call["Dratio"])) == (1.0, 1.0)
    hr, Dr = ap_ratios(bg, z, OMFID)
    assert float(ap_call["hratio"]) == float(hr) and float(ap_call["Dratio"]) == float(Dr), name
    if z == 0.0:
        assert (float(hr), float(Dr)) == (1.0, 1.0)          # B2: AP is the identity at z = 0
    else:
        assert abs(float(hr) - 1.0) > 1e-4, (name, float(hr))  # Omega_m != 0.31 on every grid point
    # the inputs reaching compute_ept are ept_inputs_from_clax's, cb by default
    pk_h, f = ept_inputs_from_clax(params, bg, pt, z, field="cb")
    for call in calls:
        np.testing.assert_allclose(np.asarray(call["pk_lin_h"]), np.asarray(pk_h), rtol=0, atol=0)
        assert float(call["f"]) == float(f)
        assert np.isclose(float(call["rs_h"]), float(sound_horizon_drag(params)) * float(params.h))
        assert call["ir"] is not None                          # traced IR splitter still in use


def test_th_rs_d_wiring(lcdm_cosmology, monkeypatch):
    """With the thermodynamics result passed, the IR-resummation damping scale is
    its integrated r_s(z_d) (smsharma/clax#44), not the Aubourg et al. fit the
    function falls back to without it; and clax.lensing passes it."""
    name, params = lcdm_cosmology
    bg = background_solve(params, BG_PREC)
    th = thermodynamics_solve(params, BG_PREC, bg)
    pt = _synthetic_pt(bg)
    calls = _record_compute_ept(monkeypatch)
    compute_ept_from_clax(params, bg, pt, z=Z, th=th)
    compute_ept_from_clax(params, bg, pt, z=Z)
    with_th, without = calls
    h = float(params.h)
    assert np.isclose(float(with_th["rs_h"]), float(th.rs_d) * h, rtol=1e-12), name
    assert np.isclose(float(without["rs_h"]), float(sound_horizon_drag(params)) * h, rtol=1e-12), name
    # the fit and the integral differ at the 1e-4 to 1e-3 level (0.15% at omega_b +20%, #44)
    assert abs(float(with_th["rs_h"]) / float(without["rs_h"]) - 1.0) > 1e-5, name
    import clax.lensing
    assert 'compute_ept_from_clax(params, bg, pt_mpk, z=0.0, th=th, field="m")' in inspect.getsource(clax.lensing)


def test_defaults_and_lensing_field():
    sig = inspect.signature(compute_ept_from_clax)
    assert sig.parameters["field"].default == "cb"
    assert sig.parameters["omfid"].default is None
    assert sig.parameters["field"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["omfid"].kind is inspect.Parameter.KEYWORD_ONLY
    import clax.lensing
    src = inspect.getsource(clax.lensing)
    assert 'compute_ept_from_clax(params, bg, pt_mpk, z=0.0, th=th, field="m")' in src, (
        "clax.lensing must request the total-matter field explicitly, "
        "the tau0-complete pt_mpk (smsharma/clax#42) and the thermodynamics "
        "result for r_s(z_d) (smsharma/clax#44): "
        "the CMB-lensing nonlinear ratio is P_mm,NL / P_mm,lin")


# ---------------------------------------------------------------------------
# delta_cb physics (spec §7 Phase 3; sign corrected in C0: delta_cb > delta_m
# above the free-streaming scale because delta_nu < delta_cb there)
# ---------------------------------------------------------------------------

DELTA_PREC = _dc_replace(PrecisionParams.fast_cl(), ncdm_q_size=5, pt_k_max_cl=0.3,
                         pt_k_chunk_size=20)
K_LOW_H, K_HIGH_H = 1e-3, 0.3          # h/Mpc
NU_MASSES = (0.06, 0.15, 0.30)         # COSMOLOGY_GRID_NULCDM
_SOLVES: dict[float, tuple] = {}       # m_ncdm -> (params, bg, pt): one solve per mass


def _solve(params):
    key = float(params.m_ncdm)
    if key not in _SOLVES:
        bg = background_solve(params, DELTA_PREC)
        th = thermodynamics_solve(params, DELTA_PREC, bg)
        # z=0 lookups need the tau0-complete matter solve (smsharma/clax#42).
        pt = perturbations_solve_mpk(params, DELTA_PREC, bg, th)
        _SOLVES[key] = (params, bg, pt)
    return _SOLVES[key]


def _cb_over_m(params, bg, pt, z):
    pk_cb, _ = ept_inputs_from_clax(params, bg, pt, z, field="cb")
    pk_m, _ = ept_inputs_from_clax(params, bg, pt, z, field="m")
    return np.sqrt(np.asarray(pk_cb) / np.asarray(pk_m))       # = delta_cb / delta_m


def _f_nu(params):
    omega_nu = float(params.m_ncdm) / 93.14
    return omega_nu / (float(params.omega_b) + float(params.omega_cdm) + omega_nu)


@pytest.mark.slow
def test_delta_cb_over_delta_m_physics(nulcdm_cosmology):
    """delta_cb/delta_m -> 1 as k -> 0 (k <= 1e-3 h/Mpc, below every free-
    streaming scale on the grid) and sits in (1, 1/(1 - f_nu)] at 0.3 h/Mpc
    (delta_m = (1 - f_nu) delta_cb + f_nu delta_nu with 0 <= delta_nu < delta_cb)."""
    name, params = nulcdm_cosmology
    params, bg, pt = _solve(params)
    k_h = ept_kgrid()
    r = _cb_over_m(params, bg, pt, 0.0)
    low = k_h <= K_LOW_H
    assert np.max(np.abs(r[low] - 1.0)) < 1e-3, (name, float(np.max(np.abs(r[low] - 1.0))))
    i = int(np.argmin(np.abs(k_h - K_HIGH_H)))
    f_nu = _f_nu(params)
    assert 1e-4 < r[i] - 1.0 <= f_nu / (1.0 - f_nu) + 1e-6, (name, float(r[i] - 1.0), f_nu)
    print(f"{name}: delta_cb/delta_m - 1 = {r[i] - 1.0:.3e} at k = {k_h[i]:.3f} h/Mpc (f_nu = {f_nu:.4f})")


@pytest.mark.slow
def test_delta_cb_suppression_grows_with_m_ncdm(fast_mode):
    """|delta_cb/delta_m - 1| at 0.3 h/Mpc increases monotonically over
    m_ncdm = 0.06, 0.15, 0.30 eV (free-streaming suppression grows with f_nu)."""
    if fast_mode:
        pytest.skip("needs three neutrino masses (full mode)")
    k_h = ept_kgrid()
    i = int(np.argmin(np.abs(k_h - K_HIGH_H)))
    vals = []
    for m in NU_MASSES:
        params, bg, pt = _solve(CosmoParams(m_ncdm=m))
        vals.append(float(_cb_over_m(params, bg, pt, 0.0)[i] - 1.0))
    assert vals == sorted(vals) and vals[0] < vals[-1], dict(zip(NU_MASSES, vals))
