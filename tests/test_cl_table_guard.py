"""Guards of the fixed-range Bessel table in compute_cls_all_fast (#54).
Cosmology-independent plumbing: exempt from the multi-cosmology rule."""
from dataclasses import replace

import jax
import pytest

from clax import PrecisionParams, harmonic
from clax.harmonic import _resolve_n_k_fine, compute_cls_all_fast


def test_tau0_beyond_bound_raises(pipeline_fast_cl):
    params, prec, bg, th, pt = pipeline_fast_cl
    small = replace(prec, hr_tau0_max=1000.0)          # fiducial tau0 ~ 14100 Mpc
    with pytest.raises(Exception, match="hr_tau0_max"):
        jax.block_until_ready(compute_cls_all_fast(pt, params, bg, l_max=50, prec=small)["tt"])


def test_kmax_mismatch_raises(pipeline_fast_cl):
    params, prec, bg, th, pt = pipeline_fast_cl
    wrong = replace(prec, pt_k_max_cl=0.5 * prec.pt_k_max_cl)
    with pytest.raises(Exception, match="pt_k_max_cl"):
        jax.block_until_ready(compute_cls_all_fast(pt, params, bg, l_max=50, prec=wrong)["tt"])


def test_traced_without_prec_asks_for_prec(pipeline_fast_cl):
    params, prec, bg, th, pt = pipeline_fast_cl
    f = jax.jit(lambda pt_, bg_: compute_cls_all_fast(pt_, params, bg_, l_max=50)["tt"])
    with pytest.raises(TypeError, match="prec="):
        f(pt, bg)


def test_guard_silent_at_fiducial(pipeline_fast_cl):
    params, prec, bg, th, pt = pipeline_fast_cl
    tt = compute_cls_all_fast(pt, params, bg, l_max=50, prec=prec)["tt"]
    assert float(tt[10]) > 0, "ERROR C_l not positive with prec given"


def test_resolve_n_k_fine():
    """The fine k-grid of compute_cls_all_fast follows the preset when prec is
    given and n_k_fine is not (at fit_cl 5000 points sample |T_l|^2 about once
    per oscillation near l ~ 1200 and dF/dh is 6x off, #54); an explicit
    n_k_fine wins; the eager no-prec path keeps 5000. Pure resolver, no solve."""
    fit = PrecisionParams.fit_cl()
    assert fit.hr_n_k_fine == 20000
    assert _resolve_n_k_fine(None, fit) == fit.hr_n_k_fine
    assert _resolve_n_k_fine(None, replace(fit, hr_n_k_fine=777)) == 777
    assert _resolve_n_k_fine(1234, fit) == 1234
    assert _resolve_n_k_fine(None, None) == 5000
    assert _resolve_n_k_fine(1234, None) == 1234


def test_default_k_grid_follows_prec(pipeline_fast_cl, monkeypatch):
    """compute_cls_all_fast hands the resolved size to the fine k-grid builder:
    prec.hr_n_k_fine when n_k_fine is not passed, the explicit value otherwise."""
    params, prec, bg, th, pt = pipeline_fast_cl
    seen = []
    real = harmonic._fine_log_k_grid

    def spy(log_k_coarse, n_k_fine):
        seen.append(n_k_fine)
        return real(log_k_coarse, n_k_fine)

    monkeypatch.setattr(harmonic, "_fine_log_k_grid", spy)
    custom = replace(prec, hr_n_k_fine=777)
    jax.block_until_ready(compute_cls_all_fast(pt, params, bg, l_max=50, prec=custom)["tt"])
    jax.block_until_ready(compute_cls_all_fast(pt, params, bg, l_max=50, prec=custom, n_k_fine=555)["tt"])
    assert seen == [777, 555], f"ERROR fine k-grid sizes {seen}, expected [777, 555]"
