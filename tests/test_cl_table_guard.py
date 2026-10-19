"""Guards of the fixed-range Bessel table in compute_cls_all_fast (#54).
Cosmology-independent plumbing: exempt from the multi-cosmology rule."""
from dataclasses import replace

import jax
import pytest

from clax.harmonic import compute_cls_all_fast


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
