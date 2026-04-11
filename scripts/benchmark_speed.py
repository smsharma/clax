"""Benchmark clax pipeline speed by stage.

Times each stage (background, thermodynamics, perturbations, harmonic)
separately. Runs twice: first call includes JIT compilation, second call
is cached execution time.

Compares C_l at key multipoles against CLASS reference for accuracy.

Usage:
    python scripts/benchmark_speed.py [preset]

    preset: planck_cl (default), fit_cl, medium_cl, fast_cl
"""
import sys
import time
import argparse

sys.path.insert(0, ".")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from clax.params import CosmoParams, PrecisionParams
from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve
from clax.perturbations import perturbations_solve
from clax.harmonic import compute_cls_all_interp, compute_cls_all, compute_cls_all_fast

# Reference ells for accuracy check
CHECK_ELLS = [20, 100, 500, 1000]


def load_class_reference():
    """Load CLASS reference C_l data."""
    ref = np.load("reference_data/cls_massless_recfast.npz")
    return {
        'ell': ref['ell'],
        'tt': ref['tt'],
        'ee': ref['ee'],
        'te': ref['te'],
    }


def get_preset(name):
    """Get PrecisionParams preset by name."""
    if hasattr(PrecisionParams, name):
        return getattr(PrecisionParams, name)()
    raise ValueError(f"Unknown preset: {name}")


def count_k_modes(prec):
    """Count number of k-modes for a preset."""
    import math
    n_k = int(math.log10(prec.pt_k_max_cl / prec.pt_k_min) * prec.pt_k_per_decade)
    return n_k


def _compute_cls(pt, params, bg, prec):
    """Compute C_l using the fastest available method."""
    l_max = prec.hr_l_max
    n_k_fine = prec.hr_n_k_fine

    if n_k_fine == 0:
        # Coarse-grid: uses perturbation k-grid directly (fast, less accurate)
        return compute_cls_all(pt, params, bg, l_max=l_max)
    else:
        # Fast all-l-at-once: single upward Bessel pass + source interpolation
        return compute_cls_all_fast(pt, params, bg, l_max=l_max,
                                     n_k_fine=n_k_fine)


def run_benchmark(preset_name):
    """Run full benchmark for a given preset."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {preset_name}")
    print(f"{'='*60}")

    prec = get_preset(preset_name)
    params = CosmoParams(m_ncdm=0.0)
    n_k = count_k_modes(prec)

    mode = "coarse-grid" if prec.hr_n_k_fine == 0 else f"fast-all-l(n_k_fine={prec.hr_n_k_fine})"
    print(f"  k_per_decade={prec.pt_k_per_decade}, k_max={prec.pt_k_max_cl}, n_k={n_k}")
    print(f"  l_max_g={prec.pt_l_max_g}, tau_n={prec.pt_tau_n_points}, th_n={prec.th_n_points}")
    print(f"  ode_max_steps={prec.ode_max_steps}, hr_l_max={prec.hr_l_max}")
    print("  perturbation PID: fixed DISCO-EB-style filtered scalar norm")
    print(f"  harmonic mode: {mode}")
    print()

    # --- First call (JIT compile + run) ---
    print("First call (compile + run):")

    t0 = time.time()
    bg = background_solve(params, prec)
    jax.block_until_ready(bg.conformal_age)
    t_bg = time.time() - t0
    print(f"  Background:      {t_bg:7.1f}s")

    t0 = time.time()
    th = thermodynamics_solve(params, prec, bg)
    jax.block_until_ready(th.tau_star)
    t_th = time.time() - t0
    print(f"  Thermodynamics:  {t_th:7.1f}s")

    t0 = time.time()
    pt = perturbations_solve(params, prec, bg, th)
    jax.block_until_ready(pt.source_T0)
    t_pt = time.time() - t0
    print(f"  Perturbations:   {t_pt:7.1f}s")

    t0 = time.time()
    cls_result = _compute_cls(pt, params, bg, prec)
    jax.block_until_ready(cls_result['tt'])
    t_hr = time.time() - t0
    print(f"  Harmonic:        {t_hr:7.1f}s")

    t_total_1 = t_bg + t_th + t_pt + t_hr
    print(f"  TOTAL (1st):     {t_total_1:7.1f}s")

    # --- Second call (cached) ---
    print("\nSecond call (cached):")
    params2 = CosmoParams(m_ncdm=0.0, h=0.70, omega_cdm=0.13)

    t0 = time.time()
    bg2 = background_solve(params2, prec)
    jax.block_until_ready(bg2.conformal_age)
    t_bg2 = time.time() - t0
    print(f"  Background:      {t_bg2:7.1f}s")

    t0 = time.time()
    th2 = thermodynamics_solve(params2, prec, bg2)
    jax.block_until_ready(th2.tau_star)
    t_th2 = time.time() - t0
    print(f"  Thermodynamics:  {t_th2:7.1f}s")

    t0 = time.time()
    pt2 = perturbations_solve(params2, prec, bg2, th2)
    jax.block_until_ready(pt2.source_T0)
    t_pt2 = time.time() - t0
    print(f"  Perturbations:   {t_pt2:7.1f}s")

    t0 = time.time()
    cls2 = _compute_cls(pt2, params2, bg2, prec)
    jax.block_until_ready(cls2['tt'])
    t_hr2 = time.time() - t0
    print(f"  Harmonic:        {t_hr2:7.1f}s")

    t_total_2 = t_bg2 + t_th2 + t_pt2 + t_hr2
    print(f"  TOTAL (2nd):     {t_total_2:7.1f}s")

    # --- Accuracy check (first call, fiducial params) ---
    l_max = prec.hr_l_max
    ref = load_class_reference()
    print(f"\nAccuracy vs CLASS (fiducial, massless ncdm, RECFAST):")
    print(f"  {'l':>5s}  {'TT err%':>8s}  {'EE err%':>8s}  {'TE err%':>8s}")

    for l in CHECK_ELLS:
        if l > l_max:
            print(f"  {l:5d}  (skipped, l > hr_l_max={l_max})")
            continue
        tt_jax = float(cls_result['tt'][l])
        ee_jax = float(cls_result['ee'][l])
        te_jax = float(cls_result['te'][l])

        tt_ref = float(ref['tt'][l])
        ee_ref = float(ref['ee'][l])
        te_ref = float(ref['te'][l])

        tt_err = (tt_jax - tt_ref) / abs(tt_ref) * 100 if abs(tt_ref) > 0 else 0
        ee_err = (ee_jax - ee_ref) / abs(ee_ref) * 100 if abs(ee_ref) > 0 else 0
        te_err = (te_jax - te_ref) / abs(te_ref) * 100 if abs(te_ref) > 0 else 0

        print(f"  {l:5d}  {tt_err:+8.3f}  {ee_err:+8.3f}  {te_err:+8.3f}")

    return {
        'preset': preset_name,
        'n_k': n_k,
        'first': {'bg': t_bg, 'th': t_th, 'pt': t_pt, 'hr': t_hr, 'total': t_total_1},
        'cached': {'bg': t_bg2, 'th': t_th2, 'pt': t_pt2, 'hr': t_hr2, 'total': t_total_2},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("preset", nargs="?", default="planck_cl",
                       help="Preset name (planck_cl, fit_cl, medium_cl, fast_cl)")
    parser.add_argument("--all", action="store_true",
                       help="Run all presets")
    args = parser.parse_args()

    print(f"GPU: {jax.devices()}")
    print(f"JAX version: {jax.__version__}")

    if args.all:
        presets = ['fit_cl', 'medium_cl', 'planck_cl']
        results = []
        for p in presets:
            try:
                r = run_benchmark(p)
                results.append(r)
            except Exception as e:
                print(f"ERROR running {p}: {e}")

        if results:
            print(f"\n{'='*60}")
            print("SUMMARY (cached execution)")
            print(f"{'='*60}")
            print(f"  {'Preset':<15s} {'n_k':>5s} {'BG':>6s} {'TH':>6s} {'PT':>7s} {'HR':>6s} {'TOTAL':>7s}")
            for r in results:
                c = r['cached']
                print(f"  {r['preset']:<15s} {r['n_k']:5d} {c['bg']:6.1f} {c['th']:6.1f} {c['pt']:7.1f} {c['hr']:6.1f} {c['total']:7.1f}")
    else:
        run_benchmark(args.preset)
