"""Multi-cosmology validation: test jaxCLASS at varied LCDM parameter points.

Runs the full pipeline (bg → thermo → pert → C_l → lensing) at 10 parameter
points and compares against CLASS reference data.

Usage:
    python scripts/diag_multicosmo.py
"""
import sys
import time
sys.path.insert(0, '.')

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from dataclasses import replace

from jaxclass import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import perturbations_solve
from jaxclass.harmonic import compute_cls_all_interp
from jaxclass.lensing import lens_cls

# Parameter variations (name → CosmoParams kwargs)
VARIATIONS = {
    'omega_b_high': {'omega_b': 0.02237 * 1.20},
    'omega_b_low':  {'omega_b': 0.02237 * 0.80},
    'omega_cdm_high': {'omega_cdm': 0.1200 * 1.20},
    'omega_cdm_low':  {'omega_cdm': 0.1200 * 0.80},
    'h_high': {'h': 0.6736 * 1.10},
    'h_low':  {'h': 0.6736 * 0.90},
    'ns_high': {'n_s': 0.9649 * 1.05},
    'ns_low':  {'n_s': 0.9649 * 0.95},
    'tau_high': {'tau_reio': 0.0544 * 1.30},
    'tau_low':  {'tau_reio': 0.0544 * 0.70},
}

# Test multipoles
TEST_ELLS = [20, 50, 100, 200, 300, 500, 700, 1000]

# planck_cl with chunked vmap for V100-32GB memory
prec = PrecisionParams.planck_cl()
from dataclasses import replace as dc_replace
prec = dc_replace(prec, pt_k_chunk_size=30)  # 30 k-modes per chunk


def run_one_cosmology(name, overrides):
    """Run jaxCLASS pipeline and compare with CLASS reference."""
    refdir = f'reference_data/{name}'
    try:
        cls_ref = dict(np.load(f'{refdir}/cls.npz'))
        cls_lensed_ref = dict(np.load(f'{refdir}/cls_lensed.npz'))
    except FileNotFoundError:
        print(f"  {name}: SKIP (no reference data)")
        return None

    # Set up parameters
    params = replace(CosmoParams(), **overrides)

    # Run pipeline
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    pt = perturbations_solve(params, prec, bg, th)

    # Unlensed C_l
    cls = compute_cls_all_interp(pt, params, bg, l_max=2500, n_k_fine=10000)

    # Lensed C_l
    cl_tt_u = jnp.array(cls['tt'])
    cl_ee_u = jnp.array(cls['ee'])
    cl_te_u = jnp.array(cls['te'])
    cl_bb_u = jnp.zeros_like(cl_tt_u)
    cl_pp = jnp.array(cls_ref['pp'][:2501])  # Use CLASS pp for now

    tt_l, ee_l, te_l, bb_l = lens_cls(
        cl_tt_u, cl_ee_u, cl_te_u, cl_bb_u, cl_pp, l_max=2500
    )
    tt_l = np.array(tt_l)
    ee_l = np.array(ee_l)

    te_l = np.array(te_l)

    # Compute errors
    results = {}
    for l in TEST_ELLS:
        if l > 2500:
            continue
        # Unlensed
        tt_ref = cls_ref['tt'][l]
        ee_ref = cls_ref['ee'][l]
        te_ref = cls_ref['te'][l]
        tt_us = cls['tt'][l]
        ee_us = cls['ee'][l]
        te_us = cls['te'][l]
        tt_err_u = (tt_us - tt_ref) / abs(tt_ref) * 100 if abs(tt_ref) > 1e-30 else 0
        ee_err_u = (ee_us - ee_ref) / abs(ee_ref) * 100 if abs(ee_ref) > 1e-30 else 0
        te_err_u = (te_us - te_ref) / abs(te_ref) * 100 if abs(te_ref) > 1e-30 else 0

        # Lensed
        tt_ref_l = cls_lensed_ref['tt'][l]
        ee_ref_l = cls_lensed_ref['ee'][l]
        te_ref_l = cls_lensed_ref['te'][l]
        tt_err_l = (tt_l[l] - tt_ref_l) / abs(tt_ref_l) * 100 if abs(tt_ref_l) > 1e-30 else 0
        ee_err_l = (ee_l[l] - ee_ref_l) / abs(ee_ref_l) * 100 if abs(ee_ref_l) > 1e-30 else 0
        te_err_l = (te_l[l] - te_ref_l) / abs(te_ref_l) * 100 if abs(te_ref_l) > 1e-30 else 0

        results[l] = {
            'tt_u': float(tt_err_u), 'ee_u': float(ee_err_u), 'te_u': float(te_err_u),
            'tt_l': float(tt_err_l), 'ee_l': float(ee_err_l), 'te_l': float(te_err_l),
        }

    return results


def main():
    print("=" * 90)
    print("Multi-cosmology validation: jaxCLASS vs CLASS at varied LCDM parameters")
    print("=" * 90)

    all_results = {}
    for i, (name, overrides) in enumerate(VARIATIONS.items()):
        print(f"\n[{i+1}/{len(VARIATIONS)}] {name}: {overrides}")
        t0 = time.time()
        results = run_one_cosmology(name, overrides)
        dt = time.time() - t0
        if results is not None:
            all_results[name] = results
            # Print results for this cosmology
            print(f"  Time: {dt:.0f}s")
            print(f"  {'l':>6}  {'TT_u%':>8}  {'EE_u%':>8}  {'TE_u%':>8}  {'TT_l%':>8}  {'EE_l%':>8}  {'TE_l%':>8}")
            for l in TEST_ELLS:
                if l in results:
                    r = results[l]
                    print(f"  {l:6d}  {r['tt_u']:+8.3f}  {r['ee_u']:+8.3f}  {r['te_u']:+8.3f}  "
                          f"{r['tt_l']:+8.3f}  {r['ee_l']:+8.3f}  {r['te_l']:+8.3f}")

    # Summary: max errors across all cosmologies
    print("\n" + "=" * 90)
    print("SUMMARY: Max absolute errors across all cosmologies")
    print("=" * 90)
    print(f"{'l':>6}  {'TT_u%':>10}  {'EE_u%':>10}  {'TE_u%':>10}  {'TT_l%':>10}  {'EE_l%':>10}  "
          f"{'worst_TT':>15}  {'worst_EE':>15}")

    for l in TEST_ELLS:
        max_tt_u = max_ee_u = max_te_u = max_tt_l = max_ee_l = 0
        worst_tt = worst_ee = ""
        for name, results in all_results.items():
            if l in results:
                r = results[l]
                if abs(r['tt_u']) > max_tt_u:
                    max_tt_u = abs(r['tt_u'])
                    worst_tt = name
                if abs(r['ee_u']) > max_ee_u:
                    max_ee_u = abs(r['ee_u'])
                    worst_ee = name
                max_te_u = max(max_te_u, abs(r['te_u']))
                max_tt_l = max(max_tt_l, abs(r['tt_l']))
                max_ee_l = max(max_ee_l, abs(r['ee_l']))
        print(f"{l:6d}  {max_tt_u:10.3f}  {max_ee_u:10.3f}  {max_te_u:10.3f}  {max_tt_l:10.3f}  "
              f"{max_ee_l:10.3f}  {worst_tt:>15}  {worst_ee:>15}")

    # Save results to file
    np.savez('reference_data/multicosmo_results.npz',
             **{f'{name}_{l}_{key}': v
                for name, results in all_results.items()
                for l, r in results.items()
                for key, v in r.items()})


if __name__ == '__main__':
    main()
