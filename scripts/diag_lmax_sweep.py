"""Phase 1 Diagnostic: l_max sweep on H100 to confirm hierarchy truncation causes high-l TT degradation.

Runs C_l TT/EE/TE at l_max=50,65,80 with IDENTICAL k_max=1.0, n_k_fine=5000.
Measures accuracy at l=20,30,50,100,300,500,700,1000,2000 vs CLASS reference.
"""
import sys
sys.path.insert(0, '.')
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxclass import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import perturbations_solve
from jaxclass.harmonic import compute_cl_tt_interp, compute_cl_ee_interp, compute_cl_te_interp

print(f"Devices: {jax.devices()}", flush=True)

# Shared settings
params = CosmoParams()
N_K_FINE = 5000
ELLS = [20, 30, 50, 100, 300, 500, 700, 1000, 2000]
L_MAX_VALUES = [50, 65, 80]

# Load CLASS reference
ref_cls = np.load('reference_data/lcdm_fiducial/cls.npz')
ell_ref = ref_cls['ell']

# Run background + thermo once (independent of l_max)
print("Running background...", flush=True)
prec_base = PrecisionParams.planck_cl()  # just for bg/thermo
bg = background_solve(params, prec_base)
print("Running thermodynamics...", flush=True)
th = thermodynamics_solve(params, prec_base, bg)

# Store results: {l_max: {spec: {ell: err%}}}
results = {}

for l_max in L_MAX_VALUES:
    print(f"\n{'='*60}", flush=True)
    print(f"  l_max = {l_max}", flush=True)
    print(f"{'='*60}", flush=True)

    prec = PrecisionParams(
        pt_k_max_cl=1.0,
        pt_k_per_decade=60,
        pt_tau_n_points=5000,
        pt_l_max_g=l_max,
        pt_l_max_pol_g=l_max,
        pt_l_max_ur=l_max,
        pt_ode_rtol=1e-6,
        pt_ode_atol=1e-11,
        ode_max_steps=131072,
    )

    print(f"Running perturbations (l_max={l_max})...", flush=True)
    t0 = time.time()
    pt = perturbations_solve(params, prec, bg, th)
    t1 = time.time()
    print(f"  Perturbations done in {t1-t0:.1f}s, n_k={len(pt.k_grid)}", flush=True)

    print(f"  Computing C_l^TT...", flush=True)
    t0 = time.time()
    cl_tt = compute_cl_tt_interp(pt, params, bg, ELLS, n_k_fine=N_K_FINE)
    t1 = time.time()
    print(f"  TT done in {t1-t0:.1f}s", flush=True)

    print(f"  Computing C_l^EE...", flush=True)
    t0 = time.time()
    cl_ee = compute_cl_ee_interp(pt, params, bg, ELLS, n_k_fine=N_K_FINE)
    t1 = time.time()
    print(f"  EE done in {t1-t0:.1f}s", flush=True)

    print(f"  Computing C_l^TE...", flush=True)
    t0 = time.time()
    cl_te = compute_cl_te_interp(pt, params, bg, ELLS, n_k_fine=N_K_FINE)
    t1 = time.time()
    print(f"  TE done in {t1-t0:.1f}s", flush=True)

    # Compute errors
    errs = {'TT': {}, 'EE': {}, 'TE': {}}
    for i, ell in enumerate(ELLS):
        idx = np.argmin(np.abs(ell_ref - ell))
        for spec, cl_arr, ref_key in [('TT', cl_tt, 'tt'), ('EE', cl_ee, 'ee'), ('TE', cl_te, 'te')]:
            cl_class = ref_cls[ref_key][idx]
            if abs(cl_class) > 1e-30:
                errs[spec][ell] = (float(cl_arr[i]) - cl_class) / abs(cl_class) * 100
            else:
                errs[spec][ell] = 0.0

    results[l_max] = errs

    # Print per-l_max table
    print(f"\n  {'l':>6} {'TT err%':>10} {'EE err%':>10} {'TE err%':>10}")
    print(f"  {'-'*40}")
    for ell in ELLS:
        tt_e = errs['TT'][ell]
        ee_e = errs['EE'][ell]
        te_e = errs['TE'][ell]
        print(f"  {ell:6d} {tt_e:+10.3f} {ee_e:+10.3f} {te_e:+10.3f}")

# Summary comparison table
print(f"\n\n{'='*80}")
print(f"  SUMMARY: TT error (%) vs l_max")
print(f"{'='*80}")
header = f"{'l':>6}"
for l_max in L_MAX_VALUES:
    header += f" {'lmax='+str(l_max):>12}"
print(header)
print("-" * (6 + 13 * len(L_MAX_VALUES)))
for ell in ELLS:
    row = f"{ell:6d}"
    for l_max in L_MAX_VALUES:
        row += f" {results[l_max]['TT'][ell]:+12.3f}"
    print(row)

print(f"\n{'='*80}")
print(f"  SUMMARY: EE error (%) vs l_max")
print(f"{'='*80}")
header = f"{'l':>6}"
for l_max in L_MAX_VALUES:
    header += f" {'lmax='+str(l_max):>12}"
print(header)
print("-" * (6 + 13 * len(L_MAX_VALUES)))
for ell in ELLS:
    row = f"{ell:6d}"
    for l_max in L_MAX_VALUES:
        row += f" {results[l_max]['EE'][ell]:+12.3f}"
    print(row)

print(f"\n{'='*80}")
print(f"  SUMMARY: TE error (%) vs l_max")
print(f"{'='*80}")
header = f"{'l':>6}"
for l_max in L_MAX_VALUES:
    header += f" {'lmax='+str(l_max):>12}"
print(header)
print("-" * (6 + 13 * len(L_MAX_VALUES)))
for ell in ELLS:
    row = f"{ell:6d}"
    for l_max in L_MAX_VALUES:
        row += f" {results[l_max]['TE'][ell]:+12.3f}"
    print(row)

print("\nDone!", flush=True)
