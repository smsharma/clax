#!/usr/bin/env python3
"""Generate CLASS Halofit reference data for C_l^pp validation.

Requires CLASS v3.2.2+ (with the full Limber scheme for accurate C_l^pp
at l > 1000). Older CLASS versions underestimate C_l^pp by ~20% at
l = 2500 due to insufficient k_max in the standard q-grid.

    python scripts/generate_classpt_halofit_reference.py

Generates reference_data/classpt_clpp_halofit.npz with:
    ell, pp_lin, pp_halofit, tt_lensed_lin, tt_lensed_halofit
"""
import numpy as np
from classy import Class

# Matches clax CosmoParams defaults exactly
common_settings = {
    'A_s': 2.1e-9,
    'n_s': 0.9649,
    'tau_reio': 0.052,
    'omega_b': 0.02237,
    'omega_cdm': 0.12,
    'h': 0.6736,
    'YHe': 0.2425,
    'N_ur': 2.0328,
    'N_ncdm': 1,
    'm_ncdm': 0.06,
}

l_max = 2500

# Linear (no NL corrections)
print("Computing linear C_l...")
cosmo_lin = Class()
cosmo_lin.set(common_settings)
cosmo_lin.set({
    'output': 'lCl,tCl',
    'lensing': 'Yes',
    'l_switch_limber': 9,
    'non linear': 'none',
})
cosmo_lin.compute()
cl_lin = cosmo_lin.lensed_cl(l_max)
cosmo_lin.struct_cleanup()

# Halofit NL corrections
print("Computing Halofit C_l...")
cosmo_hf = Class()
cosmo_hf.set(common_settings)
cosmo_hf.set({
    'output': 'lCl,tCl',
    'lensing': 'Yes',
    'l_switch_limber': 9,
    'non linear': 'Halofit',
})
cosmo_hf.compute()
cl_hf = cosmo_hf.lensed_cl(l_max)
cosmo_hf.struct_cleanup()

ell = cl_lin['ell'][2:l_max + 1]
pp_lin = cl_lin['pp'][2:l_max + 1]
pp_halofit = cl_hf['pp'][2:l_max + 1]
tt_lensed_lin = cl_lin['tt'][2:l_max + 1]
tt_lensed_halofit = cl_hf['tt'][2:l_max + 1]

outfile = 'reference_data/classpt_clpp_halofit.npz'
np.savez(outfile,
         ell=ell,
         pp_lin=pp_lin,
         pp_halofit=pp_halofit,
         tt_lensed_lin=tt_lensed_lin,
         tt_lensed_halofit=tt_lensed_halofit)

print(f"Saved to {outfile}")
print(f"  ell: [{ell[0]:.0f}, {ell[-1]:.0f}], N={len(ell)}")

# Print diagnostic ratios
print("\nPP ratio (Halofit/linear):")
for l_val in [10, 100, 200, 500, 1000, 1500, 2000, 2500]:
    idx = l_val - 2
    r = pp_halofit[idx] / pp_lin[idx]
    print(f"  l={l_val:5d}: {r:.6f}")

print("\nTT lensed ratio (Halofit/linear):")
for l_val in [500, 1000, 1500, 2000, 2500]:
    idx = l_val - 2
    r = tt_lensed_halofit[idx] / tt_lensed_lin[idx]
    print(f"  l={l_val:5d}: {r:.6f}")
