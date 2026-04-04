#!/usr/bin/env python3
"""Generate CLASS-PT reference on the full EPT k-grid (256 pts, 5e-5 to 100 h/Mpc) at z=0.38.

Saves reference_data/classpt_z0.38_fullrange.npz with all 9 spectra.
No massive neutrinos, pure LCDM (Planck 2018), b1=2, b4=500, all others=0.

Usage:
    ~/miniconda3/envs/sbi_pytorch_osx-arm64-py310forge/bin/python3 scripts/gen_reference_fullrange.py
"""
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classy import Class
from clax.ept import ept_kgrid, EPTPrecisionParams

z = 0.38
cosmo_params = {
    'h': 0.6736, 'omega_b': 0.02237, 'omega_cdm': 0.1200,
    'A_s': 2.0989e-9, 'n_s': 0.9649, 'tau_reio': 0.0544,
}
classpt_settings = {
    'output': 'mPk', 'non linear': 'PT',
    'IR resummation': 'Yes', 'Bias tracers': 'Yes', 'RSD': 'Yes',
    'P_k_max_h/Mpc': 100., 'z_pk': z,
}
bias = {'b1': 2.0, 'b2': 0.0, 'bG2': 0.0, 'bGamma3': 0.0,
        'cs': 0.0, 'cs0': 0.0, 'cs2': 0.0, 'cs4': 0.0,
        'Pshot': 0.0, 'b4': 500.0}

h = cosmo_params['h']

prec = EPTPrecisionParams()
k_h = ept_kgrid(prec)        # 256-pt, [5e-5, 100] h/Mpc
k_1Mpc = k_h * h             # convert to 1/Mpc for CLASS

print(f"k grid: {len(k_h)} pts [{k_h.min():.2e}, {k_h.max():.2e}] h/Mpc")
print(f"Running CLASS-PT at z={z} ...")

M = Class()
M.set({**cosmo_params, **classpt_settings})
M.compute()

M.initialize_output(k_1Mpc, z, len(k_h))

pk_mult = M.get_pk_mult(k_1Mpc, z, len(k_h))
pk_lin  = np.array([M.pk_lin(ki, z) for ki in k_1Mpc]) * h**3

pk_mm_real = M.pk_mm_real(cs=bias['cs'])
pk_gg_real = M.pk_gg_real(bias['b1'], bias['b2'], bias['bG2'], bias['bGamma3'],
                           bias['cs'], bias['cs0'], bias['Pshot'])
pk_mg_real = M.pk_gm_real(bias['b1'], bias['b2'], bias['bG2'], bias['bGamma3'],
                           bias['cs'], bias['cs0'])
pk_mm_l0 = M.pk_mm_l0(cs0=bias['cs0'])
pk_mm_l2 = M.pk_mm_l2(cs2=bias['cs2'])
pk_mm_l4 = M.pk_mm_l4(cs4=bias['cs4'])
pk_gg_l0 = M.pk_gg_l0(bias['b1'], bias['b2'], bias['bG2'], bias['bGamma3'],
                        bias['cs0'], bias['Pshot'], bias['b4'])
pk_gg_l2 = M.pk_gg_l2(bias['b1'], bias['b2'], bias['bG2'], bias['bGamma3'],
                        bias['cs2'], bias['b4'])
pk_gg_l4 = M.pk_gg_l4(bias['b1'], bias['b2'], bias['bG2'], bias['bGamma3'],
                        bias['cs4'], bias['b4'])

fz = M.scale_independent_growth_factor_f(z)
print(f"f(z={z}) = {fz:.6f}")
M.struct_cleanup(); M.empty()

os.makedirs('reference_data', exist_ok=True)
outpath = 'reference_data/classpt_z0.38_fullrange.npz'
np.savez(outpath, k_h=k_h, z=np.array(z), h=np.array(h), fz=np.array(fz),
         pk_lin=pk_lin, pk_mm_real=pk_mm_real, pk_gg_real=pk_gg_real,
         pk_mg_real=pk_mg_real, pk_mm_l0=pk_mm_l0, pk_mm_l2=pk_mm_l2,
         pk_mm_l4=pk_mm_l4, pk_gg_l0=pk_gg_l0, pk_gg_l2=pk_gg_l2, pk_gg_l4=pk_gg_l4,
         pk_mult=pk_mult, **{f'bias_{k}': v for k, v in bias.items()})
print(f"Saved {outpath}")

# Quick sanity check
print(f"pk_mm_real at k=0.1 h/Mpc: {np.interp(0.1, k_h, pk_mm_real):.2f} (Mpc/h)^3")
print(f"pk_gg_l4 at k=0.1 h/Mpc:   {np.interp(0.1, k_h, pk_gg_l4):.2f}")
