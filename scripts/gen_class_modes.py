"""Generate CLASS C_l with different source combinations to compare T1/T2."""
import numpy as np
from classy import Class

base_params = {
    'h': 0.6736,
    'omega_b': 0.02237,
    'omega_cdm': 0.1200,
    'A_s': 2.1e-9,
    'n_s': 0.9649,
    'tau_reio': 0.0544,
    'N_ur': 2.0328,
    'N_ncdm': 1,
    'm_ncdm': 0.06,
    'T_ncdm': 0.71611,
    'output': 'tCl pCl',
    'l_max_scalars': 2500,
    'tol_background_integration': 1e-12,
}

ells = [20, 30, 50, 100, 200, 300, 500]

# Full (default)
cosmo = Class()
cosmo.set(base_params)
cosmo.compute()
cls_full = cosmo.raw_cl(2500)
cosmo.struct_cleanup(); cosmo.empty()

# T0 only: disable ISW, Doppler is part of T0 IBP form
# CLASS: switch_sw=1, switch_eisw=0, switch_lisw=0, switch_dop=1, switch_pol=0
# This gives: T0 = SW + Doppler (IBP), no ISW, no T2
# Actually T0 IBP INCLUDES the ISW terms (both SW+ISW+Doppler in the IBP source)
# T1 = ISW dipole (switch_isw controlled separately, only affects T1)
# T2 = polarization quadrupole (switch_pol)

# Let me try disabling T2 only (switch_pol=0)
params_noT2 = dict(base_params)
params_noT2['temperature contributions'] = 'tsw,eisw,lisw,dop'
# This should give T0+T1 without T2
cosmo2 = Class()
cosmo2.set(params_noT2)
cosmo2.compute()
cls_noT2 = cosmo2.raw_cl(2500)
cosmo2.struct_cleanup(); cosmo2.empty()

# Try T0 only: no ISW dipole, no T2
# 'temperature contributions' = 'tsw,dop' (SW + Doppler in IBP = T0 only)
params_T0 = dict(base_params)
params_T0['temperature contributions'] = 'tsw,dop'
cosmo3 = Class()
cosmo3.set(params_T0)
cosmo3.compute()
cls_T0 = cosmo3.raw_cl(2500)
cosmo3.struct_cleanup(); cosmo3.empty()

print(f"{'l':>5} {'CLASS full':>12} {'CLASS T0+T1':>12} {'CLASS T0':>12} {'T1 frac':>10} {'T2 frac':>10}")
for l in ells:
    full = cls_full['tt'][l]
    noT2 = cls_noT2['tt'][l]
    T0only = cls_T0['tt'][l]
    if abs(full) > 1e-30:
        t1_frac = (noT2 - T0only) / T0only * 100  # T1 as fraction of T0
        t2_frac = (full - noT2) / noT2 * 100  # T2 as fraction of T0+T1
        print(f"{l:5d} {full:12.4e} {noT2:12.4e} {T0only:12.4e} {t1_frac:+10.2f}% {t2_frac:+10.2f}%")

print("\nDone!")
