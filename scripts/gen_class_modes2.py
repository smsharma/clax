"""Generate CLASS C_l with matching source decomposition.

CLASS temperature contributions:
  tsw: Sachs-Wolfe (g*(delta_g/4 + alpha'))
  eisw: Early ISW (visibility g * ISW terms)
  lisw: Late ISW (exp(-kappa) * ISW terms)
  dop: Doppler (g*theta_b' + g'*theta_b)/k²
  pol: Polarization quadrupole T2

Our T0 = tsw + eisw + lisw + dop  (all j_l-radial terms in IBP form)
Our T1 = ISW dipole (j_l' radial)  -- this is controlled by CLASS's transfer type, not these switches
Our T2 = pol (quadrupole radial)
"""
import numpy as np
from classy import Class

base_params = {
    'h': 0.6736, 'omega_b': 0.02237, 'omega_cdm': 0.1200,
    'A_s': 2.1e-9, 'n_s': 0.9649, 'tau_reio': 0.0544,
    'N_ur': 2.0328, 'N_ncdm': 1, 'm_ncdm': 0.06, 'T_ncdm': 0.71611,
    'output': 'tCl pCl', 'l_max_scalars': 2500,
    'tol_background_integration': 1e-12,
}

ells = [20, 30, 50, 100, 200, 300, 500, 700]

# Full (T0+T1+T2)
cosmo = Class()
cosmo.set(base_params)
cosmo.compute()
cls_full = cosmo.raw_cl(2500)
cosmo.struct_cleanup(); cosmo.empty()

# T0 = SW + eISW + lISW + Doppler (all IBP j_l terms, no T1 dipole, no T2 pol)
params = dict(base_params)
params['temperature contributions'] = 'tsw,eisw,lisw,dop'
cosmo = Class()
cosmo.set(params)
cosmo.compute()
cls_T0 = cosmo.raw_cl(2500)
cosmo.struct_cleanup(); cosmo.empty()

# T0+T1 = all except T2 (pol)
params2 = dict(base_params)
params2['temperature contributions'] = 'tsw,eisw,lisw,dop'  # T0
# Actually CLASS T1 is separate from these switches. These switches control
# what goes into T0. T1 is always the ISW dipole j_l' term.
# To get T0+T1 (no T2), I need to disable just the polarization:
params2['temperature contributions'] = 'tsw,eisw,lisw,dop'  # No pol = no T2
cosmo = Class()
cosmo.set(params2)
cosmo.compute()
cls_noT2 = cosmo.raw_cl(2500)
cosmo.struct_cleanup(); cosmo.empty()

# T0 only = disable T1 (ISW j_l' dipole) and T2 (pol)
# CLASS doesn't have a direct switch for T1. The T1 is the "ISW dipole"
# which uses the j_l' radial function. It's always included when ISW is on.
# To truly get T0-only, we'd need to disable T1 in the transfer integral.
# This isn't possible with standard CLASS switches.
# So "tsw,eisw,lisw,dop" WITH "switch_pol=0" gives T0+T1 (no T2).
# "tsw,dop" gives T0 without ISW, and we cannot get T0 with ISW but without T1.

print(f"{'l':>5} {'Full':>12} {'T0(noISW)':>12} {'T0+T1(noT2)':>12}")
print(f"{'':>5} {'':>12} {'err vs full':>12} {'err vs full':>12}")
for l in ells:
    full = cls_full['tt'][l]
    T0_noISW = cls_T0['tt'][l]
    noT2 = cls_noT2['tt'][l]
    err_T0 = (T0_noISW - full) / abs(full) * 100
    err_noT2 = (noT2 - full) / abs(full) * 100
    print(f"{l:5d} {full:12.4e} {err_T0:+12.3f}% {err_noT2:+12.3f}%")

# Now save both for comparison
np.savez('reference_data/cls_class_modes.npz',
         ell=np.arange(len(cls_full['tt'])),
         tt_full=cls_full['tt'],
         tt_T0_noISW=cls_T0['tt'],
         tt_noT2=cls_noT2['tt'])
print("\nSaved to reference_data/cls_class_modes.npz")
print("Done!")
