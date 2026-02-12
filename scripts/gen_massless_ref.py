"""Generate CLASS reference with all massless neutrinos (matching our approximation)."""
import numpy as np
from classy import Class

# Same as fiducial but with all neutrinos massless
params = {
    'h': 0.6736,
    'omega_b': 0.02237,
    'omega_cdm': 0.1200,
    'A_s': 2.1e-9,
    'n_s': 0.9649,
    'tau_reio': 0.0544,
    # ALL massless neutrinos (N_ur = 3.044, no massive)
    'N_ur': 3.044,
    'N_ncdm': 0,
    'output': 'tCl pCl',
    'l_max_scalars': 2500,
    'tol_background_integration': 1e-12,
}

cosmo = Class()
cosmo.set(params)
cosmo.compute()

cls = cosmo.raw_cl(2500)
ell = cls['ell']
# CLASS raw_cl gives C_l (not l(l+1)/(2pi)*C_l)
np.savez('reference_data/cls_all_massless.npz',
         ell=ell, tt=cls['tt'], ee=cls['ee'], te=cls['te'])

print("Generated all-massless reference")
print(f"  ell range: {ell[2]} to {ell[-1]}")

# Also print comparison at key l values
ref_massive = np.load('reference_data/lcdm_fiducial/cls.npz')
ells_test = [20, 30, 50, 100, 200, 300, 500, 700, 1000]
print(f"\n{'l':>6} {'massive/massless TT':>20} {'massive/massless EE':>20}")
for l in ells_test:
    idx = np.argmin(np.abs(ell - l))
    idx_ref = np.argmin(np.abs(ref_massive['ell'] - l))
    if abs(cls['tt'][idx]) > 1e-30:
        ratio_tt = ref_massive['tt'][idx_ref] / cls['tt'][idx]
        ratio_ee = ref_massive['ee'][idx_ref] / cls['ee'][idx]
        print(f"{l:6d} {ratio_tt:20.6f} {ratio_ee:20.6f}")

cosmo.struct_cleanup()
cosmo.empty()
print("\nDone!")
