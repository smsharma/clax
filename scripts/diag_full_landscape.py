"""Full error landscape at all ells from saved spectra."""
import numpy as np

jax_data = np.load('figures/jaxclass_spectra_h100.npz')
ref = np.load('reference_data/lcdm_fiducial/cls.npz')

ells = jax_data['ells']
ell_ref = ref['ell']

header = f"{'l':>6} {'TT%':>8} {'EE%':>8} {'TE%':>8}"
print(header)
print("-" * 40)

n_fail_tt, n_fail_ee = 0, 0
max_tt, max_ee = 0, 0
worst_tt_l, worst_ee_l = 0, 0

for i, ell in enumerate(ells):
    idx = np.argmin(np.abs(ell_ref - ell))
    tt_c = ref['tt'][idx]
    ee_c = ref['ee'][idx]
    te_c = ref['te'][idx]

    tt_e = (jax_data['cl_tt'][i] - tt_c) / abs(tt_c) * 100 if abs(tt_c) > 1e-30 else 0
    ee_e = (jax_data['cl_ee'][i] - ee_c) / abs(ee_c) * 100 if abs(ee_c) > 1e-30 else 0
    te_e = (jax_data['cl_te'][i] - te_c) / abs(te_c) * 100 if abs(te_c) > 1e-30 else 0

    if abs(tt_e) > 0.1: n_fail_tt += 1
    if abs(ee_e) > 0.1: n_fail_ee += 1
    if abs(tt_e) > max_tt: max_tt = abs(tt_e); worst_tt_l = ell
    if abs(ee_e) > max_ee: max_ee = abs(ee_e); worst_ee_l = ell

    # Print all ells where errors exceed 0.1%, plus every 200th ell
    show = abs(tt_e) > 0.1 or abs(ee_e) > 0.1 or ell <= 30 or ell >= 1800
    if show:
        tt_m = " *" if abs(tt_e) > 0.1 else ""
        ee_m = " *" if abs(ee_e) > 0.1 else ""
        print(f"{ell:6d} {tt_e:+8.3f}{tt_m:>3} {ee_e:+8.3f}{ee_m:>3} {te_e:+8.3f}")

print(f"\nFailing (>0.1%): TT={n_fail_tt}/{len(ells)}, EE={n_fail_ee}/{len(ells)}")
print(f"Worst TT: {max_tt:.3f}% at l={worst_tt_l}")
print(f"Worst EE: {max_ee:.3f}% at l={worst_ee_l}")

# Error by range
for lo, hi in [(2,30), (30,100), (100,500), (500,1000), (1000,2001)]:
    mask = (ells >= lo) & (ells < hi)
    if mask.sum() == 0: continue
    tt_errs = []
    ee_errs = []
    for i in np.where(mask)[0]:
        idx = np.argmin(np.abs(ell_ref - ells[i]))
        tt_c = ref['tt'][idx]
        ee_c = ref['ee'][idx]
        if abs(tt_c) > 1e-30:
            tt_errs.append(abs((jax_data['cl_tt'][i] - tt_c) / tt_c * 100))
        if abs(ee_c) > 1e-30:
            ee_errs.append(abs((jax_data['cl_ee'][i] - ee_c) / ee_c * 100))
    print(f"  l={lo}-{hi}: TT max={max(tt_errs):.3f}%, EE max={max(ee_errs):.3f}%")
