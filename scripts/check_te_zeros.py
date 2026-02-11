"""Check TE spectrum zero crossings in CLASS reference data."""
import numpy as np
ref = np.load('reference_data/lcdm_fiducial/cls.npz')
ell = ref['ell']
te = ref['te']
# Show TE values near l=10-100
for l in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 150, 200]:
    idx = np.argmin(np.abs(ell - l))
    ll = ell[idx]
    val = te[idx]
    dl = ll*(ll+1)/(2*np.pi)*val
    print(f"l={int(ll):4d}: C_l^TE = {val:+.6e}  D_l^TE = {dl:+.6e}")
