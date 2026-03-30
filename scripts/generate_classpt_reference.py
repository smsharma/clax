#!/usr/bin/env python3
"""Generate CLASS-PT reference data for clax-pt validation.

Requires: classy (CLASS Python wrapper with CLASS-PT patches)
  pip install classy   # or: cd ~/CLASS-PT && pip install .

Produces reference_data/classpt_*.npz files containing:
  - pk_mm_real   : matter-matter P(k) at cs0=0
  - pk_gm_real   : galaxy-matter P(k) at fiducial bias
  - pk_gg_real   : galaxy-galaxy P(k) at fiducial bias
  - pk_mm_l0/l2/l4 : matter multipoles
  - pk_gg_l0/l2/l4 : galaxy multipoles
  - k_h          : wavenumber grid in h/Mpc
  - h, f, z      : cosmological parameters

Usage:
    python scripts/generate_classpt_reference.py            # full
"""

import argparse
import os
import sys

import numpy as np

# Fiducial parameter values match the example in https://github.com/Michalychforever/CLASS-PT/blob/master/notebooks/nonlinear_pt.ipynb

# ---------------------------------------------------------------------------
# Cosmological parameters
# ---------------------------------------------------------------------------

FIDUCIAL_PARAMS = {
    'h': 0.6736,
    'omega_b': 0.02237,
    'omega_cdm': 0.1200,
    'A_s': 2.089e-9,
    'n_s': 0.9649,
    'tau_reio': 0.052,
    'YHe':0.2425,
    'N_ur': 2.0328,
    'N_ncdm': 1,
    'm_ncdm': 0.06,
    # 'T_ncdm': 0.71611,
}

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'reference_data')

# Fiducial bias / EFT parameters for galaxy reference data
BIAS_PARAMS = {
    'b1': 2.0, 'b2': -1.0, 'bG2': 0.1, 'bGamma3': -0.1,
    'cs0': 5.0, 'cs2': 15.0, 'cs4': -5.0, 'Pshot': 5e3, 'b4': 100.0,
    'cs': 1.0,  # EFT sound speed (set to zero for matter spectra)
}

Z_VALUES = [0.61,]


# ---------------------------------------------------------------------------
# Generation via classy (CLASS-PT)
# ---------------------------------------------------------------------------

def generate_classpt_reference():
    """Run CLASS-PT via classy and save reference spectra."""
    try:
        from classy import Class
    except ImportError:
        print("ERROR: classy not installed. Run: cd ~/CLASS-PT && pip install .")
        sys.exit(1)

    os.makedirs(OUTDIR, exist_ok=True)

    for z in Z_VALUES:
        print(f"\n--- Generating CLASS-PT reference at z={z} ---")

        # Build CLASS parameter dict including CLASS-PT PT options
        params = dict(FIDUCIAL_PARAMS)
        params.update({
            'output': 'mPk',
            'non linear': 'PT',
            'IR resummation': 'Yes',
            'Bias tracers': 'Yes',
            'cb': 'Yes',
            'RSD': 'Yes',
            'AP': 'Yes',
            'P_k_max_h/Mpc': 100.0,
            'Omfid':'0.31',
            'z_pk':z,
        })

        cosmo = Class()
        cosmo.set(params)
        cosmo.compute()

        h = cosmo.h()
        f = cosmo.scale_independent_growth_factor_f(z)

        # k grid from CLASS-PT internal grid (h/Mpc)
        # Use ept.ept_kgrid to match internal k sampling
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from clax.ept import ept_kgrid, EPTPrecisionParams
            prec = EPTPrecisionParams()
            k_h = ept_kgrid(prec)
            print("Using ept.ept_kgrid to match internal k sampling.")
        except Exception:
            print("Using k_h=np.logspace(-3, np.log10(0.5), 200) (not matching internal sampling).")
            k_h = np.logspace(-3, np.log10(0.5), 200)

        # Matter spectra
        b = BIAS_PARAMS
        cosmo.initialize_output(k_h,z,len(k_h))
        pk_mm_real = np.array([cosmo.pk_mm_real(b['cs'])])

        # Galaxy spectra at fiducial bias
        pk_gg_real = np.array([
            cosmo.pk_gg_real(b['b1'], b['b2'], b['bG2'], b['bGamma3'],
                             b['cs'], b['cs0'], b['Pshot'])
        ])
        pk_gm_real = np.array([
            cosmo.pk_gm_real(b['b1'], b['b2'], b['bG2'], b['bGamma3'],
                             b['cs'], b['cs0'])
        ])

        # RSD multipoles — matter
        pk_mm_l0 = np.array([cosmo.pk_mm_l0(b['cs0'])])
        pk_mm_l2 = np.array([cosmo.pk_mm_l2(b['cs2'])])
        pk_mm_l4 = np.array([cosmo.pk_mm_l4(b['cs4'])])

        # RSD multipoles — galaxy
        pk_gg_l0 = np.array([
            cosmo.pk_gg_l0(b['b1'], b['b2'], b['bG2'], b['bGamma3'],
                           b['cs0'], b['Pshot'], b['b4'])
        ])
        pk_gg_l2 = np.array([
            cosmo.pk_gg_l2(b['b1'], b['b2'], b['bG2'], b['bGamma3'],
                           b['cs2'], b['b4'])
        ])
        pk_gg_l4 = np.array([
            cosmo.pk_gg_l4(b['b1'], b['b2'], b['bG2'], b['bGamma3'],
                           b['cs4'], b['b4'])
        ])

        # Also save raw linear P(k) at this z
        pk_lin = np.array([cosmo.pk_lin(k / h, z) * h**3 for k in k_h])

        outfile = os.path.join(OUTDIR, f'classpt_z{z:.1f}.npz')
        np.savez(outfile,
                 k_h=k_h, z=z, h=h, f=f,
                 pk_lin=pk_lin,
                 pk_mm_real=pk_mm_real,
                 pk_gg_real=pk_gg_real,
                 pk_gm_real=pk_gm_real,
                 pk_mm_l0=pk_mm_l0, pk_mm_l2=pk_mm_l2, pk_mm_l4=pk_mm_l4,
                 pk_gg_l0=pk_gg_l0, pk_gg_l2=pk_gg_l2, pk_gg_l4=pk_gg_l4,
                 **{f'bias_{k}': v for k, v in BIAS_PARAMS.items()})
        print(f"  Saved {outfile}")

        cosmo.struct_cleanup()
        cosmo.empty()

    print("\nDone. Reference data saved to reference_data/classpt_z*.npz")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate CLASS-PT reference data')
    args = parser.parse_args()
    generate_classpt_reference()
