#!/usr/bin/env python3
"""Generates CLASS-PT reference data for accuracy validation.

Requires CLASS-PT Python wrapper (classy with PT extension).
Install via:  cd ~/CLASS-PT && pip install .

Produces reference_data/classpt_z{z}_fullrange.npz files containing:
  - pk_mm_real   : matter-matter P(k) real-space
  - pk_mg_real   : galaxy-matter P(k) real-space  (NB: "mg" not "gm")
  - pk_gg_real   : galaxy-galaxy P(k) real-space
  - pk_mm_l0/l2/l4 : matter RSD multipoles
  - pk_gg_l0/l2/l4 : galaxy RSD multipoles
  - pk_mult      : raw multipole array from get_pk_mult
  - pk_lin        : linear P(k) in (Mpc/h)^3
  - k_h          : wavenumber grid in h/Mpc (from ept_kgrid)
  - h, fz, z     : cosmological parameters
  - bias_*       : bias/EFT parameters used

The output filename format is classpt_z{z}_fullrange.npz, matching
what scripts/accuracy_classpt.py loads.

Usage:
    python scripts/generate_classpt_reference.py
"""

import argparse
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Cosmological parameters — Planck 2018 best-fit LCDM (no massive neutrinos)
# ---------------------------------------------------------------------------

FIDUCIAL_PARAMS = {
    'h': 0.6736,
    'omega_b': 0.02237,
    'omega_cdm': 0.1200,
    'A_s': 2.0989e-9,
    'n_s': 0.9649,
    'tau_reio': 0.0544,
}

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'reference_data')

# Fiducial bias / EFT parameters for galaxy reference data.
# Set to zero (except b1, b4) for a clean baseline; non-zero values
# can be added for richer validation.
BIAS_PARAMS = {
    'b1': 2.0, 'b2': 0.0, 'bG2': 0.0, 'bGamma3': 0.0,
    'cs0': 0.0, 'cs2': 0.0, 'cs4': 0.0, 'Pshot': 0.0, 'b4': 500.0,
    'cs': 0.0,
}

Z_VALUES = [0.38]


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

    # Import ept_kgrid to use the same k grid as clax
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from clax.ept import ept_kgrid, EPTPrecisionParams
        prec = EPTPrecisionParams()
        k_h = ept_kgrid(prec)  # 256-pt, [5e-5, 100] h/Mpc
        print(f"Using ept_kgrid: {len(k_h)} pts [{k_h.min():.2e}, {k_h.max():.2e}] h/Mpc")
    except Exception as e:
        print(f"WARNING: could not import ept_kgrid ({e}), using fallback grid")
        k_h = np.logspace(np.log10(5e-5), np.log10(100.0), 256)

    os.makedirs(OUTDIR, exist_ok=True)

    for z in Z_VALUES:
        print(f"\n--- Generating CLASS-PT reference at z={z} ---")

        h = FIDUCIAL_PARAMS['h']
        k_1Mpc = k_h * h  # convert to 1/Mpc for CLASS internal use

        # Build CLASS parameter dict including CLASS-PT options.
        # AP=Yes with Omfid=0.31: enables Alcock-Paczynski effect in the
        # multipole computation. The fiducial Omega_m (Omfid) sets the
        # reference cosmology for the AP distortion; 0.31 is a standard
        # choice matching typical survey analyses (e.g. BOSS).
        params = dict(FIDUCIAL_PARAMS)
        params.update({
            'output': 'mPk',
            'non linear': 'PT',
            'IR resummation': 'Yes',
            'Bias tracers': 'Yes',
            'RSD': 'Yes',
            'AP': 'Yes',
            'Omfid': '0.31',
            'P_k_max_h/Mpc': 100.0,
            'z_pk': z,
        })

        cosmo = Class()
        cosmo.set(params)
        cosmo.compute()

        h_out = cosmo.h()
        fz = cosmo.scale_independent_growth_factor_f(z)

        print(f"  h = {h_out:.4f}, f(z={z}) = {fz:.6f}")

        # Initialize output on the ept k grid
        cosmo.initialize_output(k_1Mpc, z, len(k_h))

        b = BIAS_PARAMS

        # Raw multipole array
        pk_mult = cosmo.get_pk_mult(k_1Mpc, z, len(k_h))

        # Linear P(k) in (Mpc/h)^3
        pk_lin = np.array([cosmo.pk_lin(ki, z) for ki in k_1Mpc]) * h_out**3

        # Matter spectra
        pk_mm_real = np.asarray(cosmo.pk_mm_real(cs=b['cs']))

        # Galaxy spectra at fiducial bias
        pk_gg_real = np.asarray(
            cosmo.pk_gg_real(b['b1'], b['b2'], b['bG2'], b['bGamma3'],
                             b['cs'], b['cs0'], b['Pshot'])
        )
        # NB: the CLASS-PT method is pk_gm_real, but we save as "pk_mg_real"
        # to match the convention used by accuracy_classpt.py
        pk_mg_real = np.asarray(
            cosmo.pk_gm_real(b['b1'], b['b2'], b['bG2'], b['bGamma3'],
                             b['cs'], b['cs0'])
        )

        # RSD multipoles -- matter
        pk_mm_l0 = np.asarray(cosmo.pk_mm_l0(cs0=b['cs0']))
        pk_mm_l2 = np.asarray(cosmo.pk_mm_l2(cs2=b['cs2']))
        pk_mm_l4 = np.asarray(cosmo.pk_mm_l4(cs4=b['cs4']))

        # RSD multipoles -- galaxy
        pk_gg_l0 = np.asarray(
            cosmo.pk_gg_l0(b['b1'], b['b2'], b['bG2'], b['bGamma3'],
                           b['cs0'], b['Pshot'], b['b4'])
        )
        pk_gg_l2 = np.asarray(
            cosmo.pk_gg_l2(b['b1'], b['b2'], b['bG2'], b['bGamma3'],
                           b['cs2'], b['b4'])
        )
        pk_gg_l4 = np.asarray(
            cosmo.pk_gg_l4(b['b1'], b['b2'], b['bG2'], b['bGamma3'],
                           b['cs4'], b['b4'])
        )

        # Save with key names matching what accuracy_classpt.py reads:
        #   - "fz" (not "f") for the growth rate
        #   - "pk_mg_real" (not "pk_gm_real") for the galaxy-matter cross
        #   - "pk_mult" for the raw multipole array
        outfile = os.path.join(OUTDIR, f'classpt_z{z}_fullrange.npz')
        np.savez(outfile,
                 k_h=k_h, z=np.array(z), h=np.array(h_out), fz=np.array(fz),
                 pk_lin=pk_lin,
                 pk_mm_real=pk_mm_real,
                 pk_gg_real=pk_gg_real,
                 pk_mg_real=pk_mg_real,
                 pk_mm_l0=pk_mm_l0, pk_mm_l2=pk_mm_l2, pk_mm_l4=pk_mm_l4,
                 pk_gg_l0=pk_gg_l0, pk_gg_l2=pk_gg_l2, pk_gg_l4=pk_gg_l4,
                 pk_mult=pk_mult,
                 **{f'bias_{k}': v for k, v in BIAS_PARAMS.items()})
        print(f"  Saved {outfile}")

        # Quick sanity check
        print(f"  pk_mm_real at k=0.1 h/Mpc: "
              f"{np.interp(0.1, k_h, pk_mm_real):.2f} (Mpc/h)^3")

        cosmo.struct_cleanup()
        cosmo.empty()

    print("\nDone. Reference data saved to reference_data/classpt_z*_fullrange.npz")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate CLASS-PT reference data')
    args = parser.parse_args()
    generate_classpt_reference()
