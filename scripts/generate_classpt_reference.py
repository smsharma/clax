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
    python scripts/generate_classpt_reference.py --synthetic # synthetic fallback (no classy)
"""

import argparse
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Cosmological parameters
# ---------------------------------------------------------------------------

FIDUCIAL_PARAMS = {
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
}

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'reference_data')

# Fiducial bias / EFT parameters for galaxy reference data
BIAS_PARAMS = {
    'b1': 1.5, 'b2': -0.5, 'bG2': 0.1, 'bGamma3': 0.0,
    'cs0': 0.0, 'cs2': 0.0, 'cs4': 0.0, 'Pshot': 500.0, 'b4': 0.0,
}

Z_VALUES = [0.0, 0.5, 1.0]


# ---------------------------------------------------------------------------
# Generation via classy (CLASS-PT)
# ---------------------------------------------------------------------------

def generate_classpt_reference():
    """Run CLASS-PT via classy and save reference spectra."""
    try:
        from classy import Class
    except ImportError:
        print("ERROR: classy not installed. Run: cd ~/CLASS-PT && pip install .")
        print("Use --synthetic flag to generate synthetic fallback data instead.")
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
            'RSD': 'Yes',
            'P_k_max_h/Mpc': 100.0,
            'z_pk': str(z),
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
        except Exception:
            k_h = np.logspace(-3, np.log10(0.5), 200)

        # Matter spectra (cs0=0)
        pk_mm_real = np.array([cosmo.pk_mm_real(k, z) for k in k_h])

        # Galaxy spectra at fiducial bias
        b = BIAS_PARAMS
        pk_gg_real = np.array([
            cosmo.pk_gg_real(k, z, b['b1'], b['b2'], b['bG2'], b['bGamma3'],
                             b['cs0'], 0., b['Pshot']) for k in k_h
        ])
        pk_gm_real = np.array([
            cosmo.pk_gm_real(k, z, b['b1'], b['b2'], b['bG2'], b['bGamma3'],
                             b['cs0'], 0.) for k in k_h
        ])

        # RSD multipoles — matter
        pk_mm_l0 = np.array([cosmo.pk_mm_l0(k, z) for k in k_h])
        pk_mm_l2 = np.array([cosmo.pk_mm_l2(k, z) for k in k_h])
        pk_mm_l4 = np.array([cosmo.pk_mm_l4(k, z) for k in k_h])

        # RSD multipoles — galaxy
        pk_gg_l0 = np.array([
            cosmo.pk_gg_l0(k, z, b['b1'], b['b2'], b['bG2'], b['bGamma3'],
                           b['cs0'], b['Pshot'], b['b4']) for k in k_h
        ])
        pk_gg_l2 = np.array([
            cosmo.pk_gg_l2(k, z, b['b1'], b['b2'], b['bG2'], b['bGamma3'],
                           b['cs2'], b['b4']) for k in k_h
        ])
        pk_gg_l4 = np.array([
            cosmo.pk_gg_l4(k, z, b['b1'], b['b2'], b['bG2'], b['bGamma3'],
                           b['cs4'], b['b4']) for k in k_h
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
# Synthetic fallback (no classy)
# ---------------------------------------------------------------------------

def generate_synthetic_reference():
    """Generate synthetic reference data for unit testing without classy.

    Uses a power-law P(k) with exponential cutoff to produce reference
    spectra via the clax EPT pipeline itself. This enables testing the
    pipeline structure (shapes, signs, bias scalings) even without
    CLASS-PT installed.

    NOTE: These are NOT validated against CLASS-PT — they are only useful
    for regression testing of the clax-pt code itself.
    """
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from clax.ept import (compute_ept, ept_kgrid, EPTPrecisionParams,
                               pk_mm_real, pk_gm_real, pk_gg_real,
                               pk_mm_l0, pk_mm_l2, pk_mm_l4,
                               pk_gg_l0, pk_gg_l2, pk_gg_l4)
        import jax.numpy as jnp
    except ImportError as e:
        print(f"ERROR: Cannot import clax.ept or JAX: {e}")
        sys.exit(1)

    os.makedirs(OUTDIR, exist_ok=True)

    prec = EPTPrecisionParams()
    k_h = ept_kgrid(prec)

    # Synthetic P(k): k^0.96 exp(-(k/5)^2), amplitude normalized at k=0.1
    def synthetic_pk(k):
        pk = k**0.96 * np.exp(-(k / 5.0)**2)
        pk /= (0.1**0.96 * np.exp(-(0.1 / 5.0)**2))
        return pk * 1e4  # (Mpc/h)^3

    h_val = 0.6736
    f_val = 0.46  # approximate f at z=0.5

    for z, f in zip(Z_VALUES, [0.54, 0.46, 0.37]):
        print(f"Generating synthetic reference at z={z}...")

        pk_lin_h = synthetic_pk(k_h)

        ept = compute_ept(jnp.array(pk_lin_h), jnp.array(k_h), h_val, f, prec)

        b = BIAS_PARAMS
        b1, b2, bG2, bGamma3 = b['b1'], b['b2'], b['bG2'], b['bGamma3']
        cs0, cs2, cs4, Pshot, b4 = b['cs0'], b['cs2'], b['cs4'], b['Pshot'], b['b4']

        outfile = os.path.join(OUTDIR, f'classpt_synthetic_z{z:.1f}.npz')
        np.savez(outfile,
                 k_h=np.array(k_h), z=z, h=h_val, f=f,
                 pk_lin=np.array(pk_lin_h),
                 pk_mm_real=np.array(pk_mm_real(ept, cs0=0.)),
                 pk_gg_real=np.array(pk_gg_real(ept, b1, b2, bG2, bGamma3, cs0, 0., Pshot)),
                 pk_gm_real=np.array(pk_gm_real(ept, b1, b2, bG2, bGamma3, cs0, 0.)),
                 pk_mm_l0=np.array(pk_mm_l0(ept, cs0=0.)),
                 pk_mm_l2=np.array(pk_mm_l2(ept, cs2=0.)),
                 pk_mm_l4=np.array(pk_mm_l4(ept, cs4=0.)),
                 pk_gg_l0=np.array(pk_gg_l0(ept, b1, b2, bG2, bGamma3, cs0, Pshot, b4)),
                 pk_gg_l2=np.array(pk_gg_l2(ept, b1, b2, bG2, bGamma3, cs2, b4)),
                 pk_gg_l4=np.array(pk_gg_l4(ept, b1, b2, bG2, bGamma3, cs4, b4)),
                 **{f'bias_{k}': v for k, v in b.items()},
                 synthetic=True)
        print(f"  Saved {outfile}")

    print("\nDone. Synthetic reference data saved to reference_data/classpt_synthetic_z*.npz")
    print("WARNING: synthetic data is self-referential; use only for regression testing.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate CLASS-PT reference data')
    parser.add_argument('--synthetic', action='store_true',
                        help='Generate synthetic fallback data (no classy required)')
    args = parser.parse_args()

    if args.synthetic:
        generate_synthetic_reference()
    else:
        generate_classpt_reference()
