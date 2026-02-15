#!/usr/bin/env python3
"""Generate CLASS reference data at varied LCDM parameter points.

For multi-cosmology validation: omega_b ±20%, omega_cdm ±20%, h ±10%,
n_s ±5%, tau_reio ±30%.

Generates unlensed + lensed C_l and P(k) at each point.
"""
import os
import numpy as np

# Planck 2018 best-fit LCDM parameters
FIDUCIAL = {
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
    'output': 'tCl pCl lCl mPk',
    'lensing': 'yes',
    'l_max_scalars': 2500,
    'P_k_max_1/Mpc': 50.0,
    'tol_background_integration': 1e-12,
    'tol_ncdm': 1e-10,
}

# Parameter variations
VARIATIONS = {
    'omega_b_high': {'omega_b': 0.02237 * 1.20},
    'omega_b_low':  {'omega_b': 0.02237 * 0.80},
    'omega_cdm_high': {'omega_cdm': 0.1200 * 1.20},
    'omega_cdm_low':  {'omega_cdm': 0.1200 * 0.80},
    'h_high': {'h': 0.6736 * 1.10},
    'h_low':  {'h': 0.6736 * 0.90},
    'ns_high': {'n_s': 0.9649 * 1.05},
    'ns_low':  {'n_s': 0.9649 * 0.95},
    'tau_high': {'tau_reio': 0.0544 * 1.30},
    'tau_low':  {'tau_reio': 0.0544 * 0.70},
}


def generate_point(params, name, outdir):
    """Generate CLASS reference at one parameter point."""
    from classy import Class

    dirname = os.path.join(outdir, name)
    os.makedirs(dirname, exist_ok=True)

    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    # Unlensed C_l
    cl_raw = cosmo.raw_cl(2500)
    # Lensed C_l
    cl_lens = cosmo.lensed_cl(2500)
    # P(k)
    k_test = np.logspace(-4, np.log10(50), 500)
    pk_lin = np.array([cosmo.pk_lin(k, 0) for k in k_test])

    np.savez(
        os.path.join(dirname, 'cls.npz'),
        ell=cl_raw['ell'], tt=cl_raw['tt'], ee=cl_raw['ee'],
        te=cl_raw['te'], bb=cl_raw['bb'], pp=cl_raw['pp'],
        tp=cl_raw.get('tp', np.zeros_like(cl_raw['tt'])),
    )
    np.savez(
        os.path.join(dirname, 'cls_lensed.npz'),
        ell=cl_lens['ell'], tt=cl_lens['tt'], ee=cl_lens['ee'],
        te=cl_lens['te'], bb=cl_lens['bb'],
    )
    np.savez(
        os.path.join(dirname, 'pk.npz'),
        k=k_test, pk_lin_z0=pk_lin,
    )

    # Save parameter values
    param_vals = {k: float(params[k]) for k in
                  ['h', 'omega_b', 'omega_cdm', 'A_s', 'n_s', 'tau_reio']}
    np.savez(os.path.join(dirname, 'params.npz'), **param_vals)

    cosmo.struct_cleanup()
    cosmo.empty()

    print(f"  {name}: done (omega_b={params['omega_b']:.5f}, "
          f"omega_cdm={params['omega_cdm']:.4f}, h={params['h']:.4f}, "
          f"n_s={params['n_s']:.4f}, tau={params['tau_reio']:.4f})")


def main():
    outdir = os.path.join(os.path.dirname(__file__), '..', 'reference_data')

    print("Generating multi-cosmology reference data...")
    for name, overrides in VARIATIONS.items():
        params = dict(FIDUCIAL)
        params.update(overrides)
        generate_point(params, name, outdir)

    print("\nAll multi-cosmology reference data generated!")


if __name__ == '__main__':
    main()
