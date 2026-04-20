#!/usr/bin/env python3
"""Generate CLASS reference data at varied LCDM parameter points.

For multi-cosmology validation: omega_b ±20%, omega_cdm ±20%, h ±10%,
n_s ±5%, tau_reio ±30%.

Generates unlensed + lensed C_l plus explicit total-matter ``P_m`` and
cb-only ``P_cb`` spectra at each point.
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
    'output': 'tCl pCl lCl mPk dTk vTk',
    'lensing': 'yes',
    'l_max_scalars': 2500,
    'P_k_max_1/Mpc': 50.0,
    'z_max_pk': 3.0,
    'tol_background_integration': 1e-12,
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


def _background_density_at_z(bg, z, key):
    """Return a background density interpolated to redshift ``z``."""
    z_bg = np.asarray(bg['z'])[::-1]
    rho_bg = np.asarray(bg[key])[::-1]
    return float(np.interp(z, z_bg, rho_bg))


def _compute_pk_components(cosmo, k_eval, z):
    """Return CLASS linear total-matter and cb-only spectra at redshift ``z``."""
    pk_m = np.array([cosmo.pk_lin(k, z) for k in k_eval])
    transfer = cosmo.get_transfer(z)
    bg = cosmo.get_background()

    k_transfer = np.asarray(transfer['k (h/Mpc)']) * cosmo.h()
    rho_b = _background_density_at_z(bg, z, '(.)rho_b')
    rho_cdm = _background_density_at_z(bg, z, '(.)rho_cdm')

    delta_b = np.asarray(transfer['d_b'])
    delta_cdm = np.asarray(transfer['d_cdm'])
    delta_tot = np.asarray(transfer['d_tot'])
    delta_cb = (rho_b * delta_b + rho_cdm * delta_cdm) / (rho_b + rho_cdm)
    cb_to_m_ratio = (delta_cb / delta_tot) ** 2
    pk_cb = pk_m * np.interp(np.log(k_eval), np.log(k_transfer), cb_to_m_ratio)
    return pk_m, pk_cb


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
    pk_m, pk_cb = _compute_pk_components(cosmo, k_test, z=0.0)

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
        k=k_test,
        pk_m_lin_z0=pk_m,
        pk_cb_lin_z0=pk_cb,
        pk_lin_z0=pk_m,
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

    print('Generating multi-cosmology reference data...')
    for name, overrides in VARIATIONS.items():
        params = dict(FIDUCIAL)
        params.update(overrides)
        generate_point(params, name, outdir)

    print('\nAll multi-cosmology reference data generated!')


if __name__ == '__main__':
    main()
