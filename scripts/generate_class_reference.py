#!/usr/bin/env python3
"""Generate CLASS reference data for jaxCLASS validation.

Requires: pip install classy (CLASS Python wrapper)

Generates .npz files in reference_data/ for each cosmological model.
These are compared against jaxCLASS output in the test suite.

Usage:
    python scripts/generate_class_reference.py
"""

import json
import os
import numpy as np

# Planck 2018 best-fit LCDM parameters
FIDUCIAL_PARAMS = {
    'h': 0.6736,
    'omega_b': 0.02237,
    'omega_cdm': 0.1200,
    'A_s': 2.1e-9,
    'n_s': 0.9649,
    'tau_reio': 0.0544,
    # Neutrinos
    'N_ur': 2.0328,
    'N_ncdm': 1,
    'm_ncdm': 0.06,
    'T_ncdm': 0.71611,
    # Output
    'output': 'tCl pCl lCl mPk',
    'lensing': 'yes',
    'l_max_scalars': 2500,
    'P_k_max_1/Mpc': 50.0,
    'z_max_pk': 3.0,
    # High precision
    'tol_background_integration': 1e-12,
    'tol_ncdm': 1e-10,
}


def generate_background(cosmo, outdir):
    """Extract and save background quantities."""
    # Get full background table
    bg = cosmo.get_background()

    # Extract key quantities at specific z values
    z_test = np.concatenate([
        np.linspace(0, 10, 100),
        np.logspace(1, 4, 100),
    ])
    z_test = np.unique(np.sort(z_test))

    H_z = np.array([cosmo.Hubble(z) for z in z_test])
    DA_z = np.array([cosmo.angular_distance(z) for z in z_test])
    DL_z = np.array([cosmo.luminosity_distance(z) for z in z_test])
    Dchi_z = np.array([cosmo.comoving_distance(z) for z in z_test])
    D_z = np.array([cosmo.scale_independent_growth_factor(z) for z in z_test])
    f_z = np.array([cosmo.scale_independent_growth_factor_f(z) for z in z_test])

    # Derived parameters
    derived = cosmo.get_current_derived_parameters([
        'z_eq', 'tau_eq', 'z_rec', 'tau_rec', 'rs_rec',
        'z_star', 'tau_star', 'rs_star', 'da_star',
        'z_d', 'tau_d', 'rs_d',
        'age', 'conformal_age',
        'z_reio', 'tau_reio',
        'Neff',
    ])

    np.savez(
        os.path.join(outdir, 'background.npz'),
        z=z_test,
        H=H_z,
        D_A=DA_z,
        D_L=DL_z,
        chi=Dchi_z,
        D=D_z,
        f=f_z,
        # Full background table
        bg_z=bg['z'],
        bg_H=bg['H [1/Mpc]'],
        bg_conf_time=bg['conf. time [Mpc]'],
        bg_D=bg.get('gr.fac. D', np.array([])),
        bg_f=bg.get('gr.fac. f', np.array([])),
    )

    # Save derived as JSON
    with open(os.path.join(outdir, 'derived.json'), 'w') as f:
        json.dump({k: float(v) for k, v in derived.items()}, f, indent=2)

    # Also save key scalars
    scalars = {
        'H0': float(cosmo.Hubble(0)),
        'h': float(cosmo.h()),
        'Omega_b': float(cosmo.Omega_b()),
        'Omega_cdm': float(cosmo.Omega_cdm()),
        'Omega_m': float(cosmo.Omega_m()),
        'Omega_r': float(cosmo.Omega_r()),
        'Omega_Lambda': float(cosmo.Omega_Lambda()),
        'Omega_nu': float(cosmo.Omega_nu()),
        'Omega_g': float(cosmo.Omega_g()),
        'T_cmb': float(cosmo.T_cmb()),
        'Neff': float(cosmo.Neff()),
        'age_Gyr': float(cosmo.age()),
        'z_eq': float(cosmo.z_eq()),
        'conformal_age': float(derived['conformal_age']),
    }
    with open(os.path.join(outdir, 'scalars.json'), 'w') as f:
        json.dump(scalars, f, indent=2)

    print(f"  Background: {len(z_test)} z-points, {len(bg['z'])} table rows")
    print(f"  H0 = {scalars['H0']:.6e} Mpc^-1")
    print(f"  conformal_age = {scalars['conformal_age']:.2f} Mpc")
    print(f"  age = {scalars['age_Gyr']:.4f} Gyr")
    print(f"  z_eq = {scalars['z_eq']:.1f}")


def generate_thermodynamics(cosmo, outdir):
    """Extract and save thermodynamics quantities."""
    th = cosmo.get_thermodynamics()

    # Individual z evaluations
    z_test = np.logspace(-1, 4, 200)
    xe_z = np.array([cosmo.ionization_fraction(z) for z in z_test])
    Tb_z = np.array([cosmo.baryon_temperature(z) for z in z_test])

    np.savez(
        os.path.join(outdir, 'thermodynamics.npz'),
        z=z_test,
        x_e=xe_z,
        T_b=Tb_z,
        # Full table
        th_z=th['z'],
        th_x_e=th['x_e'],
        th_kappa_dot=th.get("kappa' [Mpc^-1]", np.array([])),
        th_g=th.get('g [Mpc^-1]', np.array([])),
        th_Tb=th.get('Tb [K]', np.array([])),
    )

    print(f"  Thermodynamics: {len(z_test)} z-points, {len(th['z'])} table rows")


def generate_spectra(cosmo, outdir):
    """Extract and save C_l spectra and P(k)."""
    # Unlensed C_l
    cl_raw = cosmo.raw_cl(2500)
    # Lensed C_l
    cl_lens = cosmo.lensed_cl(2500)

    np.savez(
        os.path.join(outdir, 'cls.npz'),
        ell=cl_raw['ell'],
        tt=cl_raw['tt'],
        ee=cl_raw['ee'],
        te=cl_raw['te'],
        bb=cl_raw['bb'],
        pp=cl_raw['pp'],
        tp=cl_raw['tp'],
    )

    np.savez(
        os.path.join(outdir, 'cls_lensed.npz'),
        ell=cl_lens['ell'],
        tt=cl_lens['tt'],
        ee=cl_lens['ee'],
        te=cl_lens['te'],
        bb=cl_lens['bb'],
    )

    # P(k)
    k_test = np.logspace(-4, np.log10(50), 500)
    pk_lin = np.array([cosmo.pk_lin(k, 0) for k in k_test])

    # P(k) at multiple redshifts
    z_pk = [0.0, 0.5, 1.0, 2.0]
    pk_z = {}
    for z in z_pk:
        pk_z[f'pk_z{z}'] = np.array([cosmo.pk_lin(k, z) for k in k_test])

    np.savez(
        os.path.join(outdir, 'pk.npz'),
        k=k_test,
        pk_lin_z0=pk_lin,
        **pk_z,
    )

    print(f"  C_l: l=2..2500, types: TT,EE,TE,BB,PP,TP")
    print(f"  P(k): {len(k_test)} k-points, z={z_pk}")


def generate_model(params, name, outdir_base):
    """Generate all reference data for a single cosmological model."""
    from classy import Class

    outdir = os.path.join(outdir_base, name)
    os.makedirs(outdir, exist_ok=True)
    print(f"\nGenerating reference data for: {name}")

    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    generate_background(cosmo, outdir)
    generate_thermodynamics(cosmo, outdir)
    generate_spectra(cosmo, outdir)

    cosmo.struct_cleanup()
    cosmo.empty()
    print(f"  Done: {name}")


def main():
    outdir = os.path.join(os.path.dirname(__file__), '..', 'reference_data')
    os.makedirs(outdir, exist_ok=True)

    # Fiducial LCDM
    generate_model(FIDUCIAL_PARAMS, 'lcdm_fiducial', outdir)

    # Massive neutrinos (higher mass)
    params_mnu = dict(FIDUCIAL_PARAMS)
    params_mnu['m_ncdm'] = 0.15
    generate_model(params_mnu, 'massive_nu_015', outdir)

    # w0wa dark energy
    params_w0wa = dict(FIDUCIAL_PARAMS)
    params_w0wa['Omega_fld'] = 0.685
    params_w0wa['w0_fld'] = -0.9
    params_w0wa['wa_fld'] = 0.1
    params_w0wa['Omega_Lambda'] = 0
    # CLASS requires Omega_Lambda=0 when using fluid DE
    # But can't specify both -- need to let CLASS infer
    del params_w0wa['Omega_Lambda']
    generate_model(params_w0wa, 'w0wa_m09_01', outdir)

    print("\nAll reference data generated successfully!")


if __name__ == '__main__':
    main()
