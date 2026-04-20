#!/usr/bin/env python3
"""Generate CLASS reference data for clax validation.

Requires: pip install classy (CLASS Python wrapper)

Generates .npz files in reference_data/ for each cosmological model.
These are compared against clax output in the test suite.

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
    'output': 'tCl pCl lCl mPk dTk vTk',
    'lensing': 'yes',
    'l_max_scalars': 2500,
    'P_k_max_1/Mpc': 50.0,
    'z_max_pk': 3.0,
    # High precision
    'tol_background_integration': 1e-12,
}


def _background_value_at_z(bg, z, key):
    """Return a background-table quantity interpolated to redshift ``z``."""
    z_bg = np.asarray(bg['z'])[::-1]
    value_bg = np.asarray(bg[key])[::-1]
    return float(np.interp(z, z_bg, value_bg))


def _equality_redshift_and_tau(bg):
    """Return ``(z_eq, tau_eq)`` from the background matter-radiation crossing."""
    z_bg = np.asarray(bg['z'])[::-1]
    tau_bg = np.asarray(bg['conf. time [Mpc]'])[::-1]
    rho_ncdm = np.asarray(bg['(.)rho_ncdm[0]'])[::-1]
    p_ncdm = np.asarray(bg['(.)p_ncdm[0]'])[::-1]
    rho_m = np.asarray(bg['(.)rho_b'] + bg['(.)rho_cdm'])[::-1] + (rho_ncdm - 3.0 * p_ncdm)
    rho_r = np.asarray(bg['(.)rho_g'] + bg['(.)rho_ur'])[::-1] + 3.0 * p_ncdm
    ratio = rho_m / rho_r
    idx = int(np.argmin(np.abs(np.log(ratio))))

    if 0 < idx < len(z_bg) - 1:
        if ratio[idx] >= 1.0:
            left, right = idx, idx + 1
        else:
            left, right = idx - 1, idx
        z_eq = float(np.interp(1.0, [ratio[right], ratio[left]], [z_bg[right], z_bg[left]]))
    else:
        z_eq = float(z_bg[idx])

    tau_eq = float(np.interp(z_eq, z_bg, tau_bg))
    return z_eq, tau_eq


def _visibility_peak_quantities(cosmo, bg, th):
    """Return ``(z_star, tau_star, rs_star, da_star)`` from the visibility peak."""
    z_th = np.asarray(th['z'])[::-1]
    g_th = np.asarray(th['g [Mpc^-1]'])[::-1]
    z_star = float(z_th[int(np.argmax(g_th))])
    tau_star = _background_value_at_z(bg, z_star, 'conf. time [Mpc]')
    rs_star = _background_value_at_z(bg, z_star, 'comov.snd.hrz.')
    da_star = float(cosmo.angular_distance(z_star))
    return z_star, tau_star, rs_star, da_star


def _compute_pk_components(cosmo, k_eval, z):
    """Return CLASS linear total-matter and cb-only spectra at redshift ``z``.

    ``pk_lin`` from CLASS is treated as the total-matter ``P_m`` reference.
    ``P_cb`` is derived from transfer functions as ``P_m * (delta_cb / delta_m)^2``
    using CLASS background density weights at the same redshift.
    """
    pk_m = np.array([cosmo.pk_lin(k, z) for k in k_eval])
    transfer = cosmo.get_transfer(z)
    bg = cosmo.get_background()

    k_transfer = np.asarray(transfer['k (h/Mpc)']) * cosmo.h()
    rho_b = _background_value_at_z(bg, z, '(.)rho_b')
    rho_cdm = _background_value_at_z(bg, z, '(.)rho_cdm')

    delta_b = np.asarray(transfer['d_b'])
    delta_cdm = np.asarray(transfer['d_cdm'])
    delta_tot = np.asarray(transfer['d_tot'])
    delta_cb = (rho_b * delta_b + rho_cdm * delta_cdm) / (rho_b + rho_cdm)
    cb_to_m_ratio = (delta_cb / delta_tot) ** 2
    pk_cb = pk_m * np.interp(np.log(k_eval), np.log(k_transfer), cb_to_m_ratio)
    return pk_m, pk_cb


def generate_background(cosmo, outdir):
    """Extract and save background quantities."""
    # Get full background table
    bg = cosmo.get_background()
    th = cosmo.get_thermodynamics()

    # Extract key quantities at specific z values
    z_test = np.concatenate([
        np.linspace(0, 10, 100),
        np.logspace(1, 4, 100),
    ])
    z_test = np.unique(np.sort(z_test))

    H_z = np.array([cosmo.Hubble(z) for z in z_test])
    DA_z = np.array([cosmo.angular_distance(z) for z in z_test])
    DL_z = np.array([cosmo.luminosity_distance(z) for z in z_test])
    Dchi_z = (1.0 + z_test) * DA_z
    D_z = np.array([cosmo.scale_independent_growth_factor(z) for z in z_test])
    f_z = np.array([cosmo.scale_independent_growth_factor_f(z) for z in z_test])

    # Derived parameters supported directly by this classy build
    derived = {}
    for name in [
        'z_rec', 'tau_rec', 'rs_rec',
        'z_d', 'tau_d', 'rs_d',
        'age', 'conformal_age',
        'z_reio', 'tau_reio',
        'Neff',
    ]:
        derived[name] = float(cosmo.get_current_derived_parameters([name])[name])

    z_eq, tau_eq = _equality_redshift_and_tau(bg)
    z_star, tau_star, rs_star, da_star = _visibility_peak_quantities(cosmo, bg, th)
    derived.update({
        'z_eq': z_eq,
        'tau_eq': tau_eq,
        'z_star': z_star,
        'tau_star': tau_star,
        'rs_star': rs_star,
        'da_star': da_star,
    })

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
    H0 = float(cosmo.Hubble(0))
    H0_sq = H0 * H0
    rho_b_0 = _background_value_at_z(bg, 0.0, '(.)rho_b')
    rho_cdm_0 = _background_value_at_z(bg, 0.0, '(.)rho_cdm')
    rho_ncdm_0 = _background_value_at_z(bg, 0.0, '(.)rho_ncdm[0]')
    rho_g_0 = _background_value_at_z(bg, 0.0, '(.)rho_g')
    rho_ur_0 = _background_value_at_z(bg, 0.0, '(.)rho_ur')
    rho_lambda_0 = _background_value_at_z(bg, 0.0, '(.)rho_lambda')

    scalars = {
        'H0': H0,
        'h': float(cosmo.h()),
        'Omega_b': rho_b_0 / H0_sq,
        'Omega_cdm': rho_cdm_0 / H0_sq,
        'Omega_m': (rho_b_0 + rho_cdm_0 + rho_ncdm_0) / H0_sq,
        'Omega_r': (rho_g_0 + rho_ur_0) / H0_sq,
        'Omega_Lambda': rho_lambda_0 / H0_sq,
        'Omega_nu': rho_ncdm_0 / H0_sq,
        'Omega_g': rho_g_0 / H0_sq,
        'T_cmb': float(cosmo.T_cmb()),
        'Neff': float(derived['Neff']),
        'age_Gyr': float(derived['age']),
        'z_eq': float(derived['z_eq']),
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
    """Extract and save C_l spectra, ``P_m(k)``, and ``P_cb(k)``."""
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

    # P(k) at multiple redshifts
    z_pk = [0.0, 0.5, 1.0, 2.0]
    pk_payload = {}
    for z in z_pk:
        pk_m, pk_cb = _compute_pk_components(cosmo, k_test, z)
        suffix = f'z{z}'
        pk_payload[f'pk_m_{suffix}'] = pk_m
        pk_payload[f'pk_cb_{suffix}'] = pk_cb
        pk_payload[f'pk_{suffix}'] = pk_m
        if z == 0.0:
            pk_payload['pk_m_lin_z0'] = pk_m
            pk_payload['pk_cb_lin_z0'] = pk_cb
            pk_payload['pk_lin_z0'] = pk_m

    np.savez(
        os.path.join(outdir, 'pk.npz'),
        k=k_test,
        **pk_payload,
    )

    print(f"  C_l: l=2..2500, types: TT,EE,TE,BB,PP,TP")
    print(f"  P(k): {len(k_test)} k-points, z={z_pk} (stored as P_m and P_cb)")


def generate_perturbations(cosmo, outdir, params):
    """Extract perturbation variables at specific k values for source comparison.

    Uses CLASS k_output_values to get time-series of perturbation variables.
    This enables direct comparison of source function components against clax.

    For this perturbation-layer reference we disable the CLASS ncdm fluid
    approximation so the stored `delta_ncdm`, `theta_ncdm`, and `shear_ncdm`
    are apples-to-apples with clax's approximation-free hierarchy.
    """
    # Need a separate CLASS run with k_output_values
    from classy import Class

    k_output = [0.01, 0.05, 0.1]  # Mpc^-1

    cosmo2 = Class()
    params2 = dict(params)
    params2['k_output_values'] = ','.join(f'{k:.4f}' for k in k_output)
    params2['ncdm_fluid_approximation'] = 3  # ncdmfa_none
    params2['l_max_ncdm'] = 17
    # Need perturbation output
    if 'output' in params2:
        if 'dTk' not in params2['output']:
            params2['output'] = params2['output'] + ' dTk'
        if 'vTk' not in params2['output']:
            params2['output'] = params2['output'] + ' vTk'
    cosmo2.set(params2)

    try:
        cosmo2.compute()

        pert = cosmo2.get_perturbations()
        scalar_pert = pert.get('scalar', [])

        for i, k in enumerate(k_output):
            if i < len(scalar_pert):
                p = scalar_pert[i]
                data = {}
                for key in p.keys():
                    data[key.replace(' ', '_').replace('[', '').replace(']', '').replace('/', '_')] = np.array(p[key])

                np.savez(
                    os.path.join(outdir, f'perturbations_k{k:.4f}.npz'),
                    k=k,
                    **data,
                )
                print(f"  Perturbations k={k}: {len(p.get('tau [Mpc]', []))} tau-points, {len(data)} variables")

        cosmo2.struct_cleanup()
        cosmo2.empty()
    except Exception as e:
        print(f"  Warning: perturbation output failed: {e}")
        print(f"  (This requires CLASS compiled with perturbation output support)")


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

    # Generate perturbation time-series (separate run with k_output_values)
    generate_perturbations(None, outdir, params)

    print(f"  Done: {name}")


def main():
    outdir = os.path.join(os.path.dirname(__file__), '..', 'reference_data')
    generate_model(FIDUCIAL_PARAMS, 'lcdm_fiducial', outdir)


if __name__ == '__main__':
    main()
