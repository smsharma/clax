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
import sys
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


def generate_model(params, name, outdir_base, perturbations=True):
    """Generate all reference data for a single cosmological model.

    ``perturbations=False`` skips the per-k perturbation time series (the
    second CLASS run with ``k_output_values``), which only the fiducial
    point's source-function tests read; the grid points carry background,
    thermodynamics and spectra, as ``massive_nu_015`` does.
    """
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
    if perturbations:
        generate_perturbations(None, outdir, params)

    print(f"  Done: {name}")


# Recombination staging (smsharma/clax#48): CLASS RECFAST with
# recfast_Heswitch=1 (Sobolev singlet only), the helium physics clax implements.
# Cosmology names/overrides mirror tests/conftest.py COSMOLOGY_GRID_RECOMB;
# tests/test_recomb_reference.py::test_reference_grid_matches_test_grid fails if
# the two copies drift apart.
NEFF_NCDM = (0.71611 / (4.0 / 11.0) ** (1.0 / 3.0)) ** 4
RECOMB_GRID = {
    "lcdm_fiducial": {},
    "h_high": {"h": 0.6736 * 1.10},
    "omega_b_high": {"omega_b": 0.02237 * 1.20},
    "omega_cdm_low": {"omega_cdm": 0.1200 * 0.80},
    "ns_high": {"n_s": 0.9649 * 1.05},
    "omega_b_low": {"omega_b": 0.02237 * 0.80},
    "YHe_022": {"Y_He": 0.22},
    "YHe_028": {"Y_He": 0.28},
    "Neff_2": {"N_ur": 2.0 - NEFF_NCDM},
    "Neff_4": {"N_ur": 4.0 - NEFF_NCDM},
    "massive_nu_015": {"m_ncdm": 0.15},
}
RECFAST_HESW1_BASE = {
    'h': 0.6736, 'omega_b': 0.02237, 'omega_cdm': 0.1200, 'A_s': 2.1e-9,
    'n_s': 0.9649, 'tau_reio': 0.0544, 'N_ur': 2.0328, 'N_ncdm': 1,
    'm_ncdm': 0.06, 'T_ncdm': 0.71611, 'YHe': 0.2454006,  # = clax CosmoParams default, never BBN
    'recombination': 'RECFAST', 'recfast_Heswitch': 1,
    'tol_background_integration': 1e-12,
    # CLASS's default 1e-6 leaves 3-6e-6 of its own integration error in x_e
    # (2.3e-4 at omega_b - 20%, where the Peebles C = 1 switch fires inside the
    # hydrogen ODE), comparable to the bars the tests set; at 1e-10 the error is
    # <= 3e-9 at all 11 cosmologies (vs 1e-12, measured Oct 19, 2026).
    'tol_thermo_integration': 1e-10,
}


def generate_recfast_hesw1_thermo(name, overrides, outdir_base):
    """CLASS RECFAST (recfast_Heswitch=1) thermodynamics on a 4000-point
    log(1+z) grid over z in [200, 8000], for tests/test_recomb_*.py.

    Interpolated linearly in log(1+z) from CLASS's own table (~28k rows).
    Writes thermo_recfast_hesw1.npz (z, x_e, Tb, g, kappa_dot) and a .json
    provenance file next to it.
    """
    import classy
    from classy import Class

    pars = dict(RECFAST_HESW1_BASE)
    for k, v in overrides.items():
        pars['YHe' if k == 'Y_He' else k] = v
    c = Class()
    c.set(pars)
    c.compute()
    th = c.get_thermodynamics()
    z_tab = np.asarray(th['z'])
    order = np.argsort(z_tab)
    z_out = np.expm1(np.linspace(np.log1p(200.0), np.log1p(8000.0), 4000))

    def on_grid(key):
        return np.interp(np.log1p(z_out), np.log1p(z_tab[order]), np.asarray(th[key])[order])

    outdir = os.path.join(outdir_base, name)
    os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, 'thermo_recfast_hesw1.npz'),
             z=z_out, x_e=on_grid('x_e'), Tb=on_grid('Tb [K]'),
             g=on_grid('g [Mpc^-1]'), kappa_dot=on_grid("kappa' [Mpc^-1]"))
    with open(os.path.join(outdir, 'thermo_recfast_hesw1.json'), 'w') as fh:
        # The 10 references of Oct 2026 came from the CLASS-PT tree's classy, which
        # also reports v3.3.4, so record where classy was imported from.
        json.dump({"class_params": pars,
                   "classy": getattr(classy, '__version__', 'unknown'),
                   "build": f"classy imported from {os.path.dirname(classy.__file__)}",
                   "grid": "4000 points, uniform in log(1+z), z in [200, 8000]"}, fh, indent=1)
    c.struct_cleanup()
    c.empty()
    print(f"  {name}: x_e(z=2500)={np.interp(2500.0, z_out, on_grid('x_e')):.6f}")


def generate_recfast_hesw1_drag(name, overrides, outdir_base):
    """CLASS RECFAST (recfast_Heswitch=1) derived epochs for smsharma/clax#44:
    the baryon drag epoch (kappa_b = 1: z_d, tau_d, rs_d) and CLASS's visibility
    peak (a three-point parabola on its table: z_rec, tau_rec, rs_rec; clax calls
    that epoch z_star). Same CLASS setup as generate_recfast_hesw1_thermo, so the
    two oracles share one recombination history. Writes drag_recfast_hesw1.json.
    """
    import classy
    from classy import Class

    pars = dict(RECFAST_HESW1_BASE)
    for k, v in overrides.items():
        pars['YHe' if k == 'Y_He' else k] = v
    c = Class()
    c.set(pars)
    c.compute()
    names = ['z_d', 'tau_d', 'rs_d', 'z_rec', 'tau_rec', 'rs_rec', 'conformal_age']
    out = {n: float(c.get_current_derived_parameters([n])[n]) for n in names}
    out.update({"class_params": pars,
                "classy": getattr(classy, '__version__', 'unknown'),
                "build": f"classy imported from {os.path.dirname(classy.__file__)}",
                "definitions": "z_d: kappa_b(tau_d) = 1 with dkappa_b/dtau = kappa'/R, R = 3 rho_b / 4 rho_gamma, "
                               "linear interpolation on CLASS's table (thermodynamics.c, "
                               "thermodynamics_calculate_drag_quantities); z_rec: maximum of the "
                               "visibility g by a three-point parabola (thermodynamics_calculate_"
                               "recombination_quantities); rs_* from the background sound horizon at that z"})
    outdir = os.path.join(outdir_base, name)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'drag_recfast_hesw1.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    c.struct_cleanup()
    c.empty()
    print(f"  {name}: z_d={out['z_d']:.4f} rs_d={out['rs_d']:.5f} z_rec={out['z_rec']:.4f} rs_rec={out['rs_rec']:.5f}")


# Named grid points generated with CLASS defaults (HyRec), the same settings as
# the fiducial point, at ``reference_data/<name>/`` without the perturbation
# time series: ``python scripts/generate_class_reference.py --model <name> ...``.
# Names and offsets match tests/conftest.py COSMOLOGY_GRID_NULCDM (0.15, 0.30 eV);
# 0.50 eV is outside the test grid and its files are not checked in.
MODELS = {
    "massive_nu_015": {"m_ncdm": 0.15},
    "massive_nu_030": {"m_ncdm": 0.30},   # added Oct 20, 2026 (smsharma/clax#57 follow-up)
    "massive_nu_050": {"m_ncdm": 0.50},   # beyond the test grid; fit_cl q-bin decision only (not shipped)
    # CLASS's own ncdm momentum quadrature tightened far past its defaults
    # (tol_ncdm_synchronous 1e-3 -> 1e-5, tol_ncdm 1e-3 -> 1e-5, tol_ncdm_newtonian
    # 1e-5 -> 1e-6, tol_ncdm_bg 1e-5 -> 1e-7, l_max_ncdm 17 -> 25): how converged the
    # oracle itself is in this sector (Oct 22, 2026, #57 follow-up; not shipped).
    "massive_nu_030_ncdmtight": {"m_ncdm": 0.30, "tol_ncdm_synchronous": 1e-5, "tol_ncdm": 1e-5,
                                 "tol_ncdm_newtonian": 1e-6, "tol_ncdm_bg": 1e-7, "l_max_ncdm": 25},
    "lcdm_fiducial_ncdmtight": {"tol_ncdm_synchronous": 1e-5, "tol_ncdm": 1e-5,
                                "tol_ncdm_newtonian": 1e-6, "tol_ncdm_bg": 1e-7, "l_max_ncdm": 25},
}


def main():
    outdir = os.path.join(os.path.dirname(__file__), '..', 'reference_data')
    if '--model' in sys.argv:
        names = [a for a in sys.argv[sys.argv.index('--model') + 1:] if not a.startswith('--')]
        unknown = sorted(set(names) - set(MODELS))
        if unknown or not names:
            sys.exit(f"ERROR --model needs names from {sorted(MODELS)}; got {names}")
        for name in names:
            generate_model(dict(FIDUCIAL_PARAMS, **MODELS[name]), name, outdir, perturbations=False)
        return
    if '--recfast-hesw1-thermo' in sys.argv or '--recfast-hesw1-drag' in sys.argv:
        # Optional cosmology names after the flag restrict the run to those points.
        only = [a for a in sys.argv[1:] if not a.startswith('--')]
        unknown = sorted(set(only) - set(RECOMB_GRID))
        if unknown:
            sys.exit(f"ERROR unknown RECOMB_GRID names {unknown}")
        gen = generate_recfast_hesw1_thermo if '--recfast-hesw1-thermo' in sys.argv else generate_recfast_hesw1_drag
        for name, overrides in RECOMB_GRID.items():
            if not only or name in only:
                gen(name, overrides, outdir)
        return
    generate_model(FIDUCIAL_PARAMS, 'lcdm_fiducial', outdir)


if __name__ == '__main__':
    main()
