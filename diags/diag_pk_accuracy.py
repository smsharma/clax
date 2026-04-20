"""P(k) accuracy diagnostic: top-down + bottom-up investigation.

Identifies why linear matter power spectrum is 1-4% off vs CLASS.

Sections:
  1 (TOP-DOWN)    Full k-range error map: P(k)_clax / P(k)_class at 10 k values
  2 (TOP-DOWN)    Multi-z growth rate: P(k,z) at z=0,0.5,1,2 vs CLASS
  3 (BOTTOM-UP)   Primordial spectrum normalization (analytical check)
  4 (BOTTOM-UP)   Individual δ_cdm, δ_b vs CLASS perturbation reference
  5 (BOTTOM-UP)   Neutrino (ncdm) contribution: how much does δ_ncdm close the gap?
  6 (BOTTOM-UP)   ODE tolerance sensitivity at k=0.05
  7 (BOTTOM-UP)   Background density weighting sanity check

Decision tree (see plan for details):
  - Constant error across k → amplitude/normalization bug
  - Error largest at high k → missing ν free-streaming suppression
  - Error largest at low k → metric perturbation (η, h') or IC error
  - δ_cdm off → CDM evolution wrong; δ_b off → baryon-photon coupling wrong

Usage:
    python diags/diag_pk_accuracy.py         # full run (~5-10 min on V100)
    python diags/diag_pk_accuracy.py --fast  # skip tolerance sweep
"""
import sys, time, argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax

import clax
from clax.params import CosmoParams, PrecisionParams
from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve
from clax.perturbations import (
    perturbations_solve, _build_indices, _adiabatic_ic,
    _perturbation_rhs, _ncdm_quadrature, _ncdm_integrated_moments,
    _ncdm_fluid_mode_code,
)

parser = argparse.ArgumentParser()
parser.add_argument('--fast', action='store_true', help='Skip tolerance sweep (Sec 6)')
args = parser.parse_args()

# ── Reference data ──────────────────────────────────────────────────────────
REFERENCE_DIR = REPO_ROOT / 'reference_data' / 'lcdm_fiducial'
ref_pk = np.load(REFERENCE_DIR / 'pk.npz')
k_ref = ref_pk['k']
pk_m_z0 = ref_pk['pk_m_lin_z0'] if 'pk_m_lin_z0' in ref_pk.files else ref_pk['pk_lin_z0']
pk_m_z05 = ref_pk['pk_m_z0.5'] if 'pk_m_z0.5' in ref_pk.files else ref_pk['pk_z0.5']
pk_m_z10 = ref_pk['pk_m_z1.0'] if 'pk_m_z1.0' in ref_pk.files else ref_pk['pk_z1.0']
pk_m_z20 = ref_pk['pk_m_z2.0'] if 'pk_m_z2.0' in ref_pk.files else ref_pk['pk_z2.0']
pk_cb_z0 = ref_pk['pk_cb_lin_z0'] if 'pk_cb_lin_z0' in ref_pk.files else pk_m_z0
has_pk_cb_ref = 'pk_cb_lin_z0' in ref_pk.files

ref_pt05 = np.load(REFERENCE_DIR / 'perturbations_k0.0500.npz')
ref_pt01 = np.load(REFERENCE_DIR / 'perturbations_k0.0100.npz')
ref_pt10 = np.load(REFERENCE_DIR / 'perturbations_k0.1000.npz')

def interp_loglog(k, k_arr, pk_arr):
    """Interpolate P(k) in log-log space."""
    return float(np.exp(np.interp(np.log(k), np.log(k_arr), np.log(pk_arr))))


def batch_tau_ini(prec):
    """Return the fixed batch-solver initial conformal time."""
    return 0.01 / prec.pt_k_max_cl


def direct_tau_ini(k):
    """Return the production single-mode initial conformal time."""
    return min(0.5, 0.01 / k)

# ── Pipeline setup ────────────────────────────────────────────────────────────
params = CosmoParams()
prec   = PrecisionParams.planck_fast()   # ~75s on V100, l_max=50, rtol=1e-6

t0 = time.time()
bg = background_solve(params, prec)
th = thermodynamics_solve(params, prec, bg)
pt = perturbations_solve(params, prec, bg, th)
print(f"Pipeline: {time.time()-t0:.0f}s   k_modes={len(pt.k_grid)}", flush=True)

t_api = time.time()
pk_api = clax.compute_pk_interpolator(params, prec)
print(
    f"Public P(k) interpolator: {time.time()-t_api:.0f}s   "
    f"solve_k_modes={len(pk_api.solve_k_grid)}",
    flush=True,
)

A_s    = float(jnp.exp(params.ln10A_s) / 1e10)
k_grid = np.array(pt.k_grid)
tau_end = float(bg.conformal_age) * 0.999

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TOP-DOWN: Full k-range error map
# ════════════════════════════════════════════════════════════════════════════
print("\n=== SEC 1: Full k-range error map (public table P_m vs CLASS P_m) ===")
k_probe = [0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0]
pk_probe_clax = np.asarray(pk_api.pk(jnp.asarray(k_probe), z=0.0))
ratios = []
for k, pk_c in zip(k_probe, pk_probe_clax, strict=True):
    pk_cl = interp_loglog(k, k_ref, pk_m_z0)
    r = pk_c / pk_cl
    ratios.append(r)
    print(f"  k={k:.3f}: clax/class={r:.4f}  ({100*(r-1):+.2f}%)")

ratios = np.array(ratios)
worst_k = k_probe[np.argmax(np.abs(ratios - 1))]
print(f"  => max_err={100*np.max(np.abs(ratios-1)):.2f}% at k={worst_k:.3f}, "
      f"mean={100*np.mean(ratios-1):+.2f}%, bias={'over' if np.mean(ratios)>1 else 'under'}")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TOP-DOWN: Multi-z growth rate
# ════════════════════════════════════════════════════════════════════════════
print("\n=== SEC 2: Multi-z growth rate (public interpolator P(k,z) vs CLASS) ===")
k_test = jnp.array([0.05, 0.1, 0.3])

for zz, pk_cls_z, label in [
    (0.5, pk_m_z05, "z=0.5"),
    (1.0, pk_m_z10, "z=1.0"),
    (2.0, pk_m_z20, "z=2.0"),
]:
    pk_clax_z = np.asarray(pk_api.pk(k_test, z=zz))
    for ki, k in enumerate(np.array(k_test)):
        pk_c  = pk_clax_z[ki]
        pk_cl = interp_loglog(k, k_ref, pk_cls_z)
        r = pk_c / pk_cl
        print(f"  {label} k={k:.2f}: ratio={r:.4f}  ({100*(r-1):+.2f}%)")

# Redshift-independence check: ratio of ratios at z=0.5 vs z=0
k_check = 0.05
pk_ratio_clax = float(
    pk_api.pk(jnp.array([k_check]), z=0.5)[0]
    / pk_api.pk(jnp.array([k_check]), z=0.0)[0]
)
pk_ratio_cls  = interp_loglog(k_check, k_ref, pk_m_z05) / interp_loglog(k_check, k_ref, pk_m_z0)
print(f"  => P(z=0.5)/P(z=0) at k=0.05: clax={pk_ratio_clax:.4f} class={pk_ratio_cls:.4f} "
      f"delta={100*(pk_ratio_clax/pk_ratio_cls - 1):+.2f}%")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BOTTOM-UP: Primordial spectrum normalization
# ════════════════════════════════════════════════════════════════════════════
print("\n=== SEC 3: Primordial spectrum normalization ===")
A_s_check = np.exp(3.044) / 1e10   # reference value for ln10A_s=3.044
print(f"  A_s: actual={A_s:.8e}  ref(ln10As=3.044)={A_s_check:.8e}  "
      f"ratio={A_s/A_s_check:.6f}")
print(f"  k_pivot={params.k_pivot:.4f} Mpc^-1  (CLASS default=0.05)")
print(f"  n_s={params.n_s:.6f}  (Planck fiducial=0.9649)")
PR_kpivot = 2*np.pi**2 / params.k_pivot**3 * A_s
print(f"  P_R(k_pivot) = {PR_kpivot:.6e} Mpc^3  (CLASS convention check)")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — BOTTOM-UP: Individual δ_cdm, δ_b vs CLASS at τ_end
# ════════════════════════════════════════════════════════════════════════════
print("\n=== SEC 4: Individual δ_cdm, δ_b, δ_ncdm vs CLASS ===")

# Set up ncdm args
n_q_ncdm = prec.ncdm_q_size if params.N_ncdm > 0 else 0
l_max = prec.pt_l_max_g
idx_pert = _build_indices(l_max, prec.pt_l_max_pol_g, prec.pt_l_max_ur,
                          n_q_ncdm, prec.pt_l_max_ncdm)
if n_q_ncdm > 0:
    q_ncdm, w_ncdm, M_ncdm, dlnf0_ncdm = _ncdm_quadrature(params, prec)
else:
    q_ncdm = w_ncdm = dlnf0_ncdm = jnp.zeros(1)
    M_ncdm = 0.0
args_ncdm = (q_ncdm, w_ncdm, M_ncdm, dlnf0_ncdm)
ncdmfa_mode_code = _ncdm_fluid_mode_code(prec.ncdm_fluid_approximation)
ncdmfa_trigger = prec.ncdm_fluid_trigger_tau_over_tau_k

for k_val, ref_pert in [(0.01, ref_pt01), (0.05, ref_pt05), (0.1, ref_pt10)]:
    # CLASS values at τ_end
    tau_cls = ref_pert['tau_Mpc']
    dc_cls = float(np.interp(tau_end, tau_cls, ref_pert['delta_cdm']))
    db_cls = float(np.interp(tau_end, tau_cls, ref_pert['delta_b']))
    dn_cls = float(np.interp(tau_end, tau_cls, ref_pert['delta_ncdm0'])) if 'delta_ncdm0' in ref_pert else None

    # clax single-k ODE — use the production direct-path tau_ini rule
    tau_ini_k = direct_tau_ini(k_val)
    y0 = _adiabatic_ic(k_val, jnp.array(tau_ini_k), bg, params, idx_pert,
                       idx_pert['n_eq'], args_ncdm=args_ncdm)
    ode_args = (k_val, bg, th, params, idx_pert, l_max, prec.pt_l_max_pol_g,
                prec.pt_l_max_ur, ncdmfa_mode_code, ncdmfa_trigger,
                q_ncdm, w_ncdm, M_ncdm, dlnf0_ncdm)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(_perturbation_rhs), solver=diffrax.Kvaerno5(),
        t0=tau_ini_k, t1=tau_end, dt0=tau_ini_k * 0.1, y0=y0,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=diffrax.PIDController(rtol=prec.pt_ode_rtol, atol=prec.pt_ode_atol),
        adjoint=diffrax.RecursiveCheckpointAdjoint(), max_steps=prec.ode_max_steps, args=ode_args,
    )
    y_f = np.array(sol.ys[-1])

    dc_clax = float(y_f[idx_pert['delta_cdm']])
    db_clax = float(y_f[idx_pert['delta_b']])

    print(f"  k={k_val:.3f}: δ_cdm clax/class={dc_clax/dc_cls:.4f}  "
          f"δ_b clax/class={db_clax/db_cls:.4f}", end="")

    if n_q_ncdm > 0 and dn_cls is not None:
        rho_delta_n, _, _, _, rho_unnorm_n, _ = _ncdm_integrated_moments(
            sol.ys[-1], q_ncdm, w_ncdm, M_ncdm, 1.0, k_val, idx_pert)
        dn_clax = float(rho_delta_n / jnp.maximum(rho_unnorm_n, 1e-30))
        print(f"  δ_ncdm clax/class={dn_clax/dn_cls:.4f}", end="")

    print()

# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — BOTTOM-UP: Neutrino contribution to δ_m
# ════════════════════════════════════════════════════════════════════════════
print("\n=== SEC 5: Neutrino contribution to δ_m, P_cb, and P_m ===")
rho_b_z0   = float(bg.rho_b_of_loga.evaluate(jnp.array(0.0)))
rho_cdm_z0 = float(bg.rho_cdm_of_loga.evaluate(jnp.array(0.0)))
rho_ncdm_z0 = float(bg.rho_ncdm_of_loga.evaluate(jnp.array(0.0)))
f_nu = rho_ncdm_z0 / (rho_b_z0 + rho_cdm_z0 + rho_ncdm_z0)
print(f"  f_nu=Omega_ncdm/Omega_m={f_nu:.5f}  ({100*f_nu:.3f}%)")
print(f"  Predicted P_cb/P_m impact at high k: +{100*2*f_nu:.2f}% if ν clustering is omitted")

if n_q_ncdm > 0:
    print("  k-mode tests (direct P_cb, direct P_m, CLASS reference):")
    for k_val, ref_pert in [(0.01, ref_pt01), (0.05, ref_pt05), (0.1, ref_pt10)]:
        tau_ini_k = direct_tau_ini(k_val)
        y0 = _adiabatic_ic(k_val, jnp.array(tau_ini_k), bg, params, idx_pert,
                           idx_pert['n_eq'], args_ncdm=args_ncdm)
        ode_args = (k_val, bg, th, params, idx_pert, l_max, prec.pt_l_max_pol_g,
                    prec.pt_l_max_ur, ncdmfa_mode_code, ncdmfa_trigger,
                    q_ncdm, w_ncdm, M_ncdm, dlnf0_ncdm)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(_perturbation_rhs), solver=diffrax.Kvaerno5(),
            t0=tau_ini_k, t1=tau_end, dt0=tau_ini_k * 0.1, y0=y0,
            saveat=diffrax.SaveAt(t1=True),
            stepsize_controller=diffrax.PIDController(rtol=prec.pt_ode_rtol, atol=prec.pt_ode_atol),
            adjoint=diffrax.RecursiveCheckpointAdjoint(), max_steps=prec.ode_max_steps, args=ode_args,
        )
        y_f = sol.ys[-1]
        dc = float(y_f[idx_pert['delta_cdm']])
        db = float(y_f[idx_pert['delta_b']])
        rho_delta_n, _, _, _, rho_unnorm_n, _ = _ncdm_integrated_moments(
            y_f, q_ncdm, w_ncdm, M_ncdm, 1.0, k_val, idx_pert)
        dn = float(rho_delta_n / jnp.maximum(rho_unnorm_n, 1e-30))

        dm_cb   = (rho_b_z0 * db + rho_cdm_z0 * dc) / (rho_b_z0 + rho_cdm_z0)
        dm_full = (rho_b_z0 * db + rho_cdm_z0 * dc + rho_ncdm_z0 * dn) / (rho_b_z0 + rho_cdm_z0 + rho_ncdm_z0)

        prim = 2*np.pi**2 / k_val**3 * A_s * (k_val/params.k_pivot)**(params.n_s-1)
        pk_cb   = prim * dm_cb**2
        pk_full = prim * dm_full**2
        pk_cls_m = interp_loglog(k_val, k_ref, pk_m_z0)
        pk_cls_cb = interp_loglog(k_val, k_ref, pk_cb_z0)

        r_cb = pk_cb / pk_cls_cb
        r_full = pk_full / pk_cls_m
        cb_label = 'P_cb/P_cb_class' if has_pk_cb_ref else 'P_cb/P_m_class'
        print(f"  k={k_val:.3f}: {cb_label}={r_cb:.4f} ({100*(r_cb-1):+.2f}%)  "
              f"P_m/P_m_class={r_full:.4f} ({100*(r_full-1):+.2f}%)  "
              f"nu_improvement={100*((pk_full / pk_cls_m) - (pk_cb / pk_cls_m)):+.2f}pp")
else:
    print("  SKIPPED: ncdm_q_size=0 in preset (run with ncdm_q_size=5)")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — BOTTOM-UP: ODE tolerance sensitivity at k=0.05
# ════════════════════════════════════════════════════════════════════════════
if not args.fast:
    print("\n=== SEC 6: ODE tolerance sensitivity at k=0.05 ===")
    pk_cls_k05 = interp_loglog(0.05, k_ref, pk_m_z0)
    tau_ini_k05 = direct_tau_ini(0.05)
    for rtol, atol in [(1e-3, 1e-6), (1e-4, 1e-7), (1e-5, 1e-8), (1e-6, 1e-9)]:
        y0_ = _adiabatic_ic(0.05, jnp.array(tau_ini_k05), bg, params, idx_pert,
                            idx_pert['n_eq'], args_ncdm=args_ncdm)
        ode_args_ = (0.05, bg, th, params, idx_pert, l_max, prec.pt_l_max_pol_g,
                     prec.pt_l_max_ur, ncdmfa_mode_code, ncdmfa_trigger,
                     q_ncdm, w_ncdm, M_ncdm, dlnf0_ncdm)
        sol_ = diffrax.diffeqsolve(
            diffrax.ODETerm(_perturbation_rhs), solver=diffrax.Kvaerno5(),
            t0=tau_ini_k05, t1=tau_end, dt0=tau_ini_k05*0.1, y0=y0_,
            saveat=diffrax.SaveAt(t1=True),
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            adjoint=diffrax.RecursiveCheckpointAdjoint(), max_steps=131072, args=ode_args_,
        )
        y_f_ = sol_.ys[-1]
        dc_ = float(y_f_[idx_pert['delta_cdm']])
        db_ = float(y_f_[idx_pert['delta_b']])
        rho_delta_n_, _, _, _, rho_unnorm_n_, _ = _ncdm_integrated_moments(
            y_f_, q_ncdm, w_ncdm, M_ncdm, 1.0, 0.05, idx_pert)
        dn_ = float(rho_delta_n_ / jnp.maximum(rho_unnorm_n_, 1e-30))
        dm_full_ = (rho_b_z0 * db_ + rho_cdm_z0 * dc_ + rho_ncdm_z0 * dn_) / (rho_b_z0 + rho_cdm_z0 + rho_ncdm_z0)
        prim_  = 2*np.pi**2 / 0.05**3 * A_s * (0.05/params.k_pivot)**(params.n_s-1)
        pk_c_  = prim_ * dm_full_**2
        r_     = pk_c_ / pk_cls_k05
        print(f"  rtol={rtol:.0e}: P_m(k=0.05)/P_m,class={r_:.5f}  ({100*(r_-1):+.3f}%)")
else:
    print("\n=== SEC 6: Tolerance sweep SKIPPED (use --no-fast to enable) ===")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — BOTTOM-UP: Density weighting sanity check
# ════════════════════════════════════════════════════════════════════════════
print("\n=== SEC 7: Background density weighting at z=0 ===")
print(f"  tau_ini(batch)={batch_tau_ini(prec):.6e}  tau_ini(direct,k=0.05)={direct_tau_ini(0.05):.6e}")
H0 = float(bg.H_of_loga.evaluate(jnp.array(0.0)))
omega_b_clax   = rho_b_z0 / H0**2
omega_cdm_clax = rho_cdm_z0 / H0**2
omega_ncdm_clax = rho_ncdm_z0 / H0**2
print(f"  omega_b   clax={omega_b_clax:.6f}  params={params.omega_b:.6f}  "
      f"ratio={omega_b_clax/params.omega_b:.6f}")
print(f"  omega_cdm clax={omega_cdm_clax:.6f}  params={params.omega_cdm:.6f}  "
      f"ratio={omega_cdm_clax/params.omega_cdm:.6f}")
print(f"  omega_ncdm(bg)={omega_ncdm_clax:.6f}  f_nu={f_nu:.5f}")

# Weight check: if one compares clustered cb matter to total matter, the expected
# large-k offset is set by the neutrino background fraction.
f_cb = (rho_b_z0 + rho_cdm_z0) / (rho_b_z0 + rho_cdm_z0 + rho_ncdm_z0)
print(f"  f_cb = (Omega_b+Omega_cdm)/Omega_m = {f_cb:.5f}")
print(f"  => expected P_cb/P_m large-k ratio ≈ 1/{f_cb:.4f}^2 = {(1/f_cb)**2:.4f}")
print("  => current public table/interpolator path already includes ncdm, so Sec 1 is a P_m-to-P_m comparison")

print("\n=== DONE ===")
