"""Compare perturbation variables at k=0.1 against CLASS reference.

CLASS reference data is in NEWTONIAN gauge (default CLASS output).
Our ODE is in SYNCHRONOUS gauge. We apply the gauge transformation:
  delta_g_N = delta_g_S - 4*aH*alpha
  theta_g_N = theta_g_S + k^2*alpha
  delta_b_N = delta_b_S - 3*aH*alpha
  theta_b_N = theta_b_S + k^2*alpha
  delta_cdm_N = delta_cdm_S - 3*aH*alpha
where alpha = (h' + 6*eta') / (2*k^2)

Also: phi = eta - aH*alpha, psi = -alpha - a^2*pi_tot/k^2 (anisotropic stress)
"""
import sys
sys.path.insert(0, '.')

import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)

print(f"JAX device: {jax.devices()}", flush=True)

from jaxclass import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import _adiabatic_ic, _perturbation_rhs, _build_indices
import diffrax

# Load CLASS reference for k=0.1
ref = np.load('reference_data/lcdm_fiducial/perturbations_k0.1000.npz')
tau_ref = ref['tau_Mpc']

params = CosmoParams()
prec = PrecisionParams.planck_cl()

print("Solving background + thermo...", flush=True)
bg = background_solve(params, prec)
th = thermodynamics_solve(params, prec, bg)

tau_star = float(th.tau_star)
print(f"tau_star = {tau_star:.2f} Mpc", flush=True)

# Build idx and solve single-k ODE
l_max_g = prec.pt_l_max_g
l_max_pol = prec.pt_l_max_pol_g
l_max_ur = prec.pt_l_max_ur

idx = _build_indices(l_max_g, l_max_pol, l_max_ur)
n_eq = idx['n_eq']

k_target = 0.1
k = jnp.array(k_target)
k2 = k_target**2
tau_ini = 0.1 / prec.pt_k_max_cl

print(f"Solving ODE for k={k_target}...", flush=True)
y0 = _adiabatic_ic(k, jnp.array(tau_ini), bg, params, idx, n_eq)

ode_args = (k, bg, th, params, idx, l_max_g, l_max_pol, l_max_ur)

# Sample at tau values near recombination
tau_compare = []
for offset in [-100, -50, -20, -10, -5, 0, 5, 10, 20, 50, 100]:
    tau_val = tau_star + offset
    if tau_val > tau_ini * 1.01 and tau_val < float(bg.conformal_age) * 0.99:
        tau_compare.append(tau_val)
tau_compare_arr = jnp.array(tau_compare)

sol = diffrax.diffeqsolve(
    diffrax.ODETerm(_perturbation_rhs),
    solver=diffrax.Kvaerno5(),
    t0=tau_ini,
    t1=float(bg.conformal_age) * 0.999,
    dt0=tau_ini * 0.1,
    y0=y0,
    saveat=diffrax.SaveAt(ts=tau_compare_arr),
    stepsize_controller=diffrax.PIDController(
        rtol=prec.pt_ode_rtol, atol=prec.pt_ode_atol,
    ),
    adjoint=diffrax.RecursiveCheckpointAdjoint(),
    max_steps=prec.ode_max_steps,
    args=ode_args,
)

print("\n" + "=" * 110)
print("Gauge-transformed (synchronous -> Newtonian) comparison at k=0.1")
print("=" * 110)

# Table header
print(f"\n{'tau':>8s}  {'var':>8s}  {'ours_N':>14s}  {'CLASS_N':>14s}  {'err%':>10s}")
print("-" * 70)

for i, tau_val in enumerate(tau_compare):
    y = sol.ys[i]

    # Get background quantities
    loga = float(bg.loga_of_tau.evaluate(jnp.array(tau_val)))
    a = np.exp(loga)
    H = float(bg.H_of_loga.evaluate(jnp.array(loga)))
    aH = a * H  # = a'/a in conformal time units? No, aH = da/dt = a'/a * a?
    # Actually: a' = da/dtau = a*H (where H = da/(a*dt) = (1/a)(da/dt))
    # So a'/a = H*a/a = H? No. conformal Hubble = aH.
    # a' = a * H_conformal where H_conformal = aH
    # So a'/a = aH. Yes, a_prime_over_a = aH.
    a_prime_over_a = aH

    # Extract synchronous gauge variables
    eta = float(y[idx['eta']])
    # NOTE: h_prime in state vector is ALWAYS 0 (not evolved, constraint only)
    # Must recompute from Einstein constraint: h' = (k²η + 1.5 a² δρ) / (0.5 a'/a)
    delta_g_S = float(y[idx['F_g_0']])
    delta_b_S = float(y[idx['delta_b']])
    delta_cdm_S = float(y[idx['delta_cdm']])
    theta_b_S = float(y[idx['theta_b']])
    F_g_1 = float(y[idx['F_g_1']])
    theta_g_S = 3.0 * k_target * F_g_1 / 4.0
    F_ur_0 = float(y[idx['F_ur_0']])
    F_ur_1 = float(y[idx['F_ur_1']])
    theta_ur_S = 3.0 * k_target * F_ur_1 / 4.0
    shear_g_S = float(y[idx['F_g_2']]) / 2.0  # CLASS shear = F_2/2
    shear_ur_S = float(y[idx['F_ur_2']]) / 2.0

    # Background densities for Einstein equations
    rho_g = float(bg.rho_g_of_loga.evaluate(jnp.array(loga)))
    rho_b = float(bg.rho_b_of_loga.evaluate(jnp.array(loga)))
    rho_cdm = float(bg.rho_cdm_of_loga.evaluate(jnp.array(loga)))
    rho_ur = float(bg.rho_ur_of_loga.evaluate(jnp.array(loga)))
    rho_ncdm = float(bg.rho_ncdm_of_loga.evaluate(jnp.array(loga)))

    # Total density perturbation (synchronous gauge)
    a2 = a**2
    delta_rho = (rho_g * delta_g_S + rho_b * delta_b_S
                 + rho_cdm * delta_cdm_S + rho_ur * F_ur_0 + rho_ncdm * F_ur_0)

    # h' from 00 Einstein CONSTRAINT (same as perturbations.py line 400)
    h_prime = (k2 * eta + 1.5 * a2 * delta_rho) / (0.5 * a_prime_over_a)

    # (rho+P)*theta for eta' (0i Einstein equation)
    # CDM: theta_cdm = 0 in synchronous gauge
    rpt_sum = (4./3.*rho_g*theta_g_S + rho_b*theta_b_S
               + 4./3.*rho_ur*theta_ur_S + 4./3.*rho_ncdm*theta_ur_S)

    eta_prime = 1.5 * a2 * rpt_sum / k2

    # Gauge transformation parameter: alpha = (h' + 6 eta') / (2 k^2)
    alpha = (h_prime + 6.0 * eta_prime) / (2.0 * k2)

    # Transform to Newtonian gauge
    delta_g_N = delta_g_S - 4.0 * a_prime_over_a * alpha
    delta_b_N = delta_b_S - 3.0 * a_prime_over_a * alpha
    delta_cdm_N = delta_cdm_S - 3.0 * a_prime_over_a * alpha
    theta_g_N = theta_g_S + k2 * alpha
    theta_b_N = theta_b_S + k2 * alpha

    # Newtonian potentials
    phi_N = eta - a_prime_over_a * alpha
    # psi requires anisotropic stress

    # Find closest CLASS reference point
    idx_ref = np.argmin(np.abs(tau_ref - tau_val))

    class_dg = ref['delta_g'][idx_ref]
    class_db = ref['delta_b'][idx_ref]
    class_dc = ref['delta_cdm'][idx_ref]
    class_tg = ref['theta_g'][idx_ref]
    class_tb = ref['theta_b'][idx_ref]
    class_phi = ref['phi'][idx_ref]
    class_psi = ref['psi'][idx_ref]

    def pct(ours, theirs):
        if abs(theirs) < 1e-30:
            return 0.0
        return (ours - theirs) / abs(theirs) * 100

    print(f"{tau_val:8.1f}  {'delta_g':>8s}  {delta_g_N:14.6f}  {class_dg:14.6f}  {pct(delta_g_N, class_dg):+10.4f}")
    print(f"{'':>8s}  {'theta_g':>8s}  {theta_g_N:14.6e}  {class_tg:14.6e}  {pct(theta_g_N, class_tg):+10.4f}")
    print(f"{'':>8s}  {'delta_b':>8s}  {delta_b_N:14.6f}  {class_db:14.6f}  {pct(delta_b_N, class_db):+10.4f}")
    print(f"{'':>8s}  {'theta_b':>8s}  {theta_b_N:14.6e}  {class_tb:14.6e}  {pct(theta_b_N, class_tb):+10.4f}")
    print(f"{'':>8s}  {'d_cdm':>8s}  {delta_cdm_N:14.6f}  {class_dc:14.6f}  {pct(delta_cdm_N, class_dc):+10.4f}")
    print(f"{'':>8s}  {'phi':>8s}  {phi_N:14.6f}  {class_phi:14.6f}  {pct(phi_N, class_phi):+10.4f}")
    if i < len(tau_compare) - 1:
        print()

# Also print shear comparison
print("\n" + "=" * 70)
print("Photon shear (sigma_g = F_2/2) comparison")
print("=" * 70)
for i, tau_val in enumerate(tau_compare):
    y = sol.ys[i]
    our_shear = float(y[idx['F_g_2']]) / 2.0
    idx_ref = np.argmin(np.abs(tau_ref - tau_val))
    class_shear = ref['shear_g'][idx_ref]
    err = (our_shear - class_shear) / max(abs(class_shear), 1e-30) * 100
    print(f"  tau={tau_val:8.1f}: ours={our_shear:14.6e}  CLASS={class_shear:14.6e}  err={err:+.4f}%")

print("\nDone!", flush=True)
