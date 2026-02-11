"""Detailed comparison of (psi-phi) and individual perturbation variables at k=0.01.

Goal: identify what causes the 2-3% error in (psi-phi) near recombination.
This error propagates to T1 source and the C_l integral.

We reconstruct psi-phi from our ODE state (eta, h', etc) and compare
component by component against CLASS reference.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

print("Devices:", jax.devices(), flush=True)

from jaxclass.params import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import perturbations_solve

p = CosmoParams()
pp = PrecisionParams.planck_cl()
bg = background_solve(p, pp)
th = thermodynamics_solve(p, pp, bg)
pt = perturbations_solve(p, pp, bg, th)
print(f"n_k={len(pt.k_grid)}, n_tau={len(pt.tau_grid)}", flush=True)

# Load CLASS reference at k=0.01
k_test = 0.01
ref = np.load(f'reference_data/lcdm_fiducial/perturbations_k{k_test:.4f}.npz')
tau_ref = ref['tau_Mpc']

# Find our closest k
k_idx = int(jnp.argmin(jnp.abs(pt.k_grid - k_test)))
k_actual = float(pt.k_grid[k_idx])
print(f"k_test={k_test}, k_actual={k_actual:.6f}", flush=True)

tau_ours = np.array(pt.tau_grid)

# Compare individual perturbation variables
# CLASS reference variables:
delta_g_ref = ref['delta_g']
theta_g_ref = ref['theta_g']
theta_b_ref = ref['theta_b']
delta_b_ref = ref['delta_b']
delta_cdm_ref = ref['delta_cdm']
shear_g_ref = ref['shear_g']  # F_g_2/2
shear_ur_ref = ref['shear_ur']

# Our ODE state stores F_g_0 (=delta_g), F_g_1 (related to theta_g), etc.
# Let's extract these from the state vector
# The state index layout is in perturbations.py
from jaxclass.perturbations import _build_state_indices

n_species = 3  # g, ur, ncdm (massless)
idx = _build_state_indices(pp.pt_l_max_g, pp.pt_l_max_pol_g, pp.pt_l_max_ur)

# Our state at each tau
state = pt.ode_states  # shape [n_k, n_tau, n_states]
if state is not None:
    y_k = np.array(state[k_idx, :, :])  # [n_tau, n_states]

    eta_ours = y_k[:, idx['eta']]
    delta_g_ours = y_k[:, idx['F_g_0']]
    F_g_1_ours = y_k[:, idx['F_g_1']]
    theta_g_ours = 3.0 * k_actual * F_g_1_ours / 4.0
    F_g_2_ours = y_k[:, idx['F_g_start'] + 2]
    shear_g_ours = F_g_2_ours / 2.0
    delta_b_ours = y_k[:, idx['delta_b']]
    theta_b_ours = y_k[:, idx['theta_b']]
    delta_cdm_ours = y_k[:, idx['delta_cdm']]
    F_ur_2_ours = y_k[:, idx['F_ur_start'] + 2]
    shear_ur_ours = F_ur_2_ours / 2.0

    tau_star = float(th.tau_star)
    tau_compare = [200, 250, 270, 280, 285, 290, 295, 300, 310, 350]

    print(f"\n=== Perturbation variable comparison at k={k_test} (tau_star={tau_star:.1f}) ===")

    for var_name, ours, ref_arr in [
        ("delta_g", delta_g_ours, delta_g_ref),
        ("theta_g", theta_g_ours, theta_g_ref),
        ("theta_b", theta_b_ours, theta_b_ref),
        ("delta_b", delta_b_ours, delta_b_ref),
        ("delta_cdm", delta_cdm_ours, delta_cdm_ref),
        ("shear_g", shear_g_ours, shear_g_ref),
        ("shear_ur", shear_ur_ours, shear_ur_ref),
        ("eta", eta_ours, None),  # No direct CLASS reference
    ]:
        print(f"\n  --- {var_name} ---")
        print(f"  {'tau':>6} | {'CLASS':>12} {'ours':>12} {'err%':>8}")
        for tc in tau_compare:
            v_ours = float(np.interp(tc, tau_ours, ours))
            if ref_arr is not None:
                v_class = float(np.interp(tc, tau_ref, ref_arr))
                err = (v_ours / v_class - 1) * 100 if abs(v_class) > 1e-30 else 0
                print(f"  {tc:>6.0f} | {v_class:>12.6e} {v_ours:>12.6e} {err:>+8.4f}")
            else:
                print(f"  {tc:>6.0f} | {'N/A':>12} {v_ours:>12.6e}")

    # Reconstruct phi and psi from our variables
    # phi = eta - aH*alpha = eta - aH*(h'+6*eta')/(2*k^2)
    # We need h' and eta' at each tau
    # h' is a CONSTRAINT: h' = (k^2*eta + 1.5*a^2*delta_rho) / (0.5*aH)
    # This requires background quantities at each tau

    print(f"\n=== Reconstructing phi, psi ===")
    for tc in tau_compare:
        idx_o = np.argmin(np.abs(tau_ours - tc))
        tau_v = tau_ours[idx_o]

        loga_v = float(bg.loga_of_tau.evaluate(jnp.array(tau_v)))
        a_v = np.exp(loga_v)
        a2_v = a_v**2
        H_v = float(bg.H_of_loga.evaluate(jnp.array(loga_v)))
        aH_v = a_v * H_v
        k2 = k_actual**2

        # Background densities at this time
        rho_g_v = float(bg.rho_g_of_loga.evaluate(jnp.array(loga_v)))
        rho_b_v = float(bg.rho_b_of_loga.evaluate(jnp.array(loga_v)))
        rho_cdm_v = float(bg.rho_cdm_of_loga.evaluate(jnp.array(loga_v)))
        rho_ur_v = float(bg.rho_ur_of_loga.evaluate(jnp.array(loga_v)))
        rho_ncdm_v = float(bg.rho_ncdm_of_loga.evaluate(jnp.array(loga_v)))

        eta_v = float(eta_ours[idx_o])
        delta_g_v = float(delta_g_ours[idx_o])
        delta_b_v = float(delta_b_ours[idx_o])
        delta_cdm_v = float(delta_cdm_ours[idx_o])

        F_ur_0_v = float(y_k[idx_o, idx['F_ur_0']])  # delta_ur

        # delta_rho (using raw hierarchy values, not RSA)
        delta_rho = (rho_g_v * delta_g_v + rho_b_v * delta_b_v
                     + rho_cdm_v * delta_cdm_v
                     + rho_ur_v * F_ur_0_v + rho_ncdm_v * F_ur_0_v)

        h_prime = (k2 * eta_v + 1.5 * a2_v * delta_rho) / (0.5 * aH_v)

        # theta_g, theta_b for rho_plus_p_theta
        theta_g_v = float(theta_g_ours[idx_o])
        theta_b_v = float(theta_b_ours[idx_o])
        F_ur_1_v = float(y_k[idx_o, idx['F_ur_1']])
        theta_ur_v = 3.0 * k_actual * F_ur_1_v / 4.0

        rho_plus_p_theta = (4.0/3 * rho_g_v * theta_g_v + rho_b_v * theta_b_v
                           + 4.0/3 * rho_ur_v * theta_ur_v
                           + 4.0/3 * rho_ncdm_v * theta_ur_v)

        eta_prime = 1.5 * a2_v * rho_plus_p_theta / k2
        alpha = (h_prime + 6.0 * eta_prime) / (2.0 * k2)

        # Anisotropic stress
        F_g_2_v = float(F_g_2_ours[idx_o])
        F_ur_2_v = float(F_ur_2_ours[idx_o])
        rho_plus_p_shear = (2.0/3 * rho_g_v * F_g_2_v
                           + 2.0/3 * rho_ur_v * F_ur_2_v
                           + 2.0/3 * rho_ncdm_v * F_ur_2_v)

        alpha_prime = -2.0 * aH_v * alpha + eta_v - 4.5 * (a2_v / k2) * rho_plus_p_shear

        phi = eta_v - aH_v * alpha
        psi = aH_v * alpha + alpha_prime
        psi_minus_phi = alpha_prime + 2.0 * aH_v * alpha - eta_v

        # CLASS reference
        phi_c = float(np.interp(tc, tau_ref, ref['phi']))
        psi_c = float(np.interp(tc, tau_ref, ref['psi']))
        psi_phi_c = psi_c - phi_c

        print(f"  tau={tc:.0f}:")
        print(f"    phi:     CLASS={phi_c:+.6e}  ours={phi:+.6e}  err={(phi/phi_c-1)*100:+.4f}%")
        print(f"    psi:     CLASS={psi_c:+.6e}  ours={psi:+.6e}  err={(psi/psi_c-1)*100:+.4f}%")
        print(f"    psi-phi: CLASS={psi_phi_c:+.6e}  ours={psi_minus_phi:+.6e}  err={(psi_minus_phi/psi_phi_c-1)*100 if abs(psi_phi_c)>1e-30 else 0:+.4f}%")
        print(f"    shear_g={F_g_2_v/2:.4e}, shear_ur={F_ur_2_v/2:.4e}")
else:
    print("ERROR: ode_states not available in PerturbationResult. Need to modify perturbations.py to store them.")

print("\nDone!", flush=True)
