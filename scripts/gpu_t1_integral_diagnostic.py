"""Diagnose WHERE the T_l^T1 integral goes wrong.

Split the T1 transfer integral into tau ranges and compare each piece.
This tells us whether the error is at recombination or late times.
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
from jaxclass.bessel import spherical_jl_backward

p = CosmoParams()
pp = PrecisionParams.planck_cl()
bg = background_solve(p, pp)
th = thermodynamics_solve(p, pp, bg)
pt = perturbations_solve(p, pp, bg, th)
print(f"n_k={len(pt.k_grid)}, n_tau={len(pt.tau_grid)}", flush=True)

k_test = 0.01
k_idx = int(jnp.argmin(jnp.abs(pt.k_grid - k_test)))
k_actual = float(pt.k_grid[k_idx])
print(f"k={k_actual:.6f}", flush=True)

tau_0 = float(bg.conformal_age)
tau_ours = np.array(pt.tau_grid)
T1_ours = np.array(pt.source_T1[k_idx, :])
T0_ours = np.array(pt.source_T0[k_idx, :])

chi_ours = tau_0 - tau_ours
x_ours = k_actual * chi_ours

# Compute j_l and j_l' vectorized
l = 30
jl = np.array(spherical_jl_backward(l, jnp.array(x_ours)))
jl_p1 = np.array(spherical_jl_backward(l + 1, jnp.array(x_ours)))
x_s = np.where(np.abs(x_ours) < 1e-30, 1e-30, x_ours)
jl_prime = (l / x_s) * jl - jl_p1

# Integration weights
dtau = np.diff(tau_ours)
dtau_mid = np.concatenate([dtau[:1]/2, (dtau[:-1]+dtau[1:])/2, dtau[-1:]/2])

# Integrand
integrand_T1 = T1_ours * jl_prime
integrand_T0 = T0_ours * jl

# Split into tau ranges
tau_boundaries = [0, 200, 250, 280, 300, 350, 400, 500, 1000, 5000, 15000]
print(f"\n=== T_l^T1 integral contributions by tau range (k={k_actual:.6f}, l={l}) ===")
print(f"{'tau_range':>18} | {'T1_contrib':>12} {'T0_contrib':>12} {'T1/T0%':>8}")

total_T1 = 0.0
total_T0 = 0.0
for i in range(len(tau_boundaries) - 1):
    t_lo, t_hi = tau_boundaries[i], tau_boundaries[i+1]
    mask = (tau_ours >= t_lo) & (tau_ours < t_hi)
    T1_part = float(np.sum(integrand_T1[mask] * dtau_mid[mask]))
    T0_part = float(np.sum(integrand_T0[mask] * dtau_mid[mask]))
    total_T1 += T1_part
    total_T0 += T0_part
    ratio = T1_part / T0_part * 100 if abs(T0_part) > 1e-30 else 0
    print(f"  {t_lo:>6.0f}-{t_hi:<6.0f} | {T1_part:>+12.6e} {T0_part:>+12.6e} {ratio:>+8.2f}")

print(f"  {'TOTAL':>14} | {total_T1:>+12.6e} {total_T0:>+12.6e} {total_T1/total_T0*100:>+8.2f}")

# Now compare with CLASS reference
print(f"\n=== Same comparison using CLASS ref data ===")
ref = np.load(f'reference_data/lcdm_fiducial/perturbations_k{k_test:.4f}.npz')
tau_ref = ref['tau_Mpc']
phi_ref = ref['phi']
psi_ref = ref['psi']

# Get exp(-kappa) at CLASS tau values
loga_ref = np.array(jax.vmap(bg.loga_of_tau.evaluate)(jnp.array(tau_ref)))
exp_mk_ref = np.array(jax.vmap(th.exp_m_kappa_of_loga.evaluate)(jnp.array(loga_ref)))

T1_class = exp_mk_ref * k_test * (psi_ref - phi_ref)

chi_ref = tau_0 - tau_ref
x_ref = k_test * chi_ref
jl_ref = np.array(spherical_jl_backward(l, jnp.array(x_ref)))
jl_p1_ref = np.array(spherical_jl_backward(l + 1, jnp.array(x_ref)))
x_s_ref = np.where(np.abs(x_ref) < 1e-30, 1e-30, x_ref)
jl_prime_ref = (l / x_s_ref) * jl_ref - jl_p1_ref

integrand_T1_class = T1_class * jl_prime_ref
dtau_ref_d = np.diff(tau_ref)
dtau_mid_ref = np.concatenate([dtau_ref_d[:1]/2, (dtau_ref_d[:-1]+dtau_ref_d[1:])/2, dtau_ref_d[-1:]/2])

print(f"{'tau_range':>18} | {'CLASS_T1':>12} {'ours_T1':>12} {'err%':>8}")
total_class = 0.0
total_ours = 0.0
for i in range(len(tau_boundaries) - 1):
    t_lo, t_hi = tau_boundaries[i], tau_boundaries[i+1]
    mask_c = (tau_ref >= t_lo) & (tau_ref < t_hi)
    mask_o = (tau_ours >= t_lo) & (tau_ours < t_hi)
    T1_class_part = float(np.sum(integrand_T1_class[mask_c] * dtau_mid_ref[mask_c]))
    T1_ours_part = float(np.sum(integrand_T1[mask_o] * dtau_mid[mask_o]))
    total_class += T1_class_part
    total_ours += T1_ours_part
    err = (T1_ours_part / T1_class_part - 1) * 100 if abs(T1_class_part) > 1e-30 else 0
    print(f"  {t_lo:>6.0f}-{t_hi:<6.0f} | {T1_class_part:>+12.6e} {T1_ours_part:>+12.6e} {err:>+8.2f}")

print(f"  {'TOTAL':>14} | {total_class:>+12.6e} {total_ours:>+12.6e} {(total_ours/total_class-1)*100:>+8.2f}")

# Also check: what if we DISABLE RSA for T1 source extraction?
# We can reconstruct T1 = exp_mk * k * (psi - phi) from our eta, h_prime, etc.
# by checking if our phi = eta - aH*alpha and psi = aH*alpha + alpha_prime match CLASS
print(f"\n=== exp(-kappa) comparison ===")
for tc in [280, 350, 500, 1000]:
    idx_r = np.argmin(np.abs(tau_ref - tc))
    idx_o = np.argmin(np.abs(tau_ours - tc))
    # Our exp_mk at this tau
    loga_o = float(jax.vmap(bg.loga_of_tau.evaluate)(jnp.array([tau_ours[idx_o]]))[0])
    emk_o = float(jax.vmap(th.exp_m_kappa_of_loga.evaluate)(jnp.array([loga_o]))[0])
    emk_c = exp_mk_ref[idx_r]
    print(f"  tau={tc}: CLASS exp(-k)={emk_c:.6e}, ours={emk_o:.6e}, err={((emk_o/emk_c)-1)*100:+.4f}%")

print("\nDone!", flush=True)
