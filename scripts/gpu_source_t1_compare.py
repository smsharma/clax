"""Compare source_T1 integrand at k=0.01 against CLASS reference.

CLASS Newtonian gauge: T1 = exp(-kappa) * k * (psi - phi)
Our sync gauge: T1 = exp(-kappa) * k * (alpha' + 2*aH*alpha - eta)

These should be gauge-invariant and match.

Also directly compare our Newtonian gauge potentials psi, phi against CLASS.
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
from jaxclass.interpolation import CubicSpline

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
phi_ref = ref['phi']
psi_ref = ref['psi']
delta_g_ref = ref['delta_g']
theta_b_ref = ref['theta_b']

# Find our closest k
k_idx = int(jnp.argmin(jnp.abs(pt.k_grid - k_test)))
k_actual = float(pt.k_grid[k_idx])
print(f"k_test={k_test}, k_actual={k_actual:.6f} (idx={k_idx})", flush=True)

# Our source_T1 at the perturbation tau grid
T1_ours = pt.source_T1[k_idx, :]
T0_ours = pt.source_T0[k_idx, :]
tau_ours = np.array(pt.tau_grid)
tau_star = float(th.tau_star)

# Reconstruct exp(-kappa) from our thermodynamics at CLASS tau values (vectorized)
loga_ref = jax.vmap(bg.loga_of_tau.evaluate)(jnp.array(tau_ref))
exp_mk_ref = jax.vmap(th.exp_m_kappa_of_loga.evaluate)(loga_ref)

# CLASS T1 source (Newtonian gauge: exp(-kappa) * k * (psi - phi))
T1_class = np.array(exp_mk_ref) * k_test * (psi_ref - phi_ref)

print(f"\n=== source_T1 comparison at k={k_test} ===")
print(f"tau_star = {tau_star:.2f} Mpc")
print(f"{'tau':>8} | {'CLASS_T1':>12} {'ours_T1':>12} {'err%':>8}")

# Interpolate our T1 to comparison tau values using numpy (fast)
tau_compare = [200, 250, 270, 280, 290, 300, 350, 500, 1000, 5000, 10000]
for tc in tau_compare:
    t1_class = float(np.interp(tc, tau_ref, T1_class))
    t1_ours = float(np.interp(tc, tau_ours, np.array(T1_ours)))
    err = (t1_ours / t1_class - 1) * 100 if abs(t1_class) > 1e-30 else 0
    print(f"{tc:>8.1f} | {t1_class:>12.6e} {t1_ours:>12.6e} {err:>+8.2f}")

# === Direct comparison of Newtonian gauge potentials ===
# Our code computes phi_newt = eta - aH*alpha, and
# psi = alpha' + aH*alpha (from Einstein trace-free eq)
# These are stored? Let's check if we can reconstruct them.
# Actually, we need to extract individual perturbation variables.
# Since we have the full ODE state, let's reconstruct alpha, alpha_prime, eta

# First, compare (psi - phi) directly at our tau grid
# psi - phi = (alpha' + 2*aH*alpha - eta) ... which is source_T1 / (exp_m_kappa * k)
# So we can compare:  T1_ours / (exp_mk * k)  vs  CLASS's (psi - phi) interpolated

# Get exp_m_kappa at our tau grid
loga_ours = jax.vmap(bg.loga_of_tau.evaluate)(jnp.array(tau_ours))
exp_mk_ours = np.array(jax.vmap(th.exp_m_kappa_of_loga.evaluate)(loga_ours))

# Our (psi - phi) = T1 / (exp_mk * k)
psi_m_phi_ours = np.array(T1_ours) / (exp_mk_ours * k_actual + 1e-300)

# CLASS (psi - phi) at our tau points
psi_m_phi_class_at_ours = np.interp(tau_ours, tau_ref, psi_ref - phi_ref)

print(f"\n=== (psi - phi) comparison at k={k_test} ===")
print(f"{'tau':>8} | {'CLASS':>12} {'ours':>12} {'err%':>8}")
tau_compare2 = [200, 250, 270, 280, 285, 290, 295, 300, 310, 350, 500, 1000, 5000]
for tc in tau_compare2:
    v_class = float(np.interp(tc, tau_ref, psi_ref - phi_ref))
    v_ours = float(np.interp(tc, tau_ours, psi_m_phi_ours))
    err = (v_ours / v_class - 1) * 100 if abs(v_class) > 1e-30 else 0
    print(f"{tc:>8.1f} | {v_class:>12.6e} {v_ours:>12.6e} {err:>+8.2f}")

# Compare phi and psi individually
# phi from CLASS ref
# phi_ours = eta - aH*alpha = eta - aH*(h'+6eta')/(2k^2)
# We don't have eta directly, but we can check against ref
phi_class_at_ours = np.interp(tau_ours, tau_ref, phi_ref)
psi_class_at_ours = np.interp(tau_ours, tau_ref, psi_ref)

print(f"\n=== phi (Newtonian potential) at k={k_test} ===")
print(f"{'tau':>8} | {'CLASS_phi':>12} {'CLASS_psi':>12} {'psi-phi':>12}")
for tc in [200, 270, 280, 290, 300, 350, 500]:
    phi_c = float(np.interp(tc, tau_ref, phi_ref))
    psi_c = float(np.interp(tc, tau_ref, psi_ref))
    print(f"{tc:>8.1f} | {phi_c:>12.6e} {psi_c:>12.6e} {psi_c-phi_c:>12.6e}")

# === Integrated T_l^T1 comparison at l=30 ===
from jaxclass.bessel import spherical_jl_backward

tau_0 = float(bg.conformal_age)

# Our T_l^T1 integral (fully vectorized)
chi_ours = tau_0 - tau_ours
x_ours = k_actual * chi_ours
jl_ours = np.array(spherical_jl_backward(30, jnp.array(x_ours)))
jl_p1_ours = np.array(spherical_jl_backward(31, jnp.array(x_ours)))
x_s_ours = np.where(np.abs(x_ours) < 1e-30, 1e-30, x_ours)
jl_prime_ours = (30 / x_s_ours) * jl_ours - jl_p1_ours

integrand_ours = np.array(T1_ours) * jl_prime_ours
dtau_o = np.diff(tau_ours)
dtau_mid_o = np.concatenate([dtau_o[:1]/2, (dtau_o[:-1]+dtau_o[1:])/2, dtau_o[-1:]/2])
T_l_T1_ours = float(np.sum(integrand_ours * dtau_mid_o))

# CLASS T_l^T1 integral (vectorized)
chi_ref = tau_0 - tau_ref
x_ref = k_test * chi_ref
jl_ref = np.array(spherical_jl_backward(30, jnp.array(x_ref)))
jl_p1_ref = np.array(spherical_jl_backward(31, jnp.array(x_ref)))
x_s_ref = np.where(np.abs(x_ref) < 1e-30, 1e-30, x_ref)
jl_prime_ref = (30 / x_s_ref) * jl_ref - jl_p1_ref

integrand_class = T1_class * jl_prime_ref
dtau_ref_d = np.diff(tau_ref)
dtau_mid_ref = np.concatenate([dtau_ref_d[:1]/2, (dtau_ref_d[:-1]+dtau_ref_d[1:])/2, dtau_ref_d[-1:]/2])
T_l_T1_class = float(np.sum(integrand_class * dtau_mid_ref))

print(f"\n=== T_l^T1(k={k_test}) integral at l=30 ===")
print(f"  CLASS: {T_l_T1_class:+.6e}")
print(f"  Ours:  {T_l_T1_ours:+.6e}")
err = (T_l_T1_ours / T_l_T1_class - 1) * 100 if abs(T_l_T1_class) > 1e-30 else 0
print(f"  Error: {err:+.2f}%")

# Also do T0 comparison
T0_class_approx = np.interp(tau_ours, tau_ref, np.array(exp_mk_ref) * k_test * 0)  # placeholder
# The real T0 comparison needs the full source - skip for now

# Instead, compare the full T_l (T0+T1) integral
# at l=30 using only our source vs CLASS reconstruction
integrand_T0_ours = np.array(T0_ours) * jl_ours  # T0 uses j_l not j_l'
T_l_T0_ours = float(np.sum(integrand_T0_ours * dtau_mid_o))
T_l_full_ours = T_l_T0_ours + T_l_T1_ours

print(f"\n=== T_l decomposition at l=30, k={k_test} ===")
print(f"  T_l^T0:  {T_l_T0_ours:+.6e}")
print(f"  T_l^T1:  {T_l_T1_ours:+.6e}")
print(f"  T_l^T1/T_l^T0: {T_l_T1_ours/T_l_T0_ours*100:+.2f}%")

# Also try at the dominant k for l=30 (k ~ l/chi_star ~ 30/14000 ~ 0.002)
# But we only have reference data at k=0.01, so this is what we have

# === Peak region analysis ===
# Find where |T1| is largest and compare there
abs_T1_class = np.abs(T1_class)
peak_idx = np.argmax(abs_T1_class)
tau_peak = tau_ref[peak_idx]
print(f"\n=== Peak T1 analysis ===")
print(f"  T1_class peaks at tau={tau_peak:.1f}")
print(f"  Peak T1_class: {T1_class[peak_idx]:+.6e}")
t1_ours_at_peak = float(np.interp(tau_peak, tau_ours, np.array(T1_ours)))
print(f"  T1_ours at peak: {t1_ours_at_peak:+.6e}")
err_peak = (t1_ours_at_peak / T1_class[peak_idx] - 1) * 100 if abs(T1_class[peak_idx]) > 1e-30 else 0
print(f"  Error at peak: {err_peak:+.2f}%")

# Check across the whole tau range
t1_ours_interp = np.interp(tau_ref, tau_ours, np.array(T1_ours))
# Focus on recombination region where T1 matters (tau = 200-400)
mask = (tau_ref > 200) & (tau_ref < 400)
t1c = T1_class[mask]
t1o = t1_ours_interp[mask]
good = np.abs(t1c) > 1e-30
if np.any(good):
    errs = (t1o[good] / t1c[good] - 1) * 100
    print(f"\n  Recomb region (200<tau<400):")
    print(f"    Mean error: {np.mean(errs):+.2f}%")
    print(f"    Max |error|: {np.max(np.abs(errs)):.2f}% at tau={tau_ref[mask][good][np.argmax(np.abs(errs))]:.1f}")
    print(f"    RMS error: {np.sqrt(np.mean(errs**2)):.2f}%")

# Late ISW region
mask2 = (tau_ref > 5000) & (tau_ref < 14000)
t1c2 = T1_class[mask2]
t1o2 = t1_ours_interp[mask2]
good2 = np.abs(t1c2) > 1e-30
if np.any(good2):
    errs2 = (t1o2[good2] / t1c2[good2] - 1) * 100
    print(f"\n  Late ISW (5000<tau<14000):")
    print(f"    Mean error: {np.mean(errs2):+.2f}%")
    print(f"    Max |error|: {np.max(np.abs(errs2)):.2f}%")
    print(f"    RMS error: {np.sqrt(np.mean(errs2**2)):.2f}%")

print("\nDone!", flush=True)
