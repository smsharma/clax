"""Compare raw perturbation variables against CLASS at k=0.05 Mpc^-1.

This is the most direct test: if delta_g, theta_b, phi match CLASS at
tau_star, the source function should also match. Any discrepancy here
identifies the perturbation-level error.
"""
import sys, time
sys.path.insert(0, '.')
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from clax.params import CosmoParams, PrecisionParams
from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve
from clax.perturbations import perturbations_solve, _extract_sources

params = CosmoParams()
prec = PrecisionParams.planck_cl()

# Load CLASS reference at k=0.05
ref = np.load('reference_data/lcdm_fiducial/perturbations_k0.0500.npz')
tau_cls = ref['tau_Mpc']
phi_cls = ref['phi']
psi_cls = ref['psi']
delta_g_cls = ref['delta_g']
theta_g_cls = ref['theta_g']
delta_b_cls = ref['delta_b']
theta_b_cls = ref['theta_b']
delta_ur_cls = ref['delta_ur']
shear_g_cls = ref['shear_g']

t0 = time.time()
bg = background_solve(params, prec)
th = thermodynamics_solve(params, prec, bg)
pt = perturbations_solve(params, prec, bg, th)
print(f"Pipeline: {time.time()-t0:.0f}s", flush=True)

# Find k=0.05 in our grid
k_target = 0.05
k_idx = np.argmin(np.abs(np.array(pt.k_grid) - k_target))
k_actual = float(pt.k_grid[k_idx])
print(f"k target={k_target}, actual={k_actual:.6f}")

# Our source functions at this k
tau_ours = np.array(pt.tau_grid)
source_T0 = np.array(pt.source_T0[k_idx, :])
source_SW = np.array(pt.source_SW[k_idx, :])
source_Dop = np.array(pt.source_Doppler[k_idx, :])
source_ISW_fs = np.array(pt.source_ISW_fs[k_idx, :])
source_ISW_vis = np.array(pt.source_ISW_vis[k_idx, :])

# Compare at tau_star region
tau_star = float(th.tau_star)
print(f"\ntau_star = {tau_star:.2f}")

# Since we don't store raw ODE variables, compare source functions.
# CLASS phi can be compared to our Newtonian potential from source extraction.
# But we need the actual perturbation variables.

# For source function comparison, reconstruct CLASS source at tau_star:
# First find indices
idx_ours_star = np.argmin(np.abs(tau_ours - tau_star))
idx_cls_star = np.argmin(np.abs(tau_cls - tau_star))

print(f"CLASS variables at k={k_target}, tau={tau_cls[idx_cls_star]:.1f}:")
print(f"  phi = {phi_cls[idx_cls_star]:.8e}")
print(f"  psi = {psi_cls[idx_cls_star]:.8e}")
print(f"  delta_g = {delta_g_cls[idx_cls_star]:.8e}")
print(f"  theta_g = {theta_g_cls[idx_cls_star]:.8e}")
print(f"  theta_b = {theta_b_cls[idx_cls_star]:.8e}")
print(f"  shear_g = {shear_g_cls[idx_cls_star]:.8e}")

print(f"\nOur source functions at k={k_actual:.6f}, tau={tau_ours[idx_ours_star]:.1f}:")
print(f"  source_T0 = {source_T0[idx_ours_star]:.8e}")
print(f"  source_SW = {source_SW[idx_ours_star]:.8e}")
print(f"  source_Dop = {source_Dop[idx_ours_star]:.8e}")
print(f"  source_ISW_vis = {source_ISW_vis[idx_ours_star]:.8e}")
print(f"  source_ISW_fs = {source_ISW_fs[idx_ours_star]:.8e}")

# Compute g at tau_star for reconstruction
loga_star = float(jnp.log(1.0/(1.0+1088.78)))
g_star = float(th.g_of_loga.evaluate(jnp.array(loga_star)))
print(f"  g(tau_star) = {g_star:.8e}")

# Reconstruct CLASS SW source = g * (delta_g/4 + alpha')
# alpha' is harder - we don't have it from CLASS output directly.
# But we can check if delta_g/4 * g matches our SW contribution partially.
sw_partial = g_star * delta_g_cls[idx_cls_star] / 4.0
print(f"\n  CLASS g*delta_g/4 = {sw_partial:.8e}")
print(f"  Our source_SW = {source_SW[idx_ours_star]:.8e} (includes g*alpha')")

# Phi comparison: our phi should equal CLASS phi/psi
# Phi_Newtonian = eta - aH*alpha in sync gauge
# But we can't reconstruct this without the raw ODE state.

# Instead, let's compare the integrated transfer function at l=500
from clax.bessel import spherical_jl_backward
chi_grid = float(bg.conformal_age) - tau_ours
dtau = np.diff(tau_ours)
dtau_mid = np.concatenate([dtau[:1]/2, (dtau[:-1]+dtau[1:])/2, dtau[-1:]/2])

for l in [300, 500, 700]:
    x = k_actual * chi_grid
    jl = np.array(spherical_jl_backward(l, jnp.array(x)))
    T_l = np.sum(source_T0 * jl * dtau_mid)
    # CLASS T_l would give C_l contribution at this k
    print(f"  T_l(k={k_actual:.3f}, l={l}) = {T_l:.8e}")

print(f"\nDone! ({time.time()-t0:.0f}s)")
