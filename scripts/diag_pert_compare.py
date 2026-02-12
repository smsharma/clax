"""Compare perturbation variables at k=0.01 against CLASS reference."""
import sys
sys.path.insert(0, '.')
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxclass import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import perturbations_solve

print("Devices:", jax.devices(), flush=True)
params = CosmoParams()
prec = PrecisionParams.planck_cl()

bg = background_solve(params, prec)
th = thermodynamics_solve(params, prec, bg)
pt = perturbations_solve(params, prec, bg, th)

# Load CLASS reference at k=0.01
ref = np.load('reference_data/lcdm_fiducial/perturbations_k0.0100.npz')

# Find k=0.01 in our k-grid
k_target = 0.01
ik = int(jnp.argmin(jnp.abs(pt.k_grid - k_target)))
k_actual = float(pt.k_grid[ik])
print(f"Our k={k_actual:.6f}, target k={k_target}", flush=True)

# Compare source functions at recombination
tau_star = float(th.tau_star)
print(f"tau_star = {tau_star:.2f} Mpc", flush=True)

# Find tau_star in our tau_grid
itau = int(jnp.argmin(jnp.abs(pt.tau_grid - tau_star)))
tau_actual = float(pt.tau_grid[itau])
print(f"Our tau at recomb: {tau_actual:.2f}", flush=True)

# Print source function values at recombination
print(f"\nSource functions at (k={k_actual:.4f}, tau={tau_actual:.1f}):")
print(f"  source_T0:     {float(pt.source_T0[ik, itau]):.6e}")
print(f"  source_T1:     {float(pt.source_T1[ik, itau]):.6e}")
print(f"  source_T2:     {float(pt.source_T2[ik, itau]):.6e}")
print(f"  source_E:      {float(pt.source_E[ik, itau]):.6e}")
print(f"  source_SW:     {float(pt.source_SW[ik, itau]):.6e}")
print(f"  source_ISW_fs: {float(pt.source_ISW_fs[ik, itau]):.6e}")
print(f"  source_Doppler:{float(pt.source_Doppler[ik, itau]):.6e}")

# Compare perturbation variables against CLASS
# CLASS stores: delta_g, theta_g, shear_g, delta_b, theta_b, phi, psi
# Our code stores source functions, not raw perturbation variables
# But we can compare the Newtonian gauge potentials

# Find corresponding CLASS tau
ref_tau = ref['tau_Mpc']
itau_class = np.argmin(np.abs(ref_tau - tau_actual))
tau_class = ref_tau[itau_class]

print(f"\nCLASS tau at match: {tau_class:.2f}")
print(f"\nPerturbation variables comparison at recombination:")
print(f"  CLASS delta_g:  {ref['delta_g'][itau_class]:.6e}")
print(f"  CLASS theta_b:  {ref['theta_b'][itau_class]:.6e}")
print(f"  CLASS phi:      {ref['phi'][itau_class]:.6e}")
print(f"  CLASS psi:      {ref['psi'][itau_class]:.6e}")
print(f"  CLASS shear_g:  {ref['shear_g'][itau_class]:.6e}")

# Show delta_g and phi over a range of tau near recombination
print(f"\nCLASS delta_g profile near recomb:")
for offset in [-10, -5, -2, 0, 2, 5, 10]:
    idx = max(0, min(itau_class + offset, len(ref_tau)-1))
    print(f"  tau={ref_tau[idx]:.1f}: delta_g={ref['delta_g'][idx]:.6f}, phi={ref['phi'][idx]:.6f}")

# Show our source T0 profile (should peak at tau_star)
print(f"\nOur source_T0 profile near recomb:")
for offset in [-50, -20, -10, -5, 0, 5, 10, 20, 50]:
    idx = max(0, min(itau + offset, len(pt.tau_grid)-1))
    print(f"  tau={float(pt.tau_grid[idx]):.1f}: T0={float(pt.source_T0[ik, idx]):.6e}, "
          f"T1={float(pt.source_T1[ik, idx]):.6e}")

print("\nDone!", flush=True)
