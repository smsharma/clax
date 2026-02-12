"""Compare Newtonian gauge potential Phi at k=0.01 vs CLASS."""
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

params = CosmoParams()
prec = PrecisionParams.planck_cl()
bg = background_solve(params, prec)
th = thermodynamics_solve(params, prec, bg)
pt = perturbations_solve(params, prec, bg, th)

# Load CLASS reference at k=0.01
ref = np.load('reference_data/lcdm_fiducial/perturbations_k0.0100.npz')

# Find k=0.01 in our k-grid
ik = int(jnp.argmin(jnp.abs(pt.k_grid - 0.01)))
k_actual = float(pt.k_grid[ik])
print(f"k={k_actual:.6f}")

# Extract Phi from our lensing source: source_lens = exp_m_kappa * 2 * phi_newt
# phi_newt = source_lens / (2 * exp_m_kappa)
loga_grid = jnp.array([bg.loga_of_tau.evaluate(t) for t in pt.tau_grid])
exp_m_kappa_grid = jnp.array([th.exp_m_kappa_of_loga.evaluate(la) for la in loga_grid])
phi_ours = pt.source_lens[ik, :] / (2.0 * jnp.maximum(exp_m_kappa_grid, 1e-30))

# Compare at specific tau values
ref_tau = ref['tau_Mpc']
ref_phi = ref['phi']
ref_delta_g = ref['delta_g']

print(f"\n{'tau':>8} {'our_phi':>12} {'CLASS_phi':>12} {'rel_err%':>10}")
print("-" * 45)
for tau_target in [100, 150, 200, 250, 270, 280, 290, 300, 350, 500, 1000, 5000, 10000]:
    # Our value
    itau_ours = int(jnp.argmin(jnp.abs(pt.tau_grid - tau_target)))
    tau_ours = float(pt.tau_grid[itau_ours])
    phi_our_val = float(phi_ours[itau_ours])

    # CLASS value
    itau_class = np.argmin(np.abs(ref_tau - tau_ours))
    tau_class = ref_tau[itau_class]
    phi_class = ref_phi[itau_class]

    if abs(phi_class) > 1e-30:
        rel_err = (phi_our_val - phi_class) / abs(phi_class) * 100
        print(f"{tau_ours:8.1f} {phi_our_val:12.6f} {phi_class:12.6f} {rel_err:+10.3f}%")

# Also compare at k=0.05
ik2 = int(jnp.argmin(jnp.abs(pt.k_grid - 0.05)))
k2 = float(pt.k_grid[ik2])
phi_ours2 = pt.source_lens[ik2, :] / (2.0 * jnp.maximum(exp_m_kappa_grid, 1e-30))
ref2 = np.load('reference_data/lcdm_fiducial/perturbations_k0.0500.npz')

print(f"\n\nk={k2:.6f} (target 0.05)")
print(f"{'tau':>8} {'our_phi':>12} {'CLASS_phi':>12} {'rel_err%':>10}")
print("-" * 45)
for tau_target in [100, 200, 270, 280, 290, 300, 500]:
    itau_ours = int(jnp.argmin(jnp.abs(pt.tau_grid - tau_target)))
    tau_ours = float(pt.tau_grid[itau_ours])
    phi_our_val = float(phi_ours2[itau_ours])
    itau_class = np.argmin(np.abs(ref2['tau_Mpc'] - tau_ours))
    phi_class = ref2['phi'][itau_class]
    if abs(phi_class) > 1e-30:
        rel_err = (phi_our_val - phi_class) / abs(phi_class) * 100
        print(f"{tau_ours:8.1f} {phi_our_val:12.6f} {phi_class:12.6f} {rel_err:+10.3f}%")

print("\nDone!")
