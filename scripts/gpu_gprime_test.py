"""Test the g_prime fix: analytic g' = (ddkappa + dkappa^2)*exp(-kappa)
vs spline derivative of g(lna).

Also runs planck_cl preset to check C_l impact.
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
from jaxclass.perturbations import perturbations_solve
from jaxclass.harmonic import compute_cl_tt_interp, compute_cl_ee_interp

params = CosmoParams()
prec = PrecisionParams.planck_cl()

print("Computing background + thermo...", flush=True)
bg = background_solve(params, prec)
th = thermodynamics_solve(params, prec, bg)

# Compare g_prime: analytic vs spline derivative
loga_star = jnp.log(1.0 / (1.0 + th.z_star))
g_at_star = th.g_of_loga.evaluate(loga_star)
gp_analytic = th.g_prime_of_loga.evaluate(loga_star)

# Spline derivative for comparison
aH_star = jnp.exp(loga_star) * bg.H_of_loga.evaluate(loga_star)
gp_spline = th.g_of_loga.derivative(loga_star) * aH_star

print(f"\nVisibility at z_star={float(th.z_star):.1f}:")
print(f"  g(tau_star) = {float(g_at_star):.6e}")
print(f"  g'_analytic = {float(gp_analytic):.6e}")
print(f"  g'_spline   = {float(gp_spline):.6e}")
if abs(gp_analytic) > 1e-30:
    rel_diff = abs(gp_spline - gp_analytic) / abs(gp_analytic)
    print(f"  Relative difference: {rel_diff:.4%}")

# Sample around recombination peak to see the pattern
loga_samples = jnp.linspace(loga_star - 0.05, loga_star + 0.05, 20)
print(f"\n{'loga':>10s}  {'g':>12s}  {'gp_anal':>12s}  {'gp_spline':>12s}  {'rel_diff':>10s}")
for lna in loga_samples:
    g_val = th.g_of_loga.evaluate(lna)
    gp_a = th.g_prime_of_loga.evaluate(lna)
    aH = jnp.exp(lna) * bg.H_of_loga.evaluate(lna)
    gp_s = th.g_of_loga.derivative(lna) * aH
    if abs(float(gp_a)) > 1e-30:
        rd = abs(float(gp_s) - float(gp_a)) / abs(float(gp_a))
        print(f"{float(lna):10.5f}  {float(g_val):12.4e}  {float(gp_a):12.4e}  {float(gp_s):12.4e}  {rd:10.4%}")

# Now test C_l impact
print("\nSolving perturbations (planck_cl)...", flush=True)
pt = perturbations_solve(params, prec, bg, th)
print(f"n_k={len(pt.k_grid)}, n_tau={len(pt.tau_grid)}", flush=True)

ref = np.load('reference_data/lcdm_fiducial/cls.npz')
ell_ref = ref['ell']
ells = [20, 30, 50, 100, 200, 300, 500, 700, 1000]

print("\nComputing TT + EE (interp -> 3000 fine k)...", flush=True)
cl_tt = compute_cl_tt_interp(pt, params, bg, ells, n_k_fine=3000)
cl_ee = compute_cl_ee_interp(pt, params, bg, ells, n_k_fine=3000)

for spec, cl, key in [("TT", cl_tt, 'tt'), ("EE", cl_ee, 'ee')]:
    print(f"\nC_l^{spec} (planck_cl, g_prime fix):")
    for i, ell in enumerate(ells):
        cl_class = ref[key][ell]
        cl_ours = float(cl[i])
        if abs(cl_class) > 1e-30:
            err = (cl_ours - cl_class) / cl_class * 100
            sub = " ***" if abs(err) < 1.0 else ""
            print(f"  l={ell:5d}: err={err:+.3f}%{sub}", flush=True)

print("\nDone!", flush=True)
