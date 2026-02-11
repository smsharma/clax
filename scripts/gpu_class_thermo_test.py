"""Test: substitute CLASS thermodynamics into our perturbation solver.

If the TT C_l error goes away, the problem is in our thermodynamics.
If it stays, the problem is in our perturbation equations or integration.

Strategy: Load CLASS kappa_dot(z) from reference data, build splines,
and replace our thermodynamics result's kappa_dot/g/g_prime splines.
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
import math

# Load CLASS reference
ref_cls = np.load('reference_data/lcdm_fiducial/cls.npz')
ref_th = np.load('reference_data/lcdm_fiducial/thermodynamics.npz')

params = CosmoParams()
prec = PrecisionParams.planck_cl()

print("Solving background + thermo...", flush=True)
bg = background_solve(params, prec)
th = thermodynamics_solve(params, prec, bg)

# ------ Test 1: Our thermodynamics ------
print("\n--- Test 1: Our thermodynamics ---", flush=True)
pt = perturbations_solve(params, prec, bg, th)

l_values = [20, 100, 200, 500, 700, 1000]
cl_tt = compute_cl_tt_interp(pt, params, bg, l_values, n_k_fine=5000)

print(f"{'l':>6s}  {'TT_err':>10s}")
for i, l in enumerate(l_values):
    err = (float(cl_tt[i]) - ref_cls['tt'][l]) / ref_cls['tt'][l] * 100
    sub = " ***" if abs(err) < 1.0 else ""
    print(f"{l:6d}  {err:+10.3f}%{sub}")

# ------ Test 2: Replace kappa_dot with CLASS values ------
print("\n--- Test 2: CLASS kappa_dot substituted ---", flush=True)

# CLASS thermodynamics reference data
class_z = ref_th['th_z']
class_kd = ref_th['th_kappa_dot']
class_g = ref_th['th_g']
class_xe = ref_th['th_x_e']

# Convert z to loga for our spline
class_a = 1.0 / (1.0 + class_z)
class_loga = np.log(np.maximum(class_a, 1e-30))

# Sort by loga (ascending)
sort_idx = np.argsort(class_loga)
class_loga_sorted = class_loga[sort_idx]
class_kd_sorted = class_kd[sort_idx]
class_g_sorted = class_g[sort_idx]
class_xe_sorted = class_xe[sort_idx]

# Compute kappa (cumulative optical depth) from CLASS kappa_dot
# κ = ∫_τ^τ_0 κ̇(τ') dτ' where κ̇ = class_kd
# We need tau values for CLASS z-grid
class_tau = np.array([float(bg.tau_of_loga.evaluate(jnp.array(la))) for la in class_loga_sorted])

# Compute kappa by backward integration of kappa_dot
dtau_class = np.diff(class_tau)
kappa_integrand = 0.5 * (class_kd_sorted[:-1] + class_kd_sorted[1:]) * dtau_class
kappa_cumulative = np.cumsum(kappa_integrand[::-1])[::-1]
class_kappa = np.concatenate([kappa_cumulative, np.array([0.0])])

class_exp_m_kappa = np.exp(-class_kappa)

# Recompute g from CLASS kappa_dot and our recomputed kappa
class_g_recomp = class_kd_sorted * class_exp_m_kappa

# g' = (κ̈ + κ̇²) * e^{-κ}
class_ddkappa = np.gradient(class_kd_sorted, class_tau)
class_g_prime = (class_ddkappa + class_kd_sorted**2) * class_exp_m_kappa

# Build new splines on CLASS's loga grid
# We need to subsample to a manageable size (our spline can handle ~10k points)
# Use every 3rd point to keep it fast
step = max(1, len(class_loga_sorted) // 10000)
loga_sub = jnp.array(class_loga_sorted[::step])
kd_sub = jnp.array(class_kd_sorted[::step])
g_sub = jnp.array(class_g_recomp[::step])
gp_sub = jnp.array(class_g_prime[::step])
exp_mk_sub = jnp.array(class_exp_m_kappa[::step])
cs2_sub = jnp.array([float(th.cs2_of_loga.evaluate(jnp.array(la))) for la in loga_sub])

print(f"CLASS thermo grid: {len(loga_sub)} points, loga range [{float(loga_sub[0]):.2f}, {float(loga_sub[-1]):.4f}]")

# Build CubicHermiteSpline objects matching our thermodynamics result format
from jaxclass.interpolation import CubicSpline

kd_spline = CubicSpline(loga_sub, kd_sub)
g_spline = CubicSpline(loga_sub, g_sub)
gp_spline = CubicSpline(loga_sub, gp_sub)
exp_mk_spline = CubicSpline(loga_sub, exp_mk_sub)
cs2_spline = CubicSpline(loga_sub, cs2_sub)

# Create modified thermodynamics result with CLASS kappa_dot
# We need to replace the splines in the ThermoResult
# Since it's a frozen dataclass, we can use dataclasses.replace,
# BUT it's also a registered pytree node with custom tree_flatten/unflatten.
# So let's manually construct a new ThermoResult instead.
from jaxclass.thermodynamics import ThermoResult
th_class = ThermoResult(
    xe_of_loga=th.xe_of_loga,  # keep our xe (not critical)
    Tb_of_loga=th.Tb_of_loga,  # keep our Tb
    kappa_dot_of_loga=kd_spline,
    exp_m_kappa_of_loga=exp_mk_spline,
    g_of_loga=g_spline,
    g_prime_of_loga=gp_spline,
    cs2_of_loga=cs2_spline,
    z_star=th.z_star,
    z_rec=th.z_rec,
    tau_star=th.tau_star,
    rs_star=th.rs_star,
    z_reio=th.z_reio,
)

# Solve perturbations with CLASS thermodynamics
print("Solving perturbations with CLASS thermo...", flush=True)
pt_class = perturbations_solve(params, prec, bg, th_class)

cl_tt_class = compute_cl_tt_interp(pt_class, params, bg, l_values, n_k_fine=5000)

print(f"{'l':>6s}  {'TT_err':>10s}")
for i, l in enumerate(l_values):
    err = (float(cl_tt_class[i]) - ref_cls['tt'][l]) / ref_cls['tt'][l] * 100
    sub = " ***" if abs(err) < 1.0 else ""
    print(f"{l:6d}  {err:+10.3f}%{sub}")

# Also compare directly: how much did substituting CLASS thermo change things?
print("\n--- Improvement from CLASS thermo ---")
for i, l in enumerate(l_values):
    err_ours = (float(cl_tt[i]) - ref_cls['tt'][l]) / ref_cls['tt'][l] * 100
    err_class = (float(cl_tt_class[i]) - ref_cls['tt'][l]) / ref_cls['tt'][l] * 100
    print(f"  l={l:5d}: ours={err_ours:+.3f}%, CLASS_thermo={err_class:+.3f}%, delta={abs(err_ours)-abs(err_class):+.3f}pp")

print("\nDone!", flush=True)
