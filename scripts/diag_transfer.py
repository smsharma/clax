"""Compare transfer functions T_l(k) at l=30 between jaxCLASS and CLASS."""
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
from jaxclass.harmonic import _exact_transfer_tt, _exact_transfer_ee
from jaxclass.primordial import primordial_scalar_pk

# Get our transfer functions
print("Computing jaxCLASS transfer...", flush=True)
params = CosmoParams()
prec = PrecisionParams.planck_cl()
bg = background_solve(params, prec)
th = thermodynamics_solve(params, prec, bg)
pt = perturbations_solve(params, prec, bg, th)

tau_0 = float(bg.conformal_age)
chi_grid = tau_0 - pt.tau_grid
dtau = jnp.diff(pt.tau_grid)
dtau_mid = jnp.concatenate([dtau[:1] / 2, (dtau[:-1] + dtau[1:]) / 2, dtau[-1:] / 2])

for l_test in [30, 100, 500]:
    print(f"\n=== l = {l_test} ===", flush=True)

    # Compute T_l(k) for T0, T0+T1, T0+T1+T2
    T_l_T0 = _exact_transfer_tt(pt.source_T0, pt.tau_grid, pt.k_grid, chi_grid, dtau_mid, l_test,
                                  source_T1=pt.source_T1, source_T2=pt.source_T2, mode="T0")
    T_l_T01 = _exact_transfer_tt(pt.source_T0, pt.tau_grid, pt.k_grid, chi_grid, dtau_mid, l_test,
                                   source_T1=pt.source_T1, source_T2=pt.source_T2, mode="T0+T1")
    T_l_full = _exact_transfer_tt(pt.source_T0, pt.tau_grid, pt.k_grid, chi_grid, dtau_mid, l_test,
                                    source_T1=pt.source_T1, source_T2=pt.source_T2, mode="T0+T1+T2")

    # Compute C_l from each
    P_R = primordial_scalar_pk(pt.k_grid, params)
    log_k = jnp.log(pt.k_grid)
    dlnk = jnp.diff(log_k)

    def integrate_cl(T_l):
        integrand = P_R * T_l**2
        return float(4.0 * jnp.pi * jnp.sum(0.5 * (integrand[:-1] + integrand[1:]) * dlnk))

    cl_T0 = integrate_cl(T_l_T0)
    cl_T01 = integrate_cl(T_l_T01)
    cl_full = integrate_cl(T_l_full)

    # Cross term: T0*T1
    cross_T01 = float(4.0 * jnp.pi * jnp.sum(
        0.5 * (P_R[:-1]*T_l_T0[:-1]*T_l_T01[:-1] + P_R[1:]*T_l_T0[1:]*T_l_T01[1:]) * dlnk))

    print(f"  C_l(T0):       {cl_T0:.6e}")
    print(f"  C_l(T0+T1):    {cl_T01:.6e}")
    print(f"  C_l(T0+T1+T2): {cl_full:.6e}")

    # Show T_l(k) at a few key k-values
    k_peak = l_test / 13700.0  # Expected peak k
    print(f"\n  Transfer T_l(k) at peak k~{k_peak:.4f}:")
    for k_target in [k_peak*0.5, k_peak, k_peak*2, k_peak*5, 0.01, 0.05]:
        ik = int(jnp.argmin(jnp.abs(pt.k_grid - k_target)))
        k_val = float(pt.k_grid[ik])
        print(f"    k={k_val:.5f}: T0={float(T_l_T0[ik]):.4e}, T0+T1={float(T_l_T01[ik]):.4e}, full={float(T_l_full[ik]):.4e}")

print("\nDone!", flush=True)
