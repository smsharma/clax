"""Generate CLASS transfer functions at l=30 for comparison."""
import numpy as np
from classy import Class

params = {
    'h': 0.6736,
    'omega_b': 0.02237,
    'omega_cdm': 0.1200,
    'A_s': 2.1e-9,
    'n_s': 0.9649,
    'tau_reio': 0.0544,
    'N_ur': 2.0328,
    'N_ncdm': 1,
    'm_ncdm': 0.06,
    'T_ncdm': 0.71611,
    'output': 'tCl pCl',
    'l_max_scalars': 2500,
    'tol_background_integration': 1e-12,
}

cosmo = Class()
cosmo.set(params)
cosmo.compute()

# Get C_l at specific l values for both T0-only and full
cls = cosmo.raw_cl(2500)

# Print specific C_l values for comparison
print("CLASS C_l values:")
for l in [20, 30, 50, 100, 200, 300, 500]:
    print(f"  l={l:5d}: TT={cls['tt'][l]:.10e}, EE={cls['ee'][l]:.10e}, TE={cls['te'][l]:.10e}")

# Get derived parameters
derived = cosmo.get_current_derived_parameters(['z_star', 'tau_star', 'rs_star', 'da_star', 'conformal_age'])
print(f"\nCLASS derived:")
for k, v in derived.items():
    print(f"  {k}: {v}")

cosmo.struct_cleanup()
cosmo.empty()

# Now do the same with all massless to check ncdm effect
params2 = dict(params)
params2['N_ur'] = 3.044
params2['N_ncdm'] = 0
del params2['m_ncdm']
del params2['T_ncdm']

cosmo2 = Class()
cosmo2.set(params2)
cosmo2.compute()
cls2 = cosmo2.raw_cl(2500)

print("\nMassive/Massless ratio:")
for l in [20, 30, 50, 100, 200, 300, 500]:
    if abs(cls2['tt'][l]) > 1e-30:
        ratio = cls['tt'][l] / cls2['tt'][l]
        print(f"  l={l:5d}: TT ratio={ratio:.6f} (effect={100*(ratio-1):+.3f}%)")

cosmo2.struct_cleanup()
cosmo2.empty()
print("\nDone!")
