"""
Generate CLASS-PT reference table for clax-pt accuracy tests.

Produces docs/classpt_reference_table.npz with:
  k_hMpc      : k grid in h/Mpc
  pk_lin      : linear P(k) in (Mpc/h)^3
  pk_mm_real  : matter-matter real space
  pk_gg_real  : galaxy-galaxy real space
  pk_mg_real  : matter-galaxy real space
  pk_mm_l0    : matter monopole (RSD)
  pk_mm_l2    : matter quadrupole (RSD)
  pk_mm_l4    : matter hexadecapole (RSD)
  pk_gg_l0    : galaxy monopole (RSD)
  pk_gg_l2    : galaxy quadrupole (RSD)
  pk_gg_l4    : galaxy hexadecapole (RSD)
  pk_mult     : raw pk_mult array [96 x Nk] for component-level comparison
  bias_params : dict of bias parameters used
  cosmo_params: dict of cosmological parameters used

Run with:
  ~/miniconda3/envs/sbi_pytorch_osx-arm64-py310forge/bin/python3 scripts/generate_classpt_reference_v2.py

Units: k in h/Mpc, all P(k) in (Mpc/h)^3.
"""

import numpy as np
import os, sys

# Ensure classy is importable
try:
    from classy import Class
except ImportError:
    print("ERROR: classy not found. Run with the conda env python.")
    sys.exit(1)

# ── Planck 2018 best-fit cosmology ──────────────────────────────────────────
z_pk = 0.38   # BOSS CMASS effective redshift

cosmo_params = {
    'A_s': 2.0989e-9,
    'n_s': 0.9649,
    'tau_reio': 0.0544,
    'omega_b': 0.02237,
    'omega_cdm': 0.1200,
    'h': 0.6736,
}
h = cosmo_params['h']

# CLASS-PT runtime settings
classpt_settings = {
    'output': 'mPk',
    'non linear': 'PT',
    'IR resummation': 'Yes',
    'Bias tracers': 'Yes',
    'RSD': 'Yes',
    'z_pk': z_pk,
    'P_k_max_h/Mpc': 100.,
}

# ── Bias parameters ──────────────────────────────────────────────────────────
# Use CLASS-PT "default" values: b1=2, rest zero. This exercises real computation
# while keeping the bias combination tractable for validation.
bias = {
    'b1': 2.0,
    'b2': 0.0,
    'bG2': 0.0,
    'bGamma3': 0.0,
    'cs': 0.0,     # matter EFT counterterm (h/Mpc)^2 — enters pk_mm_real
    'cs0': 0.0,    # galaxy monopole EFT counterterm
    'cs2': 0.0,    # galaxy quadrupole EFT counterterm
    'cs4': 0.0,    # galaxy hexadecapole EFT counterterm
    'Pshot': 0.0,  # shot noise (Mpc/h)^3
    'b4': 500.0,   # finger-of-god stochastic (CLASS-PT default; enters l0/l2/l4)
}

# ── Output k-grid ────────────────────────────────────────────────────────────
k_hMpc = np.geomspace(0.005, 0.30, 60)  # 60 log-spaced points in h/Mpc
k_1Mpc = k_hMpc * h                     # k in 1/Mpc for pk_lin

# ── Run CLASS-PT ─────────────────────────────────────────────────────────────
print("Initializing CLASS-PT with Planck 2018 fiducial params...")
print(f"  z = {z_pk}, k range = [{k_hMpc[0]:.4f}, {k_hMpc[-1]:.4f}] h/Mpc, Nk = {len(k_hMpc)}")

M = Class()
M.set({**cosmo_params, **classpt_settings})
M.compute()

print("CLASS-PT computation done. Calling initialize_output...")
# IMPORTANT: initialize_output takes k in 1/Mpc (CLASS internal units)
# k_hMpc [h/Mpc] * h = k_1Mpc [1/Mpc]
k_1Mpc_output = k_hMpc * h  # convert to 1/Mpc for initialize_output
M.initialize_output(k_1Mpc_output, z_pk, len(k_hMpc))

# ── Collect pk_mult (raw components, no bias) ─────────────────────────────
# IMPORTANT: get_pk_mult also takes k in 1/Mpc
pk_mult = M.get_pk_mult(k_1Mpc_output, z_pk, len(k_hMpc))
print(f"  pk_mult shape: {pk_mult.shape}")

# ── Linear power spectrum ────────────────────────────────────────────────────
# pk_lin takes k in 1/Mpc and returns Mpc^3
pk_lin_Mpc3 = np.array([M.pk_lin(ki, z_pk) for ki in k_1Mpc])
pk_lin = pk_lin_Mpc3 * h**3   # convert to (Mpc/h)^3

# ── Nonlinear spectra via bias assembly ──────────────────────────────────────
print("Computing nonlinear spectra...")
pk_mm_real = M.pk_mm_real(cs=bias['cs'])
pk_gg_real = M.pk_gg_real(
    bias['b1'], bias['b2'], bias['bG2'], bias['bGamma3'],
    bias['cs'], bias['cs0'], bias['Pshot'])
pk_mg_real = M.pk_gm_real(
    bias['b1'], bias['b2'], bias['bG2'], bias['bGamma3'],
    bias['cs'], bias['cs0'])

pk_mm_l0 = M.pk_mm_l0(cs0=bias['cs0'])
pk_mm_l2 = M.pk_mm_l2(cs2=bias['cs2'])
pk_mm_l4 = M.pk_mm_l4(cs4=bias['cs4'])

pk_gg_l0 = M.pk_gg_l0(
    bias['b1'], bias['b2'], bias['bG2'], bias['bGamma3'],
    bias['cs0'], bias['Pshot'], bias['b4'])
pk_gg_l2 = M.pk_gg_l2(
    bias['b1'], bias['b2'], bias['bG2'], bias['bGamma3'],
    bias['cs2'], bias['b4'])
pk_gg_l4 = M.pk_gg_l4(
    bias['b1'], bias['b2'], bias['bG2'], bias['bGamma3'],
    bias['cs4'], bias['b4'])

# Growth rate
fz = M.scale_independent_growth_factor_f(z_pk)
print(f"  f(z={z_pk}) = {fz:.6f}")
print(f"  sigma8(z={z_pk}) = {M.sigma(8.0/h, z_pk):.6f}")

M.struct_cleanup()
M.empty()
print("CLASS-PT cleanup done.")

# ── Sanity checks ──────────────────────────────────────────────────────────
print("\nSanity checks:")
print(f"  pk_lin at k=0.1 h/Mpc: {pk_lin[np.argmin(np.abs(k_hMpc-0.1))]:.3e} (Mpc/h)^3")
print(f"  pk_mm_real at k=0.1:   {pk_mm_real[np.argmin(np.abs(k_hMpc-0.1))]:.3e} (Mpc/h)^3")
print(f"  pk_gg_real at k=0.1:   {pk_gg_real[np.argmin(np.abs(k_hMpc-0.1))]:.3e} (Mpc/h)^3")
print(f"  pk_mm_l0 at k=0.1:     {pk_mm_l0[np.argmin(np.abs(k_hMpc-0.1))]:.3e} (Mpc/h)^3")

# Check k-grid alignment
assert len(pk_mm_real) == len(k_hMpc), "k-grid mismatch"
assert np.all(pk_lin > 0), "Negative pk_lin — units error?"
assert np.all(pk_mm_real > 0), "Negative pk_mm_real"

# ── Save ──────────────────────────────────────────────────────────────────
outdir = os.path.join(os.path.dirname(__file__), '..', 'docs')
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, 'classpt_reference_table.npz')

np.savez(
    outpath,
    k_hMpc=k_hMpc,
    pk_lin=pk_lin,
    pk_mm_real=pk_mm_real,
    pk_gg_real=pk_gg_real,
    pk_mg_real=pk_mg_real,
    pk_mm_l0=pk_mm_l0,
    pk_mm_l2=pk_mm_l2,
    pk_mm_l4=pk_mm_l4,
    pk_gg_l0=pk_gg_l0,
    pk_gg_l2=pk_gg_l2,
    pk_gg_l4=pk_gg_l4,
    pk_mult=pk_mult,
    z_pk=np.array(z_pk),
    h=np.array(h),
    fz=np.array(fz),
    # Store bias params as individual arrays
    b1=np.array(bias['b1']),
    b2=np.array(bias['b2']),
    bG2=np.array(bias['bG2']),
    bGamma3=np.array(bias['bGamma3']),
    cs=np.array(bias['cs']),
    cs0=np.array(bias['cs0']),
    cs2=np.array(bias['cs2']),
    cs4=np.array(bias['cs4']),
    Pshot=np.array(bias['Pshot']),
    b4=np.array(bias['b4']),
    # Store cosmo params
    A_s=np.array(cosmo_params['A_s']),
    n_s=np.array(cosmo_params['n_s']),
    tau_reio=np.array(cosmo_params['tau_reio']),
    omega_b=np.array(cosmo_params['omega_b']),
    omega_cdm=np.array(cosmo_params['omega_cdm']),
)

print(f"\nSaved reference table to: {os.path.abspath(outpath)}")
print(f"  Arrays: k_hMpc[{len(k_hMpc)}], pk_mult[96×{len(k_hMpc)}], 9 spectra")
print("Done.")
