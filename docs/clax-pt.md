# clax-pt Implementation Log

Running log for the `clax-pt-grad-project` implementation.
See CLASS-PT.md for physics reference.

---

## 2026-04-02: Phase 0 — Reference Documents and Data

### Step 0a: FFTLog reference
- Wrote `docs/FFTLog_PT.md` covering: power-law decomposition, η_m grid, cmsym symmetrization, bias cancellation, k³ normalization, DFT convention, M22/M13 kernels.

### Step 0b: CLASS-PT reference
- Wrote `docs/CLASS-PT-summary.md` covering: all 9 spectra, bias model, pk_mult indices 0–41, assembly formulas (verified against classy.pyx), IR resummation, k-grid, units.
- Key finding: `pk_lin(k, z)` takes k in 1/Mpc (not h/Mpc!) and returns Mpc³ (not (Mpc/h)³). Must convert: `pk_lin_hMpc = pk_lin(k_hMpc * h, z) / h**3`.

### Step 0c: Reference table
- Generated `docs/classpt_reference_table.npz` using CLASS-PT with Planck 2018 fiducial params at z=0.38.
- Script: `scripts/generate_classpt_reference_v2.py`
- Parameters: Planck 2018 (A_s=2.0989e-9, n_s=0.9649, omega_b=0.02237, omega_cdm=0.12, h=0.6736), z=0.38
- Bias: b1=2, b2=bG2=bGamma3=cs0=cs2=cs4=Pshot=0, b4=500
- k range: [0.005, 0.30] h/Mpc, 60 log-spaced points
- Output: pk_lin, pk_mm_real, pk_gg_real, pk_mg_real, pk_mm_l0/l2/l4, pk_gg_l0/l2/l4, pk_mult[96×60]
- Sanity: pk_lin(k=0.1) = 3825 (Mpc/h)³, pk_mm_real(k=0.1) = 2432 (Mpc/h)³
- f(z=0.38) = 0.7166, σ₈(z=0.38) = 0.6731

---

## 2026-04-02: Phase 1 — P13 Implementation

### Pre-existing code (from clax-pt branch)
`clax/ept.py` already has complete implementation:
- `_fftlog_decompose`: FFTLog decomposition into c_m k^{η_m}
- `_compute_p22`: P22 via bilinear form with M22 matrix
- `_compute_p13`: P13 via dot product with M13 + UV renormalization
- `_ir_resummation_numpy`: BAO wiggle/no-wiggle via DST-II
- `compute_ept`: main entry point

**Known bug fixed** (from CHANGELOG): M22 was loaded as Hermitian instead of symmetric (`M[j,i] = tri[idx].conj()` → `M[j,i] = tri[idx]`). Already fixed in current code.

### Phase 1 goal: Pure numpy reference test for P13

The test strategy:
1. Write `tests/test_p13.py` — pure numpy implementation of P13 as ground truth
2. Compare to JAX `_compute_p13` (which uses same matrices but JAX ops)
3. Target: < 0.01% relative error
4. Add gradient test: `jax.grad` vs finite differences

**Why numpy reference?** The JAX implementation uses `jnp.fft.fft` and matrix ops. A pure numpy implementation using `np.fft.fft` and explicit loops is easy to verify against CLASS-PT equations line by line.

---

## TODOs
- [ ] P13 value test < 0.01%
- [ ] P13 gradient test
- [ ] P22 value test < 0.01%
- [ ] P22 gradient test
- [ ] Full spectra accuracy test < 1% at k < 0.3 h/Mpc
- [ ] Gradient through full pipeline
