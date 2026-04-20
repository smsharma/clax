# P_loop Diagnostic Notes — clax-pt branch

## Ground truth (CLASS-PT at fiducial params, z=0.61)

Cosmology: h=0.6736, omega_b=0.02237, omega_cdm=0.12, A_s=2.089e-9, n_s=0.9649,
           tau_reio=0.052, YHe=0.2425, N_ur=2.0328, N_ncdm=1, m_ncdm=0.06 eV

CLASS-PT settings used for ground truth:
  IR resummation=No, Bias tracers=No, cb=No, RSD=No, P_k_max_h/Mpc=100, z_pk=0.61

P_loop (= pk_mm_real(cs=0) − P_tree, all in (Mpc/h)³):

| k [h/Mpc] | P_loop CLASS-PT | P_loop JAX | err%  |
|-----------|----------------|------------|-------|
| 0.0148    | -24.09         | -23.43     | +2.8% |
| 0.0461    | -48.64         | -42.64     | +12%  |
| 0.1440    | +92.76         | +73.33     | -21%  |
| 0.4493    | +190.69        | +108.03    | -43%  |

Values at k<0.003 h/Mpc are **extraction artifacts** (P_lin >> P_loop there; a
~0.02% interpolation mismatch in P_tree vs pk_lin creates spurious P_loop).

sigma_v² = (1/6π²) ∫ P_lin(k) dk (over EPT k-grid 0.00005..100 h/Mpc) = **18.01 (Mpc/h)²**
Changing k_cut from 1 to 100 h/Mpc changes sigma_v² by only ~0.4 (Mpc/h)², giving
<1% change in P13. So **sigma_v truncation is NOT the bug**.

P_tree at z=0.61 (IR=No):
  k=0.0148 h/Mpc: P_lin ≈ 12760 (Mpc/h)³  (sanity-checked order of magnitude ✓)

---

## Bugs found and fixed

### BUG 1: k-unit error in `scripts/generate_classpt_reference.py` (NOT YET FIXED IN COMMIT)
- **Location**: line 144 and line 111
- **Bug**: `cosmo.pk_lin(k / h, z)` (wrong) — should be `cosmo.pk_lin(k * h, z)`
- **Bug**: `cosmo.initialize_output(k_h, z, N)` (wrong) — should be `cosmo.initialize_output(k_h * h, z, N)`
- **Effect**: All reference data in `reference_data/classpt_z0.6.npz` is computed at wrong k values
  (CLASS-PT API takes k in 1/Mpc; dividing by h instead of multiplying means k is ~2× too small)
- **Status**: IDENTIFIED, not yet fixed (reference data needs regeneration)
- **Convention confirmed via**: classy.pyx source + CLASS-PT notebook (nonlinear_pt.ipynb)
  which explicitly does `khvec = kvec * h` before calling `initialize_output(khvec, ...)`

### BUG 2 (ONGOING): Scale-dependent P_loop error, grows from +3% at k=0.015 to −43% at k=0.45 h/Mpc
- **Status**: Root cause NOT YET IDENTIFIED
- **Hypotheses ruled out**:
  - sigma_v truncation (P_k_max_h/Mpc=100 gives same error as =10)
  - FFT convention mismatch (confirmed numpy.fft.fft matches CLASS-PT fft.c exactly)
  - JAX vs numpy discrepancy (JAX and numpy give identical results to machine precision)
  - Unit inconsistency in algebra (h³ factors checked algebraically)
- **Remaining suspects** (not yet tested):
  1. CLASS-PT uses k in 1/Mpc for the FFTLog decomposition (kmin_disc in 1/Mpc, not h/Mpc),
     but our JAX code uses k_h [h/Mpc] for the FFTLog. The M13/M22 matrix coefficients
     were precomputed assuming a specific k unit. Need to verify: are matrices k-unit agnostic?
  2. P_lin input: `cosmo.pk_lin(k_mpc, z)` might differ slightly from CLASS-PT's internal Pdisc
     (different interpolation from CLASS's spline tables vs CLASS-PT's own grid).

**Next diagnostic in progress** (script `diag_ptree_vs_plin.py`):
- Runs FFTLog in 1/Mpc units (as CLASS-PT does in C) vs our h-unit computation
- Extracts pk_mult[14] (P_tree from CLASS-PT C code) and compares to pk_lin(k*h, z)
- This will determine whether unit choice for FFTLog matters and whether P_lin is consistent

---

## Established numerical conventions

### CLASS-PT k-unit convention
- C code (`nonlinear_pt.c`): kdisc in 1/Mpc, range [0.00005*h, 100*h] Mpc^{-1}
- `initialize_output(k, z, N)`: k must be in **1/Mpc**
- `pk_lin(k, z)`: k in 1/Mpc, returns P in Mpc³
- `pk_mm_real(cs)`: returns in (Mpc/h)³ (multiplied by h³ internally in classy.pyx)

### CLASS-PT B parameters (confirmed from nonlinear_pt.c source)
- `B_MATTER = -0.3`  (line 5826: `double b = -0.3`)
- `B_BASIC = -1.6`   (line 11789: `double b2 = -1.6000001`)

### Class-PT pk_mm_real formula (from classy.pyx)
  pk_mm_real(cs) = (pk_mult[0] + pk_mult[14] + 2*cs*pk_mult[10]/h²) * h³
where:
  - pk_mult[0] = P_loop  (P13 + P22)
  - pk_mult[14] = P_tree
  - pk_mult[10] = P_ctr (note: appears with factor -1 somewhere; CHECK SIGN)
  All pk_mult in Mpc³ (CLASS internal units); classy multiplies by h³ to get (Mpc/h)³.

### Sigma_v formula
  sigma_v² = (1/6π²) ∫ P_lin(k) k d(ln k)   [i.e., ∫ P(k) dk / (6π²)]
  Using EPT k-grid (0.00005..100 h/Mpc) and trapezoid rule over ln k: **18.01 (Mpc/h)²**

### IR resummation (when enabled)
  Pbin (FFTLog input) = P_nw + P_w × exp(-Σ² k²)            [line 5739]
  P_tree               = P_nw + P_w × exp(-Σ² k²)(1 + Σ² k²) [line 5755]

---

## Files created/modified in this session

- `clax/ept.py`: full implementation (no changes this session)
- `scripts/generate_classpt_reference.py`: bug identified (k/h → k*h) but NOT committed
- `scripts/diag_ploop_components.py`: NEW — diagnostic comparing P13/P22 components
- `scripts/diag_ptree_vs_plin.py`: NEW — in-progress diagnostic comparing 1/Mpc vs h-unit FFTLog
- `scripts/diagnostic_notes.md`: this file

---

## What to do next

1. **Fix generate_classpt_reference.py**: `k/h → k*h` and `k_h → k_h * h` in initialize_output
2. **Run diag_ptree_vs_plin.py**: compare 1/Mpc-unit FFTLog vs h-unit — if they differ,
   the M13/M22 matrices encode a specific unit and we must match it
3. **If FFTLog unit matters**: change JAX code to use k in 1/Mpc internally
4. **Regenerate classpt_z0.6.npz** after fixing the script
5. **Write tests/test_p13.py, test_p22.py** once P_loop matches within 1%
