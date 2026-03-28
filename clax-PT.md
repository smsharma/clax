# clax-PT: Implementation Notes and Design Decisions

This document tracks design decisions, implementation status, and accuracy at each
milestone for the CLASS-PT extension to clax.

See `CLASS-PT.md` for physics reference. See `CHANGELOG.md` for progress log.

---

## Architecture

### Pipeline Extension

clax-pt extends the existing clax pipeline with a new `ept` module:

```
CosmoParams
  → background_solve  → BackgroundResult (H, D, f growth rate)
  → thermodynamics_solve → ThermoResult
  → perturbations_solve → PerturbationResult (δ_m(k,τ))
  → [new] compute_pk_ept → EPTResult (P13, P22, bias components)
```

The EPT module takes:
- Linear P(k) at redshift z (from perturbations result + primordial spectrum)
- Growth rate f = d ln D / d ln a (from background)
- Hubble parameter h (from CosmoParams)

And produces all spectral components needed for galaxy power spectra.

### Module: `clax/ept.py`

Pure JAX functions, fully differentiable. Key types:

```python
EPTPrecisionParams   # static (controls shapes, FFTLog precision)
EPTComponents        # JAX pytree (all spectral arrays at one z)

# Core computation
compute_ept(pk_lin_h, k_h, h, f, prec) → EPTComponents

# Galaxy spectra (take EPTComponents + bias params)
pk_mm_real(ept, cs0=0.) → array
pk_gm_real(ept, b1, b2, bG2, bGamma3, cs0=0.) → array
pk_gg_real(ept, b1, b2, bG2, bGamma3, cs0=0., cs=0., Pshot=0.) → array
pk_mm_l0/l2/l4(ept, cs0/2/4=0.) → array
pk_gg_l0(ept, b1, b2, bG2, bGamma3, cs0, Pshot, b4=0.) → array
pk_gg_l2(ept, b1, b2, bG2, bGamma3, cs2, b4=0.) → array
pk_gg_l4(ept, b1, b2, bG2, bGamma3, cs4, b4=0.) → array
```

### Matrix Loading

The CLASS-PT kernel matrices are loaded once at module import from
`~/CLASS-PT/pt_matrices/`. They are cosmology-independent (depend only on
N_max and the FFTLog bias parameter b).

Loading is done with numpy (outside JAX JIT), then converted to jnp.array
inside jit-compiled functions.

---

## Design Decisions

### D1: Work in h-units internally

CLASS-PT uses k in h/Mpc, P in (Mpc/h)³. We match this convention internally
to allow direct comparison with CLASS-PT reference data.

Interface with clax: caller converts k[Mpc⁻¹] → k/h and P[Mpc³] → P*h³
before passing to EPT functions.

Why: FFTLog grid k_min = 5×10⁻⁵ h/Mpc, k_max = 100 h/Mpc are defined in h-units.
Matching CLASS-PT avoids an extra unit-conversion bug class.

### D2: Load matrices from CLASS-PT directory

Path: `~/CLASS-PT/pt_matrices/M{13,22,22basic}oneline_N256.dat` and `IFG2oneline_N256.dat`.

This is a development dependency, not a runtime requirement. For deployment, matrices
should be bundled with clax-pt or recomputed from the Python generation scripts.

Alternative considered: Recompute matrices in JAX using Gamma functions.
Rejected (for now): Would require complex Gamma function evaluations, adding ~200 lines.
Will add as `clax/ept_matrices.py` in a later phase for self-contained operation.

### D3: JAX-native FFT via jnp.fft.fft

CLASS-PT uses FFTPACK (C). We use jnp.fft.fft which is XLA-backed, GPU-accelerated,
and differentiable. Convention matches (standard DFT with negative exponent).

### D4: Full matrix for M22 (unpack from triangular)

CLASS-PT stores M22 as lower-triangular packed (Hermitian). We unpack to full
(N_max+1, N_max+1) matrix at load time. Memory: 257² × 16 bytes ≈ 1 MB. Fine.

Reason: jnp.matmul is more readable and equally fast (XLA handles symmetry).

### D5: IR resummation via DST (matching CLASS-PT exactly)

The BAO separation uses Discrete Sine Transform II, zeroing coefficients [120,240],
then DST-III inverse. This matches CLASS-PT exactly.

A Gaussian-smoothing approximation (rejected): gives ~0.5% error in the BAO peak
region vs DST. Not acceptable for <0.1% accuracy target.

### D6: UV counterterm in P13

P13 diverges as q → ∞. CLASS-PT subtracts the UV piece:
  P13_UV = −(61/105) × σ_v² × k² × P_lin(k)

where σ_v² = (1/6π²) ∫ dk P_lin(k) is the 1D velocity dispersion.

This renders P13 UV-finite. The remaining EFT counterterm cs0·k²·P_lin is the
free parameter absorbed by renormalization.

### D7: Growth rate f as external input

f = d ln D / d ln a is not computed inside ept.py; it's passed by the caller.
This keeps ept.py stateless and simplifies testing.

For integration with clax: compute f from background.py growth factor D(z) using
  f ≈ Ω_m(z)^0.55  or  f = d ln D / d ln a via finite difference.

### D8: EPTComponents is a JAX pytree

All arrays in EPTComponents are JAX-traceable, enabling:
- jit compilation of the full compute_ept → pk_gg pipeline
- vmap over redshift
- grad w.r.t. b1, b2, etc.

---

## Accuracy Targets

| Observable | Target | Current |
|------------|--------|---------|
| P_mm(k, cs0=0) vs CLASS-PT | <1% at k<0.5 h/Mpc | TBD |
| P_gg_l0 vs CLASS-PT | <1% | TBD |
| P_gg_l2 vs CLASS-PT | <1% | TBD |
| P_gg_l4 vs CLASS-PT | <2% | TBD |
| IR-resummed P_mm | <1% at BAO scale | TBD |
| Gradient d(P_mm)/d(b1) | <1% vs finite diff | TBD |

---

## Implementation Phases

### Phase 1: Matter power spectrum [CURRENT]
- [x] Matrix loading (M13, M22, M22basic, IFG2)
- [x] FFTLog decomposition (cmsym, etam)
- [x] P22 via M22 matrix product
- [x] P13 via M13 vector dot product + UV counterterm
- [x] P_CTR = −k² P_lin
- [x] P_tree = P_lin (no IR resummation yet)
- [x] pk_mm_real(cs0)

### Phase 2: IR resummation
- [ ] BAO wiggle separation via DST-II
- [ ] Σ_BAO computation from P_nw
- [ ] Resummed P_tree and 1-loop inputs
- [ ] pk_mm_real with BAO damping

### Phase 3: Bias cross-spectra
- [ ] P_Id2d2, P_Id2, P_IG2, P_Id2G2, P_IG2G2 (from M22basic)
- [ ] P_IFG2 (from IFG2 matrix)
- [ ] pk_gm_real, pk_gg_real

### Phase 4: RSD multipoles
- [ ] P_0_vv/vd/dd tree and 1-loop
- [ ] Bias RSD cross-terms
- [ ] pk_mm_l0/l2/l4
- [ ] pk_gg_l0/l2/l4

---

## How EPT Maps to CLASS-PT

| clax-pt function | CLASS-PT equivalent | Notes |
|-----------------|---------------------|-------|
| `compute_ept()` | `nonlinear_pt_loop()` | Main loop |
| `_fftlog_decompose()` | lines 5859–5938 | FFT + symmetrize |
| `_compute_p13_p22()` | lines 6068–6145 | Matrix products |
| `_ir_resummation()` | lines 5315–5776 | DST-based |
| `pk_mm_real()` | `classy.pk_mm_real()` | Real-space matter |
| `pk_gg_real()` | `classy.pk_gg_real()` | Real-space galaxy |
| `pk_gg_l0()` | `classy.pk_gg_l0()` | Galaxy monopole |
| `EPTComponents` | `pnlpt->ln_pk_*` arrays | Internal storage |

---

## Known Issues and Failed Approaches

(To be updated as implementation progresses)

---

## Testing Strategy

1. **Reference data**: Generated via `scripts/generate_classpt_reference.py` using
   CLASS-PT Python wrapper (classy) with same fiducial cosmology as clax tests.

2. **Comparison**: `tests/test_ept.py` loads reference data and compares each
   spectral component with <1% tolerance.

3. **Fast mode**: `--fast` flag tests 10% of k-modes (every 10th).

4. **Gradient tests**: Finite-difference check of d(P_mm)/d(b1), d(P_gg_l0)/d(b2), etc.
   Should agree to <1% for parameters varied by 1%.
