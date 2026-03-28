# CLASS-PT: Physics, Architecture, and Algorithm Reference

**CLASS-PT** (Chudaykin, Ivanov, Philcox & Simonovic 2020; arXiv:2012.04636) is an extension
of the CLASS Boltzmann solver that computes one-loop EFT power spectra for matter, galaxy, and
matter-galaxy cross-correlations, including redshift-space distortions and BAO resummation.

This document serves as a reference for the clax-pt reimplementation. All equations reference
the CLASS-PT source at `~/CLASS-PT/source/nonlinear_pt.c` (15,842 lines).

---

## 1. Physics Overview

### 1.1 EFT of Large-Scale Structure

The one-loop EFT matter power spectrum at redshift z:
```
P_mm(k,z) = P_tree(k,z) + P_1loop(k,z) + 2 c_s^2 k^2 P_lin(k,z)
```
where:
- **P_tree** = IR-resummed linear power spectrum (BAO damping applied)
- **P_1loop = P13 + P22** = one-loop corrections computed via FFTLog
- **c_s^2 k^2 P_lin** = EFT counterterm absorbing UV sensitivity (free parameter c_s²)

### 1.2 Galaxy Bias Expansion

The galaxy overdensity in real space (Einstein-de Sitter approximation):
```
δ_g = b1 δ + b2/2 δ² + bG2 G2[Φ] + bΓ3 Γ3 + ...
```
where:
- b1 = linear bias
- b2 = second-order density bias
- bG2 = tidal (Galileon) bias
- bΓ3 = third-order operator (degenerate with bG2 at 1-loop for Gaussian IC)

The real-space galaxy power spectrum at 1-loop:
```
P_gg(k) = b1² [P_tree + P_1loop] + 2(c_s b1² + c_s0 b1) k² P_lin / h²
         + b1 b2 P_δ,δ²  +  b2²/4 P_δ²δ²
         + 2 b1 bG2 P_δ,G2  +  b1(2bG2 + 0.8 bΓ3) P_δ,FG2
         + bG2² P_G2G2  +  b2 bG2 P_δ²G2  +  P_shot
```

### 1.3 Redshift-Space Distortions

In the Kaiser limit, the RSD power spectrum is decomposed in powers of μ = k_∥/k:
```
P(k, μ) = Σ_{n=0}^{4} A_n(k) μ^{2n}
```
The multipoles (Legendre decomposition):
```
P_ℓ(k) = (2ℓ+1)/2 ∫_{-1}^{1} P(k,μ) L_ℓ(μ) dμ
```
For ℓ=0,2,4: involves up to μ^8 terms from 1-loop velocity field.

Key growth rate: f = d ln D / d ln a ≈ Ω_m(z)^0.55

RSD tree-level (Kaiser formula):
```
P_0^tree = (b1² + 2b1f/3 + f²/5) P_lin
P_2^tree = (4b1f/3 + 4f²/7) P_lin
P_4^tree = 8f²/35 P_lin
```

### 1.4 IR Resummation

The BAO peak in P(k) is damped by large-scale bulk motions. CLASS-PT uses the
displacement-field approach:
```
P_resummed(k) = P_nw(k) + P_w(k) exp(-Σ_BAO² k²)
```
where:
- **P_nw** = no-wiggle (broadband) spectrum (extracted via DST-II smoothing)
- **P_w = P_lin - P_nw** = wiggle (BAO) component
- **Σ_BAO²** = isotropic damping scale:
  ```
  Σ_BAO² = (1/6π²) ∫_0^{k_IR} dk P_nw(k)
  ```
  with k_IR ≈ 0.2 h/Mpc

**DST-II BAO separation** (CLASS-PT lines 5315–5776):
1. Compute f(k) = log(k P_lin(k)) on fine grid (N_IR = 65536, k ∈ [7×10⁻⁵, 7] h/Mpc)
2. Apply Discrete Sine Transform II to get oscillatory modes
3. Zero coefficients in range [120, 240] (corresponds to BAO period ~150 Mpc/h)
4. Inverse DST-III → P_nw

---

## 2. FFTLog Algorithm

### 2.1 Motivation

One-loop integrals have the form:
```
P_22(k) = ∫ d³q/(2π³) K(k,q) P_lin(q) P_lin(|k-q|)
```
Direct evaluation: O(k² × N_q²). FFTLog reduces this to O(k × N_q log N_q).

### 2.2 Algorithm

**Reference**: Schmittfull et al. (2016); CLASS-PT `nonlinear_pt.c` lines 5818–5948.

**Setup parameters:**
```
N_max = 256         (FFTLog modes, precision parameter)
b = -0.3            (bias for matter PS)
b_T = -0.8          (bias for transfer functions)
k_min = 5×10⁻⁵ h/Mpc
k_max = 100 h/Mpc
Δ = log(k_max/k_min)/(N_max-1)   (log-k step)
```

**Step 1: Input preparation**

The k-grid has N_max log-spaced points:
```
k_j = k_min × exp(j × Δ),  j = 0..N_max-1
```
Multiply P(k_j) by a power-law weight:
```
input[j] = P(k_j) × exp(-j × b × Δ) = P(k_j) × (k_j/k_min)^{-b}
```

**Step 2: DFT**
```
c̃[m] = DFT{input}[m] = Σ_{j=0}^{N_max-1} input[j] × exp(-2πi j m / N_max)
```

**Step 3: Symmetrize and normalize**

Define signed mode index j_m = m - N_max/2, so m ∈ [0, N_max]:
```
complex freq: η_m = b + 2πi j_m / (N_max × Δ)
```
Symmetrize (m → N_max-m corresponds to complex conjugation):
```
c_sym[m] = conj(c̃[N_max/2 - m]) / N_max   for m < N_max/2
c_sym[m] = c̃[m - N_max/2] / N_max          for m ≥ N_max/2
```
Apply k_min factor and half-weight endpoints:
```
c_m = k_min^{-η_m} × c_sym[m]
c_0 /= 2,  c_{N_max} /= 2
```

**Step 4: Evaluate at output k**

For each output k_j (on same grid), compute the mode coefficients:
```
x_j[m] = c_m × k_j^{η_m}
```

**Step 5: P22 via matrix-vector product** (CLASS-PT lines 6082–6102)
```
y = M22 · x_j        (matrix-vector, shape N_max+1)
P22(k_j) = Re{k_j³ × x_j · y} × exp[-(k_j/Λ_cut)⁶]
```
where M22 is the precomputed (N_max+1)×(N_max+1) kernel matrix stored in
`pt_matrices/M22oneline_N256.dat`.

**Step 6: P13 via vector dot product** (CLASS-PT lines 6068–6079)
```
f13 = x_j · M13       (dot product, M13 is a (N_max+1,) vector)
P13_raw(k_j) = Re{k_j³ × f13 × P_lin(k_j)}
P13(k_j) = P13_raw(k_j) - (61/105) k_j² σ_v² P_lin(k_j)   [UV renormalization]
```
where M13 is stored in `pt_matrices/M13oneline_N256.dat` and
σ_v² = (1/6π²) ∫ dk P_lin(k) is the velocity dispersion.

---

## 3. Kernel Matrices

All matrices are computed once (cosmology-independent!) from the SPT kernels
evaluated at complex frequencies η_m. Stored in `~/CLASS-PT/pt_matrices/`.

### 3.1 Matrix File Format

ASCII text, one double per line:
- First (N_max+1)(N_max+2)/2 lines: real parts (lower triangular packed)
- Next (N_max+1)(N_max+2)/2 lines: imaginary parts

For M13/IFG2: first N_max+1 lines = real parts, next N_max+1 = imaginary parts.

### 3.2 Available Matrices (N=256)

| File | Shape | Purpose |
|------|-------|---------|
| `M22oneline_N256.dat` | (257,257) packed | P22 matter-matter (F2 kernel) |
| `M22basiconeline_N256.dat` | (257,257) packed | Bare loop I(ν₁,ν₂) without kernel |
| `M13oneline_N256.dat` | (257,) | P13 matter-matter |
| `IFG2oneline_N256.dat` | (257,) | F·G2 coupling term |
| `M12oneline_N256.dat` | (257,257) packed | Primordial NG (P12 contribution) |

### 3.3 Matrix Construction

The matrices encode the PT loop kernel evaluated at complex frequencies:
```
M22[m,n] = J(η_m, η_n) × F2_kernel(-η_m/2, -η_n/2)
J(ν₁,ν₂) = Γ(1.5-ν₁)Γ(1.5-ν₂)Γ(ν₁+ν₂-1.5) / [8π^1.5 Γ(ν₁)Γ(ν₂)Γ(3-ν₁-ν₂)]
```
where F2_kernel is the SPT mode-coupling kernel in Fourier space.

M22basic contains only J(ν₁,ν₂) (no kernel), used for bias cross-terms.

---

## 4. Spectral Components (pk_mult array)

CLASS-PT computes 96 spectral components internally (pk_mult[0..95]).
The key ones for standard (non-PNG) applications:

### 4.1 Real-Space Components (indices 0–13)

| Index | Name | Formula |
|-------|------|---------|
| 0 | P_1loop | P13 + P22 (1-loop matter) |
| 1 | P_Id2d2 | δ²δ² power (from M22basic) |
| 2 | P_Id2 | b1·δ² cross (from M22basic) |
| 3 | P_IG2 | b1·G2 cross (from M22basic) |
| 4 | P_Id2G2 | δ²·G2 cross (from M22basic) |
| 5 | P_IG2G2 | G2·G2 power (from M22basic) |
| 6 | P_IFG2 | F·G2 coupling (from IFG2) |
| 7 | P_IFG2_0b1 | FG2 monopole, b1-weighted |
| 8 | P_IFG2_0 | FG2 monopole |
| 9 | P_IFG2_2 | FG2 quadrupole |
| 10 | P_CTR | counterterm: −k² P_lin |
| 11 | P_CTR_0 | counterterm monopole (RSD) |
| 12 | P_CTR_2 | counterterm quadrupole |
| 13 | P_CTR_4 | counterterm hexadecapole |
| 14 | P_tree | IR-resummed linear P |

### 4.2 RSD Multipole Components (indices 15–47)

Tree-level (indices 15–20):

| Index | Name | RSD contribution |
|-------|------|-----------------|
| 15 | P_0_tree_vv | b1² monopole tree |
| 16 | P_0_tree_vd | b1·f monopole tree |
| 17 | P_0_tree_dd | f² monopole tree |
| 18 | P_2_tree_vv | b1² quadrupole tree |
| 19 | P_2_tree_vd | b1·f quadrupole tree |
| 20 | P_4_tree_vv | b1² hexadecapole tree |

1-loop (indices 21–47): similar decomposition into vv/vd/dd for ℓ=0,2,4
plus bias cross-terms b1b2, b2, b1bG2, bG2 for each multipole.

---

## 5. Galaxy Bias Formulas

### 5.1 Real-Space Galaxy-Galaxy Power

```python
P_gg(k) = (
    b1**2 * P_tree          # Tree-level (index 14)
  + b1**2 * P_1loop         # 1-loop matter (index 0)
  + 2*(cs*b1**2 + cs0*b1) * P_CTR / h**2  # EFT counterterm (index 10)
  + b1*b2 * P_Id2           # b1×b2 cross (index 2)
  + b2**2/4 * P_Id2d2       # b2² self (index 1)
  + 2*b1*bG2 * P_IG2        # b1×G2 cross (index 3)
  + b1*(2*bG2 + 0.8*bΓ3) * P_IFG2  # tidal (index 6)
  + bG2**2 * P_IG2G2        # G2² self (index 5)
  + b2*bG2 * P_Id2G2        # b2×G2 cross (index 4)
) * h**3 + P_shot
```

### 5.2 Galaxy-Galaxy Monopole (RSD)

```python
P_gg_l0(k) = (
    P_0_tree_vv + b1*P_0_tree_vd + b1**2*P_0_tree_dd   # Tree ℓ=0
  + P_0_vv + b1*P_0_vd + b1**2*P_0_dd                   # 1-loop ℓ=0
  + b2**2/4 * P_Id2d2 + b2*bG2 * P_Id2G2 + bG2**2 * P_IG2G2
  + b1*b2*P_0_b1b2 + b2*P_0_b2 + b1*bG2*P_0_b1bG2 + bG2*P_0_bG2
  + 2*cs0 * P_CTR_0 / h**2
  + (2*bG2 + 0.8*bΓ3)*(b1*P_IFG2_0b1 + P_IFG2_0)
) * h**3 + P_shot
  + f**2 * b4 * (k/h)**2 * [loop factor] * P_lin
```

### 5.3 Quadrupole

```python
P_gg_l2(k) = (
    P_2_tree_vv + b1*P_2_tree_vd + [1-loop ℓ=2 terms]
  + b1*b2*P_2_b1b2 + ... + 2*cs2*P_CTR_2/h**2
  + (2*bG2 + 0.8*bΓ3)*P_IFG2_2
) * h**3 + f**2*b4*(k/h)**2*[factor]*P_lin
```

### 5.4 Hexadecapole

Similar structure with ℓ=4 components and cs4 counterterm.

---

## 6. Units and Conventions

CLASS-PT internally works in **h-units**:
- Wavenumber k: h/Mpc
- Power spectrum P(k): (Mpc/h)³
- Python output multiplied by h³ to give Mpc³ (optional)

Parameter conventions:
- cs0, cs2, cs4: EFT counterterms in (Mpc/h)²
- Pshot: shot noise in (Mpc/h)³
- b4: higher-order stochastic bias (dimensionless × Mpc²/h²)
- f = growth rate d ln D/d ln a

EFT parameter range (typical BOSS CMASS):
- b1 ≈ 1.5–2.5
- b2 ≈ −2 to +2
- bG2 ≈ −1 to +1
- cs0 ≈ 0–50 (Mpc/h)²
- Pshot ≈ 1000–5000 (Mpc/h)³

---

## 7. Computational Complexity

| Step | Complexity | Time (N=256) |
|------|-----------|-------------|
| FFT decompose | O(N log N) | <1 ms |
| P22 per k-mode | O(N²) | ~1 ms |
| P22 total (N modes) | O(N³) | ~0.3 s (CPU) |
| P13 total | O(N²) | <0.1 s |
| IR resummation (DST) | O(N_IR log N_IR) | <10 ms |

GPU acceleration via vmap over k-modes expected to be 10–100× faster.

---

## 8. Key References

1. **Ivanov, Simonovic & Zaldarriaga (2020)** arXiv:1909.05273 — EFT galaxy power spectrum
2. **Chudaykin, Ivanov, Philcox & Simonovic (2020)** arXiv:2012.04636 — CLASS-PT paper
3. **Schmittfull, Feng, Harikane & Zaldarriaga (2016)** arXiv:1603.04405 — FFTLog for PT
4. **d'Amico, Lewandowski, Senatore & Zhang (2022)** arXiv:2201.07241 — EFTofLSS review
5. **Senatore & Zaldarriaga (2014)** arXiv:1404.7274 — EFT of LSS original
6. **McEwen, Fang, Hirata & Blazek (2016)** arXiv:1603.04826 — FAST-PT (related FFTLog)
