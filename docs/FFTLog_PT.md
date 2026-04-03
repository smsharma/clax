# FFTLog for Perturbation Theory Loop Integrals

**Reference**: Schmittfull, Feng, Harikane & Zaldarriaga (2016), arXiv:1708.08130
**Implementation**: CLASS-PT `nonlinear_pt.c` lines 5818–5948, 6068–6102

---

## 1. Core Idea

One-loop integrals have the form:
```
P_22(k) = ∫ d³q/(2π³) K(k,q) P_lin(q) P_lin(|k-q|)
P_13(k) = P_lin(k) ∫ d³q/(2π³) K13(k,q) P_lin(q)
```
Direct evaluation: O(N² per k-mode). FFTLog reduces this to O(N log N) by exploiting that if P(k) = k^ν, all loop integrals become products of Gamma functions.

**Key insight**: Decompose P_lin(k) into a discrete sum of complex power laws:
```
P_lin(k) ≈ Σ_m c_m k^{η_m}
```
where the coefficients c_m are obtained via FFT on a log-k grid. Then each loop integral reduces to:
```
P_22(k) = Σ_{m,n} c_m c_n k³ M22[m,n] k^{η_m + η_n}
P_13(k) = P_lin(k) Σ_m c_m k³ M13[m] k^{η_m}
```
The matrices M22[m,n] and vector M13[m] depend only on the frequencies η_m and the SPT kernels — **completely independent of cosmology**. Precompute them once; reuse for all P(k).

---

## 2. Frequency Grid

**Setup parameters** (CLASS-PT defaults):
```
N_max = 256         (number of FFTLog modes)
b = -0.3            (FFTLog bias for matter PS; b_T=-0.8 for transfer)
k_min = 5×10⁻⁵ h/Mpc
k_max = 100 h/Mpc
Δ = log(k_max/k_min) / (N_max - 1)  ≈ log(2×10⁶)/255
```

**The k-grid** (N_max log-spaced points):
```
k_j = k_min × exp(j × Δ),   j = 0, ..., N_max-1
```

**Complex frequencies** η_m (N_max + 1 values, including Nyquist):
```
η_m = b + 2πi j_m / (N_max × Δ)
```
where `j_m = m - N_max/2` for m ∈ [0, N_max], so j_m ranges from -N_max/2 to +N_max/2.

The real part of η_m is always `b` (the bias). The imaginary part covers the FFT frequencies.

---

## 3. FFTLog Decomposition Algorithm

### Step 1: Weight the input
```
input[j] = P(k_j) × (k_j / k_min)^{-b}
         = P(k_j) × exp(-j × b × Δ)
```
This maps P(k) k^{-b} onto a smooth function suitable for FFT.

### Step 2: Discrete Fourier Transform
```
c̃[m] = DFT{input}[m] = Σ_{j=0}^{N_max-1} input[j] × exp(-2πi j m / N_max)
```
CLASS-PT convention: **negative exponent** in the forward DFT (numpy fft convention).

### Step 3: Symmetrize and normalize (the "cmsym" formula)
The key symmetry: conjugate in time domain = reverse+conjugate in freq domain.
For the power-law expansion to be real-valued, we need c_{-m} = c_m*.

Map from DFT output `c̃[m]` (m=0..N_max-1) to symmetric coefficients on m ∈ [0, N_max]:
```
c_sym[m] = conj(c̃[N_max/2 - m]) / N_max   for m < N_max/2
c_sym[m] = c̃[m - N_max/2] / N_max          for m ≥ N_max/2
```
(This is a "half-cycle shift" that places DC at the center, with proper conjugation.)

Then apply k_min normalization and half-weight the endpoints:
```
c_m = k_min^{-η_m} × c_sym[m]
c_0 /= 2                          (half-weight DC)
c_{N_max} /= 2                    (half-weight Nyquist)
```

### Step 4: Mode coefficients at output k
For each k in the output grid:
```
x[m] = c_m × k^{η_m}    (shape: N_max+1 complex)
```

### Step 5: P22 via bilinear form
```
y = M22 · x                   (matrix-vector, M22 is (N_max+1) × (N_max+1))
P22(k) = Re{ k³ × (x · y) } × exp[-(k/Λ_cut)⁶]
```
where Λ_cut = 10 h/Mpc is a UV cutoff (removes unphysical UV contributions).

**Critical**: M22 is **symmetric** (not Hermitian). The bilinear form `x^T M22 x` uses `zdotu` (no complex conjugation), not `zdotc`. This is because the SPT F2 kernel satisfies F2(k1,k2) = F2(k2,k1), making I(η1,η2) = I(η2,η1) symmetric.

### Step 6: P13 via dot product
```
f13 = x · M13              (dot product, M13 is (N_max+1,) complex)
P13_raw(k) = Re{ k³ × f13 × P_lin(k) }
P13(k) = P13_raw(k) - (61/105) k² σ_v² P_lin(k)   [UV renormalization]
```
σ_v² = (1/6π²) ∫ dk P_lin(k) is the velocity dispersion (1D).

---

## 4. Bias Cancellation Property

The UV divergence in P_13 is proportional to k² P_lin(k), with coefficient depending on b (the FFTLog bias). Similarly P_22 has a compensating UV divergence. For the total:
```
P_loop = P_13 + P_22
```
the UV-divergent parts cancel exactly for any value of b, regardless of cosmology. This is the "bias cancellation" that justifies varying b for numerical stability without affecting the physical answer.

In practice, b = -0.3 is chosen to minimize numerical noise in the DFT (making the integrand smoother in log-space) without being too negative (which would amplify low-k noise).

---

## 5. Normalization Convention

CLASS-PT uses **h-units**: k in h/Mpc, P(k) in (Mpc/h)³.

The dimensionless power spectrum:
```
Δ²(k) = k³ P(k) / (2π²)
```

The P22 formula has an explicit `k³` factor (see Step 5) because the kernel matrices M22 integrate over the angular part, leaving only the radial Jacobian k². Combined with the measure from the loop integral gives k³.

The normalization in the CLASS-PT code:
```
P(k) = k³ × (real part of bilinear form)   [in h/Mpc units throughout]
```
Output is then multiplied by h³ in Python interface to give (Mpc/h)³.

---

## 6. Kernel Matrix Summary

| Matrix | Shape | Content | Used for |
|--------|-------|---------|----------|
| M22 | (N+1)×(N+1) symmetric | J(η_m,η_n) × F2(-η_m/2,-η_n/2) | P_mm 1-loop |
| M22basic | (N+1)×(N+1) symmetric | J(η_m,η_n) (no kernel) | Bias cross-spectra |
| M13 | (N+1,) | P13 integral kernel | P13 1-loop |
| IFG2 | (N+1,) | F·G2 coupling | bG2 bias terms |

where J(ν₁,ν₂) = Γ(1.5-ν₁)Γ(1.5-ν₂)Γ(ν₁+ν₂-1.5) / [8π^1.5 Γ(ν₁)Γ(ν₂)Γ(3-ν₁-ν₂)]

The bias cross-spectra (P_Id2, P_IG2, etc.) use M22basic instead of M22, but the same x[m] = c_m k^η_m vectors.

---

## 7. JAX Implementation Notes

- `jnp.fft.fft` uses the same convention as numpy (negative exponent forward).
- The symmetrization (Step 3) must be done carefully with complex array slicing.
- M22 matrix-vector products dominate cost: O(N²) per k-mode → use `jnp.einsum` or `jnp.dot`.
- For gradient computation: all operations (FFT, mat-mul, real-part extraction) are differentiable in JAX. The `Re{}` via `jnp.real` has VJP that zeroes imaginary gradients.
- The σ_v² integral for P13 UV renormalization should be computed on the same FFTLog k-grid using trapezoidal rule (same as CLASS-PT).
