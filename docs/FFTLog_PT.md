# FFTLog Method for 1-Loop Power Spectra

**References**:
Schmittfull et al. (2016) [arXiv:1603.04405](https://arxiv.org/abs/1603.04405),
Chudaykin et al. (2020) [arXiv:2004.10607](https://arxiv.org/abs/2004.10607),
CLASS-PT `nonlinear_pt.c`

See also: [CLASS-PT-summary.md](CLASS-PT-summary.md) for the component index
table and assembly formulas.

---

## 1. FFTLog Power-Law Decomposition

Decompose the linear power spectrum on a log-spaced k-grid into complex
power laws:

```
P(k) = sum_m c_m k^{-2 nu_m}
```

where `nu_m = -eta_m / 2` and `eta_m` are the FFTLog complex frequencies.

**k-grid** (N_max log-spaced points, CLASS-PT defaults N_max = 256):

```
k_j = k_min exp(j Delta),   j = 0, ..., N_max - 1
Delta = log(k_max / k_min) / (N_max - 1)
k_min = 5e-5 h/Mpc,  k_max = 100 h/Mpc
```

**Complex frequencies** (N_max + 1 values including Nyquist):

```
eta_m = b + 2 pi i j_m / (N_max Delta),    j_m = m - N_max/2
```

The real part b is the FFTLog bias parameter. Three values are used:
- b = -0.3 for matter (M22, M13)
- b = -0.8 for transfer functions
- b = -1.6 for bias spectra (M22basic)

### DFT convention

The forward DFT uses the standard negative-exponent convention (numpy/JAX
`fft` default):

```
c_tilde[m] = sum_{j=0}^{N-1} input[j] exp(-2 pi i j m / N)
```

### Decomposition algorithm

1. **Weight**: `input[j] = P(k_j) * (k_j / k_min)^{-b}`
2. **DFT**: `c_tilde = FFT(input)`
3. **Symmetrize** ("cmsym"): map the one-sided DFT to a double-sided
   spectrum on m = 0, ..., N_max:
   ```
   c_sym[m] = conj(c_tilde[N/2 - m]) / N    for m < N/2
   c_sym[m] = c_tilde[m - N/2] / N           for m >= N/2
   ```
4. **Normalize**: `c_m = k_min^{-eta_m} * c_sym[m]`
5. **Half-weight endpoints**: `c_0 /= 2`, `c_{N_max} /= 2`

The mode coefficients at any output k are then:

```
x_m(k) = c_m * k^{eta_m}
```

---

## 2. The Master Integral

All 1-loop integrals reduce to the master integral over two power laws:

```
I(alpha, beta) = integral d^3q / (2pi)^3  |q|^alpha |k-q|^beta
```

which evaluates to a product of Gamma functions:

```
I(nu1, nu2) = Gamma(3/2 - nu1) Gamma(3/2 - nu2) Gamma(nu1 + nu2 - 3/2)
              / [8 pi^{3/2} Gamma(nu1) Gamma(nu2) Gamma(3 - nu1 - nu2)]
```

where `nu1 = -eta_i / 2`, `nu2 = -eta_l / 2`. This is the key
simplification: the angular integration and SPT kernel evaluation
become algebraic operations on the complex frequencies, completely
independent of cosmology.

---

## 3. Real-Space Matter: P13 and P22

### P22 as a bilinear form

```
P22(k) = Re{ k^3 * x^T M22 x } * exp[-(k / k_cut)^6]
```

M22 is an (N_max+1) x (N_max+1) **symmetric** (NOT Hermitian) matrix.
The bilinear form uses `x^T M22 x` with no complex conjugation, matching
CLASS-PT's `zdotu` convention. Symmetry follows from `F2(k1,k2) = F2(k2,k1)`.

The matrix is precomputed from I(nu1, nu2) weighted by the F2 kernel and
stored in LAPACK lower-triangular packed format (`M22_oneline_complex.dat`).

UV cutoff: `k_cut = 3 h/Mpc` (exponential damping removes UV artifacts).

### P13 as a vector contraction

```
P13_raw(k) = Re{ k^3 * (x . M13) * P_lin(k) }
P13(k)     = (P13_raw + P13_UV) * exp[-(k / k_cut)^6]
```

where M13 is an (N_max+1,) complex vector (precomputed from the F3 kernel
contracted with the master integral).

### UV renormalization

The raw P13 contains a UV divergence proportional to k^2 P_lin(k). This is
subtracted analytically:

```
P13_UV(k) = -(61/105) sigma_v^2 k^2 P_lin(k)
```

where `sigma_v^2 = (1/6pi^2) integral dk P_lin(k)` is the 1D velocity
dispersion, computed via trapezoidal rule on the FFTLog k-grid.

### Bias cancellation

The UV-divergent parts of P13 and P22 cancel exactly in the sum
`P_1loop = P13 + P22`, regardless of the FFTLog bias b. This allows
choosing b for numerical stability without affecting the physical result.

---

## 4. Extension to Tracers in Redshift Space

The same FFTLog factorization applies to the redshift-space Z_n kernels
instead of the matter-only F_n kernels. The key change is that the kernels
acquire dependence on the growth rate f and the line-of-sight direction mu,
leading to a richer set of spectral components.

### mu-power decomposition

1-loop contributions to P(k, mu) decompose into bare mu-power channels:

| Channel       | mu powers | P22 kernel   | P13 kernel   |
|---------------|-----------|--------------|--------------|
| P_dd          | mu^0      | M22          | M13          |
| P_dd (mu^2)   | mu^2      | M22_mu2_dd   | M13_mu2_dd   |
| P_vd (mu^2)   | mu^2      | M22_mu2_vd   | M13_mu2_vd   |
| P_vv (mu^4)   | mu^4      | M22_mu4_vv   | M13_mu4_vv   |
| P_vd (mu^4)   | mu^4      | M22_mu4_vd   | M13_mu4_vd   |
| P_dd (mu^4)   | mu^4      | M22_mu4_dd   | --           |
| P_vv (mu^6)   | mu^6      | M22_mu6_vv   | M13_mu6      |
| P_vd (mu^6)   | mu^6      | M22_mu6_vd   | --           |
| P_vv (mu^8)   | mu^8      | M22_mu8      | --           |

Each M22 kernel is obtained by multiplying the base M22 matrix element-wise
by a rational function of (nu1, nu2, f). For example:

```
M22_mu4_vv[i,l] = M22[i,l] * K_mu4_vv(nu_i, nu_l, f)
```

where `K_mu4_vv` is a polynomial in nu1, nu2, f divided by a common
denominator D that encodes the F2 kernel structure:

```
D = 98 nu1 nu2 nu12^2 - 91 nu12^2 + 36 nu1 nu2
    - 14 nu1 nu2 nu12 + 3 nu12 + 58
```

The M13 RSD kernels are element-wise multiplications of the base M13
vector by 1D rational functions of (nu1, f).

Each channel also has a UV counterterm coefficient (analogous to -61/105
for matter P13) that depends on sigma_v^2 and f.

### Bias cross-spectra

Bias operators (delta^2, G2, FG2) produce additional cross-spectra.
These use a separate FFTLog decomposition with bias b = -1.6 and the
M22basic matrix (the master integral I(nu1, nu2) without the F2 kernel),
multiplied by operator-specific rational kernels.

The bias spectra depend only on (nu1, nu2) through the sum s = nu1 + nu2,
making them symmetric in the two FFTLog indices.

---

## 5. IR Resummation

BAO features in the linear power spectrum receive non-perturbative corrections
from long-wavelength displacements. IR resummation damps the BAO wiggles
while preserving the broadband shape.

### No-wiggle / wiggle split

Separate `P_lin = P_nw + P_w` using a DST-II filter:

1. Evaluate P_lin on a fine linear k-grid (N = 65536 points,
   k in [7e-5/h, 7/h] h/Mpc)
2. DST-II of `log(k P_lin(k))`
3. Split into odd/even mode sub-arrays; remove modes 120-240
   from each via cubic spline interpolation
4. Inverse DST to reconstruct P_nw

Modes 120-240 on this linear grid correspond to oscillation periods
that bracket the BAO scale 2pi/r_s ~ 0.06 h/Mpc.

### BAO damping scales

**Isotropic damping** Sigma^2:

```
Sigma^2 = (1/6pi^2) integral_0^{k_s} dq P_nw(q) [1 - j_0(qr) + 2 j_2(qr)] q
```

where `k_s = 0.2 h/Mpc`, `r = r_s` (sound horizon at drag), and j_0, j_2
are spherical Bessel functions. Equivalently, the filter is:

```
F(x) = 1 - 3 sin(x)/x + 6 (sin(x)/x^3 - cos(x)/x^2),   x = q r_s
```

**Anisotropic correction** delta_Sigma^2:

```
delta_Sigma^2 = (1/2pi^2) integral_0^{k_s} dq P_nw(q)
                * [- (3 cos(x) x + (-3 + x^2) sin(x)) / x^3] q
```

### Anisotropic Sigma_tot^2(mu)

The full direction-dependent damping scale combines both:

```
Sigma_tot^2(mu) = Sigma^2 [1 + f mu^2 (2 + f)]
                + delta_Sigma^2 f^2 mu^2 (mu^2 - 1)
```

### Tree-level assembly (anisotropic)

```
P_tree(k, mu) = P_nw(k) + P_w(k) exp(-Sigma_tot^2(mu) k^2) (1 + Sigma_tot^2(mu) k^2)
```

The `(1 + Sigma_tot k^2)` factor compensates the leading-order displacement
effect that would otherwise suppress the BAO peak amplitude.

### 1-loop assembly (nw/w split)

Each bare mu-power channel is split into nw and w parts. For P22:

```
P22_IR = P22(x_nw) + Re{k^3 x_nw^T M22 (2 x_w)} exp(-Sigma^2 k^2)
```

For P13:

```
P13_IR = [Re{k^3 f13_nw P_lin} + UV k^2 P_lin]
       + Re{k^3 f13_w P_nw} exp(-Sigma^2 k^2)
```

The nw piece absorbs the UV counterterm; the w piece is damped by the
isotropic exp(-Sigma^2 k^2) for the pre-projection components. After
assembly into P(k, mu), the full anisotropic Sigma_tot(mu) applies.

---

## 6. Multipole Projection via Gauss-Legendre Quadrature

**Key design choice**: assemble the full P(k, mu) first (including
anisotropic IR damping), then project onto Legendre multipoles.

Multipole extraction:

```
P_ell(k) = (2 ell + 1)/2 integral_{-1}^{1} dmu L_ell(mu) P(k, mu)
```

evaluated by Gauss-Legendre quadrature (40 nodes from CLASS-PT
`gauss_tab.dat`). At each GL node mu_i:

1. Compute `Sigma_tot^2(mu_i)`
2. Compute `E(mu_i) = exp(-Sigma_tot^2(mu_i) k^2)`
3. For each bare mu-power channel, form `P_loop_channel(k, mu_i)`:
   - nw part times the P13 ratio `r13 = 1 + (P_w / P_nw) E`
   - w part times E
4. Sum all channels weighted by the appropriate mu power
5. Accumulate into multipoles: `P_ell += w_i * (2 ell + 1)/2 * L_ell(mu_i) * P(k, mu_i)`

This approach naturally handles the anisotropic IR damping (which mixes
mu powers) without needing analytic mu integrals for each damped channel.
It also produces tree-level multipole components (Pk_0_dd, Pk_2_dd, etc.)
that are nonzero even for channels that would vanish in the isotropic limit
(e.g., Pk_4_dd), because the mu-dependent damping breaks the factorization.

---

## 7. Units and Conventions

- **k**: h/Mpc throughout (both internal FFTLog grid and output)
- **P(k)**: (Mpc/h)^3 throughout
- The `k^3` factor in P22 and P13 formulas comes from the loop integral
  measure after angular integration
- CLASS-PT exception: `pk_lin(k, z)` in the Python wrapper takes k in
  1/Mpc and returns P in Mpc^3. Convert via
  `P_lin_h = pk_lin(k_h * h, z) / h^3`.

---

## 8. Implementation Notes

- `jnp.fft.fft` uses the same convention as numpy (negative exponent forward)
- M22 matrix-vector products dominate cost: O(N^2) per k-mode
- All operations (FFT, matmul, real-part extraction) are differentiable
  in JAX. The `Re{}` via `jnp.real` has a VJP that zeros imaginary gradients.
- The sigma_v^2 integral for UV renormalization uses `jnp.trapezoid` on
  the FFTLog k-grid
- M22 is loaded from LAPACK 'L' packed format and reconstructed as a full
  symmetric matrix (not Hermitian -- `M[j,i] = M[i,j]`, no conjugation)
- IR resummation (DST-II) runs in NumPy (not differentiable). To enable
  `jax.grad`, pass the precomputed nw/w split via `_ir_precomputed` and
  let `pk_w = pk_lin_h - pk_nw` be JAX-traced
