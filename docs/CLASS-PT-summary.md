# CLASS-PT Algorithm: Focused Summary for clax-pt Implementation

**Paper**: Chudaykin, Ivanov, Philcox & Simonovic (2020), arXiv:2004.10607
**Source**: `~/CLASS-PT/source/nonlinear_pt.c`
**Python API**: `~/CLASS-PT/python/classy.pyx`

---

## 1. Computed Spectra

| Function | Description | Output units |
|----------|-------------|--------------|
| `pk_mm_real(cs)` | Matter–matter real space | (Mpc/h)³ |
| `pk_gg_real(b1,b2,bG2,bGamma3,cs,cs0,Pshot)` | Galaxy–galaxy real space | (Mpc/h)³ |
| `pk_gm_real(b1,b2,bG2,bGamma3,cs,cs0)` | Galaxy–matter real space | (Mpc/h)³ |
| `pk_mm_l0/l2/l4(cs0/2/4)` | Matter multipoles ℓ=0,2,4 (RSD) | (Mpc/h)³ |
| `pk_gg_l0(b1,b2,bG2,bGamma3,cs0,Pshot,b4)` | Galaxy monopole | (Mpc/h)³ |
| `pk_gg_l2(b1,b2,bG2,bGamma3,cs2,b4)` | Galaxy quadrupole | (Mpc/h)³ |
| `pk_gg_l4(b1,b2,bG2,bGamma3,cs4,b4)` | Galaxy hexadecapole | (Mpc/h)³ |

All call `initialize_output(k_arr_hMpc, z, k_size)` first, which:
1. Runs `get_pk_mult(k, z, k_size)` → fills `pk_mult[0..95, k_size]`
2. Stores `self.kh = k` (in h/Mpc) and `self.fz` = growth rate f(z)

`pk_lin(k_1Mpc, z)` returns P_lin in **Mpc³** (k in 1/Mpc, not h/Mpc!).

---

## 2. Bias Model

Real-space galaxy overdensity (EFT of LSS, EdS approximation):
```
δ_g = b1 δ + b2/2 δ² + bG2 G2[Φ] + bΓ3 Γ3 + ...
```

EFT parameters:
- **b1**: linear bias (~1.5–2.5 for BOSS CMASS)
- **b2**: second-order density bias (~-2 to +2)
- **bG2**: tidal (Galileon) bias (~-1 to +1)
- **bGamma3**: third-order (degenerate with bG2 at 1-loop)
- **cs0, cs2, cs4**: EFT counterterms in (Mpc/h)²
- **Pshot**: shot noise in (Mpc/h)³
- **b4**: stochastic higher-order (finger-of-god; dimensionless × Mpc²/h²)

---

## 3. pk_mult Component Index Table

CLASS-PT computes 96 spectral components internally. Key indices:

| Index | Name | Formula |
|-------|------|---------|
| 0 | P_1loop | P13 + P22 (1-loop matter) |
| 1 | P_Id2d2 | I(δ²,δ²) from M22basic |
| 2 | P_Id2 | I(δ,δ²) from M22basic |
| 3 | P_IG2 | I(δ,G2) from M22basic |
| 4 | P_Id2G2 | I(δ²,G2) from M22basic |
| 5 | P_IG2G2 | I(G2,G2) from M22basic |
| 6 | P_IFG2 | I(δ,FG2) from IFG2 vector |
| 7 | P_IFG2_0b1 | FG2 monopole, b1-weighted |
| 8 | P_IFG2_0 | FG2 monopole |
| 9 | P_IFG2_2 | FG2 quadrupole |
| 10 | P_CTR | counterterm: −k² P_lin |
| 11 | P_CTR_0 | counterterm monopole |
| 12 | P_CTR_2 | counterterm quadrupole |
| 13 | P_CTR_4 | counterterm hexadecapole |
| 14 | P_tree | IR-resummed linear P |
| 15–17 | P_0_vv/vd/dd | tree monopole (b1², b1·f, f²) |
| 18–19 | P_2_vv/vd | tree quadrupole |
| 20 | P_4_vv | tree hexadecapole |
| 21–23 | P_0_vv1/vd1/dd1 | 1-loop monopole (vv, vd, dd) |
| 24–26 | P_2_vv1/vd1/dd1 | 1-loop quadrupole |
| 27–29 | P_4_vv1/vd1/dd1 | 1-loop hexadecapole |
| 30–33 | P_0_b1b2/b2/b1bG2/bG2 | monopole bias cross |
| 34–37 | P_2_b1b2/b2/b1bG2/bG2 | quadrupole bias cross |
| 38–41 | P_4_b2/bG2/b1b2/b1bG2 | hexadecapole bias cross |

---

## 4. Assembly Formulas (from classy.pyx)

### Real-space matter-matter
```python
pk_mm_real = (pk_mult[0] + pk_mult[14] + 2*cs*pk_mult[10]/h**2) * h**3
```

### Real-space galaxy-galaxy
```python
pk_gg_real = (
    b1**2 * pk_mult[14]
  + b1**2 * pk_mult[0]
  + 2*(cs*b1**2 + cs0*b1) * pk_mult[10] / h**2
  + b1*b2 * pk_mult[2]
  + 0.25*b2**2 * pk_mult[1]
  + 2*b1*bG2 * pk_mult[3]
  + b1*(2*bG2 + 0.8*bGamma3) * pk_mult[6]
  + bG2**2 * pk_mult[5]
  + b2*bG2 * pk_mult[4]
) * h**3 + Pshot
```

### Real-space galaxy-matter
```python
pk_gm_real = (
    b1 * pk_mult[14]
  + b1 * pk_mult[0]
  + (2*cs*b1 + cs0) * pk_mult[10] / h**2
  + (b2/2) * pk_mult[2]
  + bG2 * pk_mult[3]
  + (bG2 + 0.4*bGamma3) * pk_mult[6]
) * h**3
```

### Matter monopole (RSD)
```python
pk_mm_l0 = (
    pk_mult[15] + pk_mult[21]   # vv tree + 1-loop
  + pk_mult[16] + pk_mult[22]   # vd tree + 1-loop
  + pk_mult[17] + pk_mult[23]   # dd tree + 1-loop
  + 2*cs0 * pk_mult[11] / h**2  # counterterm
) * h**3
```

### Matter quadrupole
```python
pk_mm_l2 = (
    pk_mult[18] + pk_mult[24]   # vv tree + 1-loop
  + pk_mult[19] + pk_mult[25]   # vd tree + 1-loop
  +              pk_mult[26]    # dd 1-loop
  + 2*cs2 * pk_mult[12] / h**2
) * h**3
```

### Matter hexadecapole
```python
pk_mm_l4 = (
    pk_mult[20] + pk_mult[27]   # vv tree + 1-loop
  +              pk_mult[28]    # vd 1-loop
  +              pk_mult[29]    # dd 1-loop
  + 2*cs4 * pk_mult[13] / h**2
) * h**3
```

### Galaxy monopole (b4 term is finger-of-god stochastic)
```python
pk_gg_l0 = (
    pk_mult[15] + pk_mult[21]
  + b1*(pk_mult[16] + pk_mult[22])
  + b1**2*(pk_mult[17] + pk_mult[23])
  + 0.25*b2**2 * pk_mult[1]
  + b1*b2 * pk_mult[30] + b2 * pk_mult[31]
  + b1*bG2 * pk_mult[32] + bG2 * pk_mult[33]
  + b2*bG2 * pk_mult[4] + bG2**2 * pk_mult[5]
  + 2*cs0 * pk_mult[11] / h**2
  + (2*bG2+0.8*bGamma3)*(b1*pk_mult[7] + pk_mult[8])
) * h**3 + Pshot
  + fz**2 * b4 * (kh/h)**2 * (fz**2/9 + 2*fz*b1/7 + b1**2/5) * (35/8) * pk_mult[13] * h
```

---

## 5. IR Resummation

BAO damping via wiggle/no-wiggle decomposition:
```
P_resummed(k) = P_nw(k) + P_w(k) exp(-Σ_BAO² k²)
```
where:
- `P_nw` = no-wiggle spectrum (DST-II filter removing BAO modes 120–240)
- `P_w = P_lin - P_nw`
- `Σ_BAO² = (1/6π²) ∫₀^{k_IR} dk P_nw(k)` with k_IR = 0.2 h/Mpc

The DST-II smoothing:
1. Fine grid: N_IR = 65536 points, k ∈ [7×10⁻⁵, 7] h/Mpc
2. Apply DST-II to log(k P_lin(k))
3. Zero modes 120–240 (corresponds to BAO scale ~150 Mpc/h)
4. Inverse DST-III → P_nw

---

## 6. k-Grid and Units

**CLASS-PT internal k-grid** (for FFTLog):
- N_max = 256 log-spaced points
- k_min = 5×10⁻⁵ h/Mpc, k_max = 100 h/Mpc

**Python output grid** (user-specified via `initialize_output`):
- Any array k in h/Mpc, typically 60–200 points over [0.005, 0.5] h/Mpc

**Units throughout**: k in h/Mpc, P(k) in (Mpc/h)³

**Exception**: `pk_lin(k, z)` takes k in **1/Mpc** and returns P in **Mpc³**.
To convert: `pk_lin_hMpc = pk_lin(k_hMpc * h, z) / h**3`

---

## 7. Planck 2018 Fiducial Parameters

Standard parameters for reference run (Planck 2018 best-fit):
```python
params = {
    'A_s': 2.0989e-9,
    'n_s': 0.9649,
    'tau_reio': 0.0544,
    'omega_b': 0.02237,
    'omega_cdm': 0.1200,
    'h': 0.6736,
    'output': 'mPk',
    'non linear': 'PT',
    'IR resummation': 'Yes',
    'Bias tracers': 'Yes',
    'RSD': 'Yes',
    'z_pk': 0.38,
    'P_k_max_h/Mpc': 100.,
}
```

Bias parameters for reference table: b1=2, b2=0, bG2=0, bGamma3=0, cs0=0, cs2=0, cs4=0, Pshot=0, b4=500 (CLASS-PT defaults when all zero except b1=2).

---

## 8. Key Source Code References

| Computation | Location |
|-------------|----------|
| FFTLog decomposition | `nonlinear_pt.c` lines 5818–5948 |
| DST-II BAO separation | `nonlinear_pt.c` lines 5315–5776 |
| P22 kernel evaluation | `nonlinear_pt.c` lines 6082–6102 |
| P13 kernel evaluation | `nonlinear_pt.c` lines 6068–6079 |
| Full loop computation | `nonlinear_pt.c::nonlinear_pt_loop()` line 4914+ |
| Python bias assembly | `python/classy.pyx` lines 1093–1240 |
