# CLASS-PT Component Index and Assembly

**Source**: CLASS-PT `nonlinear_pt.c`, `python/classy.pyx`

See also: [FFTLog_PT.md](FFTLog_PT.md) for the algorithm derivation.

---

## 1. EPTComponents Index Table

The `EPTComponents` dataclass stores 52 spectral arrays (indices 0-51).
These are the building blocks from which all 9 output spectra are assembled.

### Core matter (indices 0, 10, 14)

| Index | Field       | Content                     | FFTLog basis |
|-------|-------------|-----------------------------|--------------|
| 0     | Pk_loop     | P13 + P22 (1-loop matter)   | b = -0.3     |
| 10    | Pk_ctr      | -k^2 P_lin (counterterm)    | --           |
| 14    | Pk_tree     | P_lin (IR-resummed tree)    | --           |

### Bias cross-spectra (indices 1-9)

| Index | Field        | Operator cross | FFTLog basis |
|-------|--------------|----------------|--------------|
| 1     | Pk_Id2d2     | delta^2 x delta^2 | b = -1.6 |
| 2     | Pk_Id2       | delta x delta^2   | b = -1.6 |
| 3     | Pk_IG2       | delta x G2         | b = -1.6 |
| 4     | Pk_Id2G2     | delta^2 x G2      | b = -1.6 |
| 5     | Pk_IG2G2     | G2 x G2           | b = -1.6 |
| 6     | Pk_IFG2      | delta x F.G2      | b = -1.6 |
| 7     | Pk_IFG2_0b1  | FG2 monopole (b1) | b = -1.6 |
| 8     | Pk_IFG2_0    | FG2 monopole      | b = -1.6 |
| 9     | Pk_IFG2_2    | FG2 quadrupole    | b = -1.6 |

FG2 multipoles: `Pk_IFG2_0b1 = Pk_IFG2`, `Pk_IFG2_0 = Pk_IFG2 * f/3`,
`Pk_IFG2_2 = Pk_IFG2 * 2f/3`.

### Counterterm multipoles (indices 11-13)

| Index | Field    | Formula                            |
|-------|----------|------------------------------------|
| 11    | Pk_ctr0  | -k^2 P_lin                         |
| 12    | Pk_ctr2  | -k^2 P_lin * f * 2/3              |
| 13    | Pk_ctr4  | -k^2 P_lin * f^2 * 8/35           |

Convention: the EFT counterterm contribution to multipole ell is
`2 cs_ell * Pk_ctr_ell`.

### RSD tree-level multipoles (indices 15-20, 49-51)

| Index | Field    | Content                        |
|-------|----------|--------------------------------|
| 15    | Pk_0_vv  | monopole tree, f^2 mu^4        |
| 16    | Pk_0_vd  | monopole tree, 2f mu^2         |
| 17    | Pk_0_dd  | monopole tree, 1               |
| 18    | Pk_2_vv  | quadrupole tree, f^2 mu^4      |
| 19    | Pk_2_vd  | quadrupole tree, 2f mu^2       |
| 20    | Pk_4_vv  | hexadecapole tree, f^2 mu^4    |
| 49    | Pk_2_dd  | quadrupole tree, dd (aniso IR) |
| 50    | Pk_4_vd  | hexadecapole tree, vd (aniso IR)|
| 51    | Pk_4_dd  | hexadecapole tree, dd (aniso IR)|

Indices 49-51 are zero in the isotropic approximation but nonzero when
anisotropic Sigma_tot(mu) is used, because the mu-dependent damping
generates L2 and L4 projections from the dd and vd terms.

### RSD 1-loop multipoles (indices 21-29)

| Index | Field     | Content                      |
|-------|-----------|------------------------------|
| 21    | Pk_0_vv1  | monopole 1-loop, vv          |
| 22    | Pk_0_vd1  | monopole 1-loop, vd          |
| 23    | Pk_0_dd1  | monopole 1-loop, dd          |
| 24    | Pk_2_vv1  | quadrupole 1-loop, vv        |
| 25    | Pk_2_vd1  | quadrupole 1-loop, vd        |
| 26    | Pk_2_dd1  | quadrupole 1-loop, dd        |
| 27    | Pk_4_vv1  | hexadecapole 1-loop, vv      |
| 28    | Pk_4_vd1  | hexadecapole 1-loop, vd      |
| 29    | Pk_4_dd1  | hexadecapole 1-loop, dd      |

These are computed by GL quadrature over the bare mu-power channels
with anisotropic IR damping (see FFTLog_PT.md Section 6).

### RSD bias cross-terms (indices 30-41)

| Index | Field        | Multipole | Bias coupling |
|-------|--------------|-----------|---------------|
| 30    | Pk_0_b1b2    | l=0       | b1 * b2       |
| 31    | Pk_0_b2      | l=0       | b2            |
| 32    | Pk_0_b1bG2   | l=0       | b1 * bG2      |
| 33    | Pk_0_bG2     | l=0       | bG2           |
| 34    | Pk_2_b1b2    | l=2       | b1 * b2       |
| 35    | Pk_2_b2      | l=2       | b2            |
| 36    | Pk_2_b1bG2   | l=2       | b1 * bG2      |
| 37    | Pk_2_bG2     | l=2       | bG2           |
| 38    | Pk_4_b2      | l=4       | b2            |
| 39    | Pk_4_bG2     | l=4       | bG2           |
| 40    | Pk_4_b1b2    | l=4       | b1 * b2 (AP only, zero here) |
| 41    | Pk_4_b1bG2   | l=4       | b1 * bG2 (AP only, zero here)|

### Higher-order mu-power arrays (indices 43-48)

| Index | Field        | Content                     |
|-------|--------------|-----------------------------|
| 43    | pk_nw        | no-wiggle P(k)              |
| 44    | pk_w         | wiggle P(k) = P_lin - P_nw  |
| 45    | P22_mu6_vv   | P22 bare mu^6 vv            |
| 46    | P22_mu6_vd   | P22 bare mu^6 vd            |
| 47    | P22_mu8      | P22 bare mu^8               |
| 48    | P13_mu6      | P13 bare mu^6               |

Scalars: `sigma2_bao` (Sigma^2 in (Mpc/h)^2), `delta_sigma2_bao`
(delta_Sigma^2 in (Mpc/h)^2).

---

## 2. Bias Model

Real-space galaxy overdensity (EFT of LSS, EdS approximation):

```
delta_g = b1 delta + (b2/2) delta^2 + bG2 G2[Phi] + bGamma3 Gamma3 + ...
```

**Bias parameters**:
- `b1`: linear bias
- `b2`: quadratic density bias
- `bG2`: tidal (Galileon) bias
- `bGamma3`: third-order operator (partially degenerate with bG2 at 1-loop)

**EFT counterterms** (units: (Mpc/h)^2):
- `cs0`: monopole counterterm (= cs_0 in the literature)
- `cs2`: quadrupole counterterm
- `cs4`: hexadecapole counterterm
- `cs`: matter counterterm (real-space matter-matter only)

**Stochastic parameters**:
- `Pshot`: shot noise in (Mpc/h)^3
- `b4`: higher-order stochastic (finger-of-god); dimensionless

---

## 3. Assembly Formulas

All spectra are assembled from the `EPTComponents` fields. In all formulas,
`ept.*` refers to the corresponding field.

### Real-space matter-matter

```
P_mm(k) = Pk_tree + Pk_loop + 2 cs0 Pk_ctr
```

(`pk_mm_real` in `ept.py`)

### Real-space galaxy-matter

```
P_gm(k) = b1 (Pk_tree + Pk_loop)
         + (cs b1 + cs0) Pk_ctr
         + (b2/2) Pk_Id2
         + bG2 Pk_IG2
         + (bG2 + 0.4 bGamma3) Pk_IFG2
```

(`pk_gm_real`)

### Real-space galaxy-galaxy

```
P_gg(k) = b1^2 (Pk_tree + Pk_loop)
         + 2 (cs b1^2 + cs0 b1) Pk_ctr
         + b1 b2 Pk_Id2
         + (b2^2/4) Pk_Id2d2
         + 2 b1 bG2 Pk_IG2
         + b1 (2 bG2 + 0.8 bGamma3) Pk_IFG2
         + bG2^2 Pk_IG2G2
         + b2 bG2 Pk_Id2G2
         + Pshot
```

(`pk_gg_real`)

### Matter monopole (RSD, l=0)

```
P_mm^{l=0}(k) = (Pk_0_dd + Pk_0_vd + Pk_0_vv)
              + (Pk_0_dd1 + Pk_0_vd1 + Pk_0_vv1)
              + 2 cs0 Pk_ctr0
```

(`pk_mm_l0`; analogous for `pk_mm_l2`, `pk_mm_l4` with l=2, l=4 indices)

### Galaxy monopole (RSD, l=0)

```
P_gg^{l=0}(k) = b1^2 Pk_0_dd + b1 Pk_0_vd + Pk_0_vv
              + b1^2 Pk_0_dd1 + b1 Pk_0_vd1 + Pk_0_vv1
              + (b2^2/4) Pk_Id2d2
              + b1 b2 Pk_0_b1b2 + b2 Pk_0_b2
              + b1 bG2 Pk_0_b1bG2 + bG2 Pk_0_bG2
              + b2 bG2 Pk_Id2G2 + bG2^2 Pk_IG2G2
              + (2 bG2 + 0.8 bGamma3) (b1 Pk_IFG2_0b1 + Pk_IFG2_0)
              + 2 cs0 Pk_ctr0
              + Pshot
              + P_b4
```

where the b4 stochastic term is:

```
P_b4^{l=0} = f^2 b4 k^2 (f^2/9 + 2 f b1/7 + b1^2/5) (35/8) Pk_ctr4
```

(`pk_gg_l0`)

### Galaxy quadrupole (RSD, l=2)

```
P_gg^{l=2}(k) = Pk_2_vv + b1 Pk_2_vd + b1^2 Pk_2_dd
              + Pk_2_vv1 + b1 Pk_2_vd1 + b1^2 Pk_2_dd1
              + b1 b2 Pk_2_b1b2 + b2 Pk_2_b2
              + b1 bG2 Pk_2_b1bG2 + bG2 Pk_2_bG2
              + (2 bG2 + 0.8 bGamma3) Pk_IFG2_2
              + 2 cs2 Pk_ctr2
              + P_b4
```

where:

```
P_b4^{l=2} = f^2 b4 k^2 (70 f^2 + 165 f b1 + 99 b1^2) (4/693)(35/8) Pk_ctr4
```

(`pk_gg_l2`)

### Galaxy hexadecapole (RSD, l=4)

```
P_gg^{l=4}(k) = Pk_4_vv + Pk_4_vd + Pk_4_dd
              + Pk_4_vv1 + b1 Pk_4_vd1 + b1^2 Pk_4_dd1
              + b2 Pk_4_b2 + bG2 Pk_4_bG2
              + b1 b2 Pk_4_b1b2 + b1 bG2 Pk_4_b1bG2
              + 2 cs4 Pk_ctr4
              + P_b4
```

where:

```
P_b4^{l=4} = f^2 b4 k^2 (210 f^2 + 390 f b1 + 143 b1^2) (8/5005)(35/8) Pk_ctr4
```

Note: the l=4 tree uses the matter tree (no b1 weighting on Pk_4_vd, Pk_4_dd),
matching CLASS-PT classy.pyx line 1213.

(`pk_gg_l4`)

---

## 4. EFT Counterterms

The counterterm shape for each multipole:

```
Pk_ctr0 = -k^2 P_lin
Pk_ctr2 = -k^2 P_lin f (2/3)
Pk_ctr4 = -k^2 P_lin f^2 (8/35)
```

The contribution to the spectrum is `2 cs_ell Pk_ctr_ell` where cs_ell
is in (Mpc/h)^2.

---

## 5. Key Source Code References

| Computation              | CLASS-PT location                   |
|--------------------------|-------------------------------------|
| FFTLog decomposition     | `nonlinear_pt.c` lines 5818-5948    |
| DST-II BAO separation    | `nonlinear_pt.c` lines 5315-5776    |
| P22 kernel (matter)      | `nonlinear_pt.c` lines 6082-6102    |
| P13 kernel (matter)      | `nonlinear_pt.c` lines 6068-6079    |
| M22 RSD monopole kernels | `nonlinear_pt.c` lines 6647, 6928, 7054 |
| M22 RSD quadrupole       | `nonlinear_pt.c` lines 7159, 7275, 7395 |
| M22 RSD hexadecapole     | `nonlinear_pt.c` lines 7506, 7618, 7739 |
| M13 RSD kernels          | `nonlinear_pt.c` lines 6820-7657    |
| UV counterterms          | `nonlinear_pt.c` lines 6832-7667    |
| Bare mu-power M22        | `nonlinear_pt.c` lines 8059-8067    |
| Bare mu-power M13        | `nonlinear_pt.c` lines 8155-8159    |
| Bias cross-spectra       | `nonlinear_pt.c` lines 11880-12518  |
| RSD bias cross monopole  | `nonlinear_pt.c` lines 12871-13000  |
| RSD bias cross quadrupole| `nonlinear_pt.c` lines 13173-13271  |
| RSD bias cross hexadecapole | `nonlinear_pt.c` lines 13305-13339|
| Python assembly formulas | `python/classy.pyx` lines 1093-1240 |
| Full loop computation    | `nonlinear_pt.c::nonlinear_pt_loop()` line 4914+ |

---

## 6. Fiducial Parameters for Testing

Planck 2018 best-fit cosmology:

```
h        = 0.6736
omega_b  = 0.02237
omega_cdm = 0.12
A_s      = 2.0989e-9
n_s      = 0.9649
tau_reio = 0.0544
z        = 0.38
```

Bias parameters for the primary accuracy test (`reference_data/classpt_z0.38_fullrange.npz`):

```
b1       = 2.0
b2       = 0.0
bG2      = 0.0
bGamma3  = 0.0
cs0      = 0.0       (Mpc/h)^2
cs2      = 0.0       (Mpc/h)^2
cs4      = 0.0       (Mpc/h)^2
Pshot    = 0.0       (Mpc/h)^3
b4       = 500.0     dimensionless
cs       = 0.0       (Mpc/h)^2
```

At z = 0.38: f(z) = 0.7166, sigma_8(z) ~ 0.673.

This reference uses the FFTLog k-grid (256 points, 5e-5 to 100 h/Mpc) with
AP=Yes and Omfid=0.31.  Generated by `scripts/generate_classpt_reference.py`.

A secondary reference with non-trivial biases is stored in
`docs/classpt_reference_table.npz` (60 log-spaced k in [0.005, 0.30] h/Mpc).
