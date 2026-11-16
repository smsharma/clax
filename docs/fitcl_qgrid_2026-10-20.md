# fit_cl massive-neutrino hierarchy: q-grid, fluid approximation, batched P(k) (Oct 20-22, 2026, smsharma/clax#57)

**Question.** `fit_cl` shipped with `ncdm_q_size=0`, which keeps the default 0.06 eV
species massive in the background but gives it the massless-neutrino perturbations
(#57): TT 15% off CLASS at l <= 30 at every LCDM grid point. Which massive-neutrino
hierarchy does fit_cl need, and what does it cost?

**Decision (Oct 22).** `ncdm_q_size=5`, `pt_l_max_ncdm=8`, `ncdm_fluid_approximation="none"`
(config Q5L8F0). Against CLASS (HyRec) references at 0.06 eV (five-point LCDM grid),
0.15, 0.30 and 0.50 eV it reads TT 2-30 within 0.54%, TT 2-500 within 0.69%, EE 5-500
within 1.65%, TE/sqrt(TT EE) within 1.13%, flat in the neutrino mass, at 18.5-21.6 s end
to end on a V100 (two nodes; the broken preset 19.1 s, CLASS's 5 bins / l_max 17 with the
fluid approximation 23.5 s). The Oct 20 choice of 3 q-bins with the fluid approximation
(config Q3L8) is withdrawn: it was closer to CLASS by a cancellation of two errors.

**Findings** (CHANGELOG.md, Oct 22, 2026 entry):
1. The Gauss-Laguerre momentum grid is converged at 5 bins (8 and 12 agree to 0.02%).
2. With the "class" fluid approximation TT at l <= 30 carries a deficit growing with the
   mass (0.60 / 0.71 / 0.98 / 1.29% at 0.06 / 0.15 / 0.30 / 0.50 eV) and P_m is -0.94% off
   CLASS at k = 4e-3 Mpc^-1 at 0.30 eV; with the hierarchy followed to today both are
   0.5% and <= 0.1%. CLASS with its own ncdm tolerances tightened by two orders of
   magnitude moves by <= 0.03% in TT at l <= 30 (oracle converged). This is #46's
   implementation discrepancy seen in the late ISW; planck_cl shares it.
3. The batched P(k) does not move with the solver: hierarchy-to-hierarchy differences are
   identical at rtol 1e-4 / 1e-5 / 1e-6, the batched solve takes the same 324 steps for
   every state dimension, and its error against a per-mode Kvaerno5 reference is
   < 0.006%. Truncating at l_max 8 (+0.9% at k = 1 Mpc^-1 as shipped, +0.6% with the approximation
   on) or using 3 momenta (+1.0%)
   changes the species' relativistic-era free streaming, mass-independently. The
   lensing potential's linear source comes from the C_l solve; the batched P(k) enters
   only the halofit / EPT modulator (default "none"); l_max 8 vs 17 moves C_l^phiphi by
   0.23% (linear) / 0.12% (halofit) at l = 1000 and the lensed TT, EE by <= 0.07%.
4. fit_cl's linear P_m is 1.2-1.5% off CLASS near k = 0.2 Mpc^-1 for every hierarchy (the
   20 k per decade); not changed.

Scripts: `diags/diag_fitcl_qgrid.py` (C_l and P(k) per hierarchy and cosmology) and
`diags/diag_pk_batched_steps.py` (tolerance ladder, per-mode reference, step counts,
lensing). Worktree `/lustre/work/n2minh/clax-wt-fitcl-q`. igpu cluster, Tesla
V100-SXM2-32GB, JAX 0.9.2. Raw JSON and the reports below live under
`/lustre/work/n2minh/std/clax/presets/`. CLASS references beyond the test grid
(`massive_nu_050`, `*_ncdmtight`) come from `scripts/generate_class_reference.py --model`
and are not checked in.

## Job 27173: LCDM grid (0.06 eV), `--grid lcdm`


Errors in percent: max |C_l/ref - 1| over the band (TEn = (TE - ref)/sqrt(TT_ref EE_ref)); grid columns are the maximum over the five LCDM points, with the point and l of the maximum; massless = CosmoParams(m_ncdm=0) vs the massless-RECFAST reference. Timings: median of the cached cosmologies, seconds.

| cfg | q | l_max | n_eq | pt | hr | total | grid TT 2-30 | grid TT 2-500 | grid EE 5-500 | grid TEN 2-500 | grid TT 2-1000 | massless TT 2-30 | massless TT 2-500 | massless EE 5-500 | massless TEN 2-500 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q0 | 0 | 17 | 61 | 7.2 | 7.5 | 19.1 | 15.415 (omega_b_high, l=2) | 15.415 (omega_b_high, l=2) | 4.086 (omega_cdm_low, l=14) | 19.372 (omega_cdm_low, l=2) | 15.415 (omega_b_high, l=2) | 0.461 | 0.461 | 1.194 | 0.884 |
| Q3 | 3 | 17 | 118 | 11.4 | 7.5 | 21.4 | 0.526 (omega_b_high, l=10) | 0.690 (omega_cdm_low, l=446) | 1.603 (omega_b_high, l=16) | 1.077 (omega_b_high, l=25) | 5.484 (h_high, l=993) | 0.420 | 0.420 | 1.155 | 0.846 |
| Q4 | 4 | 17 | 136 | 13.0 | 7.5 | 23.0 | 0.607 (omega_b_high, l=7) | 0.686 (omega_cdm_low, l=445) | 1.654 (omega_b_high, l=16) | 1.129 (omega_b_high, l=25) | 5.510 (h_high, l=992) | 0.495 | 0.495 | 1.217 | 0.906 |
| Q5 | 5 | 17 | 154 | 13.5 | 7.5 | 23.5 | 0.620 (omega_b_high, l=7) | 0.686 (omega_cdm_low, l=445) | 1.654 (omega_b_high, l=16) | 1.126 (omega_b_high, l=25) | 5.506 (h_high, l=992) | 0.482 | 0.482 | 1.207 | 0.897 |
| Q5L8 | 5 | 8 | 109 | 11.0 | 7.7 | 21.1 | 0.620 (omega_b_high, l=7) | 0.696 (omega_cdm_low, l=445) | 1.652 (omega_b_high, l=16) | 1.126 (omega_b_high, l=25) | 5.577 (h_high, l=993) | 0.482 | 0.482 | 1.207 | 0.897 |
| Q3L8 | 3 | 8 | 91 | 9.3 | 7.7 | 19.8 | 0.526 (omega_b_high, l=10) | 0.700 (omega_cdm_low, l=446) | 1.603 (omega_b_high, l=16) | 1.076 (omega_b_high, l=25) | 5.555 (h_high, l=993) | 0.419 | 0.419 | 1.152 | 0.846 |

Per cosmology (TT 2-30 / TT 2-500 / EE 5-500 / TEn 2-500, percent):

| cfg | lcdm_fiducial | h_high | omega_b_high | omega_cdm_low | ns_high | massless_fiducial |
|---|---|---|---|---|---|---|
| Q0 | 15.31 / 15.31 / 3.13 / 17.19 | 13.19 / 13.19 / 3.14 / 16.18 | 15.41 / 15.41 / 3.77 / 16.83 | 15.13 / 15.13 / 4.09 / 19.37 | 14.60 / 14.60 / 3.12 / 17.05 | 0.46 / 0.46 / 1.19 / 0.88 |
| Q3 | 0.48 / 0.60 / 1.15 / 0.82 | 0.48 / 0.55 / 1.23 / 0.87 | 0.53 / 0.57 / 1.60 / 1.08 | 0.50 / 0.69 / 1.51 / 0.98 | 0.47 / 0.64 / 1.15 / 0.81 | 0.42 / 0.42 / 1.15 / 0.85 |
| Q4 | 0.58 / 0.59 / 1.21 / 0.87 | 0.55 / 0.55 / 1.29 / 0.92 | 0.61 / 0.61 / 1.65 / 1.13 | 0.57 / 0.69 / 1.58 / 1.05 | 0.55 / 0.64 / 1.21 / 0.87 | 0.49 / 0.49 / 1.22 / 0.91 |
| Q5 | 0.60 / 0.60 / 1.20 / 0.87 | 0.56 / 0.56 / 1.28 / 0.92 | 0.62 / 0.62 / 1.65 / 1.13 | 0.59 / 0.69 / 1.57 / 1.05 | 0.56 / 0.64 / 1.20 / 0.86 | 0.48 / 0.48 / 1.21 / 0.90 |
| Q5L8 | 0.60 / 0.60 / 1.20 / 0.87 | 0.56 / 0.56 / 1.28 / 0.92 | 0.62 / 0.62 / 1.65 / 1.13 | 0.58 / 0.70 / 1.57 / 1.04 | 0.56 / 0.65 / 1.20 / 0.86 | 0.48 / 0.48 / 1.21 / 0.90 |
| Q3L8 | 0.48 / 0.60 / 1.15 / 0.82 | 0.48 / 0.56 / 1.23 / 0.87 | 0.53 / 0.58 / 1.60 / 1.08 | 0.50 / 0.70 / 1.51 / 0.98 | 0.47 / 0.65 / 1.15 / 0.81 | 0.42 / 0.42 / 1.15 / 0.85 |

Fiducial TT and EE vs CLASS (HyRec, 0.06 eV) at l = (2, 3, 5, 10, 20, 30, 50, 100, 200, 500) (percent):

    Q0 TT: -15.31 -14.95 -12.55 -6.38 -0.85 +0.62 +0.91 +0.35 +0.00 -0.59
    Q0 EE: -1.44 -2.38 -2.72 +2.41 -1.27 -0.42 -0.40 -0.35 -0.09 -0.14
    Q3 TT: -0.28 -0.35 -0.44 -0.47 -0.38 -0.26 -0.08 +0.10 +0.04 -0.60
    Q3 EE: +0.59 -0.19 -0.10 -0.23 -1.15 -0.26 -0.26 -0.20 +0.03 -0.10
    Q4 TT: -0.43 -0.51 -0.57 -0.54 -0.38 -0.22 -0.04 +0.10 -0.01 -0.59
    Q4 EE: +0.58 -0.19 -0.11 -0.25 -1.21 -0.33 -0.33 -0.29 -0.05 -0.17
    Q5 TT: -0.46 -0.53 -0.59 -0.55 -0.38 -0.22 -0.04 +0.10 -0.00 -0.59
    Q5 EE: +0.58 -0.20 -0.11 -0.25 -1.20 -0.32 -0.32 -0.28 -0.04 -0.16
    Q5L8 TT: -0.42 -0.53 -0.59 -0.55 -0.38 -0.22 -0.04 +0.10 -0.00 -0.60
    Q5L8 EE: +0.57 -0.20 -0.11 -0.25 -1.20 -0.32 -0.32 -0.28 -0.04 -0.16
    Q3L8 TT: -0.28 -0.35 -0.44 -0.47 -0.38 -0.26 -0.08 +0.10 +0.04 -0.60
    Q3L8 EE: +0.59 -0.19 -0.10 -0.23 -1.15 -0.26 -0.26 -0.20 +0.03 -0.10


## Jobs 27187 and 27200: massive-neutrino grid with linear P_m(k, z = 0), `--grid nu --pk`

P_m is `clax.compute_pk_table` (the `Rodas5Batched` path fit_cl uses for P(k)) on the
reference k grid restricted to 1e-3 <= k <= 1 Mpc^-1, against `pk_m_lin_z0`. "ref
added" marks the 0.30 eV point, scored against a CLASS reference generated after the
run; the 0.50 eV row was added by job 27200 for Q5, Q5L8 and Q3L8 only.


Errors in percent: max |C_l/ref - 1| over the band (TEn = (TE - ref)/sqrt(TT_ref EE_ref)); grid columns are the maximum over the grid points that have a CLASS reference, with the point and l of the maximum; massless = CosmoParams(m_ncdm=0) vs the massless-RECFAST reference. Points without a CLASS reference are scored against the Q5 run (marked). Timings: median of the cached cosmologies, seconds.

| cfg | q | l_max | n_eq | pt | hr | total | pk | grid TT 2-30 | grid TT 2-500 | grid EE 5-500 | grid TEN 2-500 | grid TT 2-1000 | massless TT 2-30 | massless TT 2-500 | massless EE 5-500 | massless TEN 2-500 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q3 | 3 | 17 | 118 | 11.6 | 8.1 | 22.2 | - | 0.482 (massive_nu_015, l=15) | 0.597 (fiducial, l=500) | 1.150 (fiducial, l=20) | 0.816 (fiducial, l=25) | 5.734 (massive_nu_015, l=995) | - | - | - | - |
| Q5 | 5 | 17 | 154 | 12.7 | 8.1 | 23.2 | - | 1.292 (massive_nu_050, l=7) | 1.292 (massive_nu_050, l=7) | 1.209 (massive_nu_050, l=19) | 0.868 (fiducial, l=25) | 5.764 (massive_nu_015, l=995) | - | - | - | - |
| Q5L8 | 5 | 8 | 109 | 11.7 | 8.1 | 22.3 | - | 1.292 (massive_nu_050, l=7) | 1.292 (massive_nu_050, l=7) | 1.208 (massive_nu_050, l=19) | 0.868 (fiducial, l=25) | 5.824 (massive_nu_015, l=995) | - | - | - | - |
| Q3L8 | 3 | 8 | 91 | 10.2 | 8.1 | 20.8 | - | 0.607 (massive_nu_050, l=16) | 0.607 (massive_nu_050, l=16) | 1.150 (fiducial, l=20) | 1.372 (massive_nu_050, l=2) | 5.793 (massive_nu_015, l=995) | - | - | - | - |

Per cosmology (TT 2-30 / TT 2-500 / EE 5-500 / TEn 2-500 [/ P_m lin z=0, 1e-3-1 Mpc^-1], percent):

| cfg | lcdm_fiducial | massive_nu_015 | massive_nu_030 | massive_nu_050 |
|---|---|---|---|---|
| Q3 | 0.48 / 0.60 / 1.15 / 0.82 / pk 1.53 (k=0.2) | 0.48 / 0.49 / 1.14 / 0.75 / pk 1.53 (k=0.2) | 0.47 / 0.63 / 1.12 / 0.93 / pk 1.56 (k=0.2) (ref added) | - |
| Q5 | 0.60 / 0.60 / 1.20 / 0.87 / pk 1.33 (k=0.18) | 0.71 / 0.71 / 1.19 / 0.86 / pk 1.33 (k=0.18) | 0.98 / 0.98 / 1.17 / 0.84 / pk 1.33 (k=0.18) (ref added) | 1.29 / 1.29 / 1.21 / 0.82 / pk 1.89 (k=0.029) |
| Q5L8 | 0.60 / 0.60 / 1.20 / 0.87 / pk 1.43 (k=0.2) | 0.71 / 0.71 / 1.19 / 0.86 / pk 1.43 (k=0.2) | 0.98 / 0.98 / 1.17 / 0.84 / pk 1.46 (k=0.2) (ref added) | 1.29 / 1.29 / 1.21 / 0.82 / pk 1.90 (k=0.029) |
| Q3L8 | 0.48 / 0.60 / 1.15 / 0.82 / pk 1.74 (k=0.2) | 0.48 / 0.50 / 1.14 / 0.75 / pk 1.75 (k=0.2) | 0.47 / 0.64 / 1.12 / 0.93 / pk 1.77 (k=0.2) (ref added) | 0.61 / 0.61 / 1.13 / 1.37 / pk 2.12 (k=0.029) |

Fiducial TT and EE vs CLASS (HyRec, 0.06 eV) at l = (2, 3, 5, 10, 20, 30, 50, 100, 200, 500) (percent):

    Q3 TT: -0.28 -0.35 -0.44 -0.47 -0.38 -0.26 -0.08 +0.10 +0.04 -0.60
    Q3 EE: +0.59 -0.19 -0.10 -0.23 -1.15 -0.26 -0.26 -0.20 +0.03 -0.10
    Q5 TT: -0.46 -0.53 -0.59 -0.55 -0.38 -0.22 -0.04 +0.10 -0.00 -0.59
    Q5 EE: +0.58 -0.20 -0.11 -0.25 -1.20 -0.32 -0.32 -0.28 -0.04 -0.16
    Q5L8 TT: -0.42 -0.53 -0.59 -0.55 -0.38 -0.22 -0.04 +0.10 -0.00 -0.60
    Q5L8 EE: +0.57 -0.20 -0.11 -0.25 -1.20 -0.32 -0.32 -0.28 -0.04 -0.16
    Q3L8 TT: -0.28 -0.35 -0.44 -0.47 -0.38 -0.26 -0.08 +0.10 +0.04 -0.60
    Q3L8 EE: +0.59 -0.19 -0.10 -0.23 -1.15 -0.26 -0.26 -0.20 +0.03 -0.10


## Jobs 27686 and 27711: q-convergence (8, 12 bins) and the fluid approximation off, `--grid nu`

Q3L8, Q5L8, Q8L8, Q12L8 with the "class" fluid approximation (job 27686); Q3L8F0, Q5L8F0,
Q12L8F0 without it, with P(k) (job 27711). All four masses have CLASS references here.


Errors in percent: max |C_l/ref - 1| over the band (TEn = (TE - ref)/sqrt(TT_ref EE_ref)); grid columns are the maximum over the grid points that have a CLASS reference, with the point and l of the maximum; massless = CosmoParams(m_ncdm=0) vs the massless-RECFAST reference. Points without a CLASS reference are scored against the Q5 run (marked). Timings: median of the cached cosmologies, seconds.

| cfg | q | l_max | n_eq | pt | hr | total | pk | grid TT 2-30 | grid TT 2-500 | grid EE 5-500 | grid TEN 2-500 | grid TT 2-1000 | massless TT 2-30 | massless TT 2-500 | massless EE 5-500 | massless TEN 2-500 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q5L8 | 5 | 8 | 109 | 10.6 | 8.0 | 20.8 | - | 1.292 (massive_nu_050, l=7) | 1.292 (massive_nu_050, l=7) | 1.208 (massive_nu_050, l=19) | 0.868 (fiducial, l=25) | 5.824 (massive_nu_015, l=995) | - | - | - | - |
| Q3L8 | 3 | 8 | 91 | 9.4 | 8.0 | 19.8 | - | 0.607 (massive_nu_050, l=16) | 0.637 (massive_nu_030, l=500) | 1.150 (fiducial, l=20) | 1.372 (massive_nu_050, l=2) | 5.793 (massive_nu_015, l=995) | - | - | - | - |
| Q8L8 | 8 | 8 | 136 | 12.1 | 8.0 | 22.4 | - | 1.260 (massive_nu_050, l=7) | 1.260 (massive_nu_050, l=7) | 1.199 (massive_nu_050, l=19) | 0.858 (massive_nu_015, l=25) | 5.816 (massive_nu_015, l=995) | - | - | - | - |
| Q12L8 | 12 | 8 | 172 | 14.4 | 8.0 | 25.0 | - | 1.260 (massive_nu_050, l=7) | 1.260 (massive_nu_050, l=7) | 1.198 (massive_nu_050, l=19) | 0.859 (massive_nu_015, l=25) | 5.816 (massive_nu_015, l=995) | - | - | - | - |
| Q3L8F0 | 3 | 8 | 88 | 9.5 | 7.9 | 20.3 | 14.4 | 0.934 (massive_nu_050, l=2) | 0.934 (massive_nu_050, l=2) | 1.147 (fiducial, l=20) | 1.375 (massive_nu_050, l=2) | 5.795 (massive_nu_015, l=995) | - | - | - | - |
| Q5L8F0 | 5 | 8 | 106 | 11.0 | 7.7 | 21.6 | 15.9 | 0.531 (massive_nu_050, l=7) | 0.636 (massive_nu_030, l=500) | 1.202 (fiducial, l=20) | 0.871 (massive_nu_015, l=25) | 5.826 (massive_nu_015, l=995) | - | - | - | - |
| Q12L8F0 | 12 | 8 | 169 | 12.4 | 8.0 | 23.1 | 22.4 | 0.502 (massive_nu_030, l=10) | 0.640 (massive_nu_030, l=500) | 1.188 (fiducial, l=20) | 0.868 (massive_nu_015, l=25) | 5.818 (massive_nu_015, l=995) | - | - | - | - |

Per cosmology (TT 2-30 / TT 2-500 / EE 5-500 / TEn 2-500 [/ P_m lin z=0, 1e-3-1 Mpc^-1], percent):

| cfg | lcdm_fiducial | massive_nu_015 | massive_nu_030 | massive_nu_050 |
|---|---|---|---|---|
| Q5L8 | 0.60 / 0.60 / 1.20 / 0.87 | 0.71 / 0.71 / 1.19 / 0.86 | 0.98 / 0.98 / 1.17 / 0.84 | 1.29 / 1.29 / 1.21 / 0.82 |
| Q3L8 | 0.48 / 0.60 / 1.15 / 0.82 | 0.48 / 0.50 / 1.14 / 0.75 | 0.47 / 0.64 / 1.12 / 0.93 | 0.61 / 0.61 / 1.13 / 1.37 |
| Q8L8 | 0.58 / 0.60 / 1.19 / 0.85 | 0.70 / 0.70 / 1.17 / 0.86 | 0.96 / 0.96 / 1.16 / 0.83 | 1.26 / 1.26 / 1.20 / 0.81 |
| Q12L8 | 0.58 / 0.60 / 1.19 / 0.85 | 0.71 / 0.71 / 1.17 / 0.86 | 0.96 / 0.96 / 1.16 / 0.83 | 1.26 / 1.26 / 1.20 / 0.81 |
| Q3L8F0 | 0.40 / 0.60 / 1.15 / 0.82 / pk 1.85 (k=0.2) | 0.33 / 0.49 / 1.13 / 0.75 / pk 1.99 (k=0.2) | 0.52 / 0.63 / 1.11 / 0.93 / pk 2.33 (k=0.2) | 0.93 / 0.93 / 1.11 / 1.38 / pk 2.78 (k=0.2) |
| Q5L8F0 | 0.51 / 0.60 / 1.20 / 0.87 / pk 1.46 (k=0.2) | 0.48 / 0.52 / 1.18 / 0.87 / pk 1.55 (k=0.2) | 0.52 / 0.64 / 1.16 / 0.86 / pk 1.77 (k=0.2) | 0.53 / 0.63 / 1.19 / 0.85 / pk 2.04 (k=0.2) |
| Q12L8F0 | 0.49 / 0.60 / 1.19 / 0.86 / pk 1.56 (k=0.2) | 0.48 / 0.50 / 1.17 / 0.87 / pk 1.64 (k=0.2) | 0.50 / 0.64 / 1.14 / 0.85 / pk 1.86 (k=0.2) | 0.50 / 0.63 / 1.18 / 0.84 / pk 2.14 (k=0.2) |

Fiducial TT and EE vs CLASS (HyRec, 0.06 eV) at l = (2, 3, 5, 10, 20, 30, 50, 100, 200, 500) (percent):

    Q5L8 TT: -0.42 -0.53 -0.59 -0.55 -0.38 -0.22 -0.04 +0.10 -0.00 -0.60
    Q5L8 EE: +0.57 -0.20 -0.11 -0.25 -1.20 -0.32 -0.32 -0.28 -0.04 -0.16
    Q3L8 TT: -0.28 -0.35 -0.44 -0.47 -0.38 -0.26 -0.08 +0.10 +0.04 -0.60
    Q3L8 EE: +0.59 -0.19 -0.10 -0.23 -1.15 -0.26 -0.26 -0.20 +0.03 -0.10
    Q8L8 TT: -0.44 -0.51 -0.57 -0.53 -0.38 -0.23 -0.05 +0.10 +0.01 -0.60
    Q8L8 EE: +0.58 -0.20 -0.11 -0.25 -1.19 -0.30 -0.30 -0.26 -0.02 -0.14
    Q12L8 TT: -0.43 -0.51 -0.57 -0.53 -0.38 -0.23 -0.05 +0.10 +0.01 -0.60
    Q12L8 EE: +0.58 -0.20 -0.11 -0.25 -1.19 -0.30 -0.31 -0.26 -0.02 -0.14
    Q3L8F0 TT: -0.23 -0.29 -0.36 -0.39 -0.34 -0.23 -0.08 +0.10 +0.04 -0.60
    Q3L8F0 EE: +0.59 -0.19 -0.10 -0.23 -1.15 -0.26 -0.26 -0.20 +0.03 -0.10
    Q5L8F0 TT: -0.40 -0.47 -0.51 -0.47 -0.33 -0.20 -0.04 +0.10 -0.00 -0.60
    Q5L8F0 EE: +0.58 -0.20 -0.11 -0.25 -1.20 -0.32 -0.32 -0.28 -0.04 -0.16
    Q12L8F0 TT: -0.38 -0.44 -0.49 -0.45 -0.33 -0.20 -0.05 +0.10 +0.01 -0.60
    Q12L8F0 EE: +0.58 -0.20 -0.11 -0.24 -1.19 -0.30 -0.31 -0.26 -0.02 -0.14



## Job 27746: the shipped configuration (Q5L8F0) on the LCDM grid and the massless fiducial, `--grid lcdm`


Errors in percent: max |C_l/ref - 1| over the band (TEn = (TE - ref)/sqrt(TT_ref EE_ref)); grid columns are the maximum over the grid points that have a CLASS reference, with the point and l of the maximum; massless = CosmoParams(m_ncdm=0) vs the massless-RECFAST reference. Points without a CLASS reference are scored against the Q5 run (marked). Timings: median of the cached cosmologies, seconds.

| cfg | q | l_max | n_eq | pt | hr | total | pk | grid TT 2-30 | grid TT 2-500 | grid EE 5-500 | grid TEN 2-500 | grid TT 2-1000 | massless TT 2-30 | massless TT 2-500 | massless EE 5-500 | massless TEN 2-500 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q5L8F0 | 5 | 8 | 106 | 8.9 | 7.3 | 18.5 | - | 0.535 (omega_b_high, l=7) | 0.692 (omega_cdm_low, l=445) | 1.652 (omega_b_high, l=16) | 1.128 (omega_b_high, l=25) | 5.576 (h_high, l=993) | 0.482 | 0.482 | 1.208 | 0.897 |

Per cosmology (TT 2-30 / TT 2-500 / EE 5-500 / TEn 2-500 [/ P_m lin z=0, 1e-3-1 Mpc^-1], percent):

| cfg | lcdm_fiducial | h_high | omega_b_high | omega_cdm_low | ns_high | massless_fiducial |
|---|---|---|---|---|---|---|
| Q5L8F0 | 0.51 / 0.60 / 1.20 / 0.87 | 0.47 / 0.56 / 1.28 / 0.92 | 0.54 / 0.57 / 1.65 / 1.13 | 0.47 / 0.69 / 1.56 / 1.05 | 0.47 / 0.64 / 1.20 / 0.87 | 0.48 / 0.48 / 1.21 / 0.90 |

Fiducial TT and EE vs CLASS (HyRec, 0.06 eV) at l = (2, 3, 5, 10, 20, 30, 50, 100, 200, 500) (percent):

    Q5L8F0 TT: -0.40 -0.47 -0.51 -0.47 -0.33 -0.20 -0.04 +0.10 -0.00 -0.60
    Q5L8F0 EE: +0.58 -0.20 -0.11 -0.25 -1.20 -0.32 -0.32 -0.28 -0.04 -0.16



## Jobs 27685 and 27762: batched P(k) versus the ncdm state dimension (`diags/diag_pk_batched_steps.py`)


Fiducial cosmology (0.06 eV). P_m(k, z=0) on 263 reference k in [0.00101, 0.994] Mpc^-1. Percent throughout.

## A: tolerance ladder, batched Rodas5 (P_m / P_m[Q5 at the same tolerance] - 1 at k = 0.01, 0.05, 0.1, 0.2, 0.5, 1; last column max |P_m / P_CLASS - 1|)

| hierarchy | rtol | k=0.01 | k=0.05 | k=0.1 | k=0.2 | k=0.5 | k=1 | vs CLASS | t (s) |
|---|---|---|---|---|---|---|---|---|---|
| Q0 | 1e-04 | -0.429 | +0.759 | +0.877 | +0.950 | +1.022 | +1.099 | 2.172 | 88.2 |
| Q3L8 | 1e-04 | +0.060 | +0.207 | +0.343 | +0.525 | +0.807 | +1.027 | 1.742 | 101.7 |
| Q5L8 | 1e-04 | +0.001 | +0.041 | +0.100 | +0.214 | +0.419 | +0.592 | 1.428 | 109.0 |
| Q5 | 1e-04 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | 1.331 | 108.2 |
| Q5L8F0 | 1e-04 | -0.240 | +0.060 | +0.110 | +0.250 | +0.580 | +0.900 | 1.464 | 105.8 |
| Q5F0 | 1e-04 | -0.240 | +0.017 | -0.014 | -0.029 | -0.009 | +0.036 | 1.358 | 103.0 |
| Q0 | 1e-05 | -0.429 | +0.759 | +0.877 | +0.950 | +1.022 | +1.099 | 2.172 | 86.3 |
| Q3L8 | 1e-05 | +0.060 | +0.207 | +0.343 | +0.525 | +0.807 | +1.027 | 1.743 | 110.4 |
| Q5L8 | 1e-05 | +0.001 | +0.041 | +0.100 | +0.214 | +0.419 | +0.592 | 1.428 | 116.6 |
| Q5 | 1e-05 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | 1.331 | 122.8 |
| Q5L8F0 | 1e-05 | -0.240 | +0.060 | +0.110 | +0.250 | +0.580 | +0.900 | 1.464 | 106.3 |
| Q5F0 | 1e-05 | -0.240 | +0.017 | -0.014 | -0.029 | -0.009 | +0.036 | 1.358 | 112.2 |
| Q0 | 1e-06 | -0.429 | +0.759 | +0.877 | +0.950 | +1.022 | +1.099 | 2.172 | 100.5 |
| Q3L8 | 1e-06 | +0.060 | +0.207 | +0.343 | +0.525 | +0.807 | +1.027 | 1.742 | 129.2 |
| Q5L8 | 1e-06 | +0.001 | +0.041 | +0.100 | +0.215 | +0.419 | +0.592 | 1.428 | 139.6 |
| Q5 | 1e-06 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | 1.331 | 149.0 |
| Q5L8F0 | 1e-06 | -0.240 | +0.060 | +0.110 | +0.250 | +0.580 | +0.900 | 1.464 | 125.9 |
| Q5F0 | 1e-06 | -0.240 | +0.017 | -0.014 | -0.029 | -0.009 | +0.036 | 1.359 | 140.5 |

## B: per-mode Kvaerno5 reference (each hierarchy's own converged answer)

| run | vs CLASS max | P_m / P_m[Q5 kv5 1e-7] - 1 at k = 0.01, 0.05, 0.1, 0.2, 0.5, 1 | t (s) |
|---|---|---|---|
| Q0@kv5@1e-07 | 2.172 | +0.0000 +0.0000 +0.0000 +0.0000 +0.0000 +0.0000 | 340.2 |
| Q3L8@kv5@1e-07 | error | - | - |
| Q5L8@kv5@1e-07 | error | - | - |
| Q5@kv5@1e-07 | error | - | - |
| Q5@kv5@1e-06 | 1.331 | +0.0000 +0.0000 +0.0000 +0.0000 +0.0000 +0.0000 | 526.4 |
| Q5L8F0@kv5@1e-06 | 1.464 | +0.0000 +0.0000 +0.0000 +0.0000 +0.0000 +0.0000 | 930.7 |
| Q5F0@kv5@1e-06 | 1.358 | +0.0000 +0.0000 +0.0000 +0.0000 +0.0000 +0.0000 | 1208.8 |
| Q5L8@kv5@1e-06 | 1.428 | +0.0000 +0.0000 +0.0000 +0.0000 +0.0000 +0.0000 | 458.6 |

Batched run minus its own Kvaerno5 reference (P_batched / P_kv5 - 1), i.e. the step-control error of each run:

| hierarchy | rtol | k=0.01 | k=0.05 | k=0.1 | k=0.2 | k=0.5 | k=1 | max over k |
|---|---|---|---|---|---|---|---|---|
| Q0 | 1e-04 | +0.000 | -0.001 | -0.001 | -0.000 | -0.000 | -0.000 | 0.003 |
| Q0 | 1e-05 | -0.000 | -0.000 | +0.003 | -0.000 | +0.000 | +0.000 | 0.006 |
| Q0 | 1e-06 | +0.000 | -0.000 | -0.000 | -0.000 | -0.000 | +0.000 | 0.001 |

## C: batched solve mirrored chunk by chunk at the preset tolerance (1e-4 / 1e-6)

| hierarchy | n_eq | n_k | batch | steps per chunk (accepted+rejected) | rejected | delta_m(tau0)^2 / Q5 - 1 at k = 0.01, 0.05, 0.1, 0.2, 0.5, 1 | mirror vs A at k=1 |
|---|---|---|---|---|---|---|---|
| Q0 | 61 | 101 | 101 | [324] | [39] | -0.414 +0.772 +0.881 +0.942 +1.019 +1.097 | A +1.099 / C +1.097 |
| Q3L8 | 91 | 101 | 101 | [324] | [39] | +0.061 +0.216 +0.361 +0.511 +0.797 +1.024 | A +1.027 / C +1.024 |
| Q5L8 | 109 | 101 | 101 | [324] | [39] | +0.001 +0.045 +0.110 +0.205 +0.411 +0.589 | A +0.592 / C +0.589 |
| Q5 | 154 | 101 | 101 | [324] | [39] | +0.000 +0.000 +0.000 +0.000 +0.000 +0.000 | - |
| Q5L8F0 | 106 | 101 | 101 | [324] | [39] | -0.238 +0.063 +0.120 +0.237 +0.567 +0.894 | A +0.900 / C +0.894 |
| Q5F0 | 151 | 101 | 101 | [324] | [39] | -0.239 +0.015 -0.016 -0.028 -0.010 +0.035 | A +0.036 / C +0.035 |

## D: lensing at the preset tolerance; max |ratio - 1| over l = 2..1000 (C_l^pp to 1500, lensed to 1500)

- Q3L8 vs CLASS: C_l^pp (linear) 10.631% (l=2); lensed TT 4.383% (l=1000), EE 1.152% (l=20) (unlensed TT 4.558% (l=1000)); halofit / linear on C_l^pp 16.167% (l=1000); lensed / unlensed TT 3.567% (l=1000)
- Q5 vs CLASS: C_l^pp (linear) 10.650% (l=2); lensed TT 4.352% (l=1000), EE 1.211% (l=20) (unlensed TT 4.522% (l=1000)); halofit / linear on C_l^pp 16.225% (l=1000); lensed / unlensed TT 3.562% (l=1000)
- Q5L8F0 vs CLASS: C_l^pp (linear) 10.646% (l=2); lensed TT 4.417% (l=1000), EE 1.208% (l=20) (unlensed TT 4.587% (l=1000)); halofit / linear on C_l^pp 16.068% (l=1000); lensed / unlensed TT 3.561% (l=1000)
- Q5F0 vs CLASS: C_l^pp (linear) 10.647% (l=2); lensed TT 4.350% (l=1000), EE 1.210% (l=20) (unlensed TT 4.519% (l=1000)); halofit / linear on C_l^pp 16.195% (l=1000); lensed / unlensed TT 3.561% (l=1000)
- Q5L8F0 / Q5F0: unlensed TT 0.072% (l=1000), EE 0.054% (l=877); C_l^pp linear 0.230% (l=1000), halofit 0.121% (l=1000); lensed TT (halofit pp) 0.071% (l=1000), EE 0.053% (l=880)
- Q3L8 / Q5: unlensed TT 0.179% (l=2), EE 0.107% (l=176); C_l^pp linear 0.463% (l=1000), halofit 0.412% (l=1000); lensed TT (halofit pp) 0.179% (l=2), EE 0.108% (l=176)


