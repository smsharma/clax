# planck_cl precision sweep toward ABCMB-like ODE settings (Oct 10-11, 2026)

**What the follow-up PRs did with these findings (Oct 20, 2026):** this file
is the sweep's write-up as it stood on `exp/precision-sweep`; the findings
below are left as written, and the outcomes are:

- Finding 5 (Rodas5 dense output): fixed by `fix/rodas5-dense-output` (#49,
  cubic Hermite dense output). `fit_cl` was not moved off rodas5; it, like
  `planck_cl` and `planck_fast`, ships on rodas5 at rtol 1e-4 / atol 1e-6
  (`fix/preset-retune`, #50), where rodas5 matches Kvaerno5 to 0.08% at l=2
  and <= 0.03% elsewhere. The sentence "Until then fit_cl (rodas5, rtol
  1e-3, atol 1e-4) is wrong at l<30" no longer applies.
- Finding 6 (Bessel table): the `x_per_unit` knob's default is now
  `JL_TABLE_X_PER_UNIT = 30` in `clax/harmonic.py` ("default 3 unchanged"
  below describes the sweep branch), and `L_SPLINE_PAD = 100` pads the
  sparse-l spline past the requested l_max (#50).
  Superseded by #54 (cubic Hermite lookup at 3 points per unit x; see ADR 0005).
- Implications: #50 chose rodas5 rather than the Kvaerno5 recommended here
  for `planck_cl` (l_max 17/17/17, rtol 1e-4, atol 1e-6, max_steps 8192,
  hr_n_k_fine 20000) and kept th_n_points 100000 for `planck_cl` (20000 in
  `planck_fast`): the 20000-vs-100000 C_l difference measured <= 0.05% to
  l=1000, ~0.13% above. Shipped preset vs CLASS massless RECFAST (job
  24668, before #53): TT 0.086 / 0.233 / 0.262% (l<=30 / 500 / 1000), EE
  0.60 / 0.46 / 0.98% (2-4 / 5-500 / 5-1000, the last a sparse-l-grid point
  at l~526); V100 cached 124 s end to end.
- `pt_ode_dtmax` stays on `exp/precision-sweep`; it is not on the branch
  that carries this document.
- The residual above l~1000 that findings 6 and 8 leave open was traced to
  the radiation-streaming trigger (`docs/highl_residual_2026-10-12.md` on
  `exp/highl-investigation`, fixed in the PR stacked on #50).

Status: COMPLETE (Oct 11, 07:50 JST); all runs done, including the fast-path
Bessel-table fix validation (G6x30k20 / H6x30k20). Script `diags/diag_precision_sweep.py`, wrapper
`slurm/precision-sweep-v100.sbatch`, branch `exp/precision-sweep` (from main
9198580), results `/lustre/work/n2minh/std/clax/sweep/results-*.{json,md}`,
igpu V100 jobs 24189-24191, 24195, 24237, 24239, 24240, 24243, 24246, 24253,
24262. Regenerate the full tables with
`python diags/diag_precision_sweep.py --report /lustre/work/n2minh/std/clax/sweep/results-*.json`.

## Question

Which perturbation-ODE settings does the planck_cl accuracy bar actually need,
and where does the planck_cl wall-clock go? Every earlier convergence test only
tightened settings (CHANGELOG: l_max 50/65/80 identical; rtol 1e-8 == 1e-6;
n_tau 10000 == 5000); nothing was ever loosened. ABCMB (Zhou, Giovanetti, Liu,
arXiv:2602.15104; `docs/differentiable_boltzmann_landscape_2026-10-10.md`)
reports ~2 permille TT agreement with CLASS to l=4000 with l_max 12/10/17/17,
rtol 1e-5 (k<0.01/Mpc) and 1e-4 above, atol 1e-10 and 1e-6, max_steps 2048,
no TCA/UFA/RSA, in 6.7 s on an H100. clax planck_cl: l_max 50, rtol 1e-6,
atol 1e-11, max_steps 131072, 300 k modes, 487 s on H100 (CHANGELOG Mar 2026),
4606 s on the igpu V100 (this sweep, perturbation stage, cached).

## Design

One factor at a time from `PrecisionParams.planck_cl()`, plus ABCMB-like
combinations, in two reference sets:

- massless: `CosmoParams(m_ncdm=0.0)` with `ncdm_q_size=0` (clax then takes
  the ncdm perturbations from the massless-neutrino hierarchy, exact at m=0)
  vs `reference_data/cls_massless_recfast.npz` (CLASS, RECFAST).
- massive: `CosmoParams()` (one 0.06 eV species) vs
  `reference_data/lcdm_fiducial/cls.npz` (CLASS default recombination, HyRec;
  clax uses RECFAST, so absolute errors in this set carry a recombination
  systematic; use the differentials).

The C_l stage is fixed for all configs unless stated (`compute_cls_all_fast`,
table Bessel, n_k_fine 5000, l_max 2500), so config-minus-baseline isolates the
perturbation settings. Timing protocol as `scripts/benchmark_speed.py`: first
call at the target cosmology (compile + run; its C_l are the accuracy sample),
second call at h=0.70, omega_cdm=0.13 for cached timings. Timings from igpu01
(jobs 24189-24191) shared the node with two to four other jobs; igpu02/05 runs
were less contended (F: cached 1477 s vs first call 1103 s). Ratios above 2x
are robust; 20-30% differences are not.

Diagnostic knob added on this branch (default off, not for merge as is):
`PrecisionParams.pt_ode_dtmax` (Mpc), passed as `dtmax` to both perturbation
PID controllers.

## Headline table (massless set; cached V100 seconds; max |config/A0 - 1| in percent)

| cfg | hierarchy g/pol/ur | rtol / atol | solver | other | d | pt (s) | TT l<=500 | TT 500-2000 | EE l<30 | EE 30-2000 | TEn l<=2000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 50/50/50 | 1e-6 / 1e-11 | kvaerno5 | planck_cl as documented | 160 | 4606 | 0 | 0 | 0 | 0 | 0 |
| E | 50/50/50 | 1e-4 / 1e-8 | kvaerno5 | | 160 | 523 | 0.005 | 0.004 | 0.002 | 0.002 | 0.003 |
| E6 | 50/50/50 | 1e-4 / 1e-6 | kvaerno5 | | 160 | 179 | 0.094 (l=2), 0.003 at l>=30 | 0.004 | 0.031 | 0.007 | 0.060 |
| B6 | 17/17/17 | 1e-4 / 1e-6 | kvaerno5 | | 61 | 100 | 0.091 (l=2), 0.006 at l>=30 | 0.053 | 0.051 | 0.044 | ~0.06 |
| G6 | 12/10/17 | 1e-4 / 1e-6 | kvaerno5 | | 49 | 101 | 0.090 (l=2), 0.008 at l>=30 | 0.054 | 0.176 | 0.064 | 0.129 |
| G6dt20 | 12/10/17 | 1e-4 / 1e-6 | kvaerno5 | step cap 20 Mpc | 49 | 152 | 0.005 | 0.054 | ~0.18 | ~0.06 | ~0.13 |
| G | 12/10/17 | 1e-4 / 1e-8 | kvaerno5 | | 49 | 256 | 0.005 | 0.051 | 0.192 | 0.06 | 0.127 |
| G4 | 12/10/17 | 1e-4 / 1e-8 | kvaerno5 | max_steps 4096 | 49 | 256 | identical to G to all digits | | | | |
| C6 | 12/10/17 | 1e-5 / 1e-8 | kvaerno5 | | 49 | 334 | 0.005 | 0.050 | 0.193 | 0.06 | 0.128 |
| F | 12/10/17 | 1e-5 / 1e-10 | kvaerno5 | | 49 | 1477 | 0.005 | 0.050 | 0.193 | 0.06 | 0.128 |
| F8 | 12/10/17 | 1e-5 / 1e-10 | kvaerno5 | max_steps 8192 | 49 | fail | "maximum number of solver steps was reached" | | | | |
| J | 12/10/17 | 1e-5 / 1e-10 | kvaerno5 | n_tau 2000 | 49 | 1303 | 0.023 | 0.079 | 0.591 | 0.12 | |
| H | 12/10/17 | 1e-5 / 1e-10 | kvaerno5 | k_max_cl 0.5 | 49 | 1099 | 0.773 | 7.3 | 6.0 (l=2) | 2-3 | 4.8 |
| I | 12/10/17 | 1e-5 / 1e-10 | rodas5 | | 49 | 88 | 4.08 (l=2) | 0.36 | 0.19 | 0.37 | 2.67 |
| Idt20 | 12/10/17 | 1e-5 / 1e-10 | rodas5 | step cap 20 Mpc | 49 | 105 | 0.46 (l=2) | 0.36 | 0.20 | 0.37 | 0.25 |
| Idt5 | 12/10/17 | 1e-5 / 1e-10 | rodas5 | step cap 5 Mpc | 49 | 200 | 0.021 | 0.36 | | | |
| I7 | 12/10/17 | 1e-7 / 1e-12 | rodas5 | | 49 | 331 | 0.77 (l=2), 0.015 at l>=30 | 0.10 | 0.19 | 0.10 | |
| I6 | 12/10/17 | 1e-4 / 1e-6 | rodas5 | | 49 | 24 | 46.2 (l=2) | 0.80 | 0.83 | 0.91 | 48 |
| I6dt20 | 12/10/17 | 1e-4 / 1e-6 | rodas5 | step cap 20 Mpc | 49 | 50 | 0.70 | 0.80 | 0.33 | 0.91 | 0.38 |
| A0r | 50/50/50 | 1e-6 / 1e-11 | rodas5 | planck_cl tolerances | 160 | 348 | 1.80 (l=2), 0.035 at l>=30 | 0.13 | 0.022 | 0.18 | |
| G6k30 / G6k120 | 12/10/17 | 1e-4 / 1e-6 | kvaerno5 | 30 / 120 k per decade | 49 | 78 / 142 | identical to G6 for l<=300 | | | | |
| G6k20 | 12/10/17 | 1e-4 / 1e-6 | kvaerno5 | harmonic n_k_fine 20000 | 49 | 94 (hr 5.4) | 0.33 | 3.8 | 0.18 | 2.0 | 3.5 |
| G6a8 / G6t10 | 12/10/17 | 1e-4 / 1e-6 | kvaerno5 | bg table from a=1e-8 / n_tau 10000 | 49 | 101 / 100 | identical to G6 (<0.01%) | | | | |

Massive set (one 0.06 eV neutrino; max |config/M6 - 1| in percent; M6 = full
photon hierarchy l_max 50 at rtol 1e-4 / atol 1e-6):

| cfg | photon/ur l_max | q-bins | l_max_ncdm | d | pt (s) | TT l<=1000 | TT 1000-2000 | EE l<30 | EE 30-500 |
|---|---|---|---|---|---|---|---|---|---|
| M6 | 50 | 5 | 17 | 253 | 271 | 0 | 0 | 0 | 0 |
| K6 | 12/10/17 | 5 | 17 | 142 | 169 | 0.006 | 0.032 | 0.203 | 0.059 |
| N6 | 12/10/17 | 5 | 50 | 307 | 330 | 0.005 | 0.031 | 0.196 | 0.059 |
| L6 | 12/10/17 | 3 | 17 | 106 | 142 | 0.18 (l<30), 0.063 | 0.056 | 0.251 | 0.116 |
| M5 | 50 | 5 | 17 | 253 | 7006 (rtol 1e-6 / atol 1e-11, the documented preset) | M6 vs M5: 0.095 (l=2), 0.003 at l>=30 | 0.008 | 0.027 | 0.005 |

Thermodynamics costs 55-77 s per call in every config (th_n_points 100000);
the harmonic stage 4 s (fast path, n_k_fine 5000).

## Conclusions

1. **The planck_cl ODE settings are over-resolved by 25-45x.** At the same
   l_max 50 hierarchy, rtol 1e-4 / atol 1e-8 (E) reproduces the preset to
   0.005% in 523 s instead of 4606 s; rtol 1e-4 / atol 1e-6 (E6) is within
   0.1% (0.094% at l=2 only; 0.003% for l>=30) in 179 s.
2. **Cost is step-count-bound and set by the tolerances, mostly atol.** At
   the CLASS-default hierarchy: 1477 s (1e-5/1e-10), 334 s (1e-5/1e-8),
   256 s (1e-4/1e-8), 101 s (1e-4/1e-6); all four agree to <0.01% for l>=30.
   Hierarchy size is a per-step cost: d 160 -> 49 is 1.8-2x at fixed
   tolerance. Mode count barely matters (150/300/600 modes: 78/101/142 s).
   The step budget changes nothing once sufficient (G4 == G exactly); the
   ABCMB budget of 2048 is unreachable at atol 1e-10 (F8 fails at 8192).
3. **Hierarchy truncation is the only setting with an effect at the 0.1%
   level, and only in polarization at low l.** 12/10/17 vs 50: EE 0.18-0.19%
   at l~20, 0.06% at l=30-500, TE 0.13% at l<=50, TT 0.05-0.06% above
   l=1000. 17/17/17 (B6) halves that: EE 0.05% at l<30, 0.03-0.04% elsewhere,
   at the same 100 s. The 12/10/17 run is the one closer to CLASS at low l
   (CLASS truncates there too).
4. **Cheapest settings holding the bars against the documented preset
   (Kvaerno5, l<=2000):** within 0.1%: B6 = l_max 17/17/17, rtol 1e-4,
   atol 1e-6 (100 s; the 0.09% is TT at l=2 and drops to 0.005% with a
   20 Mpc step cap at +50% cost); within 0.2%: G6 (CLASS-default hierarchy,
   101 s). E6 (l_max 50, 179 s) if the 0.05% low-l EE truncation effect
   matters. Massive neutrinos: K6 settings (5 q-bins, l_max_ncdm 17) at
   169 s are within 0.03% of the full hierarchy outside the photon
   truncation effect; l_max_ncdm 50 is converged-from-above waste (N6 ==
   K6, 330 s); 3 q-bins cost 0.1-0.25%.
5. **Rodas5 (the fit_cl production solver) has a dense-output defect.**
   `clax/rosenbrock.py` evaluates output times by `LocalLinearInterpolation`
   (lines 123, 235); the late-ISW source needs the metric derivatives at the
   output times, and Rodas5's late-time steps are hundreds of Mpc. Relative
   to the Kvaerno5 baseline its TT is off by 4.1% at l=2 at rtol 1e-5 /
   atol 1e-10, 46% at rtol 1e-4 / atol 1e-6, and 1.8% at the documented
   planck_cl tolerances (A0r); the error scales with the late-time step:
   4.08% -> 0.46% (20 Mpc cap) -> 0.021% (5 Mpc cap) at fixed tolerance,
   while Kvaerno5 is unaffected by the cap (G6dt20 0.005%). Mechanism from
   the linrec session, confirmed here. A second, tolerance-driven residual
   of 0.36-0.44% above l=1000 at 1e-5/1e-10 (0.10% at 1e-7/1e-12) is
   genuine solver error. Rodas5 is 4-17x faster than Kvaerno5 at equal
   tolerance (24 vs 101 s; 88 vs 1477 s), so a higher-order dense output
   (Hermite from y, f at step ends, or the Rodas5P continuous extension)
   plus a tolerance retune would make it the fastest correct path. Until
   then fit_cl (rodas5, rtol 1e-3, atol 1e-4) is wrong at l<30 beyond its
   documented 1.5%; its accuracy was spot-checked at l=20,100,500,1000 only.
6. **Confirmed: the pre-existing -1% offset, the +20% EE / +10% TE at l=2
   and the k_max dependence are all the fast path's linearly interpolated
   Bessel table.** `compute_cls_all_fast` builds j_l on a uniform x grid with
   `n_x = max(30000, 3*x_max)`, `x_max = 1.05*k_max*tau_0`, and interpolates
   linearly; linear interpolation of a period-2pi oscillation at spacing h
   biases the amplitude by about -h^2/12 on average (-0.9% in Delta_l at
   h=0.33 for k_max 1.0, -0.5% at h=0.245 for k_max 0.5). With the identical
   perturbation solve (G6 settings) and exact j_l (`compute_cls_all_interp`,
   n_k_fine 10000; G6i at k_max 1.0, H6i at k_max 0.5), clax vs CLASS reads
   TT +0.07% (l=2), -0.06% (l=20), +0.005% (l=50), +0.05% (100-300), +0.21%
   (500), +0.17% (700-1000); EE -0.16% (l=2), -0.1..-0.3% (10-100), +0.02%
   (300), -0.08% (500); TE within 0.17% to l=1000; and G6i and H6i agree with
   each other to 0.01% at l<=1000. Band maxima: TT 0.07% (2-30), 0.24%
   (30-500), 0.27% (500-1000); EE 0.30%, 0.31%, 0.99% (one point, rms 0.29%).
   Every other candidate was excluded by direct test: ODE tolerance,
   hierarchy, k density (30/60/120 per decade identical at l<=300), fine k
   grid (5000 vs 20000 identical at l<=100), background-table start (a=1e-8
   identical), tau grid (10000 identical; 2000 only 0.1% worse at l<=3).
   Above l=1000 the exact-j_l path still shows TT rms 1.7% (1000-2000) and
   2.2% (2000-2500), EE rms 3% and 8%: that is the k sampling / source
   interpolation (finding 8), not the table. Cost: the slow path's harmonic
   stage is 4085 s per call on the V100 (per-l JIT over the sparse l grid)
   against 4 s for the fast path, so the fix belongs in the fast path (finer
   x table and/or higher-order table interpolation, or exact j_l on the fly);
   Finer-table test (fast path, `x_per_unit` knob on `compute_cls_all_fast`,
   default 3 unchanged): 10 per unit x (harmonic 7 s) brings TT at l=2-300
   to within 0.15% of CLASS and EE at l=2 to +3.6%; 30 per unit x (harmonic
   12 s) matches the exact-j_l path to 0.012% in TT at l<=30 and sits within
   0.07% of CLASS over l=2-300 (EE l=2: +0.6%). Above l~500 the fast path
   still differs from the exact path by 1.6% (500-1000), 4.0% (1000-2000)
   and 7.6% (2000-2500) in TT at n_k_fine 5000, which is the fine k grid:
   table 30/unit x plus n_k_fine 20000 (G6x30k20, harmonic 13 s) reproduces
   the exact-j_l path to 0.011% in TT at l<=500, 0.024% at 500-1000, 0.17%
   at 1000-2000 and 0.48% at 2000-2500 (EE within 0.01% at l=30-1000; the
   l=2 EE residual of 0.76% is the only low-l difference left), and is
   k_max-independent (H6x30k20 identical to l=1000). Against CLASS the fast
   path then reads TT max 0.08% (l=2-30), 0.23% (30-500), 0.26% (500-1000),
   EE max 0.31% (30-500), rms 0.29% (500-1000). So the planck_cl "0.2%"
   claim is reproducible to l~1000 in the fast path, at the loose ODE
   settings: 100 s of perturbations + 13 s of harmonic on the V100. What
   remains above l~1000 (TT rms 1.7% at 1000-2000, 2.0% at 2000-2500; EE rms
   3% and 8%) is common to the exact path and the fast path and is the next
   C_l-stage item (perturbation k sampling / source interpolation, finding
   8, and whatever else sits at l>1000).
7. **Thermodynamics is now the dominant stage** at 55-77 s per call
   (th_n_points 100000); planck_fast documents 20000 points at ~5 s with
   matching accuracy.

## Implications

- planck_cl can be re-tuned to Kvaerno5, l_max 17/17/17 (or 50), rtol 1e-4,
  atol 1e-6, max_steps 8192, th_n_points 20000: about 100-180 s of
  perturbations plus 5 s of thermodynamics on this V100 instead of 4606 + 69 s,
  with the spectra within 0.1% of today's preset. That is a preset change,
  not a solver change, and should land before any solver work.
- The linrec parallel-in-time target is therefore a ~100 s sequential floor
  (step-count-bound, d~50-60), not 4600 s; linrec's C_l validation must use
  converged Kvaerno5 and CLASS as oracles, not the rodas5 production path.
- The integral-equation route's remaining case is massive neutrinos and
  extensions: the ncdm block costs 70% on top of the massless solve at 5
  q-bins (169 vs 101 s) and the photon hierarchy at CLASS defaults is cheap.
- Before any 0.1%-vs-CLASS claim, fix the C_l stage: the Bessel table
  (finding 6: finer x grid or higher-order interpolation; this alone takes
  TT/EE to within 0.3% of CLASS up to l=1000 and removes the l=2 anomaly),
  then the k sampling / source interpolation above l~1000 (finding 8).

## Pending

- G6i / H6i done (Oct 11, 06:59): Bessel-table hypothesis confirmed (finding 6).
  G6t20 (n_tau 20000) == G6.
- G6x10 / G6x30 / G6x30k20 / H6x30k20 done (Oct 11, 07:47): see finding 6. Nothing pending.
- M5 done (Oct 11, 03:13): 7006 s cached; M6 (rtol 1e-4 / atol 1e-6, 271 s) matches it to
  0.095% at l=2 and 0.003% for l>=30 (EE 0.027% at l<30), so the massive
  preset is 26x over-resolved exactly like the massless one. K6 vs M5: TT
  0.094% (l=2), 0.008% at l>=30; EE 0.187% (truncation) at l<30, 0.06% at
  30-500.

## Auto-generated tables

See `/lustre/work/n2minh/std/clax/sweep/report-final.md` (regenerate with the
`--report` command above).
