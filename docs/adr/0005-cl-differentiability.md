# ADR 0005: C_l, C_l^phiphi and lensed C_l are differentiable at fit_cl

Date: 2026-10-25
Status: Accepted
Issue: https://github.com/smsharma/clax/issues/54 (related: #53 / ADR 0003,
#59 / ADR 0004, #57, #61)

Every number below was measured on the branch's base, upstream main at 5ac3df3
(after #53, #49-#51 and #59; before #58, which gives `fit_cl` a massive-neutrino
hierarchy, and before #60 and #61), on V100-32GB GPUs with JAX 0.9.2 unless a CPU
job is named. Main has since moved to 6ff1303 with #58, #60 and #61 merged
(Oct 25, 2026); nothing here has been re-measured on it (Not done). The lookup
comparison of Decision 2 and some value measurements ran on 99b62fe, just before
#59; #59 changes derivatives through the thermodynamics but leaves every value
bit-identical (ADR 0004), and the lookup comparison sets two lookups against each
other on one and the same solve. A rebase onto a later main re-measures the C_l-versus-CLASS baseline
(Consequences).

F_TT, F_pp and F_lTT below are sum_{l=30}^{1200} of ln C_l^TT (unlensed),
ln C_l^phiphi and ln (lensed C_l^TT), at `fit_cl`. The range l >= 30 avoids
`fit_cl`'s known low-l instability.

## Context

A NUTS, Fisher or MAP analysis of CMB spectra at `fit_cl` needs dC_l/dtheta for
theta in {h, omega_b, omega_cdm, N_eff, m_nu, ...}. Before this change there was
no such derivative, for five reasons.

- **Nothing could be differentiated.** `compute_cls_all_fast` sized its table of
  j_l(x), x = k chi, from `float()`/`int()` of 1.05 max(k) tau_0
  (`clax/harmonic.py:836-837` at 2717b5c). `jax.jvp`, `jax.grad` and `jax.jit`
  raised ConcretizationTypeError for every parameter that moves the conformal
  age tau_0, i.e. every parameter except the primordial ones.
- **Reverse mode would not have fitted on the GPU.** With the `float()` removed,
  reverse mode stores the residuals of the loop over the sparse multipoles:
  5.74 GiB per stored intermediate and about 46 GB in total by XLA's estimate
  (scratch job 24261), more than a 32 GB V100 holds.
- **The derivative of the lookup was a staircase.** The table was interpolated
  linearly in x, so the AD derivative of j_l(k chi) was the chord slope between
  neighbouring table points, constant on each interval, not j_l'(k chi). With
  main's 30 points per unit x (#50), dF_TT/dh wobbled by 1.4% about its smooth
  trend along h (cubic-fit residual 1.36e-2 over h = 0.64-0.70) and sat 1.7% low
  (-246.73, against -251.12 and -251.38 from finite differences at relative
  steps 1e-3 and 3e-3; job 27212; the Hermite lookup of Decision 2 gives
  -251.51, job 27213).
- **The k-integral was undersampled.** `fit_cl` integrated C_l over 5000 fine k
  points, about one sample per oscillation of |T_l(k)|^2 near l ~ 1200 (spacing
  2.0e-4 Mpc^-1 against a period pi/chi_* ~ 2.3e-4 Mpc^-1). As h moves tau_0
  the oscillations slide past the nodes, so dF_TT/dh changed sign along h
  (+1128, -2107, -1304, +1478, +1005, -1518, -1909 at h = 0.64 ... 0.70) with
  either lookup, and at fiducial it was -1492.5 against the converged -251.5
  (jobs 27211, 27227).
- **Lensing could not be traced.** The Halofit correction built its redshift
  grid from `float(th.z_rec)` and branched in Python on `float(pt.k_grid[-1])`
  (`clax/lensing.py:183-194`), and its linear-P(k) calls validated the k-grid
  with numpy (`clax/transfer.py:54`). The EPT correction took `float(params.h)`
  and picked EPT nodes with an h-dependent boolean mask, so its array shapes
  changed with cosmology (`clax/lensing.py:312-347`). The EPT IR-resummation
  grid was built from a concrete copy of h (`clax/ept.py:2705`). The linear
  lensing path had a gradient test along A_s only.

One more defect, pre-existing and found on the way: **every Halofit gradient was
wrong.** Halofit's nonlinear scale R_nl, the root of sigma^2(R_nl) = 1, was
found by a fixed-bounds `jnp.where` bisection, which has zero AD tangent. With A
scaling P_lin, d k_sigma/dA was 0 by AD against -0.195 by finite differences,
and d sum ln P_NL/dA 294.4 against 529.4 (CPU job 28762). Downstream, dF_pp/dh
was 11-14% off finite differences, dF_pp/d omega_b 31-32%, and dF_lTT 0.13-0.17%
(h) and 0.9-2.3% (omega_b) (jobs 28542-28547). The same zero tangent sits in
every P_NL gradient through `compute_pk_nonlinear`, lensing or not.

## Decision

1. **A fixed-range Bessel table, built once per process.**
   - The table covers x in [0, 1.05 pt_k_max_cl hr_tau0_max]. `hr_tau0_max` is
     a new `PrecisionParams` field, the largest tau_0 the table serves
     (default 20000 Mpc; fiducial tau_0 ~ 14100 Mpc).
   - Its shape depends on precision settings only, so it is built once per
     (l_max, n_x, x_max) and reused across cosmologies and calls
     (`harmonic._jl_tables`). At `fit_cl`: x_max = 21000, 63000 points, 85 rows
     (the sparse l grid to l_max + 100), 82 MiB; 101 MiB at l_max 2500. A cold
     build takes 3.6 s on a V100 and 10.6 s on CPU, a cached call 0.1 ms
     (jobs 26870/26871, measured with 83 rows).
   - Two `eqx.error_if` guards raise at run time, under jit too:
     - tau_0 > `hr_tau0_max` raises, naming the bound;
     - k chi beyond the table raises. That happens when `prec.pt_k_max_cl` does
       not match the perturbation grid.
   - `compute_cls_all_fast` and `compute_cl_pp` gain a keyword-only
     `prec: PrecisionParams | None = None`. With `prec` the table bounds are
     static, so jit, jvp and grad work. Without it, eager callers work as before
     (k_max read from `pt.k_grid`), and a traced call raises a TypeError that
     asks for `prec`.

2. **Cubic Hermite lookup at 3 points per unit x.**
   - On each table interval, j_l and j_l' come from the cubic Hermite
     interpolant of the stored values and slopes at both nodes. The slope of
     j_l' is j_l'' from Bessel's equation, [l(l+1)/x^2 - 1] j_l - (2/x) j_l'.
     The error falls as (spacing)^4 instead of (spacing)^2, and the AD
     derivative is continuous in x and close to j_l'.
   - Against direct spherical Bessel functions, at 3 points per unit x
     (x in [0, 21000], l = 2, 30, 300, 1200; job 27124):
     - the worst error is 3.6e-6 in j_l, 3.3e-6 in j_l' and 3.4e-5 in the AD
       derivative, all at l = 2;
     - the linear lookup at the same density errs by 2.0e-3, 1.7e-3 and 2.4e-2.
   - The density was chosen by measurement, on F_TT at fiducial (jobs
     27208-27214, 27227). Three lookups were compared: linear at 30 points per
     unit x (main), Hermite at 3, and Hermite at 30.
     - Smoothness of dF_TT/dh along h (cubic-fit residual of 7 points over
       h = 0.64-0.70, bar 3e-4). At 20000 fine k points: linear at 30 gives
       1.36e-2, Hermite at 3 gives 5.6e-6, Hermite at 30 gives 5.9e-6. At 5000
       fine k points every lookup gives 1.26-1.27 (Decision 4).
     - Hermite at 3 against Hermite at 30:
       - gradients agree to <= 6e-6 (h), 2.5e-6 (omega_b) and 9e-7 (omega_cdm);
       - C_l^TT agrees to <= 1.6e-5;
       - C_l^EE agrees to 1.4e-4 at l = 2 and <= 1.1e-5 above l = 30.
     - Cost of Hermite at 3 against Hermite at 30:
       - table memory 82 MiB against 0.80 GiB;
       - compiled-in table copy 78 / 162 MiB (forward / gradient program)
         against 762 / 1529 MiB;
       - compile time 48 / 340 s against 62 / 360 s.
   - So the table stays at 3 points per unit x (`JL_TABLE_X_PER_UNIT = 3.0`;
     #50 had raised it to 30 for the linear lookup). The linear lookup is gone.

3. **Bounded reverse-mode memory.**
   - The body of the loop over multipoles is wrapped in `jax.checkpoint`. The
     backward pass recomputes each multipole's lookups instead of storing them,
     and the forward result is unchanged.
   - Peak device memory with the jitted forward and gradient programs both alive
     is 6.20 GiB by the allocator and 13.7 GiB by NVML (job 30153).

4. **`fit_cl` integrates C_l on 20000 fine k points (`hr_n_k_fine` 5000 ->
   20000), as `planck_cl` and `planck_fast` already did.**
   - Without it dF_TT/dh changes sign along h for any lookup. With it the
     derivative is smooth (Decision 2).
   - Cost: +3.7% of eager forward time (job 27208).
   - `compute_cls_all_fast` takes `n_k_fine=None`: with `prec` the grid follows
     `prec.hr_n_k_fine`, so a caller that passes `prec` and forgets `n_k_fine`
     no longer integrates on 5000 points (dF/dh 6x off, sign-changing). Without
     `prec` (the eager path) the default stays 5000. An explicit `n_k_fine` wins.

5. **Lensing traces everything that depends on cosmology, on fixed shapes.**
   - **Halofit.**
     - The redshift grid is linspace(0, ln(1 + z_rec), n_z) with z_rec traced.
     - The size of the high-k extension comes from `prec.pt_k_max_cl`.
     - Both linear-P(k) calls pass `validate=False`: validation needs concrete
       arrays, the lensing path is traced, and the k-grid was validated when
       the perturbations were solved.
   - **EPT.**
     - h stays traced. P_NL is cubic-splined over the fixed EPT nodes in
       ln(k/h) and divided by P_lin at the exact perturbation k, so the ratio
       R = P_NL/P_lin is smooth in h. Before, R was splined through the EPT
       nodes that happened to fall inside the perturbation k-range, a set that
       changes with h.
     - R = 1 at the last perturbation node, k = pt_k_max_cl (a static condition
       in Mpc^-1). The one-loop EFT is not valid at 1.48 h/Mpc, where it gives
       R ~ 9.7. With this exclusion the EPT C_l^phiphi reproduces the pre-#54
       spectra to <= 8.7e-6 at every h in [0.62, 0.72] (jobs
       28562/28571/28576). Applying the ratio there shifted C_l^phiphi by
       +2.9e-3 at l = 1200.
     - The EPT ratio against the pre-#54 code: |Delta R/R| <= 6.9e-4 for
       k <= pt_k_max_cl (at k = 0.156 Mpc^-1; bar 1e-3); C_l^phiphi(EPT) moves
       by <= 3.5e-6 per l band (job 28614).
     - The IR-resummation grid in `clax/ept.py` (DST grid 7e-5/h ... 7/h)
       follows the traced h. The change is 10 lines with static shapes.
       - The primal moves by roundoff from reordered arithmetic: EPT P(k) by
         <= 1.6e-12, C_l^phiphi(EPT) by 7e-16 (job 28536).
       - The EPT h-derivative gains the term from the grid's motion, which
         central finite differences always saw. The stage-level per-k
         dP_mm/dh now matches finite differences to a median 1.4e-5 (max
         2.6e-4) over 31 modes in 0.05-0.3 h/Mpc, against 1.9e-3 (max 1.3e-2)
         with the frozen grid on main (CPU jobs 28628, 30245); end to end,
         d sum P_mm/dh matches to 9.8e-5 (job 28628).
       - This is what makes EPT lensing jit- and grad-able. The Sep 1, 2026
         IR-resummation plan had frozen the endpoints on purpose; the reason to
         trace them is this ADR's.
   - **Halofit's nonlinear scale carries its derivative.**
     - log10 R_nl is a `jax.custom_jvp`: the primal is the unchanged bisection,
       and the tangent is the implicit-function one, d R_nl/d theta =
       -(d sigma^2/d theta)/(d sigma^2/d R) at the root.
     - The primal is bit-identical, eagerly and under jit, at the five LCDM grid
       points (jobs 28762, 28763).
     - d k_sigma/dA is now -0.19519 against FD -0.19520 (8.5e-5), and
       d sum ln P_NL/dA 529.403 against 529.419 (3.0e-5).
     - Unit test `tests/test_nonlinear.py::TestHaloFitTangent`: 35 cases
       (k_sigma, n_eff, C along the amplitude of P_lin and along a shift of
       ln k), worst 2.3e-4 against a bar of 1e-3. All 30 derivative cases fail
       on the old code (job 30067).
   - **Two guards keep the eager k-range checks as run-time checks under jit.**
     Main's lensing path could not be jitted at all. Reading P_lin with
     `validate=False` (the eager support check needs concrete arrays) would have
     let a bad input pass silently, so each check is repeated as a run-time guard:
     - A P_lin query outside the k-range that pt_mpk was solved on raises.
       Without the guard the spline clamps at pt_mpk's k_max, and R(k),
       C_l^phiphi and lensed TT come out silently wrong.
     - A `prec` whose `pt_k_max_cl` differs from `pt.k_grid[-1]` raises. Without
       the guard it silently resizes the Halofit k-extension.
     - The guards change no formula, but they change the compiled program: the
       Halofit and EPT C_l^phiphi and lensed TT move by <= 1.4e-15, and their
       d/dh by <= 4.7e-14, deterministically (jobs 30065/30072). The compiled
       Halofit program gains four run-time checks and two branches, the EPT
       program two checks and one branch.
   - `lens_cls` and the linear Limber C_l^phiphi needed no change.

6. **The gradient cost is accepted as measured: 4.8x a forward evaluation.**
   - Measured at fiducial, both programs jitted, in one process (job 30153):
     - the reverse gradient of F_TT with respect to all 20 `CosmoParams` fields
       takes 31.8 s;
     - the jitted production forward takes 6.67 s in that same process, on
       the tree with the Rodas5 fix that reverse mode needs (6.70 s on this
       branch's solver);
     - the ratio is 4.77 (2.93 against an eager forward).
   - The spec's budget was 3x. The cost is not in this change: the reverse pass
     of the perturbation stage alone, timed in its own process (job 27227),
     takes 33.7 s, as long as the whole gradient here. That is the
     checkpointed adjoint through the Rodas5 solve. The gradient time does not
     depend on the lookup or on the number of fine k points (31.18 s at 5000,
     31.19 s at 20000).
   - The user waived the 3x budget. A follow-up issue on the adjoint's cost is
     drafted. Until it lands, NUTS at `fit_cl` pays about five forward
     evaluations per gradient.

7. **Rodas5 reverse mode and Rodas5's step sequence.** Two findings decide how
   the gradients are verified.
   - **Reverse mode through Rodas5 at `fit_cl` was NaN on main.** In a vmapped
     batch of k-modes, a mode that has reached the end time takes masked
     dt = 0 steps. The Rosenbrock stage matrix I/(dt gamma) - J is then
     infinite, and the backward pass returns NaN (job 26946). The fix (a
     guarded divisor, forward bit-identical) is its own PR, branch
     `fix/rodas5-reverse-nan-v2`. Here every reverse-mode gradient test through
     the perturbation solve is `xfail(strict=True)` until it merges.
   - **Rodas5's accepted-step sequence does not reproduce between
     compilations.**
     - Rodas5 is linearly implicit, so compile-order roundoff reaches its
       accept/reject decisions. The jvp program and the grad program of one
       function integrate different step sequences: the primals differ by
       4.7e-6 relative, against 2.5e-11 for Kvaerno5 (jobs 27675/27676).
     - So no check tighter than ~1e-5 rests on Rodas5's step sequence. The
       jit-of-grad and vmap checks of the harmonic stage hold the ODE solve
       fixed (bar 1e-10). The AD-against-finite-difference gate runs on
       Kvaerno5, and its Rodas5 companion's bar adds the finite difference's
       own spread (a 9-point stencil) to the spec's 1e-3.
     - The same sensitivity makes the primal wobble along a parameter. At
       `fit_cl` tolerance (rtol 1e-4, atol 1e-6), F_TT(omega_b) departs from a
       cubic by 1.9e-8 RMS (9.7e-8 max) of |F|, and a 1% stencil turns that
       into ~1e-3 in the derivative (job 28117). At 100x tighter tolerance the
       wobble is 1.6e-9 (job 28245), and at 10^4x 1.5e-9 (jobs 28262-28265).
     - The wobble is not tau_star (its grid index does not change along
       omega_b), not the background solve, not the radiation-streaming switch,
       and not the T_b Taylor switch (jobs 28117, 28133, 28168, 28190, 28244).
     - The AD-against-FD gate therefore evaluates both at the 100x tighter
       perturbation tolerance, a precision setting of the reference like the
       8000-point background grid. Its bar is unchanged.

8. **The C_l-versus-CLASS criterion.** Each (preset, spectrum, l band) is
   compared with CLASS at equal k-grid. The base is the pre-#54 code at
   `fit_cl` with 20000 fine k points too, so that only the lookup and the table
   differ. A band passes if either:
   - its error against CLASS, |C/C_CLASS - 1|, grew by <= 1e-4 (the plan's
     criterion); or
   - the spectrum moved by <= 1e-4 of itself relative to the pre-#54 code.

   TE is normalised by sqrt(TT EE) of the reference in both. Bands from l = 30
   up are checked multipole by multipole; 2-30 is checked on the band maximum,
   where the CLASS reference is noisy and `fit_cl` unstable.

   Why the second branch: the Hermite lookup raises TT near l ~ 40-50 by
   9.6e-5 to 9.9e-5 of itself at every cosmology. That is the removal of the
   linear 30-point table's amplitude bias, -(dx)^2/12 ~ -9.3e-5 at table
   spacing dx = 1/30. At the 0.15 eV
   massive-neutrino point, `fit_cl` misses CLASS by 4.94% in TT, because it
   follows the neutrino as massless (#57). Measured there as a difference of
   relative errors, the same shift reads 1.0086e-4. The first branch alone
   would flag the removal of a bias.

## Alternatives considered

1. **Keep the linear lookup at 30 points per unit x (main) and only fix the
   tracing.** It runs, but its derivative is the chord slope: dF_TT/dh is
   1.7% low and rough at 1.4e-2, against the 3e-4 bar (Decision 2). It would
   also hold a 0.8 GiB table.
2. **Hermite at 30 points per unit x.** Its numbers are those of Hermite at 3
   to <= 6e-6 in every gradient, at ten times the memory and a slower compile.
3. **Exact j_l on the fly** (`compute_cls_all_interp`). Its harmonic stage takes
   about 4100 s per call on a V100, against 4 s for the table (precision sweep,
   Oct 2026). Out of scope.
4. **A cosmology-dependent table range under `stop_gradient`.** Its shape
   would still depend on tau_0, so it could not be jitted (array shapes must be
   static), and it would be rebuilt on every call. The fixed range costs 41%
   more points than main's range at fiducial (63000 against 44583 at 3 points
   per unit x) and is built once.

## Consequences

**Values.**
- The comparison with CLASS covers lcdm_fiducial, omega_b_high, h_high, h_low
  and massive_nu_015, at `fit_cl` and `planck_cl`, for unlensed TT/EE/TE,
  C_l^phiphi and lensed TT/EE/TE/BB. It passes in every band (jobs
  30237-30241; base 30137-30139, 30149, 30150).
  - `planck_cl`: every band passes on the first branch. The tightest is
    omega_b_high TT 30-100 at l = 47: +9.93e-5 against 1e-4.
  - `planck_cl` EE at l = 2-4 against CLASS improves from 0.596% to 0.151%:
    the removal of the linear table's l = 2 bias. Every other band of
    `tests/test_preset_accuracy.py::TestPlanckClPreset` moves by <= 0.04
    percentage points (7/7 on both trees, jobs 30151/30152).
  - `fit_cl`: the tightest band is omega_b_high TT 30-100 at l = 46, +9.97e-5
    (margin 3.2e-7 on the first branch, 1.2e-6 on the second, which is the
    better one). massive_nu_015 TT and lensed TT 30-100 pass on the
    second branch (+1.01e-4 in error, 9.62e-5 shift).
  - The margins in TT 30-100 are about 1e-6: the tightest band of each case
    keeps 0.75-2.3e-6 to the 1e-4 slack on its better branch (3.8e-6 at
    massive_nu_015 `fit_cl`). They are small by construction, because the slack
    equals the removed bias. A rebase onto #58, #60 or #61, a JAX/XLA upgrade or a
    CPU run re-measures them.
- **Two documented exceptions: `fit_cl` TE and lensed TE at l = 2-30.**
  - They grow by +2.50e-3 (massive_nu_015), +3.65e-3 (h_high), +4.01e-3
    (fiducial), +4.43e-3 (h_low) and +4.48e-3 (omega_b_high), with the maximum
    at l = 2. The allowance is 5e-3.
  - The Hermite lookup moves TE at l = 2 by -4.0e-3 of sqrt(TT EE) against the
    linear table. Lensing passes that shift through unchanged to 1.4e-7.
- **`fit_cl` EE at l = 2 moves 0.74-0.88% further from CLASS at all five
  cosmologies** (base -1.2 ... -4.5%, now -2.1 ... -5.3%). It is the same l = 2
  bias removal that improves `planck_cl`, where the base sat above CLASS; at
  `fit_cl` the base already sat below it. EE 2-30 passes only because that
  band is judged on its maximum, which sits at l = 10-15. In the same way
  `planck_cl` EE and TE at l = 3 grow by 1.0-1.6e-4 inside bands whose maximum
  is at l = 13-23. l < 30 is outside every gradient functional, and cosmic
  variance at l = 2 is 63%.
- **The k-grid change moves `fit_cl` on its own**, measured against the base at
  5000 fine k points (job 27208, fiducial):
  - TT at l = 100-1200 moves by up to 1.6% toward CLASS;
  - EE and TE at l = 500-1200 move by +2.2e-3 and +6.5e-3 away from CLASS. The
    coarser grid had been cancelling another error near l ~ 1140-1200.
  - This is recorded, not gated. Over five cosmologies, TT 100-500 improves by
    2.1-2.7e-3 at four of them; the 500-1500 band maxima, at l ~ 1400-1500 next
    to `fit_cl`'s l_max, move by 0.06-1.7% with either sign.
- `fit_cl`'s accuracy test still passes with its margins: TT 2-30 0.455%, TT
  2-500 0.455%, EE 5-500 1.189% (bars 1.5 / 2 / 2%; before 0.461 / 0.461 /
  1.194; job 27651).
- **Reference files.**
  - The C_l^phiphi comparison uses new CLASS references
    (`reference_data/<name>/cl_pp.npz` with provenance JSON) at the five
    cosmologies.
  - The unlensed and lensed references of lcdm_fiducial, omega_b_high and
    h_high predate that CLASS build. A fresh run differs from them by up to
    1.5e-3 (TT), 2.1e-3 (EE) and 2.5e-3 (lensed TT) at fiducial, and by up to
    1.5e-3 in EE at the other two (job 26859).
  - The gate compares fix and base against the same files, so that offset
    cancels to first order. The direct fix-versus-base shift printed per band
    is the primary number.

**Gradients** (F_TT at fiducial, omega_b + 20% and Y_He = 0.28; `fit_cl`
tolerance unless stated).
- **Step placement.** Forward-mode dF_TT/dtheta does not depend on where the
  solver places its steps: a 0.1% jitter of the perturbation rtol times jit or
  eager gives spreads <= 6.5e-6 (h), <= 1.65e-4 (omega_b) and <= 1.3e-5
  (omega_cdm), against a bar of 3e-4 (jobs 28454, 28487). The omega_b value is
  fiducial and reflects how that derivative depends on the `fit_cl` tolerance
  itself (690.198 at rtol x 1.0, 690.31 at x 1.001).
- **Smoothness.** dF_TT/dh is smooth along h = 0.64-0.70: cubic residual
  <= 7.4e-6, bar 3e-4 (jobs 27739-27741).
- **Against finite differences.** At 100x tighter perturbation tolerance
  (Kvaerno5, 9-point stencil):
  - |AD/FD - 1| <= 3.3e-4 (h), 6.2e-4 (omega_b) and 3.5e-5 (omega_cdm), bar
    1e-3; stencil spread <= 1.4e-4, bar 3e-4.
  - The Rodas5 companion passes 9/9.
  - AD at `fit_cl` tolerance sits within 2.8e-4 (omega_b), 7.9e-6 (h) and
    2.1e-5 (omega_cdm) of AD at the tight tolerance (jobs 28321-28323).
- **The harmonic stage on its own** (ODE solve held fixed; theta = n_s and
  tau_0): jit(grad) = grad to 3.7e-14 and vmap = loop to 1.9e-16, bar 1e-10
  (job 27782).
- **A small floor on Kvaerno5, origin not established.** On the matter-power
  functional sum_k ln delta_m^2, Kvaerno5 reverse and forward mode differ by
  1.6-7.2e-6 at `fit_cl` tolerance and by 1.5-2.7e-6 at 100x tighter
  perturbation tolerance (jobs 27725/27726/27786); jit(grad) and grad of F_TT
  differ by 2.4e-9 to 3.7e-8 (jobs 27739-27741). With the background and
  thermodynamics held fixed, the perturbation solve alone is an exact
  transpose (6e-15, job 27848), so the floor arises upstream of it. It is 100x
  under every gradient bar, and it is why the 1e-10 jit and vmap checks hold
  the ODE solve fixed.
- **Reverse mode.** NaN on this branch until the Rodas5 PR. With that fix
  (scratch tree, jobs 27924-27926), reverse mode under the direct adjoint
  equals forward mode to <= 1.2e-5 (9/9).
- **Lensing**, nonlinear none / Halofit / EPT, theta = h and omega_b:
  - Forward-mode dF_lTT matches finite differences to <= 8.7e-4 (bar 1e-3) at
    all three cosmologies.
  - dF_pp/d omega_b matches to <= 2.0e-4 at omega_b + 20% and Y_He = 0.28.
  - The production reverse dF_pp equals forward to <= 5.9e-5 on the Rodas5-fix
    scratch tree.
- **Not yet: dF_pp/dh, and dF_pp/d omega_b at fiducial.**
  - `th.z_rec`, where the lensing kernel starts, is a knot of the
    thermodynamics grid (`clax/thermodynamics.py:1004`). AD follows the knot's
    smooth motion, while finite differences also see its one-index jumps.
    dF_pp/dh therefore misses finite differences by 0.55-2.9%; with z_rec held
    fixed it matches to 4.9e-5 (job 28583).
  - PR #61 (merged in main 6ff1303, Oct 24, 2026) makes z_rec continuous. On a
    #61 overlay all 24 affected cases
    pass: <= 2.7e-4 against finite differences on Kvaerno5, <= 4.8e-4 in the
    Rodas5 companion (jobs 28623-28625, 28772-28777).
  - On this base the 18 affected cases are marked, per cosmology, as expected
    to fail, strictly: once a rebase onto #61 makes them pass, pytest reports
    an error, the signal to remove the markers. Until then an HMC user of C_l^phiphi gets dF_pp/dh with the knot's
    drift term, about 0.9% at omega_b + 20% (-934.12 against -942.19 with z_rec
    frozen, job 28583).
- **The production reverse mode is wrong at the 1e-4-1e-2 level**, on top of
  the NaN (which hid it until now).
  - `ode_adjoint="recursive_checkpoint"`, the default, uses diffrax's
    RecursiveCheckpointAdjoint. Through Rodas5, with the NaN fix, its dF_TT
    differs from forward mode by:
    - h: 3.1e-4 / 3.1e-4 / 2.0e-3;
    - omega_b: 1.2e-3 / 1.6e-3 / 7.2e-3;
    - omega_cdm: 2.1e-4 / 1.9e-4 / 1.5e-3;
    at fiducial / omega_b + 20% / Y_He = 0.28 (jobs 27742-27744). Lensed TT
    inherits it: 3.4e-4 to 6.4e-3. On the matter-power functional
    sum_k ln delta_m^2 the same adjoint misses forward mode by 2.2e-4 /
    1.3e-2 / 2.4e-4 at the same three points (job 27788), the worst case
    measured: the defect is not specific to C_l.
  - The defect needs Rodas5 (Kvaerno5's two adjoints are bit-identical) and
    vmapped k-batches (with one mode per batch the gap is 3.1e-6; job 28012).
    It is not the thermodynamics gradient mode (job 27802), and not the moving
    end time or save grid (jobs 27915, 27916). Its mechanism is not
    established.
  - The direct adjoint is correct, at 76.7 s and 0.86 GiB per full-chain d/dh
    of the matter-power functional, against 29.9 s and 0.41 GiB (job 27916).

**Cost** (fiducial, eager unless stated; job 30153):
- Eager forward: 0.703x the pre-#54 code at equal k-grid, the gated budget
  being 1.05x; 0.721x against the pre-#54 `fit_cl` at its own 5000 fine k
  points. The old code rebuilt its 30-point table on every call (harmonic stage
  6.3 s, now 1.8 s).
- Jitted forward 6.70 s. Gradient 31.8 s (Decision 6).
- Peak device memory 6.20 GiB with both programs alive, so it fits in 32 GB.

**Interface.**
- To jit or differentiate, callers pass `prec=` to `compute_cls_all_fast` and
  `compute_cl_pp`; `compute_cls_all_fast` then takes its fine k-grid from
  `prec.hr_n_k_fine`.
- A cosmology with tau_0 > `hr_tau0_max` (20000 Mpc) raises instead of reading
  past the table. Raise the bound to serve it.
- The EPT C_l^phiphi correction is built differently (Decision 5). It agrees
  with the old construction to <= 3.5e-6 per band and to <= 6.9e-4 in R(k).

## Not done

- **The gradient's cost** (4.8x a forward evaluation, Decision 6): the
  checkpointed adjoint through Rodas5. A follow-up issue is drafted.
- **The production adjoint's error through batched Rodas5** (Consequences).
  Its strict expected failures record it. Issue to be filed.
- **Rodas5's step-sequence reproducibility** (Decision 7). At `fit_cl`
  tolerance it puts a ~1e-8 wobble on the primal, which floors finite
  differences along omega_b near 1e-3, and it limits agreement between two
  compilations of one derivative to ~1e-5. Candidates: acceptance hysteresis,
  error-norm rounding. A follow-up issue is drafted.
- **The Rodas5 reverse-NaN fix** is its own PR. When it merges, the strict
  expected failures flip to hard failures and their markers must be removed:
  - `tests/test_cl_gradients.py`: all 9 direct-adjoint reverse-vs-forward cases; the
    production-adjoint cases at lcdm_fiducial and omega_b_high for omega_cdm
    (2.1e-4 and 1.9e-4, under the 3e-4 bar); the jit-of-grad Rodas5 companion
    at omega_b_high (1.4e-4). The other production-adjoint cases stay marked,
    per case.
  - `tests/test_cl_lensing_gradients.py`: the 18 C_l^phiphi reverse-mode
    cases. The 18 lensed-TT cases stay marked.
- **The rebase onto main 6ff1303.** #58, #60 and #61 are merged there (Oct 25,
  2026), and every number in this record was measured at 5ac3df3, before them.
  The rebase has to:
  - re-measure every `fit_cl` number: #58's neutrino hierarchy takes the
    perturbation state from 61 to 106 variables, which moves the forward time,
    the 4.8x gradient cost, the 6.2 GiB peak and the G1-G5 values;
  - regenerate the C_l-versus-CLASS baseline with the generator (to be
    committed, next bullet) and re-derive the TE allowances;
  - remove the 18 z_rec expected-fail markers: #61 makes z_rec, tau_star and
    the other epochs continuous, so they pass (pytest reports an error until
    the markers go).
- **The massless-neutrino treatment of `fit_cl`** (#57), which #58 changes; every
  `fit_cl` number above predates #58.
- **Memory and time of the lensed gradients** (Decision 6 asks for them). The
  overlay reverse jobs of C_l^phiphi and lensed TT (28549-28553, 28769-28771)
  ran to completion on V100-32GB; no per-case time or peak memory was recorded.
- **The unlensed and lensed CLASS references** of lcdm_fiducial, omega_b_high
  and h_high are not regenerated on the current CLASS build.
- **The generator of the C_l-versus-CLASS baseline**
  (`tests/data/cl_vs_class_baseline.{json,npz}`) is not in the repository. It
  is a scratch script, recorded in the JSON's `_provenance` by path and sha1,
  with the call recipe in `_provenance.calls`. Regenerating the baseline after
  #58 or #60 (Status) needs it committed; follow-up.
- Out of scope, as specified:
  - the 800-point background-table AD noise (#56);
  - `planck_cl`'s absolute accuracy and `fit_cl`'s l < 30 values;
  - differentiability of the slow paths `compute_cls_all` and
    `compute_cls_all_interp`, and of tensor B-modes;
  - the EPT ratio of ~7-9 at k = 0.70-0.89 Mpc^-1, beyond the one-loop EFT's
    range. The pre-#54 code applies it too;
  - the accuracy and timing figures in CLAUDE.md, which are stale.
- Pre-existing test failures, identical on the base:
  - `tests/test_solver_selection.py::TestRosenbrockPk::test_pk_rosenbrock_vs_kvaerno5`
    (max_steps);
  - `tests/test_chebyshev_interpolation.py::test_cls_chebyshev_path_matches_spline_path`
    (TT ratio 1.0170 at l = 500 against a bar of 0.005; jobs 30166/30167).

## Status

Accepted on branch `fix/cl-differentiability`, on main 5ac3df3. Main is now at
6ff1303 with #58, #60 and #61 merged (Oct 25, 2026); every number here was
measured at 5ac3df3, before them (Not done: the rebase). It closes #54.

Depends on:
- **ADR 0003** (#53, recombination staging). The h-gradient smoothness this ADR
  measures depends on it. Before it, a derivative along h through the
  thermodynamics jumped by up to 6% along a smooth line in h.
- **ADR 0004** (#59, opacity-table tangents). The omega_b and omega_cdm
  gradients gated here depend on it. Without it (on 99b62fe), forward-mode AD
  of F_TT along omega_cdm was 149% off finite differences and along omega_b had
  the wrong sign (jobs 27212-27214).

Interacts with:
- the Rodas5 reverse-NaN PR, which removes the NaN expected failures;
- #61 (merged in main), which removes the z_rec ones;
- #58 and #60 (merged in main), which move `fit_cl`'s spectra and, with #61, require
  `tests/data/cl_vs_class_baseline.{json,npz}` (whose generator is not yet
  committed, Not done) to be regenerated before the
  C_l-versus-CLASS gate means anything.
