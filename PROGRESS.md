# jaxCLASS Development Progress

## Status: Science-grade C_l at acoustic peaks — TT/EE/TE <0.1% at l=150-300

**End-to-end differentiable pipeline from cosmological parameters to P(k),
C_l^TT/EE/TE/BB, and lensed C_l. AD gradients verified to 0.03%.**

### External review (Feb 11, 2026)

**Assessment: decent and close, but not done.** Meaningful validation and
physics-consistency work remains before production-grade for Planck-like TT.

- **EE**: very good (near/sub-percent over wide l range)
- **TT**: good at acoustic peaks/mid-l, residual issues at low-l and high-l
- **Main risk**: model consistency (RSA/hybrid switching logic), not basic numerics
- **Tooling/diagnostics**: mature, can iterate quickly

**Remaining for science-robust:**
1. Lock down RSA strategy (consistency + differentiability + validation)
2. Tight regression grid across presets/cosmologies (not just fiducial LCDM)
3. Resolve remaining TT systematics (ncdm dynamics / approximation boundary)
4. Clean up and harden API paths (mode handling / interp path edge cases)

### High-l TT accuracy — critical for HMC/Planck

**This is a blocker for HMC with Planck-like data** (most constraining power
at l=30-1500). Current high-l status:

| l | TT error | EE error |
|---|----------|----------|
| 500 | -0.57% | -0.25% |
| 700 | -1.58% | -0.96% |
| 1000 | -7.23% | -0.89% |
| 2000 | ~17% | ~1% |

EE is fine across all l. TT degrades above l~700 due to **hierarchy truncation
at l_max=50** — photon moments above l=50 are zeroed, corrupting metric
potentials at high k through Einstein equations. Agent confirmed l_max=80
OOMs on V100-32GB.

**Options (ordered by likely effectiveness):**
1. **H100-80GB** — 2.5x more memory, could fit l_max=80-100. Costs 2 SU/hr.
2. **Hard RSA switch** — replace hierarchy with algebraic expressions
   post-recombination (what CLASS does). Eliminates truncation entirely.
   But hard to make differentiable (jax.lax.cond with same-shape branches).
3. **Gradient checkpointing** — trade compute for memory, allow higher l_max
   on V100. May add 2-3x compute overhead.
4. **Accept and mask** — exclude l>700 from likelihood. Loses information but
   works for a proof-of-concept HMC.

### Next steps (prioritized for HMC readiness)

1. **High-l TT fix** (critical) — Try H100 with l_max=80, or implement hard
   RSA switch. Required for Planck-like posteriors.
2. **RSA validation** (high value) — Four-way A/B test + gradient smoothness.
   Must pass before trusting HMC gradients.
3. **Multi-cosmology regression** (high value, easy) — Run at 5-10 param
   points to catch bugs that cancel at fiducial. No code changes, just GPU time.
4. **Full ncdm hierarchy** (high value, substantial) — Fix remaining TT ~1%
   from massless ncdm approximation. Implement Ψ_l(q) variables.
5. **API cleanup** (medium value) — Consolidate code paths, remove dead scripts,
   single `compute_cls()` entry point.
6. **HyRec upgrade** (low-medium, substantial) — Fix EE -0.15% systematic.
   Only needed for sub-0.1% EE.

### Autonomous agent work (Feb 10-11, 2026 — Bridges-2 GPU loop)

Agent running via `scripts/gpu_claude_loop.sh` (Carlini-style while-true loop).
7 sessions completed so far.

#### Changes made by agent (session 1, Feb 10-11):
1. **Analytic g' (visibility derivative)**: Replaced spline derivative of g(τ) with
   pre-computed analytic `g' = (κ'' + κ'²) e^{-κ}` matching CLASS thermodynamics.c:3482.
   Added `g_prime_of_loga` spline to ThermoResult.
2. **RSA in source functions**: Implemented CLASS-style RSA substitution in source
   extraction (perturbations.c:7553-7567). After recombination (k*τ > 45, κ'/aH < 5):
   - `delta_g` → analytic RSA expression from metric
   - `Pi` → 0 (photon anisotropic stress vanishes)
   Applied to SW, T2 quadrupole, and E-polarization sources via `jnp.where`.
3. **Fixed a''/a formula**: Was missing factor of 2 (CLASS perturbations.c:10032).
4. **Proper dtau_c/tau_c**: Now computed from dκ̇/dloga spline derivative instead
   of the approximation `2aH`.
5. **Second-order compromise_CLASS TCA corrections**: Implemented the full
   compromise_CLASS TCA scheme (perturbations.c:10303-10316) with second-order
   slip and shear corrections. Previously only had first-order.

#### Session 5:
- Further perturbations.py edits (90 lines changed), running GPU diagnostics.

#### Session 6 (Feb 11, 2026):
- **RSA in source Einstein equations (Bug 22)**: Source extraction computed h', η', α, α'
  from RAW hierarchy values, while the ODE RHS used RSA-corrected values. After
  recombination, truncated hierarchy values contaminated the metric potentials in the
  source functions. Fixed by applying the same RSA substitution as the ODE RHS.
  Impact: TT l=500 improved from -1.45% to -0.57%, l=700 from -2.65% to -1.58%,
  l=1000 from -9.05% to -7.23%.
- **RSA shear in ODE alpha_prime**: Also zeroed photon/neutrino shear (F_g_2, F_ur_2)
  in the alpha_prime computation when RSA is active, matching CLASS perturbations.c:8259.
- Confirmed T1/T2 radial functions are correct for flat space (CLASS sets
  sqrt_absK_over_k=1.0, absK_over_k2=1.0 for K=0, NOT the physical curvature).
- l_max=80 OOMs on V100-32GB. Running l_max=65 test to check hierarchy convergence.

#### Session 7 (Feb 11, 2026):
- **Critical: k-integration under-resolution (Bug 24)**: C_l k-integral with n_k_fine=3000
  was severely under-resolved at high l. The Bessel oscillation period π/χ_star ≈ 2.3e-4 Mpc^{-1}
  is constant in k, but the log-uniform fine grid spacing grows with k. At k=0.1 (l~1400),
  only ~1 point per oscillation (below Nyquist!). Increasing n_k_fine from 3000→5000→10000
  dramatically improved accuracy:
  - l=700: -1.58% → +0.41% → -0.24%
  - l=1000: -7.23% → -0.96% → -0.57%
  Default n_k_fine increased from 3000 to 5000.
- **Chunked vmap for memory-efficient k-integration**: Added `_chunked_vmap` helper to
  process k-modes in batches of 2000, enabling n_k_fine=10000+ without GPU OOM.
- **RSA theta_g reionization correction (Bug 25)**: Added the CLASS rsa_MD_with_reio
  correction for theta_g: θ_g^RSA += (3/k²)(κ̈(θ_b+h'/2) + κ̇(-ℋθ_b+cs²k²δ_b-ℋh'+k²η)).
  Applied in both ODE RHS and source extraction. cf. CLASS perturbations.c:10427-10435.
  Impact: minimal at current precision level.
- **Hierarchy truncation fix (Bug 26)**: Closure relation used tau0-tau (comoving distance)
  instead of tau (conformal time) for the cotKgen = 1/(k*tau) formula.
  cf. CLASS perturbations.c:8882-8893. Fixed in scalar+tensor hierarchies.
  Impact: minimal at l_max=50 (hierarchy well-resolved).
- **Confirmed ODE precision is converged**: rtol=1e-8 gives identical C_l to rtol=1e-6.
- **Confirmed tau-grid is converged**: n_tau=10000 gives identical C_l to n_tau=5000.
- **Identified TT l=30-50 error (+1.5%) as massive neutrino effect**: Treating ncdm as
  massless over-estimates radiation fraction at z<100, boosting early ISW at l=30-50.

#### Issues encountered:
- API 529 overload errors overnight caused sessions 2-4 to crash immediately.
- BashTool pre-flight check warnings (benign, resolved with CI=true).
- Agent not updating PROGRESS.md (fixed in prompt).

With `planck_cl` preset (k_max=1.0, 300 modes) + source interpolation + ncdm (ρ+p) correction:
- **C_l^TT/EE/TE ALL <0.1% at l=150-300** (acoustic peaks, science-grade)
- **C_l^TT sub-0.6% from l=100 to l=1000** (0.006-0.57%)
- **C_l^EE sub-0.3% from l=100 to l=1000** (0.005-0.26%)
- **C_l^TE sub-0.2% from l=100 to l=700** (0.01-0.19%)
- TT +0.8% at l=30-50 from ncdm perturbation dynamics (needs full Ψ_l(q))
- TT/EE -0.14 to -0.23% at l=500-700 from RECFAST x_e accuracy (needs HyRec)

Bessel functions accurate to machine precision at l=2500.
RSA damping in ODE for post-recombination hierarchy.
100 tests passing, ~10K lines of code.

---

## Science-grade accuracy (Planck 2018 LCDM, V100 GPU)

planck_cl preset: k_max=1.0, 60 k/decade (300 modes), l_max=50, 5000 tau,
source-interpolated to 10000 fine k-points (chunked vmap):

| l | C_l^TT error | C_l^EE error | C_l^TE error |
|---|-------------|-------------|-------------|
| 20 | **-0.61%** | **-0.29%** | -5.8% (near zero) |
| 30 | **+0.76%** | **-0.22%** | -5.0% (near zero) |
| 50 | **+0.91%** | **-0.15%** | -15% (zero crossing) |
| 100 | **+0.23%** | ***-0.07%*** | **-0.19%** |
| 150 | ***+0.006%*** | ***-0.08%*** | ***+0.04%*** |
| 200 | ***-0.05%*** | ***-0.10%*** | **+0.17%** |
| 300 | ***-0.06%*** | ***-0.005%*** | ***-0.04%*** |
| 500 | **-0.14%** | **-0.15%** | ***-0.01%*** |
| 700 | **-0.23%** | **-0.11%** | ***+0.08%*** |
| 1000 | **-0.57%** | **-0.26%** | +1.7% |

Note: TE has zero crossing near l≈52; relative errors near crossings are misleading.
*** = sub-0.1%. TT l=150-300 now at <0.1%! EE l=100-700 now at <0.15%.
Remaining TT l=30-50 error (+0.8%) from ncdm perturbation dynamics (fluid approx needed).

Source interpolation convergence verified: k/dec = 60, 120, 200 agree to 0.01%.
k-integration convergence verified: n_k_fine = 5000, 10000, 20000 tested.
Bessel functions accurate to machine precision at l=2500.

### Pipeline accuracy

| Stage | Accuracy | Notes |
|-------|----------|-------|
| Background (H, D_A, r_s) | < 0.01% | 6+ significant digits |
| Thermodynamics (x_e at z_star) | 0.25% | RECFAST + Heun stepping |
| Visibility g(tau_star) | **0.04%** | Bisection z_reio, corrected kappa |
| Perturbation ODE (Phi, Psi at tau_star) | 0.01-0.25% | Gauge-corrected |
| P(k) | 1-4% | All k from 0.001 to 0.3 Mpc^-1 |
| AD gradients dP(k)/d(params) | 0.03% | vs finite differences |

---

## Changelog

### Feb 10, 2026: High-l Bessel fix + RSA damping + Planck preset

**Bessel function rewrite.** Replaced soft sigmoid blending with hard switch
at x=l between backward (x<l) and upward (x>=l) recurrences. Both now use
jax.lax.fori_loop for O(1) compilation. Verified accurate to machine precision
at l=2500 against scipy.

**RSA hierarchy damping in ODE.** After recombination (tau*k>45, kappa'/aH<5),
photon and neutrino hierarchy moments are damped toward RSA algebraic targets:
  delta_g_rsa = 4/k² * (aH*h' - k²*eta)
  F_1_rsa = -2h'/(3k)
  F_l = 0 for l >= 2
Damping rate = rsa_crit * k (relaxation on timescale ~1/k).
Note: this had minimal impact on C_l accuracy — the dominant TT high-l error
is likely from source function normalization, not hierarchy contamination.

**planck_cl preset.** k_max=1.0, 60 k/decade, l_max=50, 5000 tau points.
With source interpolation, covers l=2-2500.

**compute_cls_all_interp.** Full-spectrum API using source-interpolated
TT+EE+TE at sparse l-values + spline to l=2..l_max.

**compute_cl_te_interp.** Source-interpolated TE cross-spectrum.

### Feb 9, 2026: Source Interpolation (sub-percent C_l)

**Discovery: CubicSpline k-integration causes aliasing.** T_l(k) oscillates
with period pi/chi_star ~ 2.3e-4 Mpc^-1, faster than any practical k-grid.
CubicSpline interpolation of T_l introduces ringing artifacts. Raw trapezoidal
gives better results but is sensitive to k-density: non-monotonic convergence
(k=200/dec: +1%, k=120: +5%, k=60: -11%, k=30: +29%).

**Fix: Source function interpolation.** Source functions S(k,tau) vary slowly
in k (BAO scale ~0.02 Mpc^-1) and are well-sampled even at 60 k/decade. We
interpolate S(k,tau) via CubicSpline to a fine k-grid (3000 points), then
compute T_l(k_fine) = int S_fine * j_l(k_fine * chi) dtau exactly. The rapid
Bessel oscillation is handled analytically. Results converge across k-densities.

**T0+T1+T2 mode (CLASS full form).** Previously only T0 (IBP monopole) was
used. Adding T1 (ISW dipole) + T2 (quadrupole) improves TT by 15-27pp at
l=15-50. T0+T1+T2 matches CLASS harmonic.c:962.

### Feb 8-9, 2026: RECFAST + Reionization Fix (3 bugs, 70x g improvement)

**Bug 19: RECFAST fudge factor misplacement.** F=1.14 was in alpha_B. CLASS
puts F=1.125 (with Hswitch delta) inside the Peebles C coefficient.

**Bug 20: Missing Gaussian K correction.** RECFAST 1.5 Hswitch corrections
from Rubino-Martin et al. (2010) were absent.

**Bug 21: Reionization tau_reio mismatch (DOMINANT ERROR).** Crude z_reio
estimate `2 + 150*tau_reio` gave tau = 0.077 instead of 0.054. Fixed with
bisection to match tau_reio exactly.

**Heun stepping.** Upgraded RECFAST ODE from Euler to Heun (predictor-corrector).
Reduced x_e error from 0.7% to 0.25% at z_star.

Result: g(tau_star) from -2.6% to **-0.04%**.

---

## Bugs found and fixed (23 total)

1. ncdm deg=1 -> g*=2 (factor of 2 in density)
2. age: divide by Gyr_over_Mpc, not multiply
3. a_ini: 1e-14 -> 1e-7 (ODE step count + high-k ICs)
4. adot = a^2*H (conformal time derivative)
5. Diffrax args: plain tuples, not custom classes
6. float() breaks JAX tracing -> use jnp values
7. h' must be CONSTRAINT (CLASS perturbations.c:6612)
8. Monopole: -2/3*h', not -1/3*h'
9. C_l formula: int dlnk P_R Delta_l^2
10. Photon dipole: -kappa'*(F_1 - 4*theta_b/(3k)) (scattering damps)
11. Bessel clip to [-1,1] during upward recurrence
12. ncdm in Einstein constraints (delta_rho and (rho+p)*theta)
13. a_ini=1e-7 for high-k perturbation ICs
14. METRIC SHEAR in l=2: 8/15*(h'+6eta')/2 source -> P(k) from 60% to 4%
15. source_E normalization: was g*Pi/(4k^2), correct is 3*g*Pi/16
16. theta_b' extraction mismatch: source used full eq, ODE used TCA
17. TCA single criterion -> dual criteria (CLASS perturbations.c:6178-6179)
18. Global TT mode leakage: module-level global contaminated TE
19. RECFAST fudge misplacement: F in alpha_B vs F in Peebles C
20. Missing Gaussian K correction (RECFAST 1.5 Hswitch)
21. Reionization tau_reio: crude z_reio gave tau=0.077 instead of 0.054
22. RSA missing in source Einstein equations: h',η',α,α' computed from raw hierarchy
23. RSA shear missing in ODE alpha_prime: F_g_2, F_ur_2 not zeroed when RSA active
24. k-integration under-resolution: n_k_fine=3000 gave 1-7% errors at l>500 due to
    under-resolving Bessel oscillation |T_l(k)|² (period π/(2χ_star) ≈ 1.15e-4 Mpc^{-1})
25. RSA theta_g missing reionization correction (rsa_MD_with_reio, CLASS 10427-10435)
26. Hierarchy truncation used tau0-tau instead of tau for closure (CLASS 8882-8893)

## Known limitations and remaining work

**Accuracy bottlenecks (ordered by impact):**

1. **TT l=30-50 at ~1.5%**: The T1 (ISW dipole, j_l' radial) contribution
   is ~7% too small at l=30. T2 (polarization quadrupole) is negligible
   at these scales (only +0.15pp). The massive neutrino effect at l=30 is
   only 0.01% (confirmed via CLASS massive/massless comparison), so this
   is a CODE error, not a physics approximation issue.
   **ROOT CAUSE FOUND**: Newtonian potential Phi is 0.5% too high at
   recombination due to massless ncdm approximation. The ncdm density
   perturbation overshoots without proper free-streaming (k>k_fs=0.004
   Mpc^{-1}), inflating delta_rho → h' → Phi. This Phi offset varies with
   tau (0.4% at tau=200 to 0.6% at tau=350 to -0.3% at tau=10000),
   affecting Phi' and the T1 ISW integral over the full conformal time
   range. z_star matches CLASS perfectly (1088.78 vs 1088.78) — the
   apparent 0.33% "error" was a naming confusion (our z_star = CLASS z_rec).
   **Fix: implement full ncdm perturbation hierarchy Ψ_l(q).**

2. **EE systematic bias ~-0.15%**: Present across l=20-700, from RECFAST
   physics accuracy (x_e error ~0.25% at z_star). Confirmed insensitive
   to thermo grid resolution (n_points=5000 vs 20000 identical). Fix:
   improve RECFAST (use HyRec/CosmoRec) or accept ~0.15% systematic.
   **Effort: substantial.**

3. **TT/EE l>1000**: Residual 0.6-1.6% error partly from k-integration
   convergence (still improving with n_k_fine), partly from ncdm mass
   effect (~0.3% at l=1000) and Silk damping accuracy.
   **Effort: easy for k-resolution, moderate for physics.**

4. **SW plateau (l<15)**: ~5% error from gauge-dependent source at
   super-horizon scales. Low priority for most applications.
   **Effort: moderate.**

5. **Single cosmology validated**: Only Planck 2018 fiducial tested.
   **Effort: trivial (GPU time only).**

6. **RSA hybrid design needs validation**: Einstein equations and source
   extraction use a hard `jnp.where(is_rsa, ...)` switch (lines 503-506,
   871-874), while hierarchy evolution uses smooth sigmoid damping (line 680).
   This hybrid is intentional (hard RSA switch isn't differentiable for the
   hierarchy ODE), but could cause gradient artifacts at the RSA boundary.
   **Recommended checks**: (a) A/B test: Einstein RSA only / damping only /
   both / neither; (b) AD vs finite-diff gradient smoothness across RSA
   threshold; (c) sweep RSA thresholds (45,5) for TT/TE stability l=30-1000.
   **Effort: moderate (GPU diagnostic runs).**

**Next steps (ordered by effort/impact):**

- [x] TE spectrum with source interpolation (done: compute_cl_te_interp)
- [x] RSA hierarchy damping in ODE (done: implemented, tested, minimal impact)
- [x] Term-by-term T1/T2 radial function check vs CLASS transfer.c — CONFIRMED
      CORRECT for flat space (verified: source_T1 formula, radial_T2 formula,
      normalization factor 1/8 all match CLASS exactly)
- [x] k-integration resolution fix — n_k_fine=3000→5000 default, chunked vmap
      for n_k_fine=10000+, TT l=700 from -1.6% to -0.24%
- [ ] **Diagnose T1 ISW dipole deficit at l=30** — #1 PRIORITY for TT accuracy.
      T1 contribution is 7% too small. Compare T_l(k) against CLASS transfer
      function. Check if z_star offset (0.33%) explains the discrepancy.
      (hours, targeted debugging)
- [ ] Full ncdm perturbation hierarchy Ψ_l(q) — needed for <0.1% at l>100
      (ncdm mass effect ~0.2-0.3% at l=100-1000). (~1 session)
- [ ] Improve RECFAST → HyRec/CosmoRec — for EE systematic bias ~0.15%
- [ ] Multi-cosmology validation at 5+ parameter points (GPU time only)
- [ ] Gradient tests for C_l: d(C_l)/d(params)
- [ ] Hybrid linear/log fine k-grid for better convergence at l>1500

## Confirmed correct (do not re-investigate)

- **T1/T2 radial functions for flat space**: CLASS sets sqrt_absK_over_k=1.0 and
  absK_over_k2=1.0 for flat space (transfer.c:4056-4064, with comment "consistent
  with chi=k*(tau0-tau) and nu=1"). So T1 radial = j_l', T2 radial = 0.5*(3j_l''+j_l)
  are CORRECT for flat space. Attempted changing to 0-radial/0.5*j_l — made TT 22% worse.
- **ODE precision converged**: rtol=1e-8 gives identical C_l to rtol=1e-6.
- **Tau-grid converged**: n_tau=10000 gives identical C_l to n_tau=5000.
- **k-integration converged**: n_k_fine=10000 and 20000 agree to <0.01pp at l=300-700.
  The remaining errors at l>100 are from physics (ncdm), not numerical resolution.
- **T0+T1+T2 source functions match CLASS**: Verified source_T0, source_T1, source_T2,
  and source_E definitions line-by-line against CLASS perturbations.c:7660-7690.
- **E-mode source normalization**: source_E = 3*g*Pi/16 is correct. CLASS has
  sqrt(6)*g*Pi/8 as source and sqrt(3/8*(l+2)(l+1)*l*(l-1)) as radial factor;
  combined with our j_l/(kχ)² and prefactor, it matches.
- **a''/a and ℋ' formulas**: Both verified to match CLASS perturbations.c:10032
  and the ISW Φ' = η' - ℋ'α - ℋα' formulation.

## Failed approaches (do not re-attempt)

- **Upgrading alpha_B alone** without matching beta_B: MB95's alpha/beta/C are
  calibrated together. Must upgrade as complete RECFAST ODE.
- **Smooth-blending TCA and full equations**: Changes the physics. Use
  jnp.where for equation selection; sigmoid only for switching criterion.
- **CubicSpline interpolation of T_l(k)**: T_l oscillates faster than k-grid.
  CubicSpline introduces aliasing. Must interpolate SOURCE functions (smooth)
  instead, then compute T_l on the fine grid.
- **Intermediate k-density (30-120 k/decade)**: Non-monotonic convergence for
  raw trapezoidal C_l integration. Either use very dense (200+) or source
  interpolation.
- **RSA as smooth damping in ODE constraints**: Blending RSA values into the
  Einstein equations while keeping the full hierarchy running creates
  inconsistency (metric uses blended values, hierarchy uses raw values).
  Also: first attempt had wrong theta_g_rsa formula (extra factor of k).
  RSA damping in the hierarchy evolution (relaxation toward targets) is
  self-consistent but had <0.1pp impact — the TT high-l error is NOT from
  hierarchy ringing. The remaining error is elsewhere (likely T1/T2 normalization).

---

## Implementation phases (all complete)

- **Phase 1-3**: Background, thermodynamics, perturbations -- COMPLETE
- **Phase 4**: Transfer + C_l (TT/EE/TE/BB, Bessel, Limber) -- COMPLETE
- **Phase 5-6**: Gradients + API (compute(), shooting, sparse l) -- COMPLETE
- **Phase 7**: Diagnostics + bug fixes (21 bugs found and fixed) -- COMPLETE
- **Phase 8**: Sub-percent accuracy (RECFAST, source interp, T0+T1+T2) -- COMPLETE
