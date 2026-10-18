# C_ℓ differentiability at `fit_cl` — design (smsharma/clax#54)

- **Date:** Oct 18, 2026
- **Branch:** `fix/cl-differentiability`, worktree `/home/n2minh/clax-cl`, stacked on #53's head `2717b5c`.
- **Issue:** #54.
- **Related:** #48/#53 (smooth recombination; prerequisite), #55 (P(k) gradient biases; out of scope), clax-72's unposted draft on 800-point background-table AD noise (`/lustre/work/n2minh/clax-probes/recomb/issue-bg-resolution-ad-noise.md`; out of scope).

## Problem Statement

A cosmologist running NUTS, a Fisher forecast or a MAP fit on CMB spectra at the `fit_cl` preset needs ∂C_ℓ/∂θ for θ ∈ {h, ω_b, ω_cdm, N_eff, m_ν, …}. Today there are no such gradients.

**Unlensed spectra.** `compute_cls_all_fast` cannot be differentiated with respect to any parameter that moves the conformal age τ₀.
- It sizes its Bessel table with `float()`/`int()` of `max(k) · τ₀ · 1.05` (`clax/harmonic.py:836-837`), so `jax.jvp`, `jax.grad` and `jax.jit` raise `ConcretizationTypeError` before the first gradient.
- With that removed, reverse mode stores the per-ℓ residuals of its loop over 77 sparse ℓ values. That is 5.74 GiB per stored intermediate and about 46 GB in total, more than a 32 GB V100 holds.
- The table lookup is piecewise-linear in x = kχ. The derivative AD would return for j_l(kχ) is therefore a staircase (the slope between neighbouring table points) rather than j_l′(kχ). That threatens derivative precision and smoothness even once the code runs.

**Lensed spectra.** The same class of problem breaks them.
- The Halofit correction builds its redshift grid from `float(th.z_rec)` and branches in Python on `float(pt.k_grid[-1])` (`clax/lensing.py:183-194`).
- The EPT correction does `float(params.h)` and selects k-values with an h-dependent boolean mask, so its array shapes change with cosmology (`clax/lensing.py:312-347`).
- The linear lensing path (`compute_cl_pp`, `lens_cls`) has no gradient test with respect to τ₀-moving parameters: the existing test only varies A_s.

No existing test differentiates `compute_cls_all_fast`.

## Solution

1. **A fixed table.** The Bessel table covers a fixed x-range set by precision settings, so it is the same for every cosmology and is built once.
2. **A smooth lookup.** j_l and j_l′ are looked up with a cubic Hermite interpolant that uses the slopes the table already holds. AD then sees a smooth, accurate derivative.
3. **Bounded reverse-mode memory.** The backward pass recomputes each ℓ-step's lookups instead of storing them.
4. **Lensing.** The nonlinear corrections keep every cosmology-dependent quantity traced, with fixed array shapes.
5. **Tests.** The C_ℓ gradient checks parked from #48 land here: independence from solver step placement, production reverse mode against forward, smoothness along h, and AD against converged finite differences. Lensing gradients get the same checks, and every spectrum is compared with CLASS before and after.

## User Stories

1. As a cosmologist running NUTS on unlensed CMB C_ℓ at `fit_cl`, I want `jax.grad` of my log-likelihood with respect to h, ω_b and ω_cdm to run, so that I can sample at all.
2. As that cosmologist, I want the gradient to fit in a 32 GB V100, so that I can run on the GPUs I have.
3. As that cosmologist, I want one gradient to cost at most ~3× a likelihood evaluation, so that NUTS is affordable.
4. As that cosmologist, I want the gradient to agree with converged finite differences to 10⁻³, so that leapfrog follows the true force.
5. As that cosmologist, I want the gradient not to depend on where the ODE solver places its steps (k-batch size, jit or not), so that chains do not lose efficiency to random kicks.
6. As that cosmologist, I want the gradient to vary smoothly along h, so that the Hamiltonian is integrated accurately.
7. As that cosmologist, I want forward and production reverse mode to agree, so that whichever mode I use, I get the same physics.
8. As that cosmologist, I want these properties at several cosmologies, not only at fiducial, so that they hold across my prior.
9. As a user of lensed spectra, I want the same for C_ℓ^φφ and lensed TT/EE/TE/BB, with nonlinear "none", "halofit" and "ept", so that a Planck-like lensed likelihood is differentiable.
10. As a user, I want C_ℓ values to be no worse against CLASS than today in any ℓ band, so that the gradient fix costs nothing in accuracy.
11. As a user, I want plain C_ℓ evaluation to be at most ~5% slower, so that non-gradient work is not penalised.
12. As a user whose cosmology has τ₀ beyond the table's range, I want a clear error naming the bound, not a silent truncation, so that I know to raise `hr_tau0_max`.
13. As a Fisher-forecast user, I want `jax.jvp` (forward mode) to work as well as `jax.grad`, so that I can use whichever is cheaper for my parameter count.
14. As a maintainer, I want each numerics change gated against CLASS and each refactor shown to be number-preserving, so that regressions are traceable.
15. As a maintainer, I want a measurement step before bars are fixed, so that the bars rest on data rather than guesses.
16. As an emulator trainer, I want derivative precision robust across the training box, so that derivative-trained emulators are not biased.

## Implementation Decisions

**D1. A fixed-range Bessel table.**
- x_max = 1.05 × k_bound × `hr_tau0_max`, where:
  - `hr_tau0_max` is a new `PrecisionParams` field: the upper bound on τ₀, default 20 000 Mpc (fiducial τ₀ ≈ 14 100 Mpc);
  - k_bound is the largest fine-grid k, a function of precision settings only.
- n_x = ceil(3 x_max): the same 3 points per unit x as today. At `fit_cl` (k_max = 1 Mpc⁻¹) this gives about 63 000 points, against about 44 000 today.
- The table depends only on (l_max, n_x, x_max), so it is built once per precision setting and reused across cosmologies and calls, not rebuilt inside every evaluation.
- A guard (`eqx.error_if`, the same pattern as #47's τ-grid guard) raises an error naming the bound when τ₀ > `hr_tau0_max`.

**D2. Cubic Hermite lookup.** On each table interval [x_i, x_i+1], spacing Δ, t = (x − x_i)/Δ:

    j_l(x) = (2t³−3t²+1) j_l,i + (t³−2t²+t) Δ j_l′,i + (−2t³+3t²) j_l,i+1 + (t³−t²) Δ j_l′,i+1

- j_l′(x) uses the same form with j_l′ and j_l″, where j_l″ = [l(l+1)/x² − 1] j_l − (2/x) j_l′ from Bessel's equation.
- The existing safe-x guard covers x → 0.
- The radial T2 term keeps its formula, `0.5 · ([3 l(l+1)/x² − 2] j_l − (6/x) j_l′)` (CLASS `transfer.c:4168-4190`), fed by the Hermite j_l and j_l′.
- The EE transfer uses the Hermite j_l.
- **Expected accuracy.** Per-lookup error falls from about Δ²|j″|/8 ≈ 1.4% of the oscillation amplitude to about Δ⁴|j⁗|/384 ≈ 3×10⁻⁵. AD's derivative of the lookup becomes C¹ in x and close to j_l′.

**D3. Bounded reverse-mode memory.** The body of the ℓ loop (`scan_body`) is wrapped in `jax.checkpoint`. The backward pass recomputes that ℓ's lookups instead of storing them. The forward result is unchanged.

**D4. Lensing.**
- **Halofit correction:** the redshift grid is `linspace(0, ln(1+z_rec), n_z)` with z_rec traced. The k-extension decision and the extension's size come from precision settings, not from `float(pt.k_grid[-1])`.
- **EPT correction:** h stays traced. The h-dependent k selection is replaced by fixed-size weights over the full EPT k-grid (zero weight outside the overlap with `pt.k_grid`, which reproduces today's masking).
- **Linear path:** `compute_cl_pp` (Limber) and `lens_cls` change only if a gradient test shows they need it.

**D5. Measure first.** The first implementation step lands D1 and D3 with a temporary internal switch between the linear and Hermite lookups. At fiducial `fit_cl` on a V100 it measures, for each lookup:
- forward time;
- time and peak memory (by stage: background and thermodynamics, perturbations, harmonic) of one reverse-mode gradient of F_TT with respect to every `CosmoParams` field;
- AD against converged finite differences;
- spread across solver step placements;
- smoothness along h.

**Decision rule.**
- Hermite is adopted if all three budgets of D6 hold and it meets the provisional bars of the Testing Decisions.
- If Hermite misses a budget or a bar, work STOPS and both tables go to the user. The bars are never loosened.
- If the perturbation stage alone exceeds 32 GB in reverse mode, that is a design question for the user. This PR is not widened.
- Once Hermite is adopted, the switch is removed.

**D6. Budgets** (`fit_cl`, V100, timed by interleaving the base 2717b5c and the fix in one job on one node, as in #48):
- forward C_ℓ evaluation ≤ 1.05× base;
- one reverse-mode gradient of F_TT with respect to all `CosmoParams` fields fits in 32 GB;
- that gradient costs ≤ 3× a forward evaluation.

Lensed-gradient memory and time are measured and reported, and must also fit in 32 GB.

**D7. Test functionals**, each summed over ℓ ∈ [30, 1200]:
- F_TT = Σ ln C_ℓ^TT (unlensed, `fit_cl`);
- F_φφ = Σ ln C_ℓ^φφ;
- F_lTT = Σ ln C̃_ℓ^TT (lensed).

The range ℓ ≥ 30 avoids `fit_cl`'s pre-existing low-ℓ instability.

**D8. New references.** CLASS C_ℓ^φφ at 5 cosmologies, generated in the `classpt` env as `reference_data/<name>/cl_pp.npz` (keys ell, pp) with provenance json, as #48 did for thermodynamics. The cosmologies are lcdm_fiducial, omega_b_high, h_high, h_low and massive_nu_015.

**D9. Base and branch.** Stacked on #53 (`2717b5c`); this branch never pushes to #53's. The PR opens after #53 merges, or as a draft based on it, only on the user's go-ahead. It closes #54.

## Testing Decisions

A good test checks behaviour a cosmologist relies on (values against CLASS, derivative precision, smoothness, cost), not implementation details. Every new test is first shown failing on #53's head (`2717b5c`). The exceptions are the comparisons against the base itself (V1, V2, P1). They are guards that pass trivially on the base and exist to show the fix costs nothing, so they are exempt, as #48's bounds guards were. Physics-facing tests run at 3–5 cosmologies.

| Seam | What | Bar | Cosmologies |
|---|---|---|---|
| L0 lookup | Hermite j_l, j_l′ against a direct high-accuracy evaluation at random x ∈ [0, x_max], l ∈ {2, 30, 300, 1200}; AD derivative of the lookup against j_l′, j_l″ | 2× Hermite's measured worst, which must sit below linear's worst (so the linear lookup is red); fixed in D5 | none (cosmology-independent numerics; exempt, stated in the docstring) |
| L1 guard | τ₀ > `hr_tau0_max` raises naming the bound; it does not fire across the cosmology grid | — | grid |
| G1 step placement | dF_TT/dθ spread over k-batch size 8 vs 64 × jit/no-jit, θ ∈ {h, ω_b, ω_cdm} | < 3×10⁻⁴ | fiducial, ω_b +20%, Y_He 0.28 |
| G2 modes | production reverse (`th_grad_mode="stable"`) against forward | < 3×10⁻⁴ | same 3 |
| G3 smoothness | dF_TT/dh at 7 points h ∈ [0.64, 0.70], cubic-fit residual, with `bg_n_points=8000` (clax-72's background-table finding) | < 3×10⁻⁴ | same 3 |
| G4 AD vs FD | dF_TT/dθ against central FD, only where two steps (rel. 10⁻³, 3×10⁻³) agree to 10⁻⁴; resolved / floor / skipped classes as in #48 | ≤ 10⁻³ | same 3 |
| G5 lensing gradients | F_φφ and F_lTT, nonlinear ∈ {none, halofit, ept}: jit/jvp/grad finite; production reverse against forward; AD against converged FD | < 3×10⁻⁴; ≤ 10⁻³ | same 3 |
| V1 values vs CLASS | unlensed TT/EE/TE (TE relative to √(TT·EE)) at `fit_cl` and `planck_cl`; C_ℓ^φφ; lensed TT/EE/TE/BB. Bands ℓ 2–30, 30–100, 100–500, 500–1500, plus 1500–2500 at `planck_cl`. The fix's worst \|ΔC_ℓ/C_ℓ\| in each band ≤ base's + 10⁻⁴; the fix-vs-base shift is reported per band | as stated | lcdm_fiducial, omega_b_high, h_high, h_low, massive_nu_015 |
| V2 refactor identity | with the lookup held linear: unlensed C_ℓ identical to base to 10⁻¹⁰ apart from the wider x-range (any measured difference reported); Halofit and EPT corrections identical to base to 10⁻¹⁰ | 10⁻¹⁰ | fiducial |
| P1 cost | D6 budgets | D6 | fiducial |

- **Provisional bars.** The 3×10⁻⁴ bars are confirmed in D5. A miss on fixed code stops the work rather than loosening a bar.
- **`--fast` coverage.** L0 and L1 run under `--fast`. G1–G5 and V1 are slow-tier GPU tests, since each is a full `fit_cl` solve under AD.
- **Regression gate.** `--fast` plus full mode of the harmonic, lensing, C_ℓ and P(k) suites, with every failure re-run on 2717b5c.
- **Prior art.** `tests/test_recomb_downstream.py` (G1–G3 structure), `tests/test_recomb_thermo.py::_fd_check` (G4 classes), `tests/test_cl_pp.py`, `tests/test_lensing.py`.

## Out of Scope

- **Background-table AD noise:** the 800-point background-table noise (clax-72's draft issue). G3 sidesteps it with `bg_n_points=8000`.
- **P(k) gradient biases:** the ω_b / ω_cdm biases of #55.
- **`planck_cl` absolute accuracy:** limited by its 5000 fine k points.
- **`fit_cl` low-ℓ accuracy:** its ℓ ≤ 30 values.
- **Slow C_ℓ paths:** differentiability of `compute_cls_all` and `compute_cls_all_interp`.
- **Tensor B-modes:** `compute_cl_bb`, unless a test shows it shares and needs the same fix.
- **Stale figures:** CLAUDE.md's accuracy and timing figures.

## Further Notes

- **Prior measurement from #48's parked S4.** Scratch job 24261: with `stop_gradient` on x_max, eager forward dF_TT/dh = −1.46598035e3 and dF_TT/dω_b = −3.00115120e4, and k-batch 8 vs 64 agree to 9 digits. The production reverse mode ran out of memory (a 5.74 GiB allocation; XLA estimated ~45.9 GiB).
- **Risks, measured in D5:**
  - the perturbation-stage reverse memory at `fit_cl`;
  - Hermite's cost against the 5% forward budget. Building the table once may pay for it; the spacing can be revisited only with a new gate on L0;
  - the choice of the τ₀ bound.
- **Docs.** ADR 0004 records D1–D4 and the τ₀ bound. There will also be a CHANGELOG entry and docstrings for `compute_cls_all_fast`, `build_jl_table`, the Halofit and EPT corrections and `PrecisionParams.hr_tau0_max`.
