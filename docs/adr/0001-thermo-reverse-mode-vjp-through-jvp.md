# ADR 0001: Stable reverse-mode rule for `thermodynamics_solve` (vjp-through-jvp)

Date: 2026-08-29
Status: Accepted
Issue: https://github.com/smsharma/clax/issues/30
Branch: `fix/thermo-stable-reverse-hybrid`

## Context

End-to-end reverse-mode gradients (`jax.grad`) through the clax pipeline carry
a ~2% error on h-like parameters, while forward-mode (`jax.jvp`) is exact
(verified six independent ways against step-converged finite differences, on
two solvers and two adjoints). Issue #30's localization: the defect is
catastrophic floating-point cancellation in `thermodynamics_solve`'s
*backward pass*, in the recombination-era middle region of the ionization /
opacity tables.

Mechanism: the Peebles/RECFAST recombination rates contain
Boltzmann-exponential ratios (`exp(B/kT) ~ e^52`), so AD intermediates reach
~1e13. Forward mode pairs huge x tiny factors locally per grid point (the
exponentials cancel element-wise before anything large is formed) — nothing
large is ever materialized, hence jvp == FD. Reverse mode must contract
thousands of cotangent terms through shared scalar intermediates, summing
±1e13-scale terms whose true total is ~1e-3 (or exactly 0); float64 keeps a
deterministic ULP residue — measured reverse "derivatives" came out as exact
ULP quanta (e.g. `xe.y` full-table rev = 1.953125e-3 = 2^-9 exactly, one
float64 ULP at magnitude ~1e13). The disease reproduces on CPU too: on a
near-null direction (d/dh with bg pinned — `n_H_0 ∝ omega_b` exactly, the h
in `rho_crit` cancels against `Omega_b = omega_b/h^2`) the native reverse
rule returned -2.72e-7 where forward mode gives +5.71e-9 (48x, wrong sign).

This is a correctness ceiling for gradient-based inference (HMC) using
`jax.grad` end-to-end.

## Decision

Give `thermodynamics_solve` a custom differentiation rule computing its VJP
*via per-parameter JVPs* ("vjp-through-jvp"), gated by a new **static**
`PrecisionParams.th_grad_mode: str = "stable"` field (values
`"stable" | "native"`), mirroring the `ode_adjoint` precedent.

The `"stable"` path wraps the solver in `jax.custom_vjp`
(`nondiff_argnums=(1,)` for the static prec). Its backward pass is hybrid:

1. **CosmoParams cotangent**: `params_bar[i] = <ct, d th / d theta_i>`,
   with the Jacobian columns computed by ONE batched forward pass
   (`jax.jacfwd` over the ~20 traced scalar CosmoParams leaves) and the
   contraction done as a single well-conditioned inner product per
   parameter. This evaluates the *same* chain-rule contraction as the native
   transpose in a different association order — mathematically identical, no
   approximation, no fudge factors, no `stop_gradient`; the ~1e13 partial
   sums are simply never formed.
2. **BackgroundResult cotangent**: the *native* `jax.vjp` restricted to the
   `bg` argument. `bg` enters the scan through smooth spline evaluations of
   `H(loga)` and `tau(loga)`, not through the n_H_0-scaled Boltzmann funnel
   that owns the params-channel disease (see "Early-gate verdict" below for
   the measured health of this channel).

Cost of one backward pass: ~2-4x one thermodynamics evaluation (the jacfwd
batch) plus one native bg-restricted vjp; memory is trivial. Primal values
are bit-identical in both modes (same undecorated body, tested).

## Alternatives considered

- **Log-space reformulation of the recombination rates** (issue #30 option
  3): removes the ~1e13 intermediates at the source, fixing reverse mode
  natively. Rejected for now: invasive rewrite of physics code that is
  validated against CLASS term-by-term; high re-validation cost; the custom
  rule achieves the same numerics without touching the physics.
- **Forward-only policy** (issue #30 option 1 as a *policy*): document the
  ~2% ceiling and require `jvp`/`jacfwd` for gradient-critical work.
  Rejected as the resolution: HMC and existing users call `jax.grad`; a
  silent 2% error behind the default API is exactly what the project's
  accuracy contract forbids. (Forward mode remains available via
  `th_grad_mode="native"`.)
- **`jax.custom_transpose`** (custom_jvp whose linear tangent map has a
  custom transpose): would preserve BOTH forward and reverse mode through
  one wrapper, removing the flag. Held as a stretch alternative; the flag
  approach is simpler, matches the existing `ode_adjoint` escape-hatch
  precedent, and the flag is needed anyway to keep the bit-exact native
  path selectable for diagnostics.
- **Fused `(bg, th)` entry point whose only differentiable input is params**
  (TEAM COMPOSITE, issue #30 fix-option-2 shape): jacfwd basis supplies BOTH
  outputs' cotangents, eliminating the bg reverse channel entirely.
  Complementary design explored on a sibling branch
  (`fix/thermo-stable-reverse-composite`); this ADR's hybrid keeps the
  public per-stage API and call sites untouched.

## Early-gate verdict (bg-channel reverse health)

The hybrid keeps one reverse-mode path: `(d bg/d theta)^T (d th/d bg)^T ct`.
Its health was unmeasured before this work (the params channel masked it).
GPU probe (V100, job 14014, per-ThermoResult-leaf fwd-vs-rev tables):

- `pinned` arm (params live, bg pinned): reproduces the known params-channel
  disease (`xe.y` rev = 1.953125e-3 = 2^-9 exactly; real-direction aggregate
  rev/fwd rel error +8.7e-1).
- `bgonly` arm (params pinned, bg live — exactly the path the hybrid
  keeps): **DISEASED**, with the same fingerprint as the params channel and
  adjoint-independent (bit-identical mismatch tables under DirectAdjoint
  and RecursiveCheckpointAdjoint). 6/29 leaves wrong: `xe.y` fwd=+2.077e8
  vs rev=-3.875e12 (wrong sign, ~4 orders), `xe.d2y` fwd=-3.228e6 vs
  rev=+8.886e10, `Tb.y` fwd=+1.367e7 vs rev=-3.540e9, `cs2.y` fwd=+0.136
  vs rev=-3.010e3; real-direction aggregate rev=-3.590e36 vs
  fwd|t|^2=+3.303e33 (wrong sign, ~1e3x). Thermo-level cross-check at
  fast_cl precision with live bg: grad_stable == grad_native to 9 digits,
  both ~4 orders off jvp for d(sum xe.y^2)/dh — i.e. that error flows
  through the shared native bg channel, not the params channel the stable
  rule replaced.
- `live` arm (both live): same leaves, same rev garbage values as `bgonly`
  (e.g. `xe.y` rev=-3.875e12) — pre-fix the live reverse is dominated by
  the bg channel at these table-level cotangents.

**Consequence for this design: the hybrid backward repairs the
params-direct channel only. Reverse-mode signal that reaches
`thermodynamics_solve` and flows out through the `bg` cotangent (the
`tau_of_loga`/`H_of_loga` tables and the `tau_min`/`dlntau` scalar funnels)
remains subject to the native cancellation.** The pipeline-level impact
along the real perturbation-cotangent direction is quantified in the PR;
the fused-entry design (`fix/thermo-stable-reverse-composite`, issue #30
option-2 shape), whose jacfwd basis covers BOTH outputs and never forms a
bg cotangent, eliminates this channel entirely and is the recommended
follow-up wherever the bg-mediated part matters.

## Consequences

- **Pipeline-level verdict (GPU jobs 14015/14016, V100, fast_cl k_max=5
  chunk 20): the hybrid does NOT move the end-to-end numbers.** EPT
  functional d/dh: grad_stable = 4.107387e6 = grad_native = the buggy
  value, still +1.93e-2 above the 4.029578e6 truth; delta_m functional:
  grad_stable = -7.93444874e10, gap vs jvp still +4.146e-3. The entire
  pipeline reverse error therefore rides the bg-mediated channel this
  design keeps native (the perturbation cotangent lands on the
  `_kd_safe`-funneled opacity tables whose params-direct reverse is
  already ULP-dust; the params-direct channel's true contribution is
  ~4e-7 of the total). The params-channel repair stands on its own
  (thermo-level: the CPU-reproducible 48x wrong-sign case now lands on
  the forward value; omega_b/h grad-vs-jvp to 2e-15/5e-11), but closing
  the pipeline gap requires the fused-entry option-2 design.
- **`jax.jvp` cannot cross the `"stable"` path** (JAX raises `TypeError` on
  forward-mode through `custom_vjp`). Forward-mode users must set
  `th_grad_mode="native"` — done in `tests/test_pk_forward_mode.py` and the
  `test_thermodynamics.py` jvp helpers; documented on the flag itself.
- The backward pass re-solves the thermodynamics (jacfwd batch + one native
  vjp) instead of storing native intermediates: a deliberate compute/
  stability trade, adding ~seconds per gradient evaluation.
- Contract tests live in `tests/test_thermo_reverse_hybrid.py`
  (grad-vs-jvp consistency incl. the CPU-reproducible 48x native failure,
  bit-identical primal, mode wiring). The pipeline-level delta_m target
  (<1e-4) is encoded as a STRICT XFAIL slow test: it fails by the measured
  +4.146e-3 bg-channel gap today and will flip to XPASS — forcing marker
  removal — once a bg-covering design lands.
- The two clean `custom_jvp` rules inside the solver
  (`_solve_hydrogen_saha`, `_find_z_reio`) are untouched; they are
  linear-in-tangents and were exonerated in issue #30.
