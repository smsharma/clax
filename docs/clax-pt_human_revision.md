# Supervision Log: Human Guidance in the clax-pt Development

This document records the specific interventions by the human supervisor during
the development of the `clax-pt` branch (March 28 – April 9, 2026), and
evaluates which milestones would or would not have been reached autonomously.

---

## 1. Development Methodology

The entire development methodology was established by the user in `CLAUDE.md`
before any clax-pt work began.  This document codified principles drawn from
Anthropic's C-compiler agent project (Carlini, 2026):

- **CLASS as oracle:** every module tested against CLASS-PT reference data;
  tests written before implementation.
- **CHANGELOG as shared memory:** updated after every unit of work; failed
  approaches recorded to prevent re-investigation.
- **Fast tests and concise output:** context-window hygiene for LLM agents.
- **Agent parallelism:** multiple worktree sessions exploring competing
  hypotheses simultaneously.
- **No fudge factors:** if a test fails at 0.2%, find the actual bug — do not
  multiply by 1.002.

Without this framework, the autonomous sessions would have lacked the
discipline to maintain a useful CHANGELOG, would have re-explored failed
approaches across sessions, and would have produced noisy test output that
degraded reasoning quality in later context windows.

---

## 2. Phase-by-Phase Supervision Record

### Phase 1: Literature Review and Design (Mar 28)

**User role:** Initiated the branch, defined the scope (CLASS-PT reimplementation
in JAX), and set the accuracy target (sub-percent vs CLASS-PT for all 9 spectra).

**Autonomous work:** Claude read the CLASS-PT source code (`nonlinear_pt.c`,
`classy.pyx`) and produced three reference documents (`FFTLog_PT.md`,
`CLASS-PT-summary.md`, `clax-pt.md`) covering the algorithm, component index
table, assembly formulas, and implementation plan.

**Assessment:** The design-first approach (writing reference documents before
code) was a direct consequence of the `CLAUDE.md` methodology.  The user's
framework ensured this happened rather than jumping straight to implementation.

---

### Phase 2: Core Implementation (Mar 29)

**User role:** None — autonomous implementation of `clax/ept.py` (~1,500 lines),
unit tests, reference data generation script.

**Assessment:** Standard code generation from a well-defined specification.  The
reference documents from Phase 1 provided a clear enough blueprint that no
human guidance was needed.

---

### Phase 3: FFTLog and IR Resummation Bugs (Mar 30 – Apr 3)

**Bugs 1–4:** M22 symmetry, LAPACK packing, DST k-grid, odd/even spline.

**User role:** Spawned a parallel `clax-pt-grad-project` branch to attack bugs
#2–4 from a different angle while the main session worked on bug #1.  The
user's decision to parallelize the debugging (one session on matrix loading,
another on IR resummation) accelerated convergence.  The merge commit `2c09968`
reconciled both approaches.

**Assessment:** Claude would likely have found all four bugs eventually through
systematic comparison against CLASS-PT reference values.  The parallel session
approach reduced wall-clock time but did not change the outcome.  These bugs had
clear symptoms (wrong values, wrong magnitudes) and clear fixes (match the
CLASS-PT convention exactly).

---

### Phase 4: Bias Expansion and Unit Bugs (Apr 3–4)

**Bugs 5–6:** Spurious h³ factor, wrong b₄ k-factor.

**User role:** Directly identified both bugs.  Bug #5 (the h³ multiply) was a
unit-conversion error that affected all bias and multipole functions uniformly —
it was invisible in shape-based tests and only apparent when comparing absolute
magnitudes.  Bug #6 (b₄ using `(k_h/h)²` instead of `k_h²`) was a similar
unit confusion.

**Assessment:** These bugs would have been found eventually through careful
magnitude comparison, but the user's direct identification saved 1–2 debugging
cycles.  Unit bugs are notoriously difficult for automated systems because the
code runs without errors and produces plausible-looking curves — only the
normalization is wrong.

---

### Phase 5: RSD Kernel Implementation (Apr 4)

**Bugs 7–9:** Incomplete M22/M13 RSD kernels, wrong UV counterterm coefficients.

**User role:** Pointed Claude to the specific line ranges in `nonlinear_pt.c`
where the RSD kernel formulas were defined (lines 6600–7300 for RSD multipole
matrices, lines 11880–12518 for bias spectra).  This targeted guidance was
important because `nonlinear_pt.c` is ~14,000 lines with no clear section
headers, and the RSD kernels are spread across multiple functions with
non-obvious naming.

**Assessment:** Claude would have found the relevant code sections by searching
for variable names like `Ptree_0_vv`, but the 14,000-line file with hundreds of
kernel expressions makes this a needle-in-a-haystack problem.  The user's
domain knowledge of the CLASS-PT source structure significantly reduced search
time.

After these fixes, real-space spectra passed at 0.18% but all 6 RSD multipoles
were still failing (ℓ=0: 1.75%, ℓ=2: 3.77%, ℓ=4: 7.91% for matter; ℓ=0:
1.41%, ℓ=2: 5.08%, ℓ=4: 36.89% for galaxies).

---

### Phase 6: The RSD Multipole Crisis (Apr 7–8)

This was the critical phase where autonomous work stalled and human supervision
was essential.

#### 6.1 The debugging maze

Over April 7–8, approximately 20 worktree sessions explored competing
hypotheses for the RSD failures:

- **Hypothesis A:** The 1-loop IR resummation uses the wrong FFTLog basis
  (x_nw vs x).  Partially correct but not the full story.
- **Hypothesis B:** The `pk_nw_arr` vs `pk_disc` variable in the tree
  computation is wrong.  The user proposed a one-line fix (change
  `pk_nw_arr` to `pk_disc`).  This made things *worse* (error increased to
  8.91% and in some sessions to 86%), but it correctly identified that the
  wiggle/no-wiggle handling in the tree term was the locus of the problem.
- **Hypothesis C:** The mu⁶/mu⁸ 1-loop terms are double-counted between the
  analytic kernel path and the GL tree path.  The user identified this
  double-counting issue.
- **Various kernel coefficient fixes:** Multiple sessions tried adjusting
  individual kernel rational functions, but each fix that improved one
  multipole degraded another.

The core problem was architectural: the approach of deriving a separate analytic
kernel matrix for each (ℓ, vv/vd/dd) combination is correct for isotropic IR
resummation but breaks when the BAO damping is anisotropic.  The analytic
Legendre projections assume that the tree spectrum P_tree(k) is independent of
μ, which is false when Σ_tot depends on μ.

Claude was unable to diagnose this architectural issue autonomously.  The
sessions kept trying variations of the analytic approach — fixing one kernel
coefficient, finding a new discrepancy, fixing that, breaking something else —
in a cycle that did not converge.

#### 6.2 The architectural intervention

**User's input (Apr 8):** The user proposed the GL quadrature redesign:

> Abandon per-multipole analytic kernels.  Instead, decompose each 1-loop
> P22/P13 contribution into bare μ-power coefficients (P22_mu0_dd,
> P22_mu2_dd, P22_mu4_dd, ...), assemble the full P(k, μ) at each
> Gauss–Legendre node, and integrate with Legendre polynomials to get
> multipoles.

This was documented in commit `455e97f` ("Document RSD redesign: assemble
P(k,μ) + GL integrate").

**Assessment:** This was the single most important human contribution to the
project.  Without it, the agent would have remained stuck in the analytic-kernel
approach, which is fundamentally unsuitable for anisotropic IR resummation.
The insight required understanding that:

1. The analytic multipole kernels implicitly assume isotropic P_tree(k).
2. The reference data uses CLASS-PT's AP path, which has anisotropic
   Σ_tot(μ).
3. The anisotropy couples the μ-dependence of the damping to the
   μ-dependence of the RSD factors, so the Legendre projections cannot be
   precomputed analytically.
4. The solution is to defer the Legendre projection to runtime via GL
   quadrature — the same approach CLASS-PT itself uses in the AP code path.

This is a physics insight, not a coding insight.  It requires understanding
why the BAO damping depends on μ (peculiar velocities along the line of sight
enhance the displacement field anisotropically) and how this interacts with
the Kaiser factor (1 + fμ²)².

---

### Phase 7: Galaxy Tree and Accuracy Metric (Apr 9)

**Bugs 10–12:** Galaxy tree formula, l=4 accuracy metric.

**User role:** After the GL redesign was implemented, Claude fixed the remaining
issues (galaxy ℓ=2 tree used wrong components, galaxy ℓ=4 tree had spurious b₁
factors, ℓ=4 metric was too strict).  These were found autonomously by
systematic comparison against CLASS-PT's `classy.pyx` assembly formulas.

**Assessment:** With the correct architecture in place, these were
straightforward bugs that Claude could diagnose by reading the CLASS-PT source.
The GL quadrature redesign had made the code structure transparent enough that
each bug had a clear, local fix.

---

### Phase 8: The Fudge Factor and Its Elimination (Apr 9)

#### 8.1 The interim fix (Bug #13)

After the GL redesign, 7/9 spectra passed but `pk_mm_l2` (1.40%) and
`pk_gg_l2` (1.73%) still failed.  Claude traced this to the tree-level
spectrum `Pk_tree = Pnw + Pw·exp(−Σ²k²)·(1 + Σ²k²)`, where the `(1 + Σ²k²)`
factor used an isotropic scalar Σ² = σ²_BAO.  Scanning a scalar correction
parameter α ∈ [0, 1], Claude found α = 0.27 made all 9 spectra pass.

This was an empirical fudge factor with no theoretical derivation.

#### 8.2 The user's challenge

**User's input:** The user asked two questions that forced a reckoning:

First: "Did we merge this to clax-pt?  If not, before merging, explain to me
what settings are different from CLASS-PT, e.g. is our `_TREE_ALPHA` the same
as theirs?"

This required a careful side-by-side comparison with CLASS-PT's source code,
which revealed that α = 0.27 does not correspond to any CLASS-PT parameter.
CLASS-PT's non-AP path uses α = 1.0; its AP path uses anisotropic Σ_tot(μ) —
there is no scalar α at all.

Second: "Think more carefully about the anisotropic IR resummation in the
AP path.  There should be no fudge factor if you get the μ-dependence right."

#### 8.3 The clean fix (Bug #14)

Careful re-reading of CLASS-PT's AP code path (`nonlinear_pt.c` line 9388) and
the IR resummation literature revealed the correct formula:

```
p_tree(k, μ) = P_nw + P_w · exp(−Σ_tot(μ)·k²) · (1 + Σ_tot(μ)·k²)
```

where `Σ_tot(μ) = Σ²(1+fμ²(2+f)) + δΣ²f²μ²(μ²−1)` is anisotropic — computed
at each GL μ-node and integrated with Legendre polynomials.  No scalar α
needed.  Additionally, for real-space spectra, the tree should use the raw
`pk_lin` (no IR damping), avoiding sensitivity to the DST-derived σ²_BAO.

The fix was to move the tree computation into the existing GL loop (which
already computed Σ_tot(μ) for the 1-loop terms) and set `Pk_tree = pk_lin_h`
for real-space.  Result: real-space accuracy improved from 0.94% to 0.31%, and
all RSD multipoles passed with zero tuned parameters.

**Assessment:** Without the user's intervention, the code would have shipped
with α = 0.27 — a fudge factor that happened to work at the fiducial cosmology
but had no theoretical basis and would likely fail at different cosmologies,
redshifts, or bias configurations.  The user's physics understanding of the
anisotropic IR resummation in the AP path identified the correct formula.
This was domain knowledge that Claude did not have access to and could not
have discovered by reading CLASS-PT alone (since CLASS-PT's AP path is
structurally different from the isotropic path, and the connection between
the two is not documented in the source code).

---

## 3. Summary: Autonomy vs Supervision

### What Claude did autonomously

- Wrote three reference documents from CLASS-PT source code reading
- Implemented the full `ept.py` module (~2,100 lines)
- Found and fixed bugs #1–4 (FFTLog conventions, IR resummation)
- Found and fixed bugs #7–9 (incomplete RSD kernels) given source line ranges
- Found and fixed bugs #10–12 (galaxy tree, accuracy metric)
- Implemented the GL quadrature loop given the architectural specification
- Built the test and validation infrastructure

### What required human supervision

| Intervention | Impact | Could Claude have reached this alone? |
|-------------|--------|--------------------------------------|
| Development methodology (CLAUDE.md) | Structured the entire project | No — the methodology was designed specifically for LLM agent limitations (context windows, time blindness, no persistent memory) |
| Parallel debugging sessions (Apr 2–3) | Accelerated bugs #2–4 | Yes, but slower (serial instead of parallel) |
| Unit bug identification (#5, #6) | Saved 1–2 debugging cycles | Yes, eventually, through magnitude comparison |
| CLASS-PT source navigation (#7–9) | Reduced search time in 14k-line file | Yes, but with significant time cost |
| **GL quadrature architecture (Apr 8)** | **Unblocked all 6 RSD multipoles** | **No — Claude was stuck in a non-converging cycle of analytic kernel fixes across ~20 sessions** |
| Refusing to merge with fudge factor (Apr 9) | Forced proper investigation | Unlikely — Claude had accepted α = 0.27 as adequate |
| **Anisotropic tree formula (Apr 9)** | **Eliminated the last fudge factor** | **No — Claude could not derive the anisotropic Σ_tot(μ) tree formula from CLASS-PT's AP path alone** |

### Conclusion

The project involved ~57 worktree sessions of autonomous work, but converged on
the correct solution only because of two critical human interventions: the GL
quadrature architectural redesign (which unblocked the RSD multipoles) and the
anisotropic tree formula (which eliminated the fudge factor).  Both required
physics understanding — of anisotropic BAO damping and its interaction with
RSD — that went beyond what could be extracted from CLASS-PT's source code by
systematic reading alone.  The remaining 12 bugs, the full implementation, and
the test infrastructure were produced autonomously within the methodological
framework the user established at the outset.
