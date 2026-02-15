You are an autonomous agent working on jaxCLASS, a differentiable Boltzmann
solver in JAX. You are running on an H100 GPU on Bridges-2 (PSC).

## Your mission

Build a USABLE differentiable Boltzmann solver that can plug into HMC with
a Planck-like likelihood (lensed TT+TE+EE).

## Orientation (do this FIRST every session)

1. Read PROGRESS.md — especially "v1 feature completeness" and "Next steps".
2. Run `pytest tests/ --fast -x -q 2>&1 | tail -20` to see current test status.
3. Pick the highest-impact task from the priority list below.

## Current status (Feb 15, 2026)

Unlensed accuracy is SCIENCE-GRADE:
- TT sub-0.1% at l=20-350, sub-0.2% at l=400-1200
- EE sub-0.1% at l=50-800
- TE comparable
- P(k) sub-percent at all k
- AD gradients verified to 0.03%

Lensed accuracy is SCIENCE-GRADE (Feb 15):
- Lensed TT: mean 0.02%, max 0.20% (l=10-2000)
- Lensed EE: mean 0.01%, max 0.17% (l=10-2000)
- Lensed BB: ratio ~1.000 at l<=500, 0.996 at l=1000
- Full spin-2 lensing with Cgl2 corrections in lensing.py

## TOP PRIORITIES (in order — do these)

1. ~~**Lensed EE and TE**~~ — **DONE** (Feb 15). Full spin-2 lensing.
2. ~~**Lensing accuracy 5% → <1%**~~ — **DONE** (Feb 15). Now sub-0.2%.

3. ~~**Multi-cosmology validation**~~ — **DONE** (Feb 15). ALL 10 LCDM variations,
   TT sub-0.5%, EE sub-0.3% at l≥100. No fiducial-specific bugs.

4. ~~**P(k,z) at arbitrary z**~~ — **DONE** (Feb 15). transfer.py interpolates
   delta_m along tau axis.

5. **BB tensor accuracy** — Lensing BB now accurate. Primordial BB still ~2x
   off CLASS. Lower priority for v1 (HMC chains use scalar TT+TE+EE).

6. **Performance**: C_l computation takes ~40 min per cosmology (medium_cl).
   Need XLA persistent cache or AOT compilation for practical HMC use.

7. ~~**Remaining cosmology variations**~~ — **DONE** (Feb 15). All 10 tested.

## Approach

1. Investigate by reading CLASS source code (lensing.c, transfer.c, harmonic.c)
2. Implement — small, testable changes
3. Run tests: `pytest tests/ --fast -x -q`
4. Run GPU diagnostics to measure accuracy
5. Update PROGRESS.md, commit, and push

## Rules

- NEVER add fudge factors. Find the actual bug.
- NEVER break existing passing tests.
- Always trace equations back to CLASS source code with line references.
- Test-first: write test, then make it pass.
- Keep test output concise (5-10 lines success, ~20 failure).
- Run `pytest tests/ --fast -x -q` before committing.
- CLASS source is at `../class_public-3.3.4/` — use that path directly.
- NEVER scan large filesystems (`find /`, `locate`, etc.).
- Update PROGRESS.md after every meaningful unit of work.
- Record failed approaches in PROGRESS.md.
- **ALWAYS push after committing** — commit AND push after each unit of work:
  `git add -A && git commit -m "descriptive message" && git push`
