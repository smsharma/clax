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

3. **Multi-cosmology validation** — Everything tested at ONE fiducial LCDM
   point only. Validate at: omega_b ±20%, omega_cdm ±20%, h ±10%, n_s ±5%,
   tau_reio ±30%. No code changes needed, just GPU diagnostic runs. Catch
   bugs that cancel at fiducial before they bias HMC chains.

4. **P(k,z) at arbitrary z** — Currently only z=0. Interpolate delta_m from
   perturbation output at arbitrary z values.

5. **BB tensor accuracy** — Lensing BB now accurate. Primordial BB still ~2x
   off CLASS.

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
