# jaxCLASS Development Guide

## CRITICAL: HPC environment rules

### Filesystem
- **NEVER** use `find /`, `find /ocean`, or scan outside the project directory.
  The HPC filesystem has millions of files and these commands will hang forever.
- **Reference CLASS source**: `../class_public-3.3.4/` — ALWAYS use this path.
  Key files: `source/perturbations.c`, `source/transfer.c`, `source/harmonic.c`,
  `source/thermodynamics.c`, `source/background.c`
- **Project root**: the current working directory (use `.` or relative paths)
- **Only search within** `.` and `../class_public-3.3.4/`
- Use `grep` or `Grep` to search file contents — never `find` with broad paths.

### GPU access
You are running **directly on a GPU compute node** (V100-32GB).
Run python and pytest directly — no wrapper needed:
```bash
python scripts/gpu_planck_test.py
pytest tests/ --fast -x -q
```
Do NOT use `gpu-run.sh` — that was for login-node use only.

## What is this?

A fully differentiable reimplementation of the CLASS Boltzmann solver in JAX.
Full architecture and physics details are in **DESIGN.md**. This file covers
development instructions and conventions.

## Quick reference

- **Reference CLASS**: `../class_public-3.3.4/`
- **Design document**: `DESIGN.md` (read this first)
- **Progress log**: `PROGRESS.md`

## Repository

- **GitHub**: https://github.com/smsharma/jaxclass (private)
- Commit at meaningful checkpoints (passing tests, bug fixes, new features)
- Keep commits focused: one logical change per commit
- Always run `pytest tests/ --fast -x -q` before committing

## Setup

```bash
# Install dependencies
pip install jax jaxlib diffrax equinox jaxtyping pytest

# Install CLASS Python wrapper (for reference data generation only)
cd ../class_public-3.3.4 && pip install .

# Generate reference data
python scripts/generate_class_reference.py

# Run tests
pytest tests/ -v
pytest tests/ -v -m "not slow"    # skip perturbation integration tests
pytest tests/test_gradients.py -v  # gradient checks only
```

## GPU Access

For medium_cl / science_cl presets (l_max=50, many k-modes), a GPU is
strongly recommended. The perturbation vmap over k-modes parallelizes well.

### Bridges-2 (PSC) — primary

Full instructions in `../BRIDGES2_ACCESS.md` (gitignored). Summary:

- **V100-32GB, H100-80GB, L40S-48GB** available via SLURM
- SSH configured as `bridges2` (ControlMaster, no password after initial auth)
- Interactive GPU via tmux: `ssh bridges2 'tmux send-keys -t gpu "command" Enter'`
- Read output: `ssh bridges2 'tmux capture-pane -t gpu -p -S -50'`
- File transfer: `scp file bridges2:/ocean/projects/phy230064p/smishrasharma/`
- `$PROJECT` = `/ocean/projects/phy230064p/smishrasharma` (1TB, use for everything)
- ~2,000 SU remaining (1 SU/gpu-hr for V100/L40S, 2 for H100)

### Paperspace P6000 (legacy)

```bash
ssh paperspace@184.105.5.21
# Quadro P6000 (24GB VRAM), 8 CPU, 32GB RAM, CUDA 12.2
# Python/JAX at ~/jaxclass/.venv/
```

**Known GPU issue**: XLA autotuner fails with l_max=50 on P6000 ("couldn't
get temp CUBIN file name"). Works with l_max=25. May need to clear /tmp or
use `XLA_FLAGS=--xla_gpu_autotune_level=0`.

---

## Agent teams (use liberally)

Agent teams are enabled. Use them aggressively to parallelize independent work:

- **Debugging**: spawn teammates to investigate competing hypotheses in parallel
  (e.g., one checks source normalization, another checks Bessel accuracy,
  a third compares against CLASS at specific (k,tau) points)
- **Testing**: one teammate writes tests while another runs diagnostics on GPU
- **Code review**: spawn a reviewer teammate to check physics equations against
  CLASS while the main agent implements
- **Research**: multiple teammates searching CLASS source code for different
  conventions simultaneously

Each teammate gets its own context window and can read/write files independently.
Assign different files to different teammates to avoid conflicts.

Example:
```
Create an agent team with 3 teammates:
- "source-check": compare source_T0 subterms against CLASS at matched (k,tau)
- "transfer-check": verify T1/T2 radial functions against CLASS transfer.c
- "integration-check": test k-integration accuracy with different quadrature methods
```

Teams are especially valuable for this project because:
1. CLASS source code exploration (reading perturbations.c, transfer.c, harmonic.c)
   can be split across agents
2. GPU diagnostic runs take 5-20 min — use that time for parallel investigation
3. The Boltzmann equations have many independent components to verify

---

## Orientation (read this first when starting a session)

When you start a new session, orient yourself:
1. Read `PROGRESS.md` to see what's done and what's next.
2. Run `pytest tests/ -v --fast 2>&1 | tail -20` to see current test status.
3. Pick the next failing test or unchecked item from PROGRESS.md.
4. When you finish a unit of work, update PROGRESS.md before stopping.

---

## Principles for autonomous development

These are drawn from lessons building the C compiler with parallel Claude agent
teams (Carlini, Anthropic 2026). That project used 16 parallel agents, ~2000
sessions, and $20k in API costs to produce a 100k-line Rust C compiler that
builds the Linux kernel. The key lessons that apply here:

### 1. CLASS is the oracle -- tests are everything

CLASS is our known-good reference, the way GCC was the oracle for the C
compiler project. The test harness is the most important part of the project.
If the tests are wrong or incomplete, agents will solve the wrong problem.

**Rules:**
- Never merge or commit code that breaks existing passing tests.
- Every new module must have a corresponding test file BEFORE implementation.
  Write the test first (specifying what CLASS produces), then make it pass.
- When you find a bug, add a test that reproduces it before fixing it.
- Tests must be nearly perfect. Invest heavily in the test harness: generate
  high-quality CLASS reference data at many parameter points, write clear
  verifiers, and watch for failure modes so you can add targeted tests.
- When a discrepancy is found, trace upstream through the pipeline to find the
  first module where things diverge. Fix there; downstream improves automatically.

For the perturbation module specifically (our "Linux kernel" -- the one giant
task where all agents hit the same bug): use CLASS as an online oracle to
isolate which k-modes or which τ-ranges are wrong. Compare individual source
function components S_T,0 / S_T,1 / S_T,2 at specific (k, τ) points to
narrow down whether the issue is in the metric equations, photon hierarchy,
neutrino hierarchy, or initial conditions.

### 2. Concise test output (context window hygiene)

LLMs have finite context windows. Every line of noisy test output displaces
useful information and degrades reasoning quality. The C compiler project
learned this the hard way.

**Rules:**
- Tests print at most 5-10 lines on success, ~20 lines on failure.
- Use `pytest -q` by default. Never dump large arrays to stdout.
- Log verbose diagnostics to `test_logs/` files, not stdout.
- Pre-compute aggregate summary statistics. Print them, not raw data.
- When comparing arrays, print: max relative error, the index/value where
  it occurs, and the overall pass rate. Not the full arrays.
- Error messages should be greppable: put ERROR and the reason on one line
  so `grep ERROR logfile` works.

Good:
```
FAILED test_background.py::test_hubble - max rel err 0.032% at z=1089.2
  Expected H=1.0183e-4 Mpc^-1, got H=1.0180e-4 Mpc^-1
  (23/25 quantities pass at <0.01%, 2 at <0.05%)
```

Bad:
```
FAILED - arrays not equal:
  [1.0183e-4, 1.0182e-4, 1.0181e-4, ...]  (500 more lines)
```

### 3. Fast tests to avoid time blindness

LLMs can't tell time and will happily spend hours running full test suites
instead of making progress. The C compiler project addressed this with a
`--fast` flag that runs a deterministic ~10% subsample.

**Rules:**
- Every test file has a `--fast` mode (via pytest fixture).
- `--fast` runs a deterministic ~10% subsample (e.g., `z_grid[::10]`).
- The subsample should be deterministic per-agent but cover different points
  across agents (use a hash of the agent ID or test name as seed).
- Default development cycle: run `--fast` after every change, full suite
  only before committing.
- For perturbation tests (the slowest): `--fast` should test ~5 k-modes
  instead of all 150, chosen to span the full k range.

```python
@pytest.fixture
def fast_mode(request):
    return request.config.getoption("--fast", default=False)

def test_background_quantities(fast_mode, class_reference):
    z_grid = class_reference["z"]
    if fast_mode:
        z_grid = z_grid[::10]  # every 10th point
    ...
```

### 4. Keep PROGRESS.md current (agent orientation)

The C compiler project found that each agent drops into a fresh context with
no memory of what happened before. PROGRESS.md is the shared memory. Without
it, agents waste time re-discovering what's done and what's broken.

**Rules:**
- Update PROGRESS.md after every meaningful unit of work.
- Check off completed items with dates.
- Note what worked, what didn't, what's blocked.
- **Record failed approaches** so they aren't re-attempted. E.g.:
  "Tried using Tsit5 for perturbation ODE -- doesn't work, system is too
  stiff. Switched to Kvaerno5."
- Add new tasks discovered during implementation.
- When stuck, maintain a running doc of attempts in PROGRESS.md.

### 5. Prevent regressions (CI discipline)

The C compiler project found that once the codebase grew, new features
frequently broke existing functionality. They built a CI pipeline with strict
enforcement. We need the same discipline.

**Rules:**
- Run `pytest tests/ -q --fast` before every commit.
- If anything regresses, fix it before committing. Never "fix it later."
- If a new feature requires changing behavior in an existing test, update the
  test explicitly (don't just delete or skip it).
- Track test pass rates over time in PROGRESS.md (e.g., "background: 25/25,
  thermo: 18/20, perturbations: 142/150 k-modes passing").

### 6. Structure work for parallelism

The C compiler project found that parallelism is easy when there are many
independent failing tests (each agent picks a different one), but hard when
there's one giant failing task (all agents hit the same bug and overwrite
each other).

**How this applies to jaxCLASS:**

Easy to parallelize (many independent tasks):
- `background.py` and `bessel.py` are fully independent.
- `primordial.py` is independent of everything except `params.py`.
- `thermodynamics.py` depends only on `background.py`.
- `nonlinear.py` and `transfer.py` can be developed in parallel once
  perturbations exist.
- Individual k-mode failures in perturbations are independent.

Hard to parallelize (one giant task):
- Getting the first k-mode to work in perturbations (everyone hits same bug).
- Lensing accuracy (single module, one set of equations).

**Mitigation for the "one giant task" problem:** Break it into sub-tests.
For perturbations, test individual equation components separately:
- Test that metric equations (η, h') are correct with dummy species inputs.
- Test that the photon hierarchy alone matches CLASS with fixed metric.
- Test initial conditions in isolation.
- Then combine. This way, multiple agents can work on different subsystems.

**Task claiming:** When working in parallel, note your task in PROGRESS.md
(e.g., "IN PROGRESS: background.py (@agent-1)"). Check PROGRESS.md before
starting to avoid duplicate work.

### 7. Small, testable commits

**Rules:**
- Each commit implements one thing (one function, one module, one bugfix).
- Each commit passes all existing tests.
- Each commit includes or updates tests for the new code.
- Avoid large commits that change multiple modules at once.
- If a refactor touches many files, do it as a separate commit from features.

### 8. Document for the next session, not for users

The C compiler project maintained extensive READMEs and progress files
because each agent starts with zero context. Documentation is not a nicety;
it's a critical coordination mechanism.

**Every module should have a docstring explaining:**
- What physics it implements (with equation references to Ma & Bertschinger,
  Seljak & Zaldarriaga, or the CLASS papers).
- What it takes as input and produces as output (types, shapes).
- Any non-obvious numerical choices (why this tolerance? why this grid size?).
- Known limitations or accuracy issues.
- What CLASS function/file this corresponds to (e.g., "mirrors
  `perturbations.c:perturbations_einstein()`").

### 9. Specialized agent roles

The C compiler project used specialized agents beyond just "write code":
one for deduplication, one for performance, one for code quality review,
one for documentation.

**For jaxCLASS, useful specializations:**
- **Implementer agents**: Write the module code to pass tests.
- **Test quality agent**: Reviews and improves the test harness. Adds edge
  cases, improves error messages, catches gaps in coverage.
- **Gradient validation agent**: Focused solely on testing AD correctness.
  Runs finite-difference checks for every module, every parameter.
- **Performance agent**: Profiles the code, identifies bottlenecks, optimizes
  JIT compilation time, reduces memory usage.
- **Code quality agent**: Looks for duplicated code, inconsistent patterns,
  missing type hints, unclear variable names. Refactors.
- **Documentation agent**: Keeps PROGRESS.md, docstrings, and DESIGN.md
  in sync with actual code.

---

## Architecture summary

Sequential pipeline of pure functions, each returning a frozen PyTree:

```
CosmoParams → background → thermodynamics → perturbations → primordial
                                                                 ↓
                                          lensing ← harmonic ← transfer ← nonlinear
```

Key principle: **CosmoParams fields are JAX-traced** (for AD), **PrecisionParams
fields are static** (control array shapes). Never branch on CosmoParams values.

## Coding conventions

- **Pure functions only**. No mutable state, no global variables, no side effects.
- **Type hints** on all public functions using `jaxtyping.Float[Array, "..."]`.
- **Frozen dataclasses** for all result types, registered as JAX PyTrees.
- **No branching on traced values**. Use `jnp.where` instead of `if`. Use
  `jax.lax.cond` only when truly necessary (both branches must be same shape).
- **Synchronous gauge** for perturbations (matching CLASS default).
- **Units**: CLASS natural units throughout (Mpc, c=1). See `constants.py`.
- **Naming**: Follow CLASS naming where possible (e.g., `omega_b`, `tau_reio`,
  `index_bg_H`). Use snake_case for everything.

## Module dependency order

When implementing, follow this order (each depends only on previous):
1. `constants.py` (no deps)
2. `params.py` (constants)
3. `interpolation.py` (no deps, pure numerics)
4. `ode.py` (diffrax wrapper)
5. `bessel.py` (no deps, pure numerics)
6. `background.py` (1-4)
7. `thermodynamics.py` (1-6)
8. `perturbations.py` (1-7)
9. `primordial.py` (1-2)
10. `nonlinear.py` (1-6, 8-9)
11. `transfer.py` (1-5, 8)
12. `harmonic.py` (8-11)
13. `lensing.py` (12)
14. `distortions.py` (1-9)
15. `shooting.py` (6, implicit diff wrapper)

## Accuracy targets (vs CLASS at fiducial LCDM)

| Module | Target |
|--------|--------|
| Background (H, D_A, r_s) | < 0.01% |
| Thermodynamics (x_e, g) | < 0.1% |
| P(k) | < 0.1% |
| C_l (TT, TE, EE) | < 0.1% |
| Lensed C_l | < 0.2% |
| Gradients vs finite diff | < 1% |

## Testing workflow

Each module gets two types of tests:
1. **Value tests**: compare output arrays against CLASS reference data
2. **Gradient tests**: compare `jax.grad` output against finite differences

Always write the test first (what CLASS produces), then make it pass.

Reference data lives in `reference_data/` as `.npz` files, generated by
`scripts/generate_class_reference.py` from the CLASS Python wrapper.

### Using CLASS as an oracle

When debugging a discrepancy, the pattern is:
1. Identify which output quantity disagrees.
2. Trace back through the pipeline to find the first module where things diverge.
3. Compare intermediate quantities (e.g., background table at specific z values)
   against CLASS reference data.
4. Fix the upstream issue; downstream should improve automatically.

This is analogous to the "GCC oracle" pattern: CLASS is always right, our code
must match it.

---

## Critical rules to prevent physics bugs

These address the main risks unique to reimplementing a Boltzmann solver
(vs. a compiler or other software project). See DESIGN.md Section 10 for
full analysis.

### Never add fudge factors

If a test fails with 0.2% error, there is a term that is wrong -- a sign
error, a missing factor, a wrong index. Find the actual bug. Do NOT multiply
by 1.002 to make the test pass. If you are tempted, it means you haven't
isolated the source of the discrepancy.

### Trace every equation to CLASS source code

Every term in the perturbation RHS, every source function expression, every
background equation must have a comment referencing the corresponding line
in CLASS. Example:

```python
# Baryon velocity: cf. perturbations.c:5070-5075
# theta_b' = -aH*theta_b + cs2*k2*delta_b + R_inv*kappa_dot*(theta_g - theta_b)
theta_b_prime = -a_prime_over_a * theta_b + cs2 * k2 * delta_b \
    + R_inv * kappa_dot * (theta_g - theta_b)
```

This makes it possible to review term-by-term and catch missing terms.

### Test at many parameter points, not just fiducial

A fudge factor or bug that cancels at fiducial LCDM will show up when
parameters change. The reference data suite must include:
- Fiducial LCDM (Planck 2018 best-fit)
- High/low omega_b (±20%)
- High/low omega_cdm (±20%)
- Massive neutrinos (0.06, 0.15, 0.3 eV)
- w0wa dark energy (w0=-0.9, wa=0.1)
- High/low N_eff (2.0, 4.0)
- Tensor modes (r=0.01, 0.1)

### Test intermediate quantities, not just final output

When C_l disagrees, bisect through the pipeline:
1. Is background correct? (H, distances)
2. Is thermodynamics correct? (x_e, visibility)
3. Are source functions correct? (S(k,τ) at specific k,τ)
4. Are transfer functions correct? (Δ_l(k) at specific l,k)
5. Is the k-integration correct? (C_l from known transfer functions)

Generate and store CLASS intermediate outputs at every stage.

### Compare against multiple references

CLASS is the primary oracle, but it uses approximations we don't. Also compare:
- **CAMB** as a second oracle (when both agree and we don't, it's our bug)
- **DISCO-EB** for P(k) (same approximation-free approach in JAX)
- **SymBoltz.jl** results if available

### Test gradients from the bottom up

Don't jump to d(C_l)/d(params). Build confidence layer by layer:
1. d(H(z=0))/d(h) -- trivial, must be exact
2. d(x_e(z=1000))/d(omega_b) -- thermodynamics gradient
3. d(S(k,τ))/d(omega_b) at a single (k,τ) -- perturbation gradient
4. d(P(k))/d(omega_b) -- integrated perturbation gradient
5. d(C_l)/d(omega_b) -- full pipeline gradient

If step N fails, the bug is between step N-1 and N. Also: forward mode
(`jax.jvp`) must match reverse mode (`jax.grad`). Disagreement pinpoints
a `custom_vjp` bug.

---

## Key design decisions (see DESIGN.md for details)

1. **Approximation-free perturbations**: Full Boltzmann hierarchy at all times,
   no TCA/RSA/UFA switching. Validated by SymBoltz.jl.
2. **vmap over k-modes**: GPU parallelism for perturbation integration.
3. **RecursiveCheckpointAdjoint**: Memory-efficient reverse-mode AD.
4. **Implicit differentiation for shooting**: `jax.custom_vjp` for H0(theta_s).
5. **Smooth Limber transition**: Sigmoid blend, no hard switch.
6. **Analytical Jacobian** for inner ODE solve; AD for outer parameter gradients.
