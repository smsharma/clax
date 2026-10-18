# C_ℓ differentiability at `fit_cl` (smsharma/clax#54) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CMB spectra at `fit_cl` (unlensed C_ℓ, C_ℓ^φφ, lensed C_ℓ) differentiable with respect to every cosmological parameter. The gradients must be precise, smooth, independent of where the solver places its steps, and fit in a 32 GB V100.

**Architecture:**
- A Bessel table whose x-range comes from precision settings (`pt_k_max_cl`, new `hr_tau0_max`), built once and guarded.
- A cubic Hermite lookup of j_l and j_l′, so AD sees a smooth derivative.
- `jax.checkpoint` on the per-ℓ step, to bound reverse-mode memory.
- Lensing corrections with traced cosmology and fixed array shapes.

The plan starts with a measurement that decides between the linear and the Hermite lookup, then adds tests at five seams: lookup, guard, unlensed gradients, lensing gradients, and values against CLASS.

**Tech Stack:** JAX 0.9.2 (float64), equinox (`eqx.error_if`), diffrax, pytest; CLASS (`classy`, `classpt` env) for references; SLURM on igpu (V100 for gradient tests).

**Spec:** `docs/superpowers/specs/2026-10-18-cl-differentiability-design.md`. Read it; this plan argues from it. Two amendments made while planning are recorded in the spec and summarised below.

## Global Constraints

- **Base and branch.** The base is #53's head `2717b5c`. The branch is `fix/cl-differentiability` in worktree `/home/n2minh/clax-cl`. Never push to #53's branch `fix/recombination-saha-ode-staging`; session clax-72 owns it. No push or PR without the user's go-ahead.
- **Oracle and bars.** CLASS is the oracle. Bars are code-accuracy gates and are never loosened; there are no fudge factors. A miss on fixed code means: STOP and report BLOCKED with the numbers.
- **Red first.** Every new test is first shown failing on the base. The exceptions are V1/V2/P1 (guards that compare against the base itself) and L0's linear-lookup red (see Task 3).
- **Multi-cosmology.**
  - Physics-facing tests run at 3–5 cosmologies.
  - Gradient seams use `recomb_downstream_cosmology` (`lcdm_fiducial`, `omega_b_high`, `YHe_028`).
  - Value gates use `cl_value_cosmology` (`lcdm_fiducial`, `omega_b_high`, `h_high`, `h_low`, `massive_nu_015`).
  - L0 is cosmology-independent and states this in its docstring.
- **Budgets (spec D6).** At `fit_cl` on a V100, timed by interleaving base and fix in one job on one node:
  - forward C_ℓ ≤ 1.05× base;
  - one reverse-mode gradient of F_TT w.r.t. all `CosmoParams` fields fits in 32 GB and costs ≤ 3× forward.
- **STOP rules.**
  - The Hermite lookup misses a D6 budget or a provisional bar.
  - The perturbation stage alone exceeds 32 GB in reverse mode.
  - The EPT gate (Task 6) misses.
  - A test fails only on the fix (a regression).
- **Commits.**
  - Explicit paths only.
  - `git commit -F /lustre/work/n2minh/clax-probes/cl/msg-<task>.txt`, run as its own command (a hook blocks `git commit` followed by ` -n`).
  - The trailer is the authoring session's own `Co-Authored-By` line; if it has none, `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  - CHANGELOG headings use `Mon D, YYYY`; ADR headers use `YYYY-MM-DD`.
- **Compute.**
  - Heavy JAX runs only in SLURM jobs. On the login node, only `--collect-only`, `py_compile`, ruff and numpy-only scripts, each under `timeout 300`.
  - Pass variables in the submit environment (`VAR=… sbatch --export=ALL …`).
  - GPU jobs use `JAX_PLATFORMS=cuda` and `--gres=gpu:1`.
  - Never touch other users' jobs or conda envs. CLASS runs only in the `classpt` env.
- **Pre-existing failure:** `tests/test_solver_selection.py::TestRosenbrockPk::test_pk_rosenbrock_vs_kvaerno5` fails on the base (max_steps). Report it as pre-existing; any other failure blocks the task.
- **Hooks.** A PreToolUse "Fact-Forcing Gate" hook may require stated facts before Bash, Edit or Write. State them and retry.

## Review Focus

1. **The whole C_ℓ function under `jax.jit`:** `jax.jit(jax.grad(F_TT))` runs and equals `jax.grad(F_TT)` to 1e-10. Task 5.
2. **No `prec` while tracing:** calling `compute_cls_all_fast` without `prec` while tracing raises a `TypeError` that tells the user to pass `prec`, not a cryptic `ConcretizationTypeError`. Task 2.
3. **Mismatched `prec`:** if `prec.pt_k_max_cl` is smaller than the perturbation grid's k_max, the lookup would run off the table. The guard must raise rather than clamp silently. Task 2.
4. **`vmap` over cosmologies** (ensembles, emulator training): `jax.vmap(F_TT)` over 3 values of h equals the Python loop to 1e-10. Task 5.
5. **Lookup across the recurrence blend:** at x within ±2 of each l, where the table switches from backward to upward recurrence, the Hermite error stays within the L0 bar. Task 3.

## Spec amendments made while planning (also edited into the spec, Oct 18, 2026)

- **A1. Where the table bounds come from.** `compute_cls_all_fast` and `compute_cl_pp` do not receive precision settings today, so the fixed table range needs them passed in. Both gain a keyword-only argument `prec: PrecisionParams | None = None`.
  - With `prec`: the bounds come from `prec.pt_k_max_cl` and `prec.hr_tau0_max`; this works under `jit`, `jvp` and `grad`.
  - Without `prec`: eager callers keep working (k_max is read from `pt.k_grid`). When tracing, they get a clear `TypeError` asking for `prec`.
- **A2. The EPT correction is rebuilt, so V2's EPT identity does not apply.** Today it splines R = P_NL/P_lin through only the EPT nodes inside the perturbation k-range. That node set changes with h, so R jumps discontinuously in h and no differentiable version can be number-identical. The new correction:
  - splines P_NL over the fixed EPT nodes in log(k/h) and divides by P_lin at the exact perturbation k;
  - is gated by |ΔR/R| ≤ 1e-3 against the base for k ∈ [1e-4, k_max] Mpc⁻¹ at fiducial;
  - must move C_ℓ^φφ(EPT) by ≤ 1e-3 per band relative to the base, measured and reported.

---

## File structure

| File | Responsibility |
|---|---|
| `clax/params.py` (modify) | New field `hr_tau0_max: float = 20000.0` |
| `clax/harmonic.py` (modify) | `_table_bounds`, `_table_grid`, cached `_jl_tables`, `_table_position`, `_hermite`, `_jl_node_second_derivative`, `_lookup_jl`; `compute_cls_all_fast(…, *, prec=None)` with guards and the checkpointed ℓ-step |
| `clax/lensing.py` (modify) | `_halofit_modulator(…, k_pt_max=None)` with traced z_rec; smooth fixed-shape `_ept_modulator`; `compute_cl_pp(…, prec=None)` |
| `scripts/generate_class_reference.py` (modify) | `CL_PP_GRID`, `generate_cl_pp`, `--cl-pp` mode |
| `reference_data/<name>/cl_pp.{npz,json}` (create ×5) | CLASS C_ℓ^φφ (unlensed φφ, ℓ 0–2500) |
| `tests/conftest.py` (modify) | `CL_VALUE_GRID`, fixture `cl_value_cosmology` |
| `tests/test_cl_reference.py` (create) | Sanity of `cl_pp.npz` |
| `tests/test_cl_table_guard.py` (create) | L1 guard; Review Focus 2, 3 |
| `tests/test_cl_bessel_lookup.py` (create) | L0 (cosmology-independent); Review Focus 5 |
| `tests/cl_test_utils.py` (create) | Functionals F_TT, F_φφ, F_lTT; spread, cubic residual, converged-FD helpers |
| `tests/test_cl_gradients.py` (create) | G1–G4; Review Focus 1, 4 |
| `tests/test_cl_lensing_gradients.py` (create) | G5 |
| `tests/data/cl_vs_class_baseline.json` (create) | The base's band errors against CLASS (bars for V1) |
| `tests/test_cl_vs_class.py` (create) | V1 |
| `docs/adr/0004-cl-differentiability.md` (create), `CHANGELOG.md` (modify) | Docs |

Job tooling, not committed, goes in `/lustre/work/n2minh/clax-probes/cl/`: `pytest.sbatch`, `run.sbatch`, `class.sbatch`, base snapshots `base-<task>/` (made with `git -C /home/n2minh/clax-cl archive 2717b5c | tar -x -C <dir>`, plus any new test files and references copied in), measurement scripts, and logs.

---

### Task 1: Job tooling, CLASS C_ℓ^φφ references, value-gate cosmology grid

**Files:**
- Modify: `scripts/generate_class_reference.py` (add the constants, the function and the `--cl-pp` branch at the top of `main()`)
- Modify: `tests/conftest.py` (append after `recomb_downstream_cosmology`)
- Create: `tests/test_cl_reference.py`, `reference_data/{lcdm_fiducial,omega_b_high,h_high,h_low,massive_nu_015}/cl_pp.{npz,json}`

**Interfaces:**
- **Produces:**
  - `CL_VALUE_GRID: dict[str, dict]`;
  - the fixture `cl_value_cosmology -> (name, CosmoParams)` (fiducial only under `--fast`);
  - `reference_data/<name>/cl_pp.npz` with `ell` (0…2500) and `pp` (unlensed C_ℓ^φφ, dimensionless as CLASS `raw_cl`);
  - job scripts.

- [ ] **Step 1: Job tooling.** Create `/lustre/work/n2minh/clax-probes/cl/`.
  - Copy `/lustre/work/n2minh/clax-probes/recomb/pytest.sbatch` and `run.sbatch` into it. In each copy change only:
    - `--job-name` → `cl-pytest` / `cl-run`;
    - `--output=/lustre/work/n2minh/clax-probes/cl/%x.out.%j`;
    - `SRC=${SRC:-/home/n2minh/clax-cl}`.
  - Create `class.sbatch`. It is the same as `run.sbatch` except `micromamba activate classpt`, no JAX exports, `--cpus-per-task=4 --mem=16GB --time=2:00:00`, and the body `cd /home/n2minh/clax-cl && python -u scripts/generate_class_reference.py --cl-pp`.

- [ ] **Step 2: Grid in `tests/conftest.py`.** Append:

```python
# C_l value gates (smsharma/clax#54): cosmologies with CLASS cls.npz and
# cls_lensed.npz references (scripts/generate_multipoint_reference.py VARIATIONS;
# massive_nu_015 from the m_ncdm grid). Names match reference_data/<name>/.
CL_VALUE_GRID = {
    "lcdm_fiducial": {},
    "omega_b_high": {"omega_b": 0.02237 * 1.20},
    "h_high": {"h": 0.6736 * 1.10},
    "h_low": {"h": 0.6736 * 0.90},
    "massive_nu_015": {"m_ncdm": 0.15},
}
cl_value_cosmology = _make_cosmology_fixture(CL_VALUE_GRID)
```

- [ ] **Step 3: Failing reference test** `tests/test_cl_reference.py`:

```python
"""CLASS C_l^phiphi references for the C_l differentiability gates (#54) exist
and are sane. Generated by scripts/generate_class_reference.py --cl-pp in the
classpt env, on the same CLASS base as reference_data/<name>/cls.npz."""
import json
import os

import numpy as np
import pytest

from tests.conftest import CL_VALUE_GRID

REF = os.path.join(os.path.dirname(__file__), "..", "reference_data")


@pytest.mark.parametrize("name", list(CL_VALUE_GRID))
def test_cl_pp_reference_is_sane(name):
    path = os.path.join(REF, name, "cl_pp.npz")
    assert os.path.isfile(path), f"ERROR missing {path}"
    d = np.load(path)
    ell, pp = d["ell"], d["pp"]
    assert ell.shape == pp.shape == (2501,), f"ERROR shape {ell.shape} {pp.shape} ({name})"
    assert np.all(ell == np.arange(2501)), f"ERROR ell grid ({name})"
    assert np.all(np.isfinite(pp)) and np.all(pp[2:] > 0), f"ERROR non-positive pp ({name})"
    # [l(l+1)]^2 C_l^pp / 2pi peaks at ~1e-7 near l ~ 40 for LCDM-like cosmologies
    d_pp = (ell * (ell + 1.0)) ** 2 * pp / (2 * np.pi)
    lpk = int(np.argmax(d_pp[2:])) + 2
    assert 20 <= lpk <= 80 and 5e-8 < d_pp[lpk] < 3e-7, f"ERROR peak l={lpk} {d_pp[lpk]:.3e} ({name})"
    with open(path.replace(".npz", ".json")) as fh:
        prov = json.load(fh)
    assert "class_params" in prov, f"ERROR provenance ({name})"
    print(f"{name}: [l(l+1)]^2 C_l^pp/2pi peak {d_pp[lpk]:.3e} at l={lpk}")
```

- [ ] **Step 4: Run it and see it fail.** Run `timeout 300 env JAX_PLATFORMS=cpu python -m pytest tests/test_cl_reference.py -q -x` (numpy-only). Expected: FAIL with `ERROR missing`.

- [ ] **Step 5: Generator.** In `scripts/generate_class_reference.py`, add above `def main()`:

```python
# C_l differentiability gates (smsharma/clax#54): unlensed C_l^phiphi on the same
# CLASS base as the cls.npz references (FIDUCIAL_PARAMS + the multipoint overrides).
CL_PP_GRID = {
    "lcdm_fiducial": {},
    "omega_b_high": {"omega_b": 0.02237 * 1.20},
    "h_high": {"h": 0.6736 * 1.10},
    "h_low": {"h": 0.6736 * 0.90},
    "massive_nu_015": {"m_ncdm": 0.15},
}


def generate_cl_pp(name, overrides, outdir_base):
    """Unlensed C_l^phiphi (CLASS raw_cl 'pp', l = 0..2500) at FIDUCIAL_PARAMS +
    overrides; writes cl_pp.npz (ell, pp) and cl_pp.json provenance."""
    import classy
    from classy import Class

    pars = dict(FIDUCIAL_PARAMS)
    pars.update(overrides)
    c = Class()
    c.set(pars)
    c.compute()
    raw = c.raw_cl(2500)
    ell = np.arange(2501)
    pp = np.asarray(raw['pp'])[:2501]
    outdir = os.path.join(outdir_base, name)
    os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, 'cl_pp.npz'), ell=ell, pp=pp)
    with open(os.path.join(outdir, 'cl_pp.json'), 'w') as fh:
        json.dump({"class_params": pars,
                   "classy": getattr(classy, '__version__', 'unknown'),
                   "build": "CLASS-PT tree (CLASS v3.3.4), classpt env",
                   "quantity": "unlensed C_l^phiphi, CLASS raw_cl()['pp'], l=0..2500"},
                  fh, indent=1)
    c.struct_cleanup()
    c.empty()
    print(f"  {name}: C_l^pp(l=100)={pp[100]:.6e}")
```

At the top of `main()`, directly after `outdir = …`, insert:

```python
    if '--cl-pp' in sys.argv:
        for name, overrides in CL_PP_GRID.items():
            generate_cl_pp(name, overrides, outdir)
        return
```

- [ ] **Step 6: Generate the references.** Run `sbatch --export=ALL /lustre/work/n2minh/clax-probes/cl/class.sbatch` and wait with bounded waits (`sacct -j <id> -X -n -o State`).
  - Cross-check fiducial against a live CLASS run with the same parameters: |Δ| must be 0 (same code path).
  - Also print the ratio to `tests/test_cl_pp.py`'s live reference at ℓ = 100 for information. That reference uses τ_reio 0.052 and YHe 0.2425, so it differs slightly.

- [ ] **Step 7: Green.** Run the Step 4 command again. Expected: 5 passed.

- [ ] **Step 8: Commit.** Paths: `scripts/generate_class_reference.py tests/conftest.py tests/test_cl_reference.py reference_data/*/cl_pp.npz reference_data/*/cl_pp.json`. Subject: `test(cl): CLASS C_l^phiphi references and value-gate cosmology grid (#54)`.

---

### Task 2: Fixed-range Bessel table, guards, checkpointed ℓ-step (linear lookup kept)

**Files:**
- Modify: `clax/params.py` (after `hr_l_limber`)
- Modify: `clax/harmonic.py` (helpers above `compute_cls_all_fast`; the body of `compute_cls_all_fast` from "3. Build j_l…" through the scan)
- Create: `tests/test_cl_table_guard.py`
- Create, not committed: `/lustre/work/n2minh/clax-probes/cl/v2_unlensed.py`

**Interfaces:**
- **Produces:**
  - `PrecisionParams.hr_tau0_max: float = 20000.0`;
  - in `clax.harmonic`:
    - `_table_bounds(pt, prec) -> (k_bound: float, tau0_max: float)`;
    - `_table_grid(k_bound, tau0_max) -> (x_max: float, n_x: int)`;
    - `_jl_tables(l_max, n_x, x_max) -> (x_table, jl_table, jlp_table)`, cached;
    - `_table_position(x_flat, dx_tab, n_x) -> (i_left: int32 array, t: float array)`;
    - `_lookup_jl(jl_row, jlp_row, l_val, ll1_val, i_left, t, dx_tab, mode) -> (jl, jlp)`, with mode `"linear"` only in this task;
    - `compute_cls_all_fast(pt, params, bg, l_max=1500, n_k_fine=5000, tt_mode=None, k_interp_method="spline", *, prec=None, _jl_lookup="linear")`.

- [ ] **Step 1: Failing guard tests** `tests/test_cl_table_guard.py`:

```python
"""Guards of the fixed-range Bessel table in compute_cls_all_fast (#54).
Cosmology-independent plumbing: exempt from the multi-cosmology rule."""
from dataclasses import replace

import jax
import pytest

from clax.harmonic import compute_cls_all_fast


def test_tau0_beyond_bound_raises(pipeline_fast_cl):
    params, prec, bg, th, pt = pipeline_fast_cl
    small = replace(prec, hr_tau0_max=1000.0)          # fiducial tau0 ~ 14100 Mpc
    with pytest.raises(Exception, match="hr_tau0_max"):
        jax.block_until_ready(compute_cls_all_fast(pt, params, bg, l_max=50, prec=small)["tt"])


def test_kmax_mismatch_raises(pipeline_fast_cl):
    params, prec, bg, th, pt = pipeline_fast_cl
    wrong = replace(prec, pt_k_max_cl=0.5 * prec.pt_k_max_cl)
    with pytest.raises(Exception, match="pt_k_max_cl"):
        jax.block_until_ready(compute_cls_all_fast(pt, params, bg, l_max=50, prec=wrong)["tt"])


def test_traced_without_prec_asks_for_prec(pipeline_fast_cl):
    params, prec, bg, th, pt = pipeline_fast_cl
    f = jax.jit(lambda pt_, bg_: compute_cls_all_fast(pt_, params, bg_, l_max=50)["tt"])
    with pytest.raises(TypeError, match="prec="):
        f(pt, bg)


def test_guard_silent_at_fiducial(pipeline_fast_cl):
    params, prec, bg, th, pt = pipeline_fast_cl
    tt = compute_cls_all_fast(pt, params, bg, l_max=50, prec=prec)["tt"]
    assert float(tt[10]) > 0, "ERROR C_l not positive with prec given"
```

- [ ] **Step 2: Run it on a base snapshot and see it fail.** Make `/lustre/work/n2minh/clax-probes/cl/base-t2/` (archive of 2717b5c, plus the new test file copied in). Submit a CPU job: `SRC=<base-t2> PYTEST_ARGS='tests/test_cl_table_guard.py' sbatch --export=ALL …/pytest.sbatch`. Expected: FAIL, because `prec` is an unexpected keyword and `hr_tau0_max` does not exist.

- [ ] **Step 3: Precision field.** In `clax/params.py`, after the `hr_l_limber` line, add:

```python
    hr_tau0_max: float = 20000.0     # Mpc; upper bound on tau_0 fixing the Bessel-table x-range (#54)
```

- [ ] **Step 4: Helpers.** In `clax/harmonic.py`:
  - Add the imports `import functools`, `import math`, `import equinox as eqx` and `from clax.params import PrecisionParams` to the imports at the top.
  - Replace the lines `# _build_jl_table is now build_jl_table, imported from clax.bessel` and `_build_jl_table = build_jl_table` with:

```python
# _build_jl_table is now build_jl_table, imported from clax.bessel
_build_jl_table = build_jl_table


def _table_bounds(pt, prec):
    """Static (k_bound, tau0_max) for the fixed-range Bessel table (#54).

    With prec: k_bound = prec.pt_k_max_cl (the fine k-grid ends at pt.k_grid[-1]
    = pt_k_max_cl) and tau0_max = prec.hr_tau0_max -- both static, so jit, jvp
    and grad work. Without prec (eager callers only), k_bound is read from
    pt.k_grid; under tracing that read fails and we ask for prec instead.
    """
    if prec is not None:
        return float(prec.pt_k_max_cl), float(prec.hr_tau0_max)
    try:
        k_bound = float(pt.k_grid[-1])
    except jax.errors.ConcretizationTypeError as err:
        raise TypeError(
            "compute_cls_all_fast: pass prec=<the PrecisionParams of the perturbation "
            "solve> to jit or differentiate; without prec the Bessel table's k-range "
            "is read from pt.k_grid, which is traced here.") from err
    return k_bound, float(PrecisionParams().hr_tau0_max)


def _table_grid(k_bound, tau0_max):
    """x_max = 1.05 k_bound tau0_max and n_x for ~3 points per unit x
    (the pre-#54 density), at least 30000 points."""
    x_max = 1.05 * k_bound * tau0_max
    return x_max, max(30000, int(math.ceil(3.0 * x_max)))


@functools.lru_cache(maxsize=8)
def _jl_tables(l_max, n_x, x_max):
    """j_l, j_l' on the fixed x-range: built once per (l_max, n_x, x_max) and
    reused across cosmologies and calls -- concrete arrays, never traced."""
    with jax.ensure_compile_time_eval():
        return build_jl_table(l_max, n_x=n_x, x_max=x_max)


def _table_position(x_flat, dx_tab, n_x):
    """Left table index and fractional position t in [0, 1] on the uniform
    x-table. The index carries no derivative; t carries d/dx = 1/dx_tab."""
    u = x_flat / dx_tab
    i_left = jnp.clip(jnp.floor(u).astype(jnp.int32), 0, n_x - 2)
    return i_left, u - i_left.astype(u.dtype)


def _lookup_jl(jl_row, jlp_row, l_val, ll1_val, i_left, t, dx_tab, mode):
    """j_l(x) and j_l'(x) from one table row at positions (i_left, t)."""
    i_right = i_left + 1
    jl0, jl1 = jl_row[i_left], jl_row[i_right]
    jp0, jp1 = jlp_row[i_left], jlp_row[i_right]
    if mode == "linear":
        return jl0 + t * (jl1 - jl0), jp0 + t * (jp1 - jp0)
    raise ValueError(f"unknown Bessel lookup mode {mode!r}")
```

- [ ] **Step 5: Use them in `compute_cls_all_fast`.**
  - Change the signature to `def compute_cls_all_fast(pt, params, bg, l_max=1500, n_k_fine=5000, tt_mode=None, k_interp_method="spline", *, prec=None, _jl_lookup="linear"):`.
  - Add to its docstring Args: `prec: PrecisionParams of the perturbation solve; required to jit or differentiate (fixes the Bessel table's x-range from pt_k_max_cl and hr_tau0_max). _jl_lookup: internal, measurement only (#54).`
  - Replace `tau_0 = bg.conformal_age` with:

```python
    k_bound, tau0_max = _table_bounds(pt, prec)
    x_max, n_x = _table_grid(k_bound, tau0_max)
    tau_0 = eqx.error_if(
        bg.conformal_age, bg.conformal_age > tau0_max,
        f"compute_cls_all_fast: conformal age tau_0 exceeds PrecisionParams.hr_tau0_max "
        f"= {tau0_max:g} Mpc, the Bessel table's design bound; raise hr_tau0_max.")
```

  - Directly after `x_query = k_fine[:, None] * chi_grid[None, :]`, insert:

```python
    x_query = eqx.error_if(
        x_query, jnp.max(x_query) > x_max,
        f"compute_cls_all_fast: k*chi exceeds the Bessel table (x_max = {x_max:g}); "
        "prec.pt_k_max_cl is smaller than the perturbation grid's k_max -- pass the "
        "prec used for the perturbation solve.")
```

  - Replace everything from `# 3. Build j_l and j_l' tables…` through the end of the `scan_body` definition and the `jax.lax.scan(...)` call with:

```python
    # 3. j_l and j_l' tables on the fixed x-range (#54): independent of cosmology
    x_table, jl_table, jlp_table = _jl_tables(l_max, n_x, x_max)
    dx_tab = x_max / (n_x - 1)

    # 4. Table positions (shared across all l and both tables)
    i_left, t_pos = _table_position(x_flat_query, dx_tab, n_x)

    # 5. Compute transfer functions via table lookup (scan over l)
    l_sparse = sparse_l_grid(l_max)
    n_l = len(l_sparse)
    l_sparse_jnp = jnp.array(l_sparse, dtype=jnp.float64)

    # Precompute l-dependent constants for radT2 and EE
    ll1 = l_sparse_jnp * (l_sparse_jnp + 1.0)  # l(l+1) for radT2
    ee_prefactors = jnp.sqrt(
        l_sparse_jnp * (l_sparse_jnp + 1.0) *
        (l_sparse_jnp - 1.0) * (l_sparse_jnp + 2.0)
    )

    # jax.checkpoint: the backward pass recomputes each l's lookups instead of
    # storing ~(n_l x n_k x n_tau) residuals (~46 GB at fit_cl; #54).
    @jax.checkpoint
    def scan_body(carry, l_idx):
        jl_flat, jlp_flat = _lookup_jl(
            jl_table[l_idx], jlp_table[l_idx], l_sparse_jnp[l_idx], ll1[l_idx],
            i_left, t_pos, dx_tab, _jl_lookup)

        # Compute radial_T2 on-the-fly from j_l and j_l':
        # j_l''(x) = [l(l+1)/x² - 1]*j_l - 2/x*j_l'
        # radial_T2 = 0.5*(3*j_l'' + j_l) = 0.5*([3*l(l+1)/x² - 2]*j_l - 6/x*j_l')
        # cf. CLASS transfer.c:4168-4190
        rT2_flat = 0.5 * ((3.0 * ll1[l_idx] * inv_x2_flat - 2.0) * jl_flat
                          - 6.0 * inv_x_flat * jlp_flat)

        jl_interp = jl_flat.reshape(n_k_fine, n_tau)        # (n_k, n_tau)
        jlp_interp = jlp_flat.reshape(n_k_fine, n_tau)
        rT2_interp = rT2_flat.reshape(n_k_fine, n_tau)

        # TT transfer: T_l(k) = Σ_τ [S_T0*j_l + S_T1*j_l' + (S_T2/8)*radial_T2] * dτ
        # cf. CLASS harmonic.c:962 (T_total = T0 + T1 + T2)
        T_l = jnp.sum(src_T0_dtau * jl_interp
                      + src_T1_dtau * jlp_interp
                      + src_T2_dtau * rT2_interp, axis=-1)  # (n_k,)
        # EE transfer: E_l(k) = prefactor * Σ_τ source_E * j_l/x² * dτ
        E_l = ee_prefactors[l_idx] * jnp.sum(src_E_dtau * jl_interp, axis=-1)

        return carry, (T_l, E_l)

    _, (T_all, E_all) = jax.lax.scan(scan_body, None, jnp.arange(n_l))
```

  - Delete the now-unused `idx_right`, `idx_left`, `idx_right_safe`, `dx`, `dx_safe` and `frac` lines. Check: `grep -n "idx_right\|frac\b" clax/harmonic.py` must show no reader left in `compute_cls_all_fast`.

- [ ] **Step 6: Green.** Run the guard tests on the worktree, CPU job (`PYTEST_ARGS='tests/test_cl_table_guard.py'`). Expected: 4 passed.

- [ ] **Step 7: V2 refactor identity, a script gate.** Write `/lustre/work/n2minh/clax-probes/cl/v2_unlensed.py`. At fiducial `fit_cl`, with an eager pipeline (`solve_background_and_thermo` with `th_grad_mode="native"`, then `perturbations_solve`), it compares this tree's unlensed TT/EE/TE (ℓ ≤ 1500) under two settings:
  - **(a)** `prec` given, with `clax.harmonic._table_grid` monkeypatched to the pre-#54 range: `lambda kb, tm: (float(kb_fine_max * tau0_actual * 1.05), max(30000, int(kb_fine_max * tau0_actual * 1.05 * 3)))`, where `kb_fine_max = float(pt.k_grid[-1])` and `tau0_actual = float(bg.conformal_age)` are computed eagerly first;
  - **(b)** the unpatched fixed range.

  It does the same on the base snapshot (`base-t2`, no `prec`). It prints max |ΔC_ℓ/C_ℓ| for (a) vs base, which must be ≤ 1e-10, and for (b) vs base, which is reported. Run both trees as GPU jobs via `run.sbatch` (`SCRIPT=…`). If (a) exceeds 1e-10, STOP: the refactor changed numbers.

- [ ] **Step 8: Gate.** Run `tests/ --fast -q` as a CPU job. The only allowed failure is the pre-existing one.

- [ ] **Step 9: Commit.** Paths: `clax/params.py clax/harmonic.py tests/test_cl_table_guard.py`. Subject: `fix(harmonic): fixed-range Bessel table, tau0/k guards, checkpointed l-step (#54)`. The body carries the red and green job IDs and the V2 numbers.

---

### Task 3: Cubic Hermite lookup (L0)

**Files:**
- Modify: `clax/harmonic.py` (`_lookup_jl` gains `"hermite"`; add `_hermite` and `_jl_node_second_derivative`)
- Create: `tests/test_cl_bessel_lookup.py`
- Create, not committed: `/lustre/work/n2minh/clax-probes/cl/l0_measure.py`

**Interfaces:**
- **Consumes:** `_jl_tables`, `_table_position`, `_lookup_jl` (Task 2).
- **Produces:** `_lookup_jl(..., mode="hermite")`; `_hermite(f0, f1, d0, d1, t, h)`; `_jl_node_second_derivative(jl_n, jlp_n, x_n, ll1_val, is_l2)`.

- [ ] **Step 1: Implement the Hermite branch.** In `clax/harmonic.py`, above `_lookup_jl`, add:

```python
def _hermite(f0, f1, d0, d1, t, h):
    """Cubic Hermite interpolant on [x_i, x_i + h] from values f and slopes d
    at both ends; t = (x - x_i)/h in [0, 1]. C^1 across intervals (#54)."""
    t2 = t * t
    t3 = t2 * t
    return ((2.0 * t3 - 3.0 * t2 + 1.0) * f0 + (t3 - 2.0 * t2 + t) * h * d0
            + (-2.0 * t3 + 3.0 * t2) * f1 + (t3 - t2) * h * d1)


def _jl_node_second_derivative(jl_n, jlp_n, x_n, ll1_val, is_l2):
    """j_l'' at table nodes from Bessel's equation,
    j_l'' = [l(l+1)/x^2 - 1] j_l - (2/x) j_l', with the x -> 0 limit
    (2/15 for l = 2, 0 for l >= 3)."""
    small = x_n < 1e-10
    x_safe = jnp.where(small, 1.0, x_n)
    jpp = (ll1_val / (x_safe * x_safe) - 1.0) * jl_n - 2.0 / x_safe * jlp_n
    return jnp.where(small, jnp.where(is_l2, 2.0 / 15.0, 0.0), jpp)
```

In `_lookup_jl`, before the `raise`, add:

```python
    if mode == "hermite":
        x0 = i_left.astype(t.dtype) * dx_tab
        x1 = x0 + dx_tab
        is_l2 = l_val == 2.0
        jpp0 = _jl_node_second_derivative(jl0, jp0, x0, ll1_val, is_l2)
        jpp1 = _jl_node_second_derivative(jl1, jp1, x1, ll1_val, is_l2)
        return (_hermite(jl0, jl1, jp0, jp1, t, dx_tab),
                _hermite(jp0, jp1, jpp0, jpp1, t, dx_tab))
```

- [ ] **Step 2: Measure L0 errors.** Write `l0_measure.py`:
  - **Setup:** imports `scipy.special.spherical_jn`; builds `x_max, n_x = _table_grid(1.0, 20000.0)` and the tables `_jl_tables(1500, n_x, x_max)`.
  - **Samples:** for l ∈ {2, 30, 300, 1200}, 20000 seeded-uniform x in [0, x_max] plus 2000 x uniform in [l−2, l+2] (clipped at 0).
  - **For each mode ∈ {linear, hermite}:**
    - max abs error of the j_l lookup against `spherical_jn(l, x)`;
    - max abs error of the j_l′ lookup against `spherical_jn(l, x, derivative=True)`;
    - max abs error of the AD derivative of the j_l lookup with respect to x, against `spherical_jn(l, x, derivative=True)`. Use `jax.vmap(jax.grad(lambda x: lookup(x)[0]))`; `lookup` calls `_table_position` and `_lookup_jl`.
  - **Output:** one line per (mode, quantity), plus the overall worst per (mode, quantity).

  Run it as a CPU job via `run.sbatch`. Record the numbers.

- [ ] **Step 3: Set the bars by the spec rule and write the test.**
  - For each quantity: `BAR = 2 × hermite_worst`, rounded up to 2 significant figures.
  - If any `BAR ≥ linear_worst` for that quantity, STOP: the linear lookup would not be red.

  Write `tests/test_cl_bessel_lookup.py`, filling the three constants from Step 2 (with their job ID in a comment):

```python
"""L0 (#54): the Bessel-table lookup in compute_cls_all_fast against direct
spherical Bessel functions, including x within +-2 of l where the table switches
from backward to upward recurrence. Cosmology-independent numerics: exempt from
the multi-cosmology rule. Bars = 2x the Hermite lookup's measured worst error
(l0_measure.py, job <ID>); the old linear lookup fails them."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import spherical_jn

from clax.harmonic import _jl_tables, _lookup_jl, _table_grid, _table_position
from clax.bessel import sparse_l_grid

L_MAX = 1500
BAR_JL = ...    # 2 x hermite worst |j_l lookup - j_l|          (job <ID>)
BAR_JLP = ...   # 2 x hermite worst |j_l' lookup - j_l'|        (job <ID>)
BAR_DJL = ...   # 2 x hermite worst |d/dx j_l lookup - j_l'|    (job <ID>)


@pytest.fixture(scope="module")
def tables():
    x_max, n_x = _table_grid(1.0, 20000.0)
    _, jl_t, jlp_t = _jl_tables(L_MAX, n_x, x_max)
    return x_max, n_x, jl_t, jlp_t


def _lookup(tables, l, x, mode):
    x_max, n_x, jl_t, jlp_t = tables
    li = int(np.nonzero(sparse_l_grid(L_MAX) == l)[0][0])
    dx_tab = x_max / (n_x - 1)
    i_left, t = _table_position(x, dx_tab, n_x)
    return _lookup_jl(jl_t[li], jlp_t[li], jnp.float64(l), jnp.float64(l * (l + 1.0)),
                      i_left, t, dx_tab, mode)


def _samples(l, x_max):
    rng = np.random.default_rng(l)
    near = np.clip(rng.uniform(l - 2.0, l + 2.0, 2000), 0.0, None)
    return np.concatenate([rng.uniform(0.0, x_max, 20000), near])


@pytest.mark.parametrize("l", [2, 30, 300, 1200])
def test_hermite_lookup_matches_spherical_bessel(tables, l):
    x = _samples(l, tables[0])
    jl, jlp = _lookup(tables, l, jnp.asarray(x), "hermite")
    e_jl = float(np.max(np.abs(np.asarray(jl) - spherical_jn(l, x))))
    e_jlp = float(np.max(np.abs(np.asarray(jlp) - spherical_jn(l, x, derivative=True))))
    print(f"l={l}: |dj_l| {e_jl:.2e} (bar {BAR_JL:.1e}), |dj_l'| {e_jlp:.2e} (bar {BAR_JLP:.1e})")
    assert e_jl <= BAR_JL, f"ERROR j_l lookup {e_jl:.2e} > {BAR_JL:.1e} (l={l})"
    assert e_jlp <= BAR_JLP, f"ERROR j_l' lookup {e_jlp:.2e} > {BAR_JLP:.1e} (l={l})"


@pytest.mark.parametrize("l", [2, 30, 300, 1200])
def test_ad_derivative_of_lookup_is_jl_prime(tables, l):
    x = _samples(l, tables[0])
    d = jax.vmap(jax.grad(lambda xx: _lookup(tables, l, xx, "hermite")[0]))(jnp.asarray(x))
    e = float(np.max(np.abs(np.asarray(d) - spherical_jn(l, x, derivative=True))))
    print(f"l={l}: |d/dx lookup - j_l'| {e:.2e} (bar {BAR_DJL:.1e})")
    assert np.all(np.isfinite(np.asarray(d))), f"ERROR non-finite AD derivative (l={l})"
    assert e <= BAR_DJL, f"ERROR AD derivative of lookup {e:.2e} > {BAR_DJL:.1e} (l={l})"
```

  The three `...` are replaced in this step by the numeric bars measured in Step 2, so the committed file has concrete numbers. These are the only values this plan leaves to measurement, and they are fixed by the stated rule.

- [ ] **Step 4: Red and green.**
  - **Red:** in a scratch copy, run the test with `"hermite"` replaced by `"linear"` in both test functions. It must fail on the AD-derivative test and on at least one value test.
  - **Green:** the test as committed must pass. Run it as a CPU job (`PYTEST_ARGS='tests/test_cl_bessel_lookup.py'`).

- [ ] **Step 5: Commit.** Paths: `clax/harmonic.py tests/test_cl_bessel_lookup.py`. Subject: `feat(harmonic): cubic Hermite Bessel lookup (switchable, measurement pending) (#54)`. The body carries the l0_measure table and the red/green job IDs.

---

### Task 4: Measure (D5), decide, remove the switch

**Files:**
- Create, not committed: `/lustre/work/n2minh/clax-probes/cl/d5_measure.py`
- Modify (after the decision): `clax/harmonic.py` (remove `_jl_lookup` and the linear branch)

**Interfaces:**
- **Consumes:** `compute_cls_all_fast(..., prec=, _jl_lookup=)`.
- **Produces:**
  - a decision recorded in the commit body and report;
  - after it, `compute_cls_all_fast(pt, params, bg, l_max=1500, n_k_fine=5000, tt_mode=None, k_interp_method="spline", *, prec=None)`, Hermite only;
  - `_lookup_jl(jl_row, jlp_row, l_val, ll1_val, i_left, t, dx_tab)`, with no mode argument.

- [ ] **Step 1: The measurement script** `d5_measure.py`. Usage: `python d5_measure.py <mode> <quantity>`, one quantity per process so each peak-memory reading is clean. Build it as follows.
  - **Setup.** `FIT = PrecisionParams.fit_cl()`. `FWD = replace(FIT, th_grad_mode="native", ode_adjoint="direct")`. Use `CosmoParams()` as the fiducial. `L = 1200`.
  - **The functional.** For a prec `pr`:

```python
def F_tt(p, pr, mode):
    bg, th = solve_background_and_thermo(p, pr)
    pt = perturbations_solve(p, pr, bg, th)
    cl = compute_cls_all_fast(pt, p, bg, l_max=L, n_k_fine=pr.hr_n_k_fine, prec=pr, _jl_lookup=mode)
    return jnp.sum(jnp.log(cl["tt"][30:L + 1]))
```

  - **Quantities.**
    - **`fwd`:** median of 5 timings of `jax.jit(lambda p: F_tt(p, FWD, mode))`, after one compile call.
    - **`grad`:** median of 3 timings of `jax.jit(jax.grad(lambda p: F_tt(p, FIT, mode)))` over the whole `as_traced(CosmoParams())` pytree, after one compile call. Then report `jax.devices()[0].memory_stats()["peak_bytes_in_use"]`.
    - **`stage1` / `stage2`:** peak memory of `jax.grad` of `sum(th.xe_of_loga.y)` (background and thermodynamics), and of `sum(pt.source_T0**2)` (plus perturbations).
    - **`fd`:** for θ ∈ {h, omega_b, omega_cdm}:
      - the forward `jax.jvp` of `x ↦ F_tt(replace(p, θ=x), FWD, mode)`;
      - central FD at relative steps 1e-3 and 3e-3 (eager);
      - print |AD/FD − 1| and whether the two FD estimates agree to 1e-4.
    - **`spread`:** θ = h, forward jvp at `pt_k_chunk_size` ∈ {8, 64} × jit/no-jit. Print (max − min)/|median|.
    - **`smooth`:** forward dF/dh at 7 points h ∈ linspace(0.64, 0.70) with `replace(FWD, bg_n_points=8000)`. Print the cubic-fit residual max|resid|/|median|.

  Every printed line starts with `D5 <mode> <quantity>:`.

- [ ] **Step 2: Forward time on the base,** for the D6 budget.
  - Add the quantity `fwd_base` to the script. It times the eager stage-wise unlensed C_ℓ forward the way the base must run it: no jit around `compute_cls_all_fast`, and no `prec` (the base has none).
  - Run it on the base snapshot, and run the same eager stage-wise timing (`fwd_eager`) on the fix in each mode.
  - All forward timings for the budget run in ONE GPU job on one node, interleaved base, linear, hermite, base, linear, hermite. Use a wrapper that runs the processes in sequence, switching `PYTHONPATH`.

- [ ] **Step 3: Run it.**
  - The timing job is one GPU job (`--gres=gpu:1 --time=6:00:00`).
  - Then, per mode, one GPU job each for `grad`, `stage1`, `stage2`, `fd`, `spread` and `smooth`. They may run concurrently.
  - Tabulate every number for both modes.

- [ ] **Step 4: Decide by the spec's rule.** Hermite is adopted if all of these hold:
  - forward `fwd_eager(hermite)` ≤ 1.05 × `fwd_base`;
  - the `grad` peak ≤ 32 GB, and its time ≤ 3 × `fwd` (both jitted, so like is compared with like);
  - |AD/FD − 1| ≤ 1e-3 wherever the FD steps agree;
  - spread < 3e-4;
  - smoothness < 3e-4.

  STOP and report BLOCKED with both tables if:
  - any condition fails for hermite;
  - `stage2`'s peak alone exceeds 32 GB (a design question).

  Linear's numbers are reported for comparison; they do not gate.

- [ ] **Step 5: Remove the switch,** once Hermite is adopted.
  - In `_lookup_jl`: drop the `mode` parameter, the linear branch and the `raise`, keeping the Hermite body.
  - In `compute_cls_all_fast`: drop `_jl_lookup` and its docstring line, and call `_lookup_jl(...)` without the mode.
  - In `tests/test_cl_bessel_lookup.py`: drop the mode argument in `_lookup`.
  - Check: `grep -n "_jl_lookup\|\"linear\"" clax/harmonic.py tests/test_cl_bessel_lookup.py` must return nothing.
  - Run `tests/test_cl_bessel_lookup.py tests/test_cl_table_guard.py` (CPU job) and the `--fast` gate.

- [ ] **Step 6: Commit.** Paths: `clax/harmonic.py tests/test_cl_bessel_lookup.py`. Subject: `fix(harmonic): adopt the Hermite Bessel lookup after the D5 measurement (#54)`. The body carries the full D5 table (both modes, every quantity, job IDs) and the rule check.

---

### Task 5: Unlensed C_ℓ gradient seams G1–G4 (+ Review Focus 1, 4)

**Files:**
- Create: `tests/cl_test_utils.py`, `tests/test_cl_gradients.py`

**Interfaces:**
- **Consumes:**
  - `compute_cls_all_fast(..., prec=)`;
  - `as_traced` from `tests.recomb_test_utils`;
  - the fixture `recomb_downstream_cosmology`.
- **Produces,** in `tests/cl_test_utils.py`:
  - `CL_LMIN = 30` and `CL_LMAX = 1200`;
  - `FIT` and `fwd(prec)`;
  - `F_tt(prec, base, theta)`, `F_pp(prec, base, theta, nonlinear)` and `F_ltt(prec, base, theta, nonlinear)`;
  - `jvp_scalar(F, x0)`;
  - `spread(values)` and `cubic_residual(xs, ys)`;
  - `converged_fd(F, x0) -> (fd, converged: bool)`.

- [ ] **Step 1: Helpers** `tests/cl_test_utils.py`:

```python
"""Shared helpers for the C_l differentiability tests (smsharma/clax#54).
Functionals sum ln C_l over l in [30, 1200], avoiding fit_cl's low-l instability."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from clax import PrecisionParams
from clax.harmonic import compute_cls_all_fast
from clax.lensing import compute_cl_pp, lens_cls
from clax.perturbations import perturbations_solve, perturbations_solve_mpk
from clax.thermodynamics import solve_background_and_thermo
from tests.recomb_test_utils import as_traced

CL_LMIN, CL_LMAX = 30, 1200
FIT = PrecisionParams.fit_cl()


def fwd(prec):
    """Forward-mode settings: native thermodynamics AD, direct ODE adjoint."""
    return replace(prec, th_grad_mode="native", ode_adjoint="direct")


def _solve(p, prec):
    bg, th = solve_background_and_thermo(p, prec)
    return bg, th, perturbations_solve(p, prec, bg, th)


def _unlensed(p, prec, bg, pt):
    return compute_cls_all_fast(pt, p, bg, l_max=CL_LMAX, n_k_fine=prec.hr_n_k_fine, prec=prec)


def F_tt(prec, base, theta):
    """F(theta) = sum_{l=30}^{1200} ln C_l^TT (unlensed)."""
    def F(x):
        p = replace(as_traced(base), **{theta: x})
        bg, th, pt = _solve(p, prec)
        return jnp.sum(jnp.log(_unlensed(p, prec, bg, pt)["tt"][CL_LMIN:CL_LMAX + 1]))
    return F


def _cl_pp(p, prec, bg, th, pt, nonlinear):
    pt_mpk = None if nonlinear == "none" else perturbations_solve_mpk(p, prec, bg, th)
    return compute_cl_pp(pt, p, bg, th, CL_LMAX, nonlinear=nonlinear, pt_mpk=pt_mpk, prec=prec)


def F_pp(prec, base, theta, nonlinear):
    """F(theta) = sum_{l=30}^{1200} ln C_l^phiphi."""
    def F(x):
        p = replace(as_traced(base), **{theta: x})
        bg, th, pt = _solve(p, prec)
        return jnp.sum(jnp.log(_cl_pp(p, prec, bg, th, pt, nonlinear)[CL_LMIN:CL_LMAX + 1]))
    return F


def F_ltt(prec, base, theta, nonlinear):
    """F(theta) = sum_{l=30}^{1200} ln (lensed C_l^TT)."""
    def F(x):
        p = replace(as_traced(base), **{theta: x})
        bg, th, pt = _solve(p, prec)
        cl = _unlensed(p, prec, bg, pt)
        pp = _cl_pp(p, prec, bg, th, pt, nonlinear)
        tt_l, _, _, _ = lens_cls(cl["tt"], cl["ee"], cl["te"], jnp.zeros_like(cl["tt"]), pp,
                                 l_max=CL_LMAX)
        return jnp.sum(jnp.log(tt_l[CL_LMIN:CL_LMAX + 1]))
    return F


def jvp_scalar(F, x0, jit=True):
    g = lambda x: jax.jvp(F, (x,), (jnp.asarray(1.0),))[1]
    return float((jax.jit(g) if jit else g)(jnp.asarray(x0)))


def spread(values):
    v = np.asarray(values, dtype=float)
    return float((v.max() - v.min()) / abs(np.median(v)))


def cubic_residual(xs, ys):
    ys = np.asarray(ys, dtype=float)
    resid = ys - np.polyval(np.polyfit(xs, ys, 3), xs)
    return float(np.max(np.abs(resid)) / abs(np.median(ys)))


def converged_fd(F, x0):
    """Central FD at relative steps 1e-3 and 3e-3; converged if they agree to 1e-4."""
    Fj = jax.jit(F)
    fds = []
    for rel in (1e-3, 3e-3):
        h = rel * abs(x0)
        fds.append((float(Fj(jnp.asarray(x0 + h))) - float(Fj(jnp.asarray(x0 - h)))) / (2 * h))
    return fds[0], abs(fds[0] - fds[1]) <= 1e-4 * abs(fds[0])
```

- [ ] **Step 2: Tests** `tests/test_cl_gradients.py`:

```python
"""G1-G4 (smsharma/clax#54): d/dtheta of F = sum_{l=30}^{1200} ln C_l^TT at fit_cl
is independent of solver step placement, agrees between forward and production
reverse mode, is smooth along h, and matches converged finite differences.
GPU (each point is a full fit_cl solve under AD). Multi-cosmology:
recomb_downstream_cosmology (3 points)."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.cl_test_utils import FIT, F_tt, converged_fd, cubic_residual, fwd, jvp_scalar, spread
from tests.recomb_test_utils import as_traced

pytestmark = pytest.mark.slow
THETAS = ("h", "omega_b", "omega_cdm")


@pytest.mark.parametrize("theta", THETAS)
def test_g1_step_placement(recomb_downstream_cosmology, theta):
    name, base = recomb_downstream_cosmology
    x0 = float(getattr(base, theta))
    v = [jvp_scalar(F_tt(replace(fwd(FIT), pt_k_chunk_size=c), base, theta), x0, jit=j)
         for c in (8, 64) for j in (True, False)]
    s = spread(v)
    print(f"{name} dF/d{theta}: {v} spread {s:.3e}")
    assert s < 3e-4, f"ERROR C_l step-placement spread {s:.3e} ({name}, {theta})"


@pytest.mark.parametrize("theta", THETAS)
def test_g2_production_reverse_matches_forward(recomb_downstream_cosmology, theta):
    name, base = recomb_downstream_cosmology
    x0 = float(getattr(base, theta))
    f = jvp_scalar(F_tt(fwd(FIT), base, theta), x0)
    r = float(jax.grad(F_tt(FIT, base, theta))(jnp.asarray(x0)))
    rel = abs(r / f - 1.0)
    print(f"{name} dF/d{theta}: forward {f:.8e} reverse {r:.8e} rel {rel:.3e}")
    assert np.isfinite(r) and rel < 3e-4, f"ERROR reverse vs forward {rel:.3e} ({name}, {theta})"


def test_g3_smooth_along_h(recomb_downstream_cosmology):
    name, base = recomb_downstream_cosmology
    hs = np.linspace(0.64, 0.70, 7)
    prec = replace(fwd(FIT), bg_n_points=8000)    # 800-point background floor (clax-72)
    d = [jvp_scalar(F_tt(prec, replace(base, h=float(h)), "h"), float(h)) for h in hs]
    r = cubic_residual(hs, d)
    print(f"{name}: dF/dh along h {d} cubic residual {r:.3e}")
    assert r < 3e-4, f"ERROR dF/dh not smooth along h: {r:.3e} ({name})"


@pytest.mark.parametrize("theta", THETAS)
def test_g4_ad_matches_converged_fd(recomb_downstream_cosmology, theta):
    name, base = recomb_downstream_cosmology
    x0 = float(getattr(base, theta))
    F = F_tt(fwd(FIT), base, theta)
    ad = jvp_scalar(F, x0)
    fd, ok = converged_fd(F, x0)
    rel = abs(ad / fd - 1.0)
    print(f"{name} dF/d{theta}: AD {ad:.8e} FD {fd:.8e} rel {rel:.3e} FD converged {ok}")
    assert ok, f"ERROR FD not converged (steps 1e-3, 3e-3 disagree > 1e-4) ({name}, {theta})"
    assert rel <= 1e-3, f"ERROR AD vs FD {rel:.3e} ({name}, {theta})"


def test_rf1_jit_of_grad_equals_grad(recomb_downstream_cosmology):
    name, base = recomb_downstream_cosmology
    F = F_tt(FIT, base, "h")
    x0 = jnp.asarray(float(base.h))
    a, b = float(jax.grad(F)(x0)), float(jax.jit(jax.grad(F))(x0))
    print(f"{name}: grad {a:.10e} jit(grad) {b:.10e}")
    assert abs(a / b - 1.0) < 1e-10, f"ERROR jit(grad) differs from grad ({name})"


def test_rf4_vmap_matches_loop(recomb_downstream_cosmology):
    name, base = recomb_downstream_cosmology
    F = jax.jit(F_tt(fwd(FIT), base, "h"))
    hs = float(base.h) * jnp.array([0.99, 1.0, 1.01])
    a = np.asarray(jax.vmap(F)(hs))
    b = np.array([float(F(h)) for h in hs])
    d = float(np.max(np.abs(a - b) / np.abs(b)))
    print(f"{name}: vmap vs loop max rel {d:.3e}")
    assert d < 1e-10, f"ERROR vmap vs loop {d:.3e} ({name})"
```

- [ ] **Step 3: Red on the base.** Make `base-t5` (archive 2717b5c, plus `tests/cl_test_utils.py`, `tests/test_cl_gradients.py` and the updated `tests/conftest.py` copied in). Run one GPU job (`PYTEST_ARGS='tests/test_cl_gradients.py -m slow -k lcdm_fiducial'`). Expected: every test errors, because `prec` is an unexpected keyword. Record the job ID and output.

- [ ] **Step 4: Green.** Run GPU jobs on the worktree, one per cosmology (`-k lcdm_fiducial`, `-k omega_b_high`, `-k YHe_028`), with `--time=12:00:00`. All must pass. If a bar is missed, STOP and report BLOCKED with the numbers.

- [ ] **Step 5: Commit.** Paths: `tests/cl_test_utils.py tests/test_cl_gradients.py`. Subject: `test(cl): C_l gradients at fit_cl are step-placement-invariant, mode-consistent, smooth and FD-exact (#54)`.

---

### Task 6: Lensing (D4 with amendment A2) and G5

**Files:**
- Modify: `clax/lensing.py` (`_halofit_modulator`, `_ept_modulator`, `compute_cl_pp`)
- Create: `tests/test_cl_lensing_gradients.py`
- Create, not committed: `/lustre/work/n2minh/clax-probes/cl/v2_lensing.py`

**Interfaces:**
- **Consumes:** `F_pp`, `F_ltt`, `fwd`, `FIT`, `jvp_scalar` and `converged_fd` from `tests/cl_test_utils.py`.
- **Produces:**
  - `compute_cl_pp(pt, params, bg, th, l_max, *, nonlinear="none", pt_mpk=None, prec=None)`;
  - `_halofit_modulator(..., *, pt_mpk, k_pt_max=None)`;
  - `_ept_modulator(..., *, pt_mpk)`, with smooth fixed shapes.

- [ ] **Step 1: Failing tests** `tests/test_cl_lensing_gradients.py`:

```python
"""G5 (smsharma/clax#54): gradients of sum ln C_l^phiphi and sum ln (lensed C_l^TT),
l in [30, 1200], at fit_cl for nonlinear none / halofit / ept: finite under jit,
production reverse matches forward, AD matches converged FD. GPU.
Multi-cosmology: recomb_downstream_cosmology (3 points)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.cl_test_utils import FIT, F_ltt, F_pp, converged_fd, fwd, jvp_scalar

pytestmark = pytest.mark.slow
FUNCS = {"pp": F_pp, "lensed_tt": F_ltt}


@pytest.mark.parametrize("nonlinear", ["none", "halofit", "ept"])
@pytest.mark.parametrize("func", list(FUNCS))
@pytest.mark.parametrize("theta", ["h", "omega_b"])
def test_g5_lensing_gradients(recomb_downstream_cosmology, func, nonlinear, theta):
    name, base = recomb_downstream_cosmology
    x0 = float(getattr(base, theta))
    Ff = FUNCS[func](fwd(FIT), base, theta, nonlinear)
    f = jvp_scalar(Ff, x0)
    r = float(jax.grad(FUNCS[func](FIT, base, theta, nonlinear))(jnp.asarray(x0)))
    fd, ok = converged_fd(Ff, x0)
    rel_mode, rel_fd = abs(r / f - 1.0), abs(f / fd - 1.0)
    print(f"{name} {func}/{nonlinear} d/d{theta}: fwd {f:.8e} rev {r:.8e} fd {fd:.8e} "
          f"(conv {ok}) rev/fwd {rel_mode:.2e} ad/fd {rel_fd:.2e}")
    assert np.isfinite(f) and np.isfinite(r), f"ERROR non-finite ({name}, {func}, {nonlinear}, {theta})"
    assert rel_mode < 3e-4, f"ERROR reverse vs forward {rel_mode:.3e} ({name}, {func}, {nonlinear}, {theta})"
    assert ok, f"ERROR FD not converged ({name}, {func}, {nonlinear}, {theta})"
    assert rel_fd <= 1e-3, f"ERROR AD vs FD {rel_fd:.3e} ({name}, {func}, {nonlinear}, {theta})"
```

- [ ] **Step 2: Red.** Run on the worktree as it stands after Task 5, as a GPU job with `-k "lcdm_fiducial and (halofit or ept)"`. Expected: FAIL. Either `prec` is an unexpected keyword to `compute_cl_pp`, or the run hits `ConcretizationTypeError` from `float(th.z_rec)` / `float(params.h)`. Record it.

- [ ] **Step 3: Halofit correction.** In `_halofit_modulator`:
  - Add `import math` at the top of `clax/lensing.py`.
  - Change the signature's keyword-only part to `*, pt_mpk: MatterPerturbationResult, k_pt_max: float | None = None,`.
  - Add a docstring Arg: `k_pt_max: static k_max of pt.k_grid (prec.pt_k_max_cl); None reads it from pt.k_grid (eager callers only).`
  - Replace the lines from `pt_k_max = float(pt.k_grid[-1])` through `log1pz_grid = jnp.linspace(jnp.log(1.0), jnp.log(1.0 + z_rec), n_z)` with:

```python
    # --- Extended k-grid: decision and size from the static k_max (#54) ---
    if k_pt_max is None:
        k_pt_max = float(pt.k_grid[-1])          # eager callers only
    k_max_use = max(k_pt_max, k_max_extend)
    if k_pt_max < k_max_extend:
        n_extra = max(int(20.0 * math.log10(k_max_use / k_pt_max)), 10)
        k_extra = jnp.logspace(
            jnp.log10(pt.k_grid[-1]), jnp.log10(k_max_use), n_extra + 1)[1:]
        k_ext = jnp.concatenate([pt.k_grid, k_extra])
    else:
        k_ext = pt.k_grid

    # --- z-grid spanning [0, z_rec], log-spaced in 1+z; z_rec stays traced ---
    z_rec = th.z_rec
    log1pz_grid = jnp.linspace(0.0, jnp.log1p(z_rec), n_z)
```

- [ ] **Step 4: EPT correction (amendment A2).** In `_ept_modulator`, replace everything from `h = float(params.h)` through the line ending `R0_pt, 1.0)` (the end of the old `R0_pt = jnp.where(...)` block) with:

```python
    h = params.h                                   # traced (#54)

    # --- P_NL at the perturbation k (Mpc^-1): cubic spline in log(k/h) over the
    # fixed EPT nodes, then R0 = P_NL / P_lin at the exact pt k. Fixed shapes:
    # no h-dependent node selection, so R0 is smooth in h (#54). ---
    ept = compute_ept_from_clax(params, bg, pt_mpk, z=0.0, field="m")
    pk_mm_h = pk_mm_real(ept)                      # (Mpc/h)^3 on the EPT k-grid (h/Mpc)
    log_k_h = jnp.log(jnp.asarray(ept_kgrid()))    # static nodes
    log_q = jnp.log(pt.k_grid) - jnp.log(h)        # pt k in h/Mpc, log
    inside = (log_q >= log_k_h[0]) & (log_q <= log_k_h[-1])
    log_q_c = jnp.clip(log_q, log_k_h[0], log_k_h[-1])
    pk_nl_pt = CubicSpline(log_k_h, pk_mm_h).evaluate(log_q_c) / h**3     # Mpc^3
    pk_lin_pt = compute_linear_matter_pk_from_perturbations(
        pt_mpk, bg, params, pt.k_grid, z=0.0)
    R0_pt = jnp.where(inside & (pk_lin_pt > 0), pk_nl_pt / pk_lin_pt, 1.0)
    R0_pt = jnp.clip(R0_pt, 0.0, 100.0)
```

  Keep the growth-factor rescaling below it unchanged. Update the docstring paragraph that described the spline of R0 so it says this.

- [ ] **Step 5: `compute_cl_pp`.**
  - Add the keyword `prec: PrecisionParams | None = None` after `pt_mpk`, with its docstring Arg: `prec: PrecisionParams of the perturbation solve; required to jit or differentiate with nonlinear="halofit" (static k-range).`
  - Add `from clax.params import PrecisionParams` to the imports if absent.
  - Change the Halofit call to `_halofit_modulator(pt, params, bg, th, pt_mpk=pt_mpk, k_pt_max=None if prec is None else float(prec.pt_k_max_cl))`.

- [ ] **Step 6: V2 for Halofit and the EPT gate, a script.** `v2_lensing.py` runs at fiducial `fit_cl`, eager, on both the worktree and `base-t6` (an archive of 2717b5c):
  - **(a) Halofit:** the `_halofit_modulator` output, fix vs base. Must be ≤ 1e-10 (V2).
  - **(b) EPT:** the ratio R0 = (modulator)² at τ nearest today, fix vs base: max |ΔR/R| for pt k ∈ [1e-4, pt_k_max]. Must be ≤ 1e-3 (A2).
  - **(c) EPT C_ℓ^φφ:** fix vs base, max |ΔC/C| per band 30–100, 100–500, 500–1200. Must be ≤ 1e-3 (A2).

  Both trees are GPU jobs. Any miss means STOP.

- [ ] **Step 7: Green.** Run G5 on the worktree, one GPU job per cosmology (`--time=24:00:00`; shard further by `-k` on `func` if needed). All must pass.

- [ ] **Step 8: Gate.** `tests/ --fast -q` CPU job, plus full mode of `tests/test_cl_pp.py tests/test_lensing.py tests/test_clpp_halofit_ratio.py` (CPU or GPU). Any failure is re-run on `base-t6`. A test that fails only on the fix means STOP.

- [ ] **Step 9: Commit.** Paths: `clax/lensing.py tests/test_cl_lensing_gradients.py`. Subject: `fix(lensing): differentiable C_l^phiphi and lensed C_l (traced z_rec, static k-range, smooth EPT ratio) (#54)`. The body carries the V2/A2 numbers and red/green.

---

### Task 7: Values against CLASS (V1), cost (P1), regression gate

**Files:**
- Create: `tests/data/cl_vs_class_baseline.json`, `tests/test_cl_vs_class.py`
- Create, not committed: `/lustre/work/n2minh/clax-probes/cl/v1_baseline.py` and `p1_bench.py`

**Interfaces:**
- **Consumes:**
  - `compute_cls_all_fast(..., prec=)` and `compute_cl_pp(..., prec=)`;
  - `lens_cls`;
  - the fixture `cl_value_cosmology`;
  - the references `cls.npz`, `cls_lensed.npz` and `cl_pp.npz`.
- **Produces:** the baseline JSON `{cosmology: {preset: {spectrum: {band: base_error}}}}`.

- [ ] **Step 1: Baseline on the base.** `v1_baseline.py <tree> <out.json>` runs at the 5 `CL_VALUE_GRID` cosmologies, eagerly (the base has no `prec`). For presets `fit_cl` and `planck_cl` it computes:
  - unlensed TT/EE/TE (ℓ ≤ 2500 for `planck_cl`, ≤ 1500 for `fit_cl`) via `compute_cls_all_fast(..., l_max=prec.hr_l_max, n_k_fine=prec.hr_n_k_fine)`;
  - C_ℓ^φφ via `compute_cl_pp(..., l_max)`;
  - lensed TT/EE/TE/BB via `lens_cls(tt, ee, te, zeros, pp, l_max)`.

  It writes max |ΔC/C| per band against `cls.npz`, `cl_pp.npz` and `cls_lensed.npz`. TE uses |ΔTE|/√(TT·EE). The bands are 2–30, 30–100, 100–500, 500–1500, plus 1500–2500 for `planck_cl`. Run it on `base-t7` (archive of 2717b5c) as GPU jobs, one per cosmology. Merge the outputs into `tests/data/cl_vs_class_baseline.json`, recording job IDs under a `"_provenance"` key.

- [ ] **Step 2: The test** `tests/test_cl_vs_class.py`:

```python
"""V1 (smsharma/clax#54): C_l (unlensed TT/EE/TE, C_l^phiphi, lensed TT/EE/TE/BB)
at fit_cl and planck_cl against CLASS is no worse than the pre-#54 code (2717b5c)
in any l band: fix error <= base error + 1e-4 (base errors in
tests/data/cl_vs_class_baseline.json). GPU. Multi-cosmology: cl_value_cosmology."""
import json
import os

import jax.numpy as jnp
import numpy as np
import pytest

from clax import PrecisionParams
from clax.harmonic import compute_cls_all_fast
from clax.lensing import compute_cl_pp, lens_cls
from clax.perturbations import perturbations_solve
from clax.thermodynamics import solve_background_and_thermo
from dataclasses import replace

pytestmark = pytest.mark.slow
HERE = os.path.dirname(__file__)
REF = os.path.join(HERE, "..", "reference_data")
BASE = json.load(open(os.path.join(HERE, "data", "cl_vs_class_baseline.json")))
PRESETS = {"fit_cl": PrecisionParams.fit_cl, "planck_cl": PrecisionParams.planck_cl}
BANDS = {"2-30": (2, 30), "30-100": (30, 100), "100-500": (100, 500),
         "500-1500": (500, 1500), "1500-2500": (1500, 2500)}


def _errors(name, p, prec):
    prec = replace(prec, th_grad_mode="native")
    bg, th = solve_background_and_thermo(p, prec)
    pt = perturbations_solve(p, prec, bg, th)
    lmax = prec.hr_l_max
    cl = compute_cls_all_fast(pt, p, bg, l_max=lmax, n_k_fine=prec.hr_n_k_fine, prec=prec)
    pp = compute_cl_pp(pt, p, bg, th, lmax, prec=prec)
    lt, le, lte, lb = lens_cls(cl["tt"], cl["ee"], cl["te"], jnp.zeros_like(cl["tt"]), pp, l_max=lmax)
    u = np.load(os.path.join(REF, name, "cls.npz"))
    ln = np.load(os.path.join(REF, name, "cls_lensed.npz"))
    rp = np.load(os.path.join(REF, name, "cl_pp.npz"))
    spectra = {
        "tt": (np.asarray(cl["tt"]), u["tt"]), "ee": (np.asarray(cl["ee"]), u["ee"]),
        "te": (np.asarray(cl["te"]), u["te"], np.sqrt(u["tt"] * u["ee"])),
        "pp": (np.asarray(pp), rp["pp"]),
        "ltt": (np.asarray(lt), ln["tt"]), "lee": (np.asarray(le), ln["ee"]),
        "lte": (np.asarray(lte), ln["te"], np.sqrt(ln["tt"] * ln["ee"])),
        "lbb": (np.asarray(lb), ln["bb"]),
    }
    out = {}
    for s, tup in spectra.items():
        mine, ref = tup[0], tup[1]
        norm = tup[2] if len(tup) == 3 else ref
        out[s] = {}
        for band, (lo, hi) in BANDS.items():
            hi = min(hi, lmax)
            if lo >= hi:
                continue
            sl = slice(lo, hi + 1)
            out[s][band] = float(np.max(np.abs(mine[sl] - ref[sl]) / np.abs(norm[sl])))
    return out


@pytest.mark.parametrize("preset", list(PRESETS))
def test_v1_no_worse_than_base_vs_class(cl_value_cosmology, preset):
    name, p = cl_value_cosmology
    errs = _errors(name, p, PRESETS[preset]())
    worst = []
    for s, bands in errs.items():
        for band, e in bands.items():
            b = BASE[name][preset][s][band]
            worst.append((e - b, s, band, e, b))
    w = max(worst)
    print(f"{name} {preset}: worst (fix - base) {w[0]:+.2e} at {w[1]} {w[2]} (fix {w[3]:.3e}, base {w[4]:.3e})")
    bad = [x for x in worst if x[0] > 1e-4]
    assert not bad, f"ERROR C_l worse than base vs CLASS: {bad[:3]} ({name}, {preset})"
```

- [ ] **Step 3: Run it (green).** Run the test on the worktree as GPU jobs, one per cosmology. Record per-band fix-vs-base shifts in the report. Any miss means STOP.

- [ ] **Step 4: P1 bench.** `p1_bench.py` repeats the D5 `fwd_eager` (both trees, interleaved, one node) and the `grad` time and peak (fix) at fiducial `fit_cl`, in one GPU job. It checks D6. A miss means STOP.

- [ ] **Step 5: Regression gate.** Run CPU jobs on the worktree:
  - `tests/ --fast -q`;
  - full mode of `tests/test_harmonic.py tests/test_lensing.py tests/test_cl_pp.py tests/test_clpp_halofit_ratio.py tests/test_chebyshev_interpolation.py tests/test_multicosmo_consistency.py tests/test_pk_accuracy.py`.

  Any failure is re-run on `base-t7`. A failure only on the fix means STOP.

- [ ] **Step 6: Commit.** Paths: `tests/data/cl_vs_class_baseline.json tests/test_cl_vs_class.py`. Subject: `test(cl): C_l, C_l^phiphi and lensed C_l no worse than before against CLASS (#54)`. The body carries the band tables, the P1 numbers and the gate summary.

---

### Task 8: ADR 0004, docstrings, CHANGELOG

**Files:**
- Create: `docs/adr/0004-cl-differentiability.md`
- Modify: `CHANGELOG.md`, `clax/harmonic.py` (docstring of `compute_cls_all_fast`), `clax/bessel.py` (docstring of `build_jl_table`), `clax/params.py` (the `hr_tau0_max` comment, if expanded)

- [ ] **Step 1: ADR 0004.** Use the same structure as ADR 0003: Context, Decision, Consequences, Status.
  - Date header `2026-10-18`, or the actual date.
  - Context: #54's two causes, the staircase derivative, and the lensing `float()` sites.
  - Decision: D1–D4 plus A1/A2, the τ₀ bound, and the D5 measurement with its numbers.
  - Consequences:
    - C_ℓ shifts per band (from Task 7);
    - cost (P1);
    - the τ₀ bound guard;
    - the `prec=` requirement for jit and grad;
    - the EPT construction change.

- [ ] **Step 2: Docstrings.**
  - `compute_cls_all_fast`: the Architecture list now says the table is fixed-range and cached, the lookup is cubic Hermite, and the ℓ-step is checkpointed. Add the `prec` requirement and the guard.
  - `build_jl_table`'s See-also: callers interpolate with a cubic Hermite lookup, not linear weights.

- [ ] **Step 3: CHANGELOG.** Add an entry above the first `### ` heading: `### <Mon D, YYYY>: C_l, C_l^phiphi and lensed C_l are differentiable at fit_cl (#54)`. It covers:
  - Bug, in physics terms;
  - Fix: D1–D4, A1, A2;
  - Measured: D5, G1–G5, V1 and P1 numbers;
  - Not done: the Out-of-Scope list.

  Keep it at about 100 lines or less, with no internal labels or commit hashes.

- [ ] **Step 4: Gate and commit.**
  - `git diff -- clax/` must be docstring-only (AST identical with docstrings stripped); run `py_compile`.
  - Commit paths: `docs/adr/0004-cl-differentiability.md CHANGELOG.md clax/harmonic.py clax/bessel.py clax/params.py`. Subject: `docs: ADR 0004 and CHANGELOG for C_l differentiability (#54)`.

---

## After the plan

- Report to the user with every measured number. Do **not** push or open a PR without approval. This PR is stacked on #53 and closes #54.
