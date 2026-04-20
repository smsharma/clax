# Test Suite Status

## Summary

The `tests/` suite is organized by contract ownership. Each file should answer one clear question about one layer of the pipeline.

For linear matter power, the suite now distinguishes:
- forward scalar `P(k)` accuracy
- scalar partial-derivative accuracy
- public table/interpolator behavior and direct single-mode spot checks

`tests/pk_test_utils.py` provides both:
- thin public-table helpers for forward `P(k)` checks and interpolation-path gradient probes
- direct single-mode helpers for gradient checks and perturbation-level spot checks

## Current Ownership

- `test_end_to_end.py`
  - public API smoke only

- `test_background.py`
  - background value and gradient contracts

- `test_thermodynamics.py`
  - thermodynamics forward-value contracts

- `test_perturbations.py`
  - perturbation-layer invariants, cheap forward checks, direct single-mode `P(k)` spot checks, and matched species-level perturbation checks

- `test_pk_accuracy.py`
  - public table-backed scalar `P(k, z=0)` forward-accuracy contract on a sparse solve grid

- `test_pk_gradients.py`
  - direct scalar and public table-backed scalar `P(k)` partial-derivative contracts

- `test_harmonic.py`
  - scalar unlensed `C_l` forward/API checks

- `test_high_l.py`
  - high-`l` helper and consistency checks

- `test_lensing.py`
  - lensing forward behavior and lensed-spectrum checks

- `test_tensor.py`
  - tensor-mode forward checks with reduced-precision tolerances

- `test_nonlinear.py`
  - nonlinear `P(k)` behavior and local differentiability checks

- `test_multipoint.py`
  - non-fiducial regression points

## Linear `P(k)` Contracts

### Forward accuracy
- owner: `test_pk_accuracy.py`
- quantity: public table-backed scalar linear `P(k, z=0)` built from one sparse perturbation-table solve
- target range: `k <= 1 Mpc^-1`
- target tolerance: `<=1%` max relative error against CLASS

### Direct single-mode spot checks
- owner: `test_perturbations.py`
- quantity: direct scalar linear `P(k, z=0)` at a tiny fixed `k` probe set
- target range: deterministic low-to-high spot checks up to `1 Mpc^-1`
- target tolerance: `<=1%` max relative error against CLASS

### Species-level perturbation checks
- owner: `test_perturbations.py`
- quantity: matched `(k, tau)` comparisons for `delta_cdm`, `delta_b`, `delta_ncdm`, `theta_ncdm`, `shear_ncdm`, and derived `delta_m`
- target range: stored fiducial perturbation series at `k = 0.01, 0.05, 0.1 Mpc^-1`
- reference convention: generated with `ncdm_fluid_approximation = none` and `l_max_ncdm = 17`, so the matched-species test precision uses the same `ncdm` hierarchy depth
- target tolerance:
  - `delta_cdm`, `delta_b`, `delta_m`: low-percent regression envelope
  - `delta_ncdm`, `theta_ncdm`, `shear_ncdm`: sub-percent contract against the no-fluid CLASS perturbation reference

### Gradients
- owner: `test_pk_gradients.py`
- quantity:
  - direct scalar partial derivatives `dP/dtheta_i` on the stable primordial subset
  - public table-backed scalar interpolation-path finite differences in `--fast`, plus default-mode table-backed AD smoke checks
- parameters:
  - full mode: stable materially non-zero direct-path subset (`ln10A_s`, `n_s`, `k_pivot`)
  - density-parameter gradient coverage stays on the public table-backed path
  - public table path: stable subset chosen to keep finite-difference runtime bounded (`h` in full mode)
  - `--fast`: deterministic stable subset
- practical AD policy:
  - if an objective depends on several `k` points, prefer differentiating one
    `compute_pk_table()` solve and aggregating `result.pk_grid`
  - reserve repeated exact `compute_pk()` solves for local diagnostics and the
    small direct-path regression subset
- target range: deterministic low-to-high `k` probe set up to `1 Mpc^-1`
- target tolerance: `<=1%` relative error vs central finite differences where the derivative is materially non-zero
- null-response parameters use an absolute null check instead of a relative-error gate
- adjoint policy: gradient contracts run on the production/default checkpointed
  perturbation adjoint, not the optional `DirectAdjoint` variant
- rationale and environment-specific validation workflow: see `README.md` and
  `DESIGN.md` before using an alternate adjoint in local experiments or tests

## Execution Guidance

Do not run heavy JAX test files concurrently.

Recommended order:
1. cheap numerics and module-local tests
2. `test_pk_accuracy.py`
3. `test_perturbations.py`
4. `test_pk_gradients.py`
5. harmonic / high-`l`
6. lensing / tensor / nonlinear

Safe local pattern:

```bash
pytest tests/test_pk_accuracy.py -q --fast
pytest tests/test_perturbations.py -q --fast
pytest tests/test_pk_gradients.py -q --fast
pytest tests/ -q --fast
```

Run these serially, one pytest process at a time. `test_pk_gradients.py` now skips at module import when xdist launches more than one worker.
