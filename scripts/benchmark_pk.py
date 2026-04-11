"""Benchmark direct single-mode vs table-backed linear matter power workflows.

Usage:
    python scripts/benchmark_pk.py [preset]

This script measures cached execution time for three practical ``P(k)`` paths:

- repeated direct ``compute_pk()`` calls over a requested ``k`` array
- ``compute_pk_table()`` with explicit full-``vmap`` over the internal solve grid
- ``compute_pk_table()`` with backend-aware auto-batching
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import replace as dataclass_replace

sys.path.insert(0, ".")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import clax
from clax import CosmoParams, PrecisionParams
from clax.perturbations import (
    _build_indices,
    _mpk_tau_n_points,
    _resolve_pt_k_batch_size,
    _pt_saved_output_count,
)


def get_preset(name: str) -> PrecisionParams:
    """Return a named ``PrecisionParams`` preset."""
    if hasattr(PrecisionParams, name):
        return getattr(PrecisionParams, name)()
    raise ValueError(f"Unknown preset: {name}")


def direct_loop_pk(params: CosmoParams, prec: PrecisionParams, k_eval: jnp.ndarray) -> jnp.ndarray:
    """Evaluate ``compute_pk()`` repeatedly over a requested grid."""
    values = [clax.compute_pk(params, prec, float(k)) for k in np.asarray(k_eval)]
    return jnp.asarray(values)


def timed_cached(label: str, fn):
    """Warm up once, then time one cached execution."""
    warm = fn()
    jax.block_until_ready(warm)

    t0 = time.time()
    out = fn()
    jax.block_until_ready(out)
    dt = time.time() - t0
    return {"label": label, "seconds": dt, "value": out}


def run_benchmark(preset_name: str, num_eval: int) -> None:
    """Run the direct-vs-table benchmark for one preset."""
    prec = get_preset(preset_name)
    params = CosmoParams()
    k_eval = jnp.geomspace(prec.pt_k_min, min(prec.pt_k_max_cl, 1.0), num_eval)

    n_q_ncdm = prec.ncdm_q_size if params.N_ncdm > 0 else 0
    include_ncdm_fluid = prec.ncdm_fluid_approximation.lower() != "none"
    idx = _build_indices(
        prec.pt_l_max_g,
        prec.pt_l_max_pol_g,
        prec.pt_l_max_ur,
        n_q_ncdm,
        prec.pt_l_max_ncdm,
        include_ncdm_fluid=include_ncdm_fluid,
    )
    full_auto_batch = _resolve_pt_k_batch_size(
        prec,
        n_k=int(math.log10(prec.pt_k_max_cl / prec.pt_k_min) * prec.pt_k_per_decade),
        n_tau=prec.pt_tau_n_points,
        n_outputs=_pt_saved_output_count(solve_kind="full"),
        solve_kind="full",
    )
    mpk_auto_batch = _resolve_pt_k_batch_size(
        prec,
        n_k=int(math.log10(prec.pt_k_max_cl / prec.pt_k_min) * prec.pt_k_per_decade),
        n_tau=_mpk_tau_n_points(prec),
        n_outputs=_pt_saved_output_count(solve_kind="mpk"),
        solve_kind="mpk",
    )

    direct = timed_cached(
        "direct_loop",
        lambda: direct_loop_pk(params, prec, k_eval),
    )
    table_full = timed_cached(
        "table_full_vmap",
        lambda: clax.compute_pk_table(params, dataclass_replace(prec, pt_k_chunk_size=-1), k_eval=k_eval).pk_grid,
    )
    table_auto = timed_cached(
        "table_auto_batch",
        lambda: clax.compute_pk_table(params, dataclass_replace(prec, pt_k_chunk_size=0), k_eval=k_eval).pk_grid,
    )

    baseline = np.asarray(table_auto["value"])
    rel_vs_direct = float(np.max(np.abs(np.asarray(direct["value"]) / baseline - 1.0)))
    rel_vs_full = float(np.max(np.abs(np.asarray(table_full["value"]) / baseline - 1.0)))

    n_k_internal = int(math.log10(prec.pt_k_max_cl / prec.pt_k_min) * prec.pt_k_per_decade)

    print(f"Backend: {jax.devices()}")
    print(f"Preset: {preset_name}")
    print(f"Requested k points: {num_eval}")
    print(f"Internal perturbation k grid: {n_k_internal}")
    print(f"pt_k_chunk_size semantics: -1=full_vmap, 0=auto-batched")
    print(f"Resolved auto batch sizes: full={full_auto_batch}, mpk={mpk_auto_batch}")
    print()
    print(f"{'Path':<18} {'Cached time [s]':>16}")
    print(f"{direct['label']:<18} {direct['seconds']:16.3f}")
    print(f"{table_full['label']:<18} {table_full['seconds']:16.3f}")
    print(f"{table_auto['label']:<18} {table_auto['seconds']:16.3f}")
    print()
    print("Agreement vs auto-batched table:")
    print(f"  direct loop max rel diff:      {rel_vs_direct:.3e}")
    print(f"  full-vmap table max rel diff:  {rel_vs_full:.3e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("preset", nargs="?", default="fit_cl", help="Precision preset name")
    parser.add_argument("--num-eval", type=int, default=32, help="Number of requested output k points")
    args = parser.parse_args()

    run_benchmark(args.preset, args.num_eval)
