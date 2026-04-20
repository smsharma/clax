"""Benchmark direct-loop vs table-backed multi-``k`` scalar ``P(k)`` gradients.

Usage:
    python scripts/benchmark_pk_gradients.py [preset]

This script compares two reverse-mode workload shapes for the same scalar
multi-``k`` objective:

- repeated direct ``compute_pk()`` solves inside the differentiated objective
- one ``compute_pk_table()`` solve reused across the requested ``k`` points
"""

from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, ".")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import clax
from clax import CosmoParams, PrecisionParams


def get_preset(name: str) -> PrecisionParams:
    """Return a named ``PrecisionParams`` preset."""
    if hasattr(PrecisionParams, name):
        return getattr(PrecisionParams, name)()
    raise ValueError(f"Unknown preset: {name}")


def direct_multi_k_objective(params: CosmoParams, prec: PrecisionParams, k_eval: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Weighted scalar objective from repeated exact single-mode solves."""
    values = [clax.compute_pk(params, prec, float(k)) for k in k_eval]
    return jnp.sum(weights * jnp.asarray(values))


def table_multi_k_objective(params: CosmoParams, prec: PrecisionParams, k_eval: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Weighted scalar objective from one reusable public PK table solve."""
    result = clax.compute_pk_table(params, prec, k_eval=k_eval)
    return jnp.sum(weights * result.pk_grid)


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
    """Run the reverse-mode benchmark for one preset."""
    params = CosmoParams()
    prec = get_preset(preset_name)
    k_eval = jnp.geomspace(prec.pt_k_min, min(prec.pt_k_max_cl, 0.3), num_eval)
    weights = jnp.linspace(1.0, float(num_eval), num_eval)
    weights = weights / jnp.sum(weights)

    direct_grad = timed_cached(
        "direct_multi_k_grad",
        lambda: jax.grad(lambda p: direct_multi_k_objective(p, prec, k_eval, weights))(params).h,
    )
    table_grad = timed_cached(
        "table_multi_k_grad",
        lambda: jax.grad(lambda p: table_multi_k_objective(p, prec, k_eval, weights))(params).h,
    )

    grad_ratio = float(table_grad["value"] / direct_grad["value"]) if float(direct_grad["value"]) != 0.0 else float("nan")

    print(f"Backend: {jax.devices()}")
    print(f"Preset: {preset_name}")
    print(f"Requested k points: {num_eval}")
    print()
    print(f"{'Path':<20} {'Cached time [s]':>16} {'d/dh':>16}")
    print(f"{direct_grad['label']:<20} {direct_grad['seconds']:16.3f} {float(direct_grad['value']):16.6e}")
    print(f"{table_grad['label']:<20} {table_grad['seconds']:16.3f} {float(table_grad['value']):16.6e}")
    print()
    print("Agreement:")
    print(f"  table/direct gradient ratio: {grad_ratio:.6e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("preset", nargs="?", default="fit_cl", help="Precision preset name")
    parser.add_argument("--num-eval", type=int, default=4, help="Number of requested output k points")
    args = parser.parse_args()

    run_benchmark(args.preset, args.num_eval)
