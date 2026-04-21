"""Spherical Bessel functions for clax.

Provides j_l(x) via several methods:

- ``spherical_jl``: upward recurrence from j_0, j_1.  Stable for x >= l;
  zeros the classically forbidden region x < 0.7*l.
- ``spherical_jl_backward``: Miller's backward recurrence (stable for all x)
  blended with upward recurrence.  Used by ``harmonic.py`` for CMB transfer
  functions where accuracy at x < l matters.
- ``spherical_jl_array``: upward recurrence producing j_0..j_{l_max} in one
  pass.  Same stability caveats as ``spherical_jl``.
- ``build_jl_table``: precomputed 2-D table of j_l(x) (and optionally j_l')
  on a uniform x-grid for a sparse set of l-values.  Uses backward + upward
  recurrence with blending.  Shared by ``harmonic.py`` and ``lensing.py``.

All recurrences use ``jax.lax.fori_loop`` for O(1) compilation time
regardless of l (no Python loop unrolling).

References:
    Numerical Recipes Ch. 6
    Abramowitz & Stegun 10.1
"""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
import math


def spherical_jl(l: int, x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Compute spherical Bessel function j_l(x) via upward recurrence.

    Uses jax.lax.fori_loop for efficient JIT compilation at any l.
    Stable for x > l. For x < l, returns 0 (j_l is exponentially small).

    Args:
        l: order (non-negative integer, static)
        x: argument(s), any shape

    Returns:
        j_l(x), same shape as x
    """
    x = jnp.asarray(x)

    if l == 0:
        return _j0(x)
    elif l == 1:
        return _j1(x)

    x_safe = jnp.where(jnp.abs(x) < 1e-30, 1e-30, x)

    def body_fn(l_curr, state):
        j_prev, j_curr = state
        j_next = (2.0 * l_curr + 1.0) / x_safe * j_curr - j_prev
        # For x < 0.7*l, j_l is exponentially small — zero it to prevent overflow
        j_next = jnp.where(jnp.abs(x) < 0.7 * (l_curr + 1), 0.0, j_next)
        # |j_l(x)| <= 1 always
        j_next = jnp.clip(j_next, -1.0, 1.0)
        return (j_curr, j_next)

    j_prev, j_curr = jax.lax.fori_loop(1, l, body_fn, (_j0(x), _j1(x)))
    return j_curr


def _j0(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """j_0(x) = sin(x) / x."""
    x_safe = jnp.where(jnp.abs(x) < 1e-8, 1e-8, x)
    return jnp.where(jnp.abs(x) < 1e-8, 1.0 - x**2 / 6.0, jnp.sin(x_safe) / x_safe)


def _j1(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """j_1(x) = sin(x)/x^2 - cos(x)/x."""
    x_safe = jnp.where(jnp.abs(x) < 1e-8, 1e-8, x)
    return jnp.where(
        jnp.abs(x) < 1e-8,
        x / 3.0,
        jnp.sin(x_safe) / x_safe**2 - jnp.cos(x_safe) / x_safe,
    )


def spherical_jl_backward(l: int, x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Compute j_l(x) via backward (Miller's) recurrence, blended with upward.

    Backward recurrence is stable for all x but especially critical for x < l.
    Upward recurrence is stable for x > l and cheaper.
    We blend the two using a smooth sigmoid transition around x = l.

    Uses jax.lax.fori_loop for O(1) compilation time at any l.

    Args:
        l: order (non-negative integer, static)
        x: argument(s), any shape

    Returns:
        j_l(x), same shape as x
    """
    x = jnp.asarray(x)
    if l == 0:
        return _j0(x)
    if l == 1:
        return _j1(x)

    # --- Backward recurrence (stable for x < l) ---
    # Start from l_start = l + extra, with j_{l_start+1} = 0, j_{l_start} = 1
    # Recur downward to l=0, recording j_l. Normalize using j_0.
    extra = min(60, max(30, l // 5))
    l_start = l + extra
    x_safe = jnp.where(jnp.abs(x) < 1e-30, 1e-30, x)

    def body_fn(i, state):
        j_curr, j_next, j_at_l = state
        n = l_start - i
        j_prev = (2.0 * n + 1.0) / x_safe * j_curr - j_next
        # Rescale to prevent over/underflow
        scale = jnp.maximum(jnp.abs(j_prev), 1e-300)
        j_prev = j_prev / scale
        j_curr = j_curr / scale
        j_at_l = j_at_l / scale
        # Record j_l when we reach it
        j_at_l = jnp.where(n - 1 == l, j_prev, j_at_l)
        return (j_prev, j_curr, j_at_l)

    init = (jnp.ones_like(x), jnp.zeros_like(x), jnp.zeros_like(x))
    j_0_backward, _, j_l_backward = jax.lax.fori_loop(0, l_start, body_fn, init)

    # Normalize: j_l = j_l_backward * (j_0_true / j_0_backward)
    j_0_true = _j0(x)
    j_0_safe = jnp.where(jnp.abs(j_0_backward) < 1e-300, 1e-300, j_0_backward)
    j_l_back = j_l_backward * (j_0_true / j_0_safe)

    # --- Upward recurrence (stable for x > l) ---
    j_l_up = spherical_jl(l, x)

    # --- Hard switch at x = l ---
    # Backward is accurate for x <= l, upward for x >= l.
    # Both agree at x = l, so the switch is smooth in practice.
    l_fl = float(l)
    result = jnp.where(jnp.abs(x) >= l_fl, j_l_up, j_l_back)

    result = jnp.where(jnp.abs(x) < 1e-10, 0.0, result)
    return result


def spherical_jl_array(l_max: int, x: Float[Array, "..."]) -> Float[Array, "L ..."]:
    """Compute j_l(x) for l = 0, 1, ..., l_max via a single upward recurrence pass.

    More efficient than calling spherical_jl for each l separately.

    Returns array of shape (l_max + 1, *x.shape).
    """
    x = jnp.asarray(x)
    x_safe = jnp.where(jnp.abs(x) < 1e-30, 1e-30, x)

    j0 = _j0(x)
    j1 = _j1(x)

    if l_max == 0:
        return j0[None, ...]
    if l_max == 1:
        return jnp.stack([j0, j1])

    def body_fn(l_curr, state):
        results, j_prev, j_curr = state
        j_next = (2.0 * l_curr + 1.0) / x_safe * j_curr - j_prev
        j_next = jnp.clip(j_next, -1.0, 1.0)
        results = results.at[l_curr + 1].set(j_next)
        return (results, j_curr, j_next)

    results = jnp.zeros((l_max + 1, *x.shape))
    results = results.at[0].set(j0)
    results = results.at[1].set(j1)

    results, _, _ = jax.lax.fori_loop(1, l_max, body_fn, (results, j0, j1))

    return results


# ---------------------------------------------------------------------------
# Sparse l-grid and precomputed Bessel tables
# ---------------------------------------------------------------------------

def sparse_l_grid(l_max=2500):
    """Generate sparse l-sampling for efficient C_l computation.

    Mirrors CLASS strategy: dense at low l, sparser at high l.
    Returns ~100-150 l-values depending on l_max.
    """
    l_list = list(range(2, min(31, l_max + 1)))
    l_list += list(range(35, min(101, l_max + 1), 5))
    l_list += list(range(120, min(501, l_max + 1), 20))
    l_list += list(range(550, l_max + 1, 50))
    return np.array(l_list, dtype=int)


def build_jl_table(l_max, n_x=30000, x_max=15000.0):
    """Build j_l(x) and j_l'(x) lookup tables for a sparse set of l-values.

    Uses backward recurrence (accurate for x < l) blended with upward
    recurrence (accurate for x >= l) at a hard switch at x = l.

    The tables are built for a sparse l-grid (from ``sparse_l_grid``) on a
    uniform x-grid.  Callers interpolate the table at query x-values using
    ``searchsorted`` + linear weights.

    j_l' is needed for TT transfer functions (T1 ISW dipole term).
    Lensing only uses j_l but the derivative cost is marginal.

    Args:
        l_max: maximum l value
        n_x: number of x-table points (default 30000, ~0.5 spacing)
        x_max: maximum x value (default 15000, > k_max * tau_0)

    Returns:
        x_table: (n_x,) uniform x-grid
        jl_table: (n_l, n_x) blended j_l values
        jlp_table: (n_l, n_x) blended j_l'(x) values

    See also:
        ``harmonic.compute_cls_all_fast`` and ``lensing.compute_cl_pp_vmap``
        which consume these tables.
    """
    l_sparse = sparse_l_grid(l_max)
    n_l = len(l_sparse)

    x_table = jnp.linspace(0.0, x_max, n_x)
    x_safe = jnp.where(x_table < 1e-30, 1e-30, x_table)

    # --- Backward recurrence (accurate for x < l) ---
    extra = min(60, max(30, l_max // 5))
    l_start = l_max + extra

    lookup_np = np.full(l_start + 2, -1, dtype=np.int32)
    for idx_l, l_val in enumerate(l_sparse):
        lookup_np[int(l_val)] = idx_l
    lookup = jnp.array(lookup_np)

    def body_back(i, state):
        j_curr, j_next, j_at_ls, jlp_at_ls = state
        n = l_start - i
        j_prev = (2.0 * n + 1.0) / x_safe * j_curr - j_next
        scale = jnp.maximum(jnp.abs(j_prev), 1e-300)
        j_prev = j_prev / scale
        j_curr = j_curr / scale
        j_at_ls = j_at_ls / scale[None, :]
        jlp_at_ls = jlp_at_ls / scale[None, :]

        l_val = n - 1
        l_fl = jnp.float64(l_val)
        jlp_val = l_fl / x_safe * j_prev - j_curr

        idx = lookup[l_val]
        idx_safe = jnp.maximum(idx, 0)
        j_at_ls = j_at_ls.at[idx_safe].set(
            jnp.where(idx >= 0, j_prev, j_at_ls[idx_safe]))
        jlp_at_ls = jlp_at_ls.at[idx_safe].set(
            jnp.where(idx >= 0, jlp_val, jlp_at_ls[idx_safe]))
        return (j_prev, j_curr, j_at_ls, jlp_at_ls)

    init_back = (jnp.ones(n_x), jnp.zeros(n_x),
                 jnp.zeros((n_l, n_x)), jnp.zeros((n_l, n_x)))
    j_0_back, _, j_at_ls, jlp_at_ls = jax.lax.fori_loop(
        0, l_start, body_back, init_back)

    j_0_true = _j0(x_table)
    j_0_safe = jnp.where(jnp.abs(j_0_back) < 1e-300, 1e-300, j_0_back)
    norm = (j_0_true / j_0_safe)[None, :]
    jl_back = j_at_ls * norm
    jlp_back = jlp_at_ls * norm

    # --- Upward recurrence (accurate for x > l) ---
    def body_up(l_curr, state):
        j_prev, j_curr, j_up_ls, jlp_up_ls = state
        j_next = (2.0 * l_curr + 1.0) / x_safe * j_curr - j_prev
        j_next = jnp.clip(j_next, -1e200, 1e200)

        l_new = l_curr + 1
        l_fl = jnp.float64(l_new)
        jlp_val = j_curr - (l_fl + 1.0) / x_safe * j_next
        jlp_val = jnp.clip(jlp_val, -1e200, 1e200)

        idx = lookup[l_new]
        idx_safe = jnp.maximum(idx, 0)
        j_up_ls = j_up_ls.at[idx_safe].set(
            jnp.where(idx >= 0, j_next, j_up_ls[idx_safe]))
        jlp_up_ls = jlp_up_ls.at[idx_safe].set(
            jnp.where(idx >= 0, jlp_val, jlp_up_ls[idx_safe]))
        return (j_curr, j_next, j_up_ls, jlp_up_ls)

    j0_vals = _j0(x_table)
    j1_vals = _j1(x_table)
    init_up = (j0_vals, j1_vals,
               jnp.zeros((n_l, n_x)), jnp.zeros((n_l, n_x)))
    _, _, jl_up, jlp_up = jax.lax.fori_loop(1, l_max, body_up, init_up)

    # --- Blend: backward for x < l, upward for x >= l ---
    l_vals = jnp.array(l_sparse, dtype=jnp.float64)
    mask_up = x_table[None, :] >= l_vals[:, None]
    jl_table = jnp.where(mask_up, jl_up, jl_back)
    jlp_table = jnp.where(mask_up, jlp_up, jlp_back)

    jl_table = jnp.where(x_table[None, :] < 1e-10, 0.0, jl_table)
    jlp_table = jnp.where(x_table[None, :] < 1e-10, 0.0, jlp_table)

    return x_table, jl_table, jlp_table
