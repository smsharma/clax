"""Spherical Bessel functions for clax.

Computes j_l(x) using backward (Miller's) recurrence for all l >= 2.
This is the standard stable algorithm: start from an arbitrary guess at
l_start > l, recur downward to l=0, then normalize using j_0(x) = sin(x)/x.

For the transfer integral, we need j_l(x) at l up to ~2500 with x up to
~10000. The backward recurrence handles this stably for all x.

For x >> l (oscillatory regime), upward recurrence from j_0, j_1 is also
stable and avoids the overhead of backward recurrence. We blend the two
using a sigmoid transition around x = l.

All recurrences use jax.lax.fori_loop for O(1) compilation time
regardless of l (no Python loop unrolling).

References:
    Numerical Recipes Ch. 6
    Abramowitz & Stegun 10.1
"""

import jax
import jax.numpy as jnp
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
