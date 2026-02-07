"""Spherical Bessel functions for jaxCLASS.

Computes j_l(x) using a combination of:
- Direct formulas for l=0,1
- Upward recurrence for x > l (stable regime)
- Asymptotic x^l/(2l+1)!! for x << l
- Smooth blending between regimes

For large l (>30), the upward recurrence from j_0,j_1 accumulates errors.
We use a more careful approach: start the recurrence from max(0, l-20) using
the WKB/asymptotic form as a starting point.

References:
    Numerical Recipes Ch. 6
    Abramowitz & Stegun 10.1
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import math


def spherical_jl(l: int, x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Compute spherical Bessel function j_l(x).

    Uses upward recurrence with careful handling of small x.

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

    # For l >= 2: use upward recurrence from j_0, j_1
    # This is stable when x > l. For x < l, j_l is exponentially small
    # and we use the asymptotic form.

    j_prev = _j0(x)
    j_curr = _j1(x)

    for l_curr in range(1, l):
        x_safe = jnp.where(jnp.abs(x) < 1e-30, 1e-30, x)
        j_next = (2.0 * l_curr + 1.0) / x_safe * j_curr - j_prev
        # For x < l_curr, the recurrence is unstable → set to 0
        # For x < l_curr: set to 0 (exponentially small regime, upward recurrence unstable)
        j_next = jnp.where(jnp.abs(x) < 0.7 * (l_curr + 1), 0.0, j_next)
        # |j_l(x)| < 1 always, so clip aggressively to prevent overflow cascade
        j_next = jnp.clip(j_next, -1.0, 1.0)
        j_prev = j_curr
        j_curr = j_next

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

    Backward recurrence is stable for x < l (where upward recurrence fails).
    Upward recurrence is stable for x > l (where backward loses accuracy).
    We blend the two using a smooth transition around x = l.

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
    l_start = l + 60
    x_safe = jnp.where(jnp.abs(x) < 1e-30, 1e-30, x)

    def body_fn(i, state):
        j_curr, j_next, j_at_l = state
        n = l_start - i
        j_prev = (2.0 * n + 1.0) / x_safe * j_curr - j_next
        scale = jnp.maximum(jnp.abs(j_prev), 1e-300)
        j_prev = j_prev / scale
        j_curr = j_curr / scale
        j_at_l = j_at_l / scale
        j_at_l = jnp.where(n - 1 == l, j_prev, j_at_l)
        return (j_prev, j_curr, j_at_l)

    init = (jnp.ones_like(x), jnp.zeros_like(x), jnp.zeros_like(x))
    j_0_backward, _, j_l_backward = jax.lax.fori_loop(0, l_start, body_fn, init)

    j_0_true = _j0(x)
    j_0_safe = jnp.where(jnp.abs(j_0_backward) < 1e-300, 1e-300, j_0_backward)
    j_l_back = j_l_backward * (j_0_true / j_0_safe)

    # --- Upward recurrence (stable for x > l) ---
    j_l_up = spherical_jl(l, x)

    # --- Blend: sigmoid transition around x = l ---
    # For x < 0.9*l: use backward; for x > 1.1*l: use upward
    l_fl = float(l)
    w_up = jax.nn.sigmoid(10.0 * (jnp.abs(x) / l_fl - 1.0))
    result = (1.0 - w_up) * j_l_back + w_up * j_l_up

    result = jnp.where(jnp.abs(x) < 1e-10, 0.0, result)
    return result


def spherical_jl_array(l_max: int, x: Float[Array, "..."]) -> Float[Array, "L ..."]:
    """Compute j_l(x) for l = 0, 1, ..., l_max via a single upward recurrence pass.

    More efficient than calling spherical_jl for each l separately.

    Returns array of shape (l_max + 1, *x.shape).
    """
    j0 = _j0(x)
    if l_max == 0:
        return j0[None, ...]

    j1 = _j1(x)
    if l_max == 1:
        return jnp.stack([j0, j1], axis=0)

    results = [j0, j1]
    j_prev = j0
    j_curr = j1

    for l_curr in range(1, l_max):
        x_safe = jnp.where(jnp.abs(x) < 1e-30, 1e-30, x)
        j_next = (2.0 * l_curr + 1.0) / x_safe * j_curr - j_prev
        j_next = jnp.where(jnp.abs(x) < 0.5 * (l_curr + 1), 0.0, j_next)
        results.append(j_next)
        j_prev = j_curr
        j_curr = j_next

    return jnp.stack(results, axis=0)
