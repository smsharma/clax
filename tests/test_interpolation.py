"""Test differentiable cubic spline interpolation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxclass.interpolation import CubicSpline


def test_spline_sin():
    """Spline of sin(x) should match to high accuracy."""
    x = jnp.linspace(0, 2 * jnp.pi, 100)
    y = jnp.sin(x)
    spl = CubicSpline(x, y)

    x_eval = jnp.linspace(0.1, 2 * jnp.pi - 0.1, 500)
    y_eval = spl.evaluate(x_eval)
    y_exact = jnp.sin(x_eval)

    max_err = float(jnp.max(jnp.abs(y_eval - y_exact)))
    assert max_err < 1e-5, f"Spline sin error: {max_err:.2e}"


def test_spline_derivative_cos():
    """Derivative of spline(sin) should be cos."""
    x = jnp.linspace(0, 2 * jnp.pi, 200)
    y = jnp.sin(x)
    spl = CubicSpline(x, y)

    x_eval = jnp.linspace(0.1, 2 * jnp.pi - 0.1, 100)
    dy = spl.derivative(x_eval)
    dy_exact = jnp.cos(x_eval)

    max_err = float(jnp.max(jnp.abs(dy - dy_exact)))
    assert max_err < 1e-3, f"Spline derivative error: {max_err:.2e}"


def test_spline_exp():
    """Spline of exp(x) on sparse grid."""
    x = jnp.linspace(0, 5, 50)
    y = jnp.exp(x)
    spl = CubicSpline(x, y)

    x_eval = jnp.linspace(0.1, 4.9, 200)
    y_eval = spl.evaluate(x_eval)
    y_exact = jnp.exp(x_eval)

    rel_err = jnp.abs(y_eval - y_exact) / y_exact
    max_rel_err = float(jnp.max(rel_err))
    assert max_rel_err < 5e-4, f"Spline exp rel error: {max_rel_err:.2e} (50 points on exp(0..5))"


def test_spline_pytree():
    """CubicSpline should work as a JAX pytree (flatten/unflatten)."""
    x = jnp.linspace(0, 1, 10)
    y = x ** 2
    spl = CubicSpline(x, y)

    # Flatten and unflatten
    leaves, treedef = jax.tree_util.tree_flatten(spl)
    spl2 = jax.tree_util.tree_unflatten(treedef, leaves)

    x_eval = jnp.array([0.5])
    assert jnp.allclose(spl.evaluate(x_eval), spl2.evaluate(x_eval))


def test_spline_grad():
    """Gradients should flow through spline evaluation."""
    x = jnp.linspace(0, 1, 20)

    def f(y_knots):
        spl = CubicSpline(x, y_knots)
        return spl.evaluate(jnp.array(0.5))

    y = jnp.sin(x)
    grad = jax.grad(f)(y)
    # Gradient should be non-zero (spline value depends on knot values)
    assert float(jnp.sum(jnp.abs(grad))) > 0, "Gradient through spline is zero"
    # No NaN
    assert jnp.all(jnp.isfinite(grad)), "NaN in spline gradient"
