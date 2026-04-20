"""Tests shooting-method forward and gradient behavior.

Contract:
- The ``theta_s -> h`` shooting map is numerically stable, approximately correct, and differentiable.

Scope:
- Covers fiducial ``theta_s``, round-trip shooting, and local gradient checks.
- Excludes unrelated background or perturbation contracts owned elsewhere.

Notes:
- Gradients remain in this file because they are local to the shooting contract and comparatively cheap.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from clax.params import CosmoParams, PrecisionParams
from clax.shooting import _compute_theta_s, make_shoot_h_from_theta_s


PREC = PrecisionParams()


@pytest.fixture(scope="module")
def theta_s_fiducial():
    """Compute the fiducial ``100*theta_s`` value once for this module."""
    params = CosmoParams()
    return float(_compute_theta_s(0.6736, params, PREC))


class TestComputeThetaS:
    """Tests direct ``theta_s`` computation."""

    def test_theta_s_reasonable_value(self, theta_s_fiducial):
        """Fiducial ``100*theta_s`` has the expected scale; expects a value in [1.03, 1.06]."""
        assert 1.03 < theta_s_fiducial < 1.06, (
            f"100*theta_s = {theta_s_fiducial:.4f} outside reasonable range [1.03, 1.06]"
        )

    def test_theta_s_close_to_class(self, theta_s_fiducial, lcdm_derived):
        """Fiducial ``100*theta_s`` matches the CLASS-derived value; expects <1% relative error."""
        rs_rec = lcdm_derived['rs_rec']
        tau_rec = lcdm_derived['tau_rec']
        conformal_age = lcdm_derived['conformal_age']
        ra_rec = conformal_age - tau_rec
        class_theta_s_100 = 100.0 * rs_rec / ra_rec

        rel_err = abs(theta_s_fiducial - class_theta_s_100) / class_theta_s_100
        assert rel_err < 0.01, (
            f"100*theta_s: got {theta_s_fiducial:.6f}, CLASS gives {class_theta_s_100:.6f}, "
            f"rel err = {rel_err:.4%}"
        )

    def test_theta_s_monotonic_in_h(self):
        """``theta_s`` increases with ``h`` on the probe grid; expects monotonic ordering."""
        params = CosmoParams()
        ts_low = _compute_theta_s(0.60, params, PREC)
        ts_mid = _compute_theta_s(0.67, params, PREC)
        ts_high = _compute_theta_s(0.75, params, PREC)
        assert float(ts_low) < float(ts_mid) < float(ts_high), (
            f"theta_s not monotonic: {float(ts_low):.4f}, {float(ts_mid):.4f}, {float(ts_high):.4f}"
        )


class TestShootingRoundTrip:
    """Tests shooting round-trip behavior."""

    def test_roundtrip_fiducial(self, theta_s_fiducial):
        """Shooting recovers fiducial ``h`` from fiducial ``theta_s``; expects <1e-4 relative error."""
        params = CosmoParams()
        shoot_fn = make_shoot_h_from_theta_s(PREC)
        h_recovered = float(shoot_fn(theta_s_fiducial, params))

        rel_err = abs(h_recovered - 0.6736) / 0.6736
        assert rel_err < 1e-4, (
            f"Round-trip failed: h_recovered={h_recovered:.8f}, expected 0.6736, "
            f"rel err = {rel_err:.2e}"
        )

    @pytest.mark.slow
    def test_roundtrip_alternative_h(self):
        """Round-trip shooting works away from the initial guess; expects <1e-4 relative error on the probe grid."""
        params = CosmoParams()
        shoot_fn = make_shoot_h_from_theta_s(PREC)

        for h_true in [0.60, 0.70, 0.75]:
            ts_100 = _compute_theta_s(h_true, params, PREC)
            h_recovered = float(shoot_fn(ts_100, params))
            rel_err = abs(h_recovered - h_true) / h_true
            assert rel_err < 1e-4, (
                f"Round-trip failed for h={h_true}: recovered={h_recovered:.8f}, "
                f"rel err = {rel_err:.2e}"
            )


class TestShootingGradient:
    """Tests shooting-method gradient behavior."""

    def test_gradient_finite(self, theta_s_fiducial):
        """``dh/dtheta_s`` is finite and positive; expects a finite positive gradient."""
        params = CosmoParams()
        shoot_fn = make_shoot_h_from_theta_s(PREC)

        grad_fn = jax.grad(lambda ts: shoot_fn(ts, params))
        grad_val = float(grad_fn(theta_s_fiducial))

        assert np.isfinite(grad_val), f"Gradient is not finite: {grad_val}"
        assert grad_val > 0, f"Gradient should be positive, got {grad_val}"

    def test_gradient_vs_finite_diff(self, theta_s_fiducial):
        """``dh/dtheta_s`` matches finite differences; expects <1% relative error."""
        params = CosmoParams()
        shoot_fn = make_shoot_h_from_theta_s(PREC)

        # AD gradient
        grad_fn = jax.grad(lambda ts: shoot_fn(ts, params))
        grad_ad = float(grad_fn(theta_s_fiducial))

        # Finite difference gradient
        eps = 1e-5
        h_plus = float(shoot_fn(theta_s_fiducial + eps, params))
        h_minus = float(shoot_fn(theta_s_fiducial - eps, params))
        grad_fd = (h_plus - h_minus) / (2 * eps)

        rel_err = abs(grad_ad - grad_fd) / abs(grad_fd)
        assert rel_err < 0.01, (
            f"dh/d(theta_s): AD={grad_ad:.6f}, FD={grad_fd:.6f}, rel err={rel_err:.4%}"
        )

    def test_gradient_reasonable_magnitude(self, theta_s_fiducial):
        """``dh/d(100*theta_s)`` has a reasonable scale; expects a value between 1 and 20."""
        params = CosmoParams()
        shoot_fn = make_shoot_h_from_theta_s(PREC)

        grad_fn = jax.grad(lambda ts: shoot_fn(ts, params))
        grad_val = float(grad_fn(theta_s_fiducial))

        # theta_s changes by ~0.01 when h changes by ~0.06, so dh/d(100ts) ~ 6
        assert 1.0 < grad_val < 20.0, (
            f"dh/d(100*theta_s) = {grad_val:.4f}, expected O(1-10)"
        )
