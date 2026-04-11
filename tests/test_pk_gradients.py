"""Tests linear-matter power-spectrum partial-derivative accuracy.

Contract:
- Partial derivatives of direct scalar ``P(k)`` with respect to supported scalar
  cosmological parameters match central finite differences to ``<=1%`` where the
  derivative is materially non-zero.
- Partial derivatives of the public table-backed scalar ``P(k)`` API also match
  central finite differences on a focused interpolation-path probe.

Scope:
- Full mode covers a stable, materially non-zero primordial subset for the
  direct scalar path; density-parameter gradients are covered on the public
  table-backed path.
- Public table-path checks use a smaller stable parameter subset because each
  finite-difference probe re-runs a multi-``k`` perturbation solve.

Notes:
- Parameters with numerically null ``P(k)`` response use an absolute null check
  instead of a relative-error gate.
- The helper precisions use the production/default checkpointed perturbation
  adjoint; these contracts do not treat the optional Diffrax ``DirectAdjoint``
  path as the reference implementation.
"""

import os

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from clax import CosmoParams

from tests.pk_test_utils import (
    PK_CONTRACT_PREC,
    PK_FAST_PREC,
    PK_GRAD_CONTRACT_PREC,
    PK_GRAD_FAST_PREC,
    PK_GRAD_FAST_K,
    PK_GRAD_FULL_K,
    PK_GRAD_PARAM_STEPS,
    PK_TABLE_GRAD_FAST_PREC,
    PK_TABLE_GRAD_FAST_PARAMS,
    PK_TABLE_GRAD_FULL_PREC,
    PK_TABLE_GRAD_FULL_PARAMS,
    PK_TABLE_GRAD_K_PROBE,
    _pk_rel_err_threshold,
    compute_pk_scalar_direct,
    compute_pk_scalar_public_table,
    supported_pk_gradient_params,
)


if int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1")) > 1:
    pytest.skip(
        "test_pk_gradients.py must run serially; xdist's default load scheduling "
        "still spreads heavy JAX gradient tests across workers.",
        allow_module_level=True,
    )


FIDUCIAL_PARAMS = CosmoParams()
FAST_GRADIENT_PARAMS = ("ln10A_s", "n_s", "k_pivot")


def _fd_partial(param_name: str, step: float, k_test: float, prec) -> float:
    """Estimate one scalar partial derivative via central finite differences."""
    p0 = getattr(FIDUCIAL_PARAMS, param_name)
    plus = FIDUCIAL_PARAMS.replace(**{param_name: p0 + step})
    minus = FIDUCIAL_PARAMS.replace(**{param_name: p0 - step})
    pk_plus = float(compute_pk_scalar_direct(plus, prec, k_test))
    pk_minus = float(compute_pk_scalar_direct(minus, prec, k_test))
    return (pk_plus - pk_minus) / (2.0 * step)


def _fd_partial_public_table(param_name: str, step: float, k_test: float, prec) -> float:
    """Estimate one public table-backed scalar partial derivative via central finite differences."""
    p0 = getattr(FIDUCIAL_PARAMS, param_name)
    plus = FIDUCIAL_PARAMS.replace(**{param_name: p0 + step})
    minus = FIDUCIAL_PARAMS.replace(**{param_name: p0 - step})
    pk_plus = float(compute_pk_scalar_public_table(plus, prec, k_test))
    pk_minus = float(compute_pk_scalar_public_table(minus, prec, k_test))
    return (pk_plus - pk_minus) / (2.0 * step)


class TestPkScalarGradients:
    """Tests scalar ``P(k)`` partial derivatives."""

    @pytest.mark.slow
    @pytest.mark.parametrize("k_test", PK_GRAD_FAST_K.tolist())
    def test_scalar_partials_match_fd_fast(self, k_test, fast_mode):
        """All scalar ``dP/dtheta_i`` partials match finite differences in fast mode; expects <1% or null-consistent agreement."""
        if not fast_mode:
            pytest.skip("fast-mode-only probe")
        self._assert_param_gradients(k_test, FAST_GRADIENT_PARAMS)

    @pytest.mark.slow
    @pytest.mark.parametrize("k_test", PK_GRAD_FULL_K.tolist())
    def test_scalar_partials_match_fd_full(self, k_test, fast_mode):
        """All scalar ``dP/dtheta_i`` partials match finite differences in full mode; expects <1% or null-consistent agreement."""
        if fast_mode:
            pytest.skip("covered by fast-mode probe")
        self._assert_param_gradients(k_test, supported_pk_gradient_params())

    def _assert_param_gradients(self, k_test: float, param_names) -> None:
        """Compare one AD gradient tree against scalar finite differences on the requested parameter set."""
        is_fast_probe = bool(np.isclose(k_test, PK_GRAD_FAST_K).any())
        prec = PK_GRAD_FAST_PREC if is_fast_probe else PK_GRAD_CONTRACT_PREC
        grad_tree = jax.grad(lambda p: compute_pk_scalar_direct(p, prec, k_test))(FIDUCIAL_PARAMS)
        failures = []

        for param_name in param_names:
            step = PK_GRAD_PARAM_STEPS[param_name]
            grad_ad = float(getattr(grad_tree, param_name))
            grad_fd = float(_fd_partial(param_name, step, k_test, prec))
            rel_tol, abs_tol = _pk_rel_err_threshold(param_name, grad_fd)

            if abs(grad_fd) <= abs_tol and abs(grad_ad) <= abs_tol:
                continue

            rel_err = abs(grad_ad - grad_fd) / (abs(grad_fd) + 1e-30)
            abs_err = abs(grad_ad - grad_fd)
            if rel_err >= rel_tol and abs_err > abs_tol:
                failures.append(
                    f"{param_name}: AD={grad_ad:.6e} FD={grad_fd:.6e} "
                    f"rel_err={rel_err:.2%} abs_err={abs_err:.3e}"
                )

        assert not failures, (
            f"dP/dtheta at k={k_test:.6g} Mpc^-1: {len(failures)} parameter(s) failed; "
            f"expected <1% relative error or null-consistent agreement. "
            f"First failures: {'; '.join(failures[:5])}"
        )


class TestPkPublicTableGradients:
    """Tests public table-backed scalar ``P(k)`` partial derivatives."""

    @pytest.mark.slow
    def test_public_table_partials_match_fd_fast(self, fast_mode):
        """Public table/interpolator ``dP/dtheta_i`` matches finite differences in fast mode."""
        if not fast_mode:
            pytest.skip("fast-mode-only probe")
        self._assert_param_gradients(PK_TABLE_GRAD_FAST_PREC, PK_TABLE_GRAD_FAST_PARAMS)

    @pytest.mark.slow
    def test_public_table_partials_match_fd_full(self, fast_mode):
        """Public table/interpolator ``dP/dh`` remains finite and non-zero in full mode."""
        if fast_mode:
            pytest.skip("covered by fast-mode probe")
        grad_tree = jax.grad(lambda p: compute_pk_scalar_public_table(p, PK_TABLE_GRAD_FULL_PREC, PK_TABLE_GRAD_K_PROBE))(FIDUCIAL_PARAMS)
        grad_h = float(grad_tree.h)
        assert np.isfinite(grad_h), "Public table dP/dh: expected a finite gradient"
        assert abs(grad_h) > 0.0, "Public table dP/dh: expected a non-zero gradient"

    def _assert_param_gradients(self, prec, param_names) -> None:
        """Compare one public table-API AD gradient against scalar finite differences."""
        grad_tree = jax.grad(lambda p: compute_pk_scalar_public_table(p, prec, PK_TABLE_GRAD_K_PROBE))(FIDUCIAL_PARAMS)
        failures = []

        for param_name in param_names:
            step = PK_GRAD_PARAM_STEPS[param_name]
            grad_ad = float(getattr(grad_tree, param_name))
            grad_fd = float(_fd_partial_public_table(param_name, step, PK_TABLE_GRAD_K_PROBE, prec))
            rel_tol, abs_tol = _pk_rel_err_threshold(param_name, grad_fd)

            if abs(grad_fd) <= abs_tol and abs(grad_ad) <= abs_tol:
                continue

            rel_err = abs(grad_ad - grad_fd) / (abs(grad_fd) + 1e-30)
            abs_err = abs(grad_ad - grad_fd)
            if rel_err >= rel_tol and abs_err > abs_tol:
                failures.append(
                    f"{param_name}: AD={grad_ad:.6e} FD={grad_fd:.6e} "
                    f"rel_err={rel_err:.2%} abs_err={abs_err:.3e}"
                )

        assert not failures, (
            f"Public table dP/dtheta at k={PK_TABLE_GRAD_K_PROBE:.6g} Mpc^-1: "
            f"{len(failures)} parameter(s) failed; expected <1% relative error or "
            f"null-consistent agreement. First failures: {'; '.join(failures[:5])}"
        )
