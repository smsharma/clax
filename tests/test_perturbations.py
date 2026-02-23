"""Test perturbations module: P(k) against CLASS reference data."""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import diffrax
import pytest

from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve
from clax.perturbations import (
    _build_indices, _adiabatic_ic, _perturbation_rhs,
)
from clax.params import CosmoParams, PrecisionParams


PREC = PrecisionParams(
    bg_n_points=400, ncdm_bg_n_points=200, bg_tol=1e-8,
    th_n_points=10000, th_z_max=5e3,
    pt_l_max_g=17, pt_l_max_pol_g=17, pt_l_max_ur=17,
)


@pytest.fixture(scope="module")
def bg():
    return background_solve(CosmoParams(), PREC)


@pytest.fixture(scope="module")
def th(bg):
    return thermodynamics_solve(CosmoParams(), PREC, bg)


def _compute_pk_single_k(k, bg, th, prec):
    """Compute P(k) at a single k-mode."""
    idx = _build_indices(prec.pt_l_max_g, prec.pt_l_max_pol_g, prec.pt_l_max_ur)
    tau_ini = 0.5
    tau_end = float(bg.conformal_age) * 0.999

    y0 = _adiabatic_ic(k, jnp.array(tau_ini), bg, CosmoParams(), idx, idx['n_eq'])
    args = (k, bg, th, CosmoParams(), idx, prec.pt_l_max_g, prec.pt_l_max_pol_g, prec.pt_l_max_ur)

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(_perturbation_rhs), solver=diffrax.Kvaerno5(),
        t0=tau_ini, t1=tau_end, dt0=tau_ini * 0.1, y0=y0,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-7),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        max_steps=65536, args=args,
    )
    y_f = sol.ys[-1]
    rho_b = float(bg.rho_b_of_loga.evaluate(jnp.array(0.0)))
    rho_cdm = float(bg.rho_cdm_of_loga.evaluate(jnp.array(0.0)))
    dm = (rho_b * float(y_f[idx['delta_b']]) + rho_cdm * float(y_f[idx['delta_cdm']])) / (rho_b + rho_cdm)

    A_s = np.exp(3.044) / 1e10
    return 2 * np.pi**2 / k**3 * A_s * (k / 0.05)**(0.9649 - 1) * dm**2


class TestPkLowK:
    """Test P(k) at low k where our solver should match CLASS to < 5%."""

    def test_pk_k0001(self, bg, th, lcdm_pk_ref):
        """P(k=0.001) should match CLASS to < 5%."""
        pk_us = _compute_pk_single_k(0.001, bg, th, PREC)
        idx_ref = np.argmin(np.abs(lcdm_pk_ref['k'] - 0.001))
        pk_class = lcdm_pk_ref['pk_lin_z0'][idx_ref]
        ratio = pk_us / pk_class
        assert abs(ratio - 1) < 0.04, f"P(k=0.001): ratio={ratio:.4f}"

    def test_pk_k001(self, bg, th, lcdm_pk_ref):
        """P(k=0.01) should match CLASS to < 15%."""
        pk_us = _compute_pk_single_k(0.01, bg, th, PREC)
        idx_ref = np.argmin(np.abs(lcdm_pk_ref['k'] - 0.01))
        pk_class = lcdm_pk_ref['pk_lin_z0'][idx_ref]
        ratio = pk_us / pk_class
        assert abs(ratio - 1) < 0.05, f"P(k=0.01): ratio={ratio:.4f}"


class TestPkGradient:
    """Test that gradients through the full pipeline are correct."""

    @pytest.mark.slow
    def test_dpk_domega_cdm(self):
        """d(P(k=0.05))/d(omega_cdm) via AD should match finite differences."""
        prec_low = PrecisionParams(
            bg_n_points=200, ncdm_bg_n_points=100, bg_tol=1e-8,
            th_n_points=5000, th_z_max=5e3,
            pt_l_max_g=6, pt_l_max_pol_g=6, pt_l_max_ur=6,
        )

        def pk_scalar(omega_cdm):
            params = CosmoParams(omega_cdm=omega_cdm)
            bg = background_solve(params, prec_low)
            th = thermodynamics_solve(params, prec_low, bg)
            idx_loc = _build_indices(6, 6, 6)
            k = 0.05
            tau_ini = 0.5
            tau_end = bg.conformal_age * 0.999
            y0 = _adiabatic_ic(k, jnp.array(tau_ini), bg, params, idx_loc, idx_loc['n_eq'])
            args = (k, bg, th, params, idx_loc, 6, 6, 6)
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(_perturbation_rhs), solver=diffrax.Kvaerno5(),
                t0=tau_ini, t1=tau_end, dt0=tau_ini*0.1, y0=y0,
                saveat=diffrax.SaveAt(t1=True),
                stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
                adjoint=diffrax.RecursiveCheckpointAdjoint(), max_steps=16384, args=args,
            )
            y_f = sol.ys[-1]
            rho_b = bg.rho_b_of_loga.evaluate(jnp.array(0.0))
            rho_cdm = bg.rho_cdm_of_loga.evaluate(jnp.array(0.0))
            dm = (rho_b*y_f[idx_loc['delta_b']] + rho_cdm*y_f[idx_loc['delta_cdm']]) / (rho_b+rho_cdm)
            return dm**2

        omega_cdm_fid = 0.12
        grad_ad = float(jax.grad(pk_scalar)(omega_cdm_fid))
        eps = 1e-4
        grad_fd = float((pk_scalar(omega_cdm_fid + eps) - pk_scalar(omega_cdm_fid - eps)) / (2*eps))
        rel_err = abs(grad_ad - grad_fd) / abs(grad_fd)
        assert rel_err < 0.05, (
            f"d(P(k))/d(omega_cdm): AD={grad_ad:.4e} FD={grad_fd:.4e} err={rel_err:.2%}"
        )


class TestPerturbationRHS:
    """Test that the RHS function is well-formed."""

    def test_rhs_finite(self, bg, th):
        """RHS should return finite values."""
        idx = _build_indices(6, 6, 6)
        k = 0.05
        tau = 100.0
        y0 = _adiabatic_ic(k, jnp.array(0.5), bg, CosmoParams(), idx, idx['n_eq'])
        args = (k, bg, th, CosmoParams(), idx, 6, 6, 6)
        dy = _perturbation_rhs(jnp.array(tau), y0, args)
        assert jnp.all(jnp.isfinite(dy)), "RHS contains NaN/Inf"

    def test_ic_eta_unity(self, bg):
        """Initial η should be close to 1 (curvature normalization)."""
        idx = _build_indices(6, 6, 6)
        y0 = _adiabatic_ic(0.01, jnp.array(0.5), bg, CosmoParams(), idx, idx['n_eq'])
        eta_ini = float(y0[idx['eta']])
        assert abs(eta_ini - 1.0) < 0.01, f"eta_ini = {eta_ini}"
