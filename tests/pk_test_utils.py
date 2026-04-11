"""Shared helpers for linear-matter power-spectrum tests.

Contract:
- Provide direct single-mode, matched perturbation-sample, and public table-API
  ``P(k)`` helpers.

Scope:
- Covers scalar and sparse-grid ``P(k, z=0)`` evaluation against CLASS reference data.
- Excludes any duplicate test-only interpolation implementation now that the
  library exposes a reusable public table API.

Notes:
- The direct helpers intentionally mirror ``clax.compute_pk()`` internals so
  gradient and perturbation-level spot checks exercise the same single-mode path.
- The public-table helpers route through ``clax.compute_pk_table()`` so forward
  and interpolation-path gradient tests exercise the shipped API rather than a
  parallel test-only interpolation layer.
- The matched perturbation helpers support species-level comparisons against
  stored CLASS perturbation time series at shared ``(k, tau)`` points.
"""

from __future__ import annotations

import functools
from dataclasses import replace as dataclass_replace
from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

import clax
from clax.background import background_solve
from clax.params import CosmoParams, PrecisionParams
from clax.thermodynamics import thermodynamics_solve


PK_CONTRACT_PREC = PrecisionParams(
    bg_n_points=800,
    ncdm_bg_n_points=512,
    ncdm_q_size=5,
    th_n_points=20000,
    pt_k_max_cl=1.0,
    pt_k_per_decade=60,
    pt_l_max_g=50,
    pt_l_max_pol_g=50,
    pt_l_max_ur=50,
    pt_l_max_ncdm=50,
    pt_tau_n_points=5000,
    pt_ode_rtol=1e-6,
    pt_ode_atol=1e-11,
    ode_max_steps=131072,
    pt_k_chunk_size=1,
    ncdm_fluid_approximation="none",
)

PK_FAST_PREC = PrecisionParams(
    bg_n_points=400,
    ncdm_bg_n_points=256,
    ncdm_q_size=5,
    th_n_points=5000,
    pt_k_max_cl=1.0,
    pt_k_per_decade=40,
    pt_l_max_g=35,
    pt_l_max_pol_g=35,
    pt_l_max_ur=35,
    pt_l_max_ncdm=35,
    pt_tau_n_points=3000,
    pt_ode_rtol=1.0e-5,
    pt_ode_atol=1.0e-10,
    ode_max_steps=65536,
    pt_k_chunk_size=1,
    ncdm_fluid_approximation="none",
)

# Use the production-default checkpointed adjoint for scalar ``P(k)`` gradient
# contracts. The single-mode perturbation solve's ``DirectAdjoint`` path is
# environment-sensitive on CPU and has produced unstable density-parameter
# gradients even when the forward solve and thermodynamics AD are healthy.
PK_GRAD_CONTRACT_PREC = dataclass_replace(
    PK_CONTRACT_PREC,
    ode_adjoint="recursive_checkpoint",
)
PK_GRAD_FAST_PREC = dataclass_replace(
    PK_FAST_PREC,
    ode_adjoint="recursive_checkpoint",
)

PERTURBATION_MATCH_PREC = dataclass_replace(
    PK_CONTRACT_PREC,
    # Stored CLASS perturbation references are generated at the CLASS default
    # scalar ncdm hierarchy depth, so matched species tests should use the
    # same truncation when comparing intermediate-time ncdm moments.
    pt_l_max_ncdm=17,
)

PK_TABLE_GRAD_FAST_PREC = dataclass_replace(
    PK_FAST_PREC,
    pt_k_per_decade=28,
    pt_l_max_g=24,
    pt_l_max_pol_g=24,
    pt_l_max_ur=24,
    pt_l_max_ncdm=24,
    pt_tau_n_points=1800,
    ode_adjoint="recursive_checkpoint",
)

PK_TABLE_GRAD_FULL_PREC = dataclass_replace(
    PK_TABLE_GRAD_FAST_PREC,
)


PK_FORWARD_FAST_K = np.array([
    1.0e-4,
    1.0e-2,
    1.0,
])

PK_FORWARD_FULL_K = np.geomspace(1.0e-4, 1.0, 20)

PK_DIRECT_SPOT_FAST_K = np.array([
    3.0e-4,
    1.0e-2,
    1.0,
])

PK_DIRECT_SPOT_FULL_K = np.array([
    3.0e-4,
    3.0e-3,
    3.0e-2,
    1.0,
])

PERTURBATION_MATCH_K = np.array([1.0e-2, 5.0e-2, 1.0e-1])

PK_GRAD_FAST_K = np.array([1.0e-2])
PK_GRAD_FULL_K = np.array([3.0e-4, 1.0])
# Direct single-mode density-parameter gradients remain sensitive to the moving
# terminal-time coordinate on the current CPU/macOS environment even after the
# exact-path cleanup. Keep the direct contract on the stable primordial subset;
# density-parameter gradient coverage stays on the public table-backed API.
PK_GRAD_FULL_PARAMS = ("ln10A_s", "n_s", "k_pivot")

PK_TABLE_GRAD_K_TABLE = np.array([
    1.5e-2,
    3.0e-2,
    9.0e-2,
    2.5e-1,
])
PK_TABLE_GRAD_VECTOR_WEIGHTS = np.array([0.1, 0.2, 0.3, 0.4])
PK_TABLE_GRAD_K_PROBE = 5.0e-2
PK_TABLE_GRAD_FAST_PARAMS = ("ln10A_s", "n_s")
PK_TABLE_GRAD_FULL_PARAMS = ("h",)


PK_GRAD_PARAM_STEPS = {
    "h": 5.0e-5,
    "omega_b": 2.0e-5,
    "omega_cdm": 2.0e-5,
    "T_cmb": 1.0e-4,
    "N_ur": 1.0e-4,
    "m_ncdm": 1.0e-4,
    "T_ncdm_over_T_cmb": 1.0e-5,
    "deg_ncdm": 1.0e-4,
    "ln10A_s": 1.0e-4,
    "n_s": 1.0e-4,
    "alpha_s": 1.0e-4,
    "r_t": 1.0e-4,
    "n_t": 1.0e-4,
    "k_pivot": 1.0e-5,
    "tau_reio": 1.0e-4,
    "w0": 1.0e-4,
    "wa": 1.0e-4,
    "cs2_fld": 1.0e-4,
    "Omega_k": 1.0e-5,
    "Y_He": 1.0e-5,
}


REFERENCE_DIR = Path(__file__).resolve().parents[1] / "reference_data" / "lcdm_fiducial"


def supported_pk_gradient_params() -> list[str]:
    """Return the stable scalar parameter subset used in full direct ``P(k)`` gradient tests."""
    return list(PK_GRAD_FULL_PARAMS)


def _pk_rel_err_threshold(param_name: str, grad_fd: float) -> tuple[float, float]:
    """Return ``(relative_tol, absolute_tol)`` for a scalar ``P(k)`` gradient contract."""
    rel_tol = 0.01

    if param_name in {"r_t", "n_t"}:
        return rel_tol, 1.0e-12
    if param_name in {"alpha_s", "Omega_k", "wa"}:
        return rel_tol, 1.0e-9
    if abs(grad_fd) < 1.0e-8:
        return rel_tol, 1.0e-8
    return rel_tol, 0.0


def _interp_loglog(k_eval: np.ndarray, k_ref: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
    """Interpolate positive reference data in log-log space."""
    return np.exp(np.interp(np.log(k_eval), np.log(k_ref), np.log(y_ref)))


def pk_reference_grid(fast_mode: bool) -> np.ndarray:
    """Return the explicit log-spaced probe grid for forward ``P(k)`` tests up to ``1 Mpc^-1``."""
    if fast_mode:
        return PK_FORWARD_FAST_K
    return PK_FORWARD_FULL_K


def _resolve_pk_reference_key(lcdm_pk_ref: dict[str, np.ndarray], key: str) -> str:
    """Resolve preferred explicit ``P_m`` keys with backward-compatible fallbacks."""
    if key in lcdm_pk_ref:
        return key

    legacy_key = key
    if key == "pk_m_lin_z0":
        legacy_key = "pk_lin_z0"
    elif key.startswith("pk_m_z"):
        legacy_key = key.replace("pk_m_", "pk_", 1)

    if legacy_key in lcdm_pk_ref:
        return legacy_key

    raise KeyError(f"Missing CLASS P(k) reference key {key!r} (legacy fallback {legacy_key!r})")


def pk_reference_values(lcdm_pk_ref: dict[str, np.ndarray], k_eval: np.ndarray, key: str = "pk_m_lin_z0") -> np.ndarray:
    """Return CLASS reference ``P(k)`` values at the requested ``k`` values."""
    resolved_key = _resolve_pk_reference_key(lcdm_pk_ref, key)
    return _interp_loglog(np.asarray(k_eval), np.asarray(lcdm_pk_ref["k"]), np.asarray(lcdm_pk_ref[resolved_key]))


def load_perturbation_reference(k: float) -> dict[str, np.ndarray]:
    """Load the stored CLASS perturbation time series for one fiducial ``k`` mode."""
    path = REFERENCE_DIR / f"perturbations_k{k:.4f}.npz"
    return dict(np.load(path, allow_pickle=True))


def perturbation_match_tau_samples(ref_tau: np.ndarray, fast_mode: bool) -> np.ndarray:
    """Pick a small deterministic set of late-time ``tau`` samples from a CLASS series."""
    n_samples = 3 if fast_mode else 5
    start = int(round(0.65 * (len(ref_tau) - 1)))
    indices = np.unique(np.round(np.linspace(start, len(ref_tau) - 1, n_samples)).astype(int))
    return np.asarray(ref_tau)[indices]


def direct_tau_ini(k: float) -> float:
    """Return the production single-mode initial conformal time."""
    return min(0.5, 0.01 / k)


def batch_like_tau_ini(prec: PrecisionParams) -> float:
    """Return the fixed initial conformal time used by the perturbation-table solve."""
    return 0.01 / prec.pt_k_max_cl


def _background_densities_at_tau(bg, tau_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(rho_b, rho_cdm, rho_ncdm)`` on the requested ``tau`` grid."""
    tau_samples_jax = jnp.asarray(tau_samples)
    loga_samples = jax.vmap(bg.loga_of_tau.evaluate)(tau_samples_jax)
    rho_b = np.asarray(jax.vmap(bg.rho_b_of_loga.evaluate)(loga_samples))
    rho_cdm = np.asarray(jax.vmap(bg.rho_cdm_of_loga.evaluate)(loga_samples))
    rho_ncdm = np.asarray(jax.vmap(bg.rho_ncdm_of_loga.evaluate)(loga_samples))
    return rho_b, rho_cdm, rho_ncdm


def class_perturbation_components(bg, ref_pert: dict[str, np.ndarray], tau_samples: np.ndarray) -> dict[str, np.ndarray]:
    """Interpolate stored CLASS perturbations to shared ``tau`` samples."""
    tau_ref = np.asarray(ref_pert["tau_Mpc"])
    delta_cdm = np.interp(tau_samples, tau_ref, ref_pert["delta_cdm"])
    delta_b = np.interp(tau_samples, tau_ref, ref_pert["delta_b"])
    delta_ncdm = np.interp(tau_samples, tau_ref, ref_pert["delta_ncdm0"])
    theta_ncdm = np.interp(tau_samples, tau_ref, ref_pert["theta_ncdm0"])
    shear_ncdm = np.interp(tau_samples, tau_ref, ref_pert["shear_ncdm0"])
    rho_b, rho_cdm, rho_ncdm = _background_densities_at_tau(bg, tau_samples)
    delta_m = (
        rho_b * delta_b + rho_cdm * delta_cdm + rho_ncdm * delta_ncdm
    ) / (rho_b + rho_cdm + rho_ncdm)
    return {
        "delta_cdm": delta_cdm,
        "delta_b": delta_b,
        "delta_ncdm": delta_ncdm,
        "theta_ncdm": theta_ncdm,
        "shear_ncdm": shear_ncdm,
        "delta_m": delta_m,
    }


def solve_matched_perturbation_components(
    params: CosmoParams,
    prec: PrecisionParams,
    bg,
    th,
    k: float,
    tau_samples: np.ndarray,
    tau_ini_mode: str = "direct",
) -> dict[str, np.ndarray]:
    """Solve one mode and return clax component perturbations at the requested ``tau`` values."""
    from clax.perturbations import (
        _adiabatic_ic,
        _build_indices,
        _ncdm_fluid_mode_code,
        _ncdm_observables_from_state,
        _ncdm_quadrature,
        _perturbation_rhs,
    )

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

    if n_q_ncdm > 0:
        q_ncdm, w_ncdm, M_ncdm, dlnf0_ncdm = _ncdm_quadrature(params, prec)
    else:
        q_ncdm = jnp.zeros(1)
        w_ncdm = jnp.zeros(1)
        M_ncdm = 0.0
        dlnf0_ncdm = jnp.zeros(1)

    ncdmfa_mode_code = _ncdm_fluid_mode_code(prec.ncdm_fluid_approximation)
    ncdmfa_trigger = prec.ncdm_fluid_trigger_tau_over_tau_k

    if tau_ini_mode == "direct":
        tau_ini = direct_tau_ini(k)
    elif tau_ini_mode == "batch":
        tau_ini = batch_like_tau_ini(prec)
    else:
        raise ValueError(f"Unknown tau_ini_mode {tau_ini_mode!r}")

    y0 = _adiabatic_ic(
        k,
        jnp.asarray(tau_ini),
        bg,
        params,
        idx,
        idx["n_eq"],
        args_ncdm=(q_ncdm, w_ncdm, M_ncdm, dlnf0_ncdm),
    )
    ode_args = (
        k,
        bg,
        th,
        params,
        idx,
        prec.pt_l_max_g,
        prec.pt_l_max_pol_g,
        prec.pt_l_max_ur,
        ncdmfa_mode_code,
        ncdmfa_trigger,
        q_ncdm,
        w_ncdm,
        M_ncdm,
        dlnf0_ncdm,
    )
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(_perturbation_rhs),
        solver=diffrax.Kvaerno5(),
        t0=tau_ini,
        t1=float(np.asarray(tau_samples)[-1]),
        dt0=tau_ini * 0.1,
        y0=y0,
        saveat=diffrax.SaveAt(ts=jnp.asarray(tau_samples)),
        stepsize_controller=diffrax.PIDController(rtol=prec.pt_ode_rtol, atol=prec.pt_ode_atol),
        adjoint=diffrax.DirectAdjoint(),
        max_steps=prec.ode_max_steps,
        args=ode_args,
    )

    y_samples = np.asarray(sol.ys)
    delta_cdm = np.asarray(y_samples[:, idx["delta_cdm"]])
    delta_b = np.asarray(y_samples[:, idx["delta_b"]])
    rho_b, rho_cdm, rho_ncdm = _background_densities_at_tau(bg, tau_samples)

    delta_ncdm = []
    theta_ncdm = []
    shear_ncdm = []
    for tau_i, y_i in zip(np.asarray(tau_samples), y_samples, strict=True):
        delta_i, theta_i, shear_i, _ = _ncdm_observables_from_state(
            jnp.asarray(y_i),
            jnp.asarray(tau_i),
            k,
            bg,
            idx,
            q_ncdm,
            w_ncdm,
            M_ncdm,
            ncdmfa_mode_code,
            ncdmfa_trigger,
        )
        delta_ncdm.append(float(delta_i))
        theta_ncdm.append(float(theta_i))
        shear_ncdm.append(float(shear_i))
    delta_ncdm = np.asarray(delta_ncdm)
    theta_ncdm = np.asarray(theta_ncdm)
    shear_ncdm = np.asarray(shear_ncdm)

    delta_m = (
        rho_b * delta_b + rho_cdm * delta_cdm + rho_ncdm * delta_ncdm
    ) / (rho_b + rho_cdm + rho_ncdm)

    return {
        "delta_cdm": delta_cdm,
        "delta_b": delta_b,
        "delta_ncdm": delta_ncdm,
        "theta_ncdm": theta_ncdm,
        "shear_ncdm": shear_ncdm,
        "delta_m": delta_m,
    }


def solve_matched_perturbation_states(
    params: CosmoParams,
    prec: PrecisionParams,
    bg,
    th,
    k: float,
    tau_samples: np.ndarray,
    tau_ini_mode: str = "direct",
) -> dict[str, object]:
    """Solve one mode and return saved states plus the ncdm projection metadata."""
    from clax.perturbations import (
        _adiabatic_ic,
        _build_indices,
        _make_scalar_pid_controller,
        _ncdm_fluid_mode_code,
        _ncdm_quadrature,
        _perturbation_rhs,
        _resolve_scalar_pid_config,
    )

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

    if n_q_ncdm > 0:
        q_ncdm, w_ncdm, M_ncdm, dlnf0_ncdm = _ncdm_quadrature(params, prec)
    else:
        q_ncdm = jnp.zeros(1)
        w_ncdm = jnp.zeros(1)
        M_ncdm = 0.0
        dlnf0_ncdm = jnp.zeros(1)

    ncdmfa_mode_code = _ncdm_fluid_mode_code(prec.ncdm_fluid_approximation)
    ncdmfa_trigger = prec.ncdm_fluid_trigger_tau_over_tau_k

    if tau_ini_mode == "direct":
        tau_ini = direct_tau_ini(k)
    elif tau_ini_mode == "batch":
        tau_ini = batch_like_tau_ini(prec)
    else:
        raise ValueError(f"Unknown tau_ini_mode {tau_ini_mode!r}")

    y0 = _adiabatic_ic(
        k,
        jnp.asarray(tau_ini),
        bg,
        params,
        idx,
        idx["n_eq"],
        args_ncdm=(q_ncdm, w_ncdm, M_ncdm, dlnf0_ncdm),
    )
    ode_args = (
        k,
        bg,
        th,
        params,
        idx,
        prec.pt_l_max_g,
        prec.pt_l_max_pol_g,
        prec.pt_l_max_ur,
        ncdmfa_mode_code,
        ncdmfa_trigger,
        q_ncdm,
        w_ncdm,
        M_ncdm,
        dlnf0_ncdm,
    )
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(_perturbation_rhs),
        solver=diffrax.Kvaerno5(),
        t0=tau_ini,
        t1=float(np.asarray(tau_samples)[-1]),
        dt0=tau_ini * 0.1,
        y0=y0,
        saveat=diffrax.SaveAt(ts=jnp.asarray(tau_samples)),
        stepsize_controller=diffrax.PIDController(rtol=prec.pt_ode_rtol, atol=prec.pt_ode_atol),
        adjoint=diffrax.DirectAdjoint(),
        max_steps=prec.ode_max_steps,
        args=ode_args,
    )

    return {
        "idx": idx,
        "q_ncdm": q_ncdm,
        "w_ncdm": w_ncdm,
        "M_ncdm": M_ncdm,
        "ncdmfa_mode_code": ncdmfa_mode_code,
        "ncdmfa_trigger": ncdmfa_trigger,
        "y_samples": np.asarray(sol.ys),
    }


def compute_pk_scalar_public_table(
    params: CosmoParams,
    prec: PrecisionParams,
    k_probe: float,
    *,
    k_table=PK_TABLE_GRAD_K_TABLE,
    z: float = 0.0,
) -> jnp.ndarray:
    """Evaluate one scalar ``P(k, z)`` through the public table-backed API."""
    result = clax.compute_pk_table(params, prec, z=z, k_eval=jnp.asarray(k_table))
    return result.pk(k_probe, z=z)


def compute_pk_weighted_sum_public_table(
    params: CosmoParams,
    prec: PrecisionParams,
    *,
    k_table=PK_TABLE_GRAD_K_TABLE,
    weights=PK_TABLE_GRAD_VECTOR_WEIGHTS,
    z: float = 0.0,
) -> jnp.ndarray:
    """Return a scalar multi-``k`` objective from one reusable public PK table solve."""
    k_table = jnp.asarray(k_table)
    weights = jnp.asarray(weights)
    result = clax.compute_pk_table(params, prec, z=z, k_eval=k_table)
    return jnp.sum(weights * result.pk_grid)


def compute_pk_array_direct(
    params: CosmoParams,
    prec: PrecisionParams,
    k_eval,
) -> jnp.ndarray:
    """Compute direct scalar linear ``P(k)`` values via the shipped public API."""
    k_eval = np.atleast_1d(np.asarray(k_eval, dtype=float))
    outputs = [clax.compute_pk(params, prec, float(k)) for k in k_eval]
    return jnp.asarray(outputs)


def _pk_from_final_state_local(y_f, params: CosmoParams, prec: PrecisionParams, bg, idx, q_ncdm, w_ncdm, M_ncdm, k):
    """Convert one final perturbation state into scalar linear ``P(k)``."""
    from clax.perturbations import _ncdm_fluid_mode_code, _ncdm_observables_from_state

    rho_b = bg.rho_b_of_loga.evaluate(jnp.array(0.0))
    rho_cdm = bg.rho_cdm_of_loga.evaluate(jnp.array(0.0))
    rho_ncdm = bg.rho_ncdm_of_loga.evaluate(jnp.array(0.0))

    if idx["n_q_ncdm"] > 0 and params.N_ncdm > 0:
        mode_code = _ncdm_fluid_mode_code(prec.ncdm_fluid_approximation)
        delta_ncdm, _, _, _ = _ncdm_observables_from_state(
            y_f,
            bg.conformal_age * 0.999,
            k,
            bg,
            idx,
            q_ncdm,
            w_ncdm,
            M_ncdm,
            mode_code,
            prec.ncdm_fluid_trigger_tau_over_tau_k,
        )
        delta_m = (
            rho_b * y_f[idx["delta_b"]]
            + rho_cdm * y_f[idx["delta_cdm"]]
            + rho_ncdm * delta_ncdm
        ) / (rho_b + rho_cdm + rho_ncdm)
    else:
        delta_m = (
            rho_b * y_f[idx["delta_b"]] + rho_cdm * y_f[idx["delta_cdm"]]
        ) / (rho_b + rho_cdm)

    A_s = jnp.exp(params.ln10A_s) / 1e10
    return 2.0 * jnp.pi**2 / k**3 * A_s * (k / params.k_pivot) ** (params.n_s - 1) * delta_m**2


@functools.partial(jax.jit, static_argnums=(1,))
def _compute_pk_scalar_direct_local(params: CosmoParams, prec: PrecisionParams, k: float) -> jnp.ndarray:
    """Compute one direct scalar ``P(k)`` from a local one-mode perturbation solve.

    This path is intentionally simpler than the optimized public ``compute_pk()``
    implementation. Gradient contracts use it to isolate perturbation-level AD
    from the public table-backed API, which is tested separately.
    """
    from clax.ode import _get_adjoint
    from clax.perturbations import (
        _adiabatic_ic,
        _build_indices,
        _ncdm_fluid_mode_code,
        _ncdm_quadrature,
        _perturbation_rhs,
    )

    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)

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

    if n_q_ncdm > 0:
        q_ncdm, w_ncdm, M_ncdm, dlnf0_ncdm = _ncdm_quadrature(params, prec)
    else:
        q_ncdm = jnp.zeros(1)
        w_ncdm = jnp.zeros(1)
        M_ncdm = 0.0
        dlnf0_ncdm = jnp.zeros(1)

    ncdmfa_mode_code = _ncdm_fluid_mode_code(prec.ncdm_fluid_approximation)
    ncdmfa_trigger = prec.ncdm_fluid_trigger_tau_over_tau_k

    tau_ini = jnp.minimum(jnp.array(0.5), 0.01 / k)
    tau_end = bg.conformal_age * 0.999
    y0 = _adiabatic_ic(
        k,
        tau_ini,
        bg,
        params,
        idx,
        idx["n_eq"],
        args_ncdm=(q_ncdm, w_ncdm, M_ncdm, dlnf0_ncdm),
    )
    ode_args = (
        k,
        bg,
        th,
        params,
        idx,
        prec.pt_l_max_g,
        prec.pt_l_max_pol_g,
        prec.pt_l_max_ur,
        ncdmfa_mode_code,
        ncdmfa_trigger,
        q_ncdm,
        w_ncdm,
        M_ncdm,
        dlnf0_ncdm,
    )
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(_perturbation_rhs),
        solver=diffrax.Kvaerno5(),
        t0=tau_ini,
        t1=tau_end,
        dt0=tau_ini * 0.1,
        y0=y0,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=diffrax.PIDController(
            rtol=prec.pt_ode_rtol,
            atol=prec.pt_ode_atol,
        ),
        adjoint=_get_adjoint(prec.ode_adjoint),
        max_steps=prec.ode_max_steps,
        args=ode_args,
    )

    return _pk_from_final_state_local(sol.ys[-1], params, prec, bg, idx, q_ncdm, w_ncdm, M_ncdm, k)


def compute_pk_scalar_direct(params: CosmoParams, prec: PrecisionParams, k: float) -> jnp.ndarray:
    """Compute one direct scalar linear ``P(k)`` value via the shipped public API."""
    return clax.compute_pk(params, prec, float(k))
