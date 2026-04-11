"""Focused ncdm perturbation diagnostic at matched (k, tau) points.

Compares clax single-mode solutions against CLASS `get_perturbations()` output
for the stored fiducial k-modes. The goal is to isolate whether the remaining
P_m residual comes from:

- the direct-vs-batch setup difference in `tau_ini`
- the ncdm hierarchy / integrated-moment mapping itself

Usage:
    python diags/diag_ncdm_perturbations.py
    python diags/diag_ncdm_perturbations.py --fast
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace as dataclass_replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

from clax.background import background_solve
from clax.params import CosmoParams, PrecisionParams
from clax.perturbations import (
    _adiabatic_ic,
    _build_indices,
    _ncdm_fluid_mode_code,
    _ncdm_integrated_moments,
    _ncdm_quadrature,
    _perturbation_rhs,
)
from clax.thermodynamics import thermodynamics_solve

jax.config.update("jax_enable_x64", True)


def batch_tau_ini(prec: PrecisionParams) -> float:
    """Return the fixed batch-solver initial conformal time."""
    return 0.01 / prec.pt_k_max_cl


def direct_tau_ini(k: float) -> float:
    """Return the production single-mode initial conformal time."""
    return min(0.5, 0.01 / k)


def rel_err(value: np.ndarray, reference: np.ndarray, eps: float = 1.0e-30) -> np.ndarray:
    """Return elementwise relative error with safe normalization."""
    return np.abs(value - reference) / (np.abs(reference) + eps)


def late_tau_samples(ref_tau: np.ndarray, fast_mode: bool) -> np.ndarray:
    """Pick a small deterministic set of late-time tau samples."""
    n_samples = 3 if fast_mode else 5
    start = int(round(0.65 * (len(ref_tau) - 1)))
    indices = np.unique(np.round(np.linspace(start, len(ref_tau) - 1, n_samples)).astype(int))
    return ref_tau[indices]


def class_components_at_tau(bg, ref_pert: dict[str, np.ndarray], tau_samples: np.ndarray) -> dict[str, np.ndarray]:
    """Interpolate CLASS perturbation outputs to requested tau samples."""
    tau_ref = np.asarray(ref_pert["tau_Mpc"])
    delta_cdm = np.interp(tau_samples, tau_ref, ref_pert["delta_cdm"])
    delta_b = np.interp(tau_samples, tau_ref, ref_pert["delta_b"])
    delta_ncdm = np.interp(tau_samples, tau_ref, ref_pert["delta_ncdm0"])
    theta_ncdm = np.interp(tau_samples, tau_ref, ref_pert["theta_ncdm0"])
    shear_ncdm = np.interp(tau_samples, tau_ref, ref_pert["shear_ncdm0"])

    rho_b = np.asarray(jax.vmap(bg.rho_b_of_loga.evaluate)(jax.vmap(bg.loga_of_tau.evaluate)(jnp.asarray(tau_samples))))
    rho_cdm = np.asarray(jax.vmap(bg.rho_cdm_of_loga.evaluate)(jax.vmap(bg.loga_of_tau.evaluate)(jnp.asarray(tau_samples))))
    rho_ncdm = np.asarray(jax.vmap(bg.rho_ncdm_of_loga.evaluate)(jax.vmap(bg.loga_of_tau.evaluate)(jnp.asarray(tau_samples))))

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


def clax_components_at_tau(
    bg,
    y_samples,
    tau_samples: np.ndarray,
    k: float,
    idx,
    q_ncdm,
    w_ncdm,
    M_ncdm,
) -> dict[str, np.ndarray]:
    """Project clax states onto component density contrasts at each tau sample."""
    tau_samples_jax = jnp.asarray(tau_samples)
    loga_samples = jax.vmap(bg.loga_of_tau.evaluate)(tau_samples_jax)
    a_samples = jnp.exp(loga_samples)
    rho_b = np.asarray(jax.vmap(bg.rho_b_of_loga.evaluate)(loga_samples))
    rho_cdm = np.asarray(jax.vmap(bg.rho_cdm_of_loga.evaluate)(loga_samples))
    rho_ncdm = np.asarray(jax.vmap(bg.rho_ncdm_of_loga.evaluate)(loga_samples))

    delta_cdm = np.asarray(y_samples[:, idx["delta_cdm"]])
    delta_b = np.asarray(y_samples[:, idx["delta_b"]])

    delta_ncdm = []
    theta_ncdm = []
    shear_ncdm = []
    for tau_i, a_i, y_i in zip(tau_samples, np.asarray(a_samples), np.asarray(y_samples), strict=True):
        rho_delta_n, rho_plus_p_theta_n, rho_plus_p_shear_n, _, rho_unnorm_n, p_unnorm_n = _ncdm_integrated_moments(
            jnp.asarray(y_i), q_ncdm, w_ncdm, M_ncdm, a_i, k, idx
        )
        rho_plus_p_n = jnp.maximum(rho_unnorm_n + p_unnorm_n, 1.0e-30)
        delta_ncdm.append(float(rho_delta_n / jnp.maximum(rho_unnorm_n, 1.0e-30)))
        theta_ncdm.append(float(rho_plus_p_theta_n / rho_plus_p_n))
        shear_ncdm.append(float(rho_plus_p_shear_n / rho_plus_p_n))
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


def solve_single_mode(
    params: CosmoParams,
    prec: PrecisionParams,
    bg,
    th,
    k: float,
    tau_ini: float,
    tau_samples: np.ndarray,
    idx,
    q_ncdm,
    w_ncdm,
    M_ncdm,
    dlnf0_ncdm,
    ncdmfa_mode_code: int,
    ncdmfa_trigger: float,
):
    """Integrate one mode and save the state at the requested tau samples."""
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
        t1=float(tau_samples[-1]),
        dt0=tau_ini * 0.1,
        y0=y0,
        saveat=diffrax.SaveAt(ts=jnp.asarray(tau_samples)),
        stepsize_controller=diffrax.PIDController(rtol=prec.pt_ode_rtol, atol=prec.pt_ode_atol),
        adjoint=diffrax.DirectAdjoint(),
        max_steps=prec.ode_max_steps,
        args=ode_args,
    )
    return np.asarray(sol.ys)


def summarize_component_errors(label: str, tau_samples: np.ndarray, clax_values: dict[str, np.ndarray], class_values: dict[str, np.ndarray]) -> None:
    """Print concise max-error summaries for each component."""
    print(f"  {label}:")
    for field in ("delta_cdm", "delta_b", "delta_ncdm", "theta_ncdm", "shear_ncdm", "delta_m"):
        err = rel_err(clax_values[field], class_values[field])
        idx = int(np.argmax(err))
        print(
            f"    {field}: max_rel={100*err[idx]:.2f}% at tau={tau_samples[idx]:.1f} "
            f"(clax={clax_values[field][idx]:+.6e}, class={class_values[field][idx]:+.6e})"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Use 3 late-time tau samples instead of 5")
    args = parser.parse_args()

    params = CosmoParams()
    prec = dataclass_replace(PrecisionParams.planck_fast(), ncdm_fluid_approximation="none")

    t0 = time.time()
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    print(f"Background+thermo: {time.time() - t0:.1f}s")

    n_q_ncdm = prec.ncdm_q_size if params.N_ncdm > 0 else 0
    idx = _build_indices(
        prec.pt_l_max_g,
        prec.pt_l_max_pol_g,
        prec.pt_l_max_ur,
        n_q_ncdm,
        prec.pt_l_max_ncdm,
    )
    if n_q_ncdm > 0:
        q_ncdm, w_ncdm, M_ncdm, dlnf0_ncdm = _ncdm_quadrature(params, prec)
    else:
        raise RuntimeError("ncdm diagnostic requires ncdm_q_size > 0")
    ncdmfa_mode_code = _ncdm_fluid_mode_code(prec.ncdm_fluid_approximation)
    ncdmfa_trigger = prec.ncdm_fluid_trigger_tau_over_tau_k

    print(
        f"tau_ini(batch)={batch_tau_ini(prec):.6e}  "
        f"tau_ini(direct,k=0.05)={direct_tau_ini(0.05):.6e}"
    )

    for k in (0.01, 0.05, 0.1):
        ref_path = REPO_ROOT / "reference_data" / "lcdm_fiducial" / f"perturbations_k{k:.4f}.npz"
        ref_pert = dict(np.load(ref_path))
        tau_samples = late_tau_samples(np.asarray(ref_pert["tau_Mpc"]), args.fast)
        class_values = class_components_at_tau(bg, ref_pert, tau_samples)

        print(f"\n=== k={k:.3f} Mpc^-1 ===")
        print("  tau samples:", ", ".join(f"{tau:.1f}" for tau in tau_samples))

        direct_states = solve_single_mode(
            params,
            prec,
            bg,
            th,
            k,
            direct_tau_ini(k),
            tau_samples,
            idx,
            q_ncdm,
            w_ncdm,
            M_ncdm,
            dlnf0_ncdm,
            ncdmfa_mode_code,
            ncdmfa_trigger,
        )
        batch_states = solve_single_mode(
            params,
            prec,
            bg,
            th,
            k,
            batch_tau_ini(prec),
            tau_samples,
            idx,
            q_ncdm,
            w_ncdm,
            M_ncdm,
            dlnf0_ncdm,
            ncdmfa_mode_code,
            ncdmfa_trigger,
        )

        direct_values = clax_components_at_tau(bg, direct_states, tau_samples, k, idx, q_ncdm, w_ncdm, M_ncdm)
        batch_values = clax_components_at_tau(bg, batch_states, tau_samples, k, idx, q_ncdm, w_ncdm, M_ncdm)

        summarize_component_errors("direct tau_ini", tau_samples, direct_values, class_values)
        summarize_component_errors("batch-like tau_ini", tau_samples, batch_values, class_values)

        final_idx = -1
        print(
            "  setup drift at final tau: "
            f"delta_ncdm={100*(direct_values['delta_ncdm'][final_idx] / batch_values['delta_ncdm'][final_idx] - 1.0):+.2f}%  "
            f"delta_m={100*(direct_values['delta_m'][final_idx] / batch_values['delta_m'][final_idx] - 1.0):+.2f}%"
        )


if __name__ == "__main__":
    main()
