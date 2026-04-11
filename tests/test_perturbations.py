"""Tests perturbation-solver forward behavior.

Contract:
- The perturbation ODE system is structurally well-posed and integrates to finite states.

Scope:
- Covers initial conditions, RHS finiteness, RHS shape, one cheap single-mode solve,
  direct single-mode ``P(k)`` spot checks, and matched species-level perturbation checks.
- Excludes public table/interpolated ``P(k)`` accuracy and gradient contracts
  owned by dedicated files.

Notes:
- Direct scalar ``P(k)`` spot checks live here because they exercise the
  expensive single-mode perturbation solve rather than the table/interpolation path.
- Massive-neutrino (`ncdm`) perturbation checks also live here because the
  remaining linear ``P_m(k)`` discrepancy is now localized to the perturbation layer,
  including the neutrino density, velocity, and shear moments.
"""

import functools
from dataclasses import replace as dataclass_replace

import jax
jax.config.update("jax_enable_x64", True)

import diffrax
import jax.numpy as jnp
import numpy as np
import pytest

import clax
from clax.background import background_solve
from clax.params import CosmoParams, PrecisionParams
from clax.perturbations import (
    SCALAR_PID_FILTERED_VARIABLE_NAMES,
    _adiabatic_ic,
    _build_indices,
    _ncdm_integrated_moments,
    _ncdm_observables_from_state,
    _perturbation_rhs,
    _pt_saved_output_count,
    _resolve_pt_k_batch_size,
    _rms_norm_safe,
    _scalar_pid_filtered_variable_indices,
    _scalar_pid_filtered_variable_weights,
    _scalar_pid_filtered_rms_norm,
)
from clax.thermodynamics import thermodynamics_solve
from tests.pk_test_utils import (
    PK_CONTRACT_PREC,
    PK_DIRECT_SPOT_FAST_K,
    PK_DIRECT_SPOT_FULL_K,
    PK_FAST_PREC,
    PERTURBATION_MATCH_PREC,
    PERTURBATION_MATCH_K,
    class_perturbation_components,
    compute_pk_array_direct,
    load_perturbation_reference,
    pk_reference_values,
    perturbation_match_tau_samples,
    solve_matched_perturbation_components,
    solve_matched_perturbation_states,
)


PREC = PrecisionParams(
    bg_n_points=400,
    ncdm_bg_n_points=200,
    bg_tol=1e-8,
    th_n_points=10000,
    th_z_max=5e3,
    pt_l_max_g=17,
    pt_l_max_pol_g=17,
    pt_l_max_ur=17,
)


@pytest.fixture(scope="module")
def bg():
    """Compute the fiducial background state once for this module."""
    return background_solve(CosmoParams(), PREC)


@pytest.fixture(scope="module")
def th(bg):
    """Compute the fiducial thermodynamics state once for this module."""
    return thermodynamics_solve(CosmoParams(), PREC, bg)


def _solve_single_mode(bg, th, k=0.05):
    """Integrate one low-cost perturbation mode; expects a finite final state."""
    idx = _build_indices(6, 6, 6)
    tau_ini = min(0.5, 0.01 / k)
    y0 = _adiabatic_ic(k, jnp.array(tau_ini), bg, CosmoParams(), idx, idx["n_eq"])
    args = (k, bg, th, CosmoParams(), idx, 6, 6, 6)
    return diffrax.diffeqsolve(
        diffrax.ODETerm(_perturbation_rhs),
        solver=diffrax.Kvaerno5(),
        t0=tau_ini,
        t1=1.0,
        dt0=tau_ini * 0.1,
        y0=y0,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        max_steps=16384,
        args=args,
    )


class TestPerturbationInitialConditions:
    """Tests perturbation initial conditions."""

    def test_ic_state_is_finite(self, bg):
        """Initial conditions are finite; expects no NaN or Inf entries."""
        idx = _build_indices(6, 6, 6)
        y0 = _adiabatic_ic(0.01, jnp.array(0.5), bg, CosmoParams(), idx, idx["n_eq"])
        assert jnp.all(jnp.isfinite(y0)), "Initial conditions: found non-finite entries; expected all finite"

    def test_ic_eta_matches_curvature_normalization(self, bg):
        """Initial ``eta`` matches curvature normalization; expects <1% offset from unity."""
        idx = _build_indices(6, 6, 6)
        y0 = _adiabatic_ic(0.01, jnp.array(0.5), bg, CosmoParams(), idx, idx["n_eq"])
        eta_ini = float(y0[idx["eta"]])
        assert abs(eta_ini - 1.0) < 0.01, f"eta_ini: value {eta_ini:.6f}; expected within 1% of unity"


class TestPerturbationRhs:
    """Tests perturbation RHS evaluation."""

    def test_rhs_is_finite(self, bg, th):
        """RHS evaluated on valid ICs is finite; expects no NaN or Inf entries."""
        idx = _build_indices(6, 6, 6)
        y0 = _adiabatic_ic(0.05, jnp.array(0.5), bg, CosmoParams(), idx, idx["n_eq"])
        args = (0.05, bg, th, CosmoParams(), idx, 6, 6, 6)
        dy = _perturbation_rhs(jnp.array(100.0), y0, args)
        assert jnp.all(jnp.isfinite(dy)), "RHS: found non-finite entries; expected all finite"

    def test_rhs_shape_matches_state(self, bg, th):
        """RHS output shape matches state shape; expects identical array shapes."""
        idx = _build_indices(6, 6, 6)
        y0 = _adiabatic_ic(0.05, jnp.array(0.5), bg, CosmoParams(), idx, idx["n_eq"])
        args = (0.05, bg, th, CosmoParams(), idx, 6, 6, 6)
        dy = _perturbation_rhs(jnp.array(100.0), y0, args)
        assert dy.shape == y0.shape, f"RHS shape: got {dy.shape}; expected {y0.shape}"


class TestPerturbationIntegration:
    """Tests cheap single-mode perturbation integration."""

    def test_single_mode_final_state_is_finite(self, bg, th):
        """A very short single-mode solve reaches a finite final state; expects no NaN or Inf entries."""
        sol = _solve_single_mode(bg, th, k=0.01)
        y_final = sol.ys[-1]
        assert jnp.all(jnp.isfinite(y_final)), "Single-mode solve: found non-finite final-state entries; expected all finite"

    def test_scalar_pid_norm_is_finite_and_differentiable_at_zero(self):
        """Scalar PID norm helpers stay finite and have finite gradients at zero-valued inputs."""
        idx = _build_indices(6, 6, 6)
        filter_indices = _scalar_pid_filtered_variable_indices(idx)
        filter_weights = _scalar_pid_filtered_variable_weights(jnp.array(0.05))
        x0 = jnp.zeros(idx["n_eq"])

        norm_val = _scalar_pid_filtered_rms_norm(x0, filter_indices, filter_weights)
        grad_val = jax.grad(lambda x: _scalar_pid_filtered_rms_norm(x, filter_indices, filter_weights))(x0)
        rms_grad = jax.grad(lambda x: _rms_norm_safe(x))(jnp.zeros(6))

        assert jnp.isfinite(norm_val), "Scalar PID norm: expected finite value at zero input"
        assert jnp.all(jnp.isfinite(grad_val)), "Scalar PID norm: expected finite gradient at zero input"
        assert jnp.all(jnp.isfinite(rms_grad)), "Safe RMS norm: expected finite gradient at zero input"

    def test_scalar_pid_filtered_variables_match_fixed_discoeb_layout(self):
        """Scalar PID control should use the fixed DISCO-EB-style six-variable filter."""
        idx = _build_indices(6, 6, 6)
        filter_indices = _scalar_pid_filtered_variable_indices(idx)

        assert SCALAR_PID_FILTERED_VARIABLE_NAMES == (
            "eta",
            "delta_cdm",
            "delta_b",
            "F_g_0",
            "theta_b",
            "F_g_1",
        )
        assert tuple(filter_indices.shape) == (6,)
        assert tuple(int(i) for i in filter_indices) == tuple(idx[name] for name in SCALAR_PID_FILTERED_VARIABLE_NAMES)

    def test_scalar_pid_filtered_weights_match_fixed_discoeb_recipe(self):
        """Scalar PID control should use the fixed DISCO-EB-style k-dependent weights."""
        k = jnp.array(0.05)
        weights = _scalar_pid_filtered_variable_weights(k)
        expected = jnp.asarray([k**2, 1.0, 1.0, 1.0, 1.0 / k**2, 1.0])

        assert tuple(weights.shape) == (6,)
        assert jnp.allclose(weights, expected), f"Scalar PID weights: got {weights}, expected {expected}"

    def test_k_batch_size_auto_mode_is_finite(self):
        """Auto-batched ``k`` execution should choose a finite positive batch size."""
        batch_size_full = _resolve_pt_k_batch_size(
            PK_FAST_PREC,
            n_k=200,
            n_tau=3000,
            n_outputs=_pt_saved_output_count(solve_kind="full"),
            solve_kind="full",
        )
        batch_size_mpk = _resolve_pt_k_batch_size(
            PK_FAST_PREC,
            n_k=200,
            n_tau=1500,
            n_outputs=_pt_saved_output_count(solve_kind="mpk"),
            solve_kind="mpk",
        )
        assert isinstance(batch_size_full, int)
        assert batch_size_full > 0, f"Auto full batch size should be positive, got {batch_size_full}"
        assert isinstance(batch_size_mpk, int)
        assert batch_size_mpk > 0, f"Auto mPk batch size should be positive, got {batch_size_mpk}"

    def test_mpk_saved_output_heuristic_is_no_more_restrictive_than_full_state_guess(self):
        """The reduced ``mPk`` path should not be penalized by a full-state output estimate."""
        saved_output_batch = _resolve_pt_k_batch_size(
            PK_FAST_PREC,
            n_k=200,
            n_tau=1500,
            n_outputs=_pt_saved_output_count(solve_kind="mpk"),
            solve_kind="mpk",
        )
        full_state_guess_batch = _resolve_pt_k_batch_size(
            PK_FAST_PREC,
            n_k=200,
            n_tau=1500,
            n_outputs=200,
            solve_kind="mpk",
        )
        assert saved_output_batch >= full_state_guess_batch, (
            f"Reduced-output heuristic regressed: saved-output batch {saved_output_batch} "
            f"should be >= full-state guess {full_state_guess_batch}"
        )

    def test_k_batch_size_negative_chunk_means_full_vmap(self):
        """Negative ``pt_k_chunk_size`` should explicitly request full-``vmap`` execution."""
        prec = dataclass_replace(PK_FAST_PREC, pt_k_chunk_size=-1)
        batch_size = _resolve_pt_k_batch_size(
            prec,
            n_k=200,
            n_tau=3000,
            n_outputs=_pt_saved_output_count(solve_kind="full"),
            solve_kind="full",
        )
        assert batch_size == 200, f"Expected explicit full-vmap batch size 200, got {batch_size}"


class TestPerturbationPkSpotChecks:
    """Tests perturbation-layer scalar ``P(k)`` spot checks and table/direct consistency."""

    @pytest.mark.slow
    def test_direct_pk_spot_checks_match_class(self, lcdm_pk_ref, fast_mode):
        """Direct single-mode ``P(k, z=0)`` spot checks match CLASS; expects <1% max relative error."""
        k_eval = PK_DIRECT_SPOT_FAST_K if fast_mode else PK_DIRECT_SPOT_FULL_K
        prec = PK_FAST_PREC if fast_mode else PK_CONTRACT_PREC
        pk_clax = np.asarray(compute_pk_array_direct(CosmoParams(), prec, k_eval))
        pk_class = pk_reference_values(lcdm_pk_ref, k_eval, key="pk_m_lin_z0")
        rel_err = np.abs(pk_clax / pk_class - 1.0)
        max_err = float(np.max(rel_err))
        worst_idx = int(np.argmax(rel_err))
        worst_k = float(k_eval[worst_idx])

        assert max_err < 0.01, (
            f"Direct perturbation P(k): relative error {max_err:.2%} at k={worst_k:.6g} Mpc^-1; "
            f"clax={pk_clax[worst_idx]:.6e}, CLASS={pk_class[worst_idx]:.6e}, expected <1%"
        )

    @pytest.mark.slow
    def test_pk_table_tracks_direct_single_mode_solves(self, fast_mode):
        """Table-backed ``P(k)`` agrees with direct scalar solves at representative grid points."""
        k_eval = PK_DIRECT_SPOT_FAST_K if fast_mode else PK_DIRECT_SPOT_FULL_K
        prec = PK_FAST_PREC if fast_mode else PK_CONTRACT_PREC
        pk_table = np.asarray(clax.compute_pk_table(CosmoParams(), prec, k_eval=k_eval).pk_grid)
        pk_direct = np.asarray(compute_pk_array_direct(CosmoParams(), prec, k_eval))
        rel_err = np.abs(pk_table / pk_direct - 1.0)
        max_err = float(np.max(rel_err))
        worst_idx = int(np.argmax(rel_err))

        assert max_err < 0.01, (
            f"Table/direct mismatch {max_err:.2%} at k={float(k_eval[worst_idx]):.6g} Mpc^-1; "
            f"table={pk_table[worst_idx]:.6e}, direct={pk_direct[worst_idx]:.6e}, expected <1%"
        )


@functools.lru_cache(maxsize=2)
def _matched_species_results(fast_mode: bool):
    """Cache matched `(k, tau)` species comparisons for the fiducial cosmology."""
    prec_base = PrecisionParams.planck_fast() if fast_mode else PERTURBATION_MATCH_PREC
    prec = dataclass_replace(prec_base, ncdm_fluid_approximation="none")
    params = CosmoParams()
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)

    results = {}
    for k in PERTURBATION_MATCH_K:
        ref_pert = load_perturbation_reference(float(k))
        tau_samples = perturbation_match_tau_samples(ref_pert["tau_Mpc"], fast_mode)
        class_values = class_perturbation_components(bg, ref_pert, tau_samples)
        clax_values = solve_matched_perturbation_components(
            params,
            prec,
            bg,
            th,
            float(k),
            tau_samples,
            tau_ini_mode="direct",
        )
        results[float(k)] = {
            "tau": tau_samples,
            "class": class_values,
            "clax": clax_values,
        }

    return results


class TestPerturbationSpeciesContracts:
    """Tests matched species-level perturbations against stored CLASS time series."""

    @pytest.mark.slow
    def test_matched_species_components_track_class(self, fast_mode):
        """``delta_cdm``, ``delta_b``, and ``delta_m`` stay within a low-percent envelope."""
        thresholds = {
            "delta_cdm": 0.03,
            "delta_b": 0.035,
            "delta_m": 0.03,
        }
        failures = []
        for k, result in _matched_species_results(bool(fast_mode)).items():
            tau = result["tau"]
            for field, threshold in thresholds.items():
                rel_err = np.abs(result["clax"][field] / result["class"][field] - 1.0)
                worst_idx = int(np.argmax(rel_err))
                max_err = float(rel_err[worst_idx])
                if max_err >= threshold:
                    failures.append(
                        f"{field} at k={k:.3f}: rel err {max_err:.2%} at tau={tau[worst_idx]:.1f} "
                        f"(clax={result['clax'][field][worst_idx]:+.6e}, "
                        f"CLASS={result['class'][field][worst_idx]:+.6e}, "
                        f"expected < {threshold:.0%})"
                    )

        assert not failures, "\n".join(failures)

    @pytest.mark.slow
    def test_matched_delta_ncdm_matches_class(self, fast_mode):
        """``delta_ncdm`` should eventually match CLASS at matched ``(k, tau)`` points."""
        threshold = 0.002
        failures = []
        for k, result in _matched_species_results(bool(fast_mode)).items():
            tau = result["tau"]
            rel_err = np.abs(result["clax"]["delta_ncdm"] / result["class"]["delta_ncdm"] - 1.0)
            worst_idx = int(np.argmax(rel_err))
            max_err = float(rel_err[worst_idx])
            if max_err >= threshold:
                failures.append(
                    f"delta_ncdm at k={k:.3f}: rel err {max_err:.2%} at tau={tau[worst_idx]:.1f} "
                    f"(clax={result['clax']['delta_ncdm'][worst_idx]:+.6e}, "
                    f"CLASS={result['class']['delta_ncdm'][worst_idx]:+.6e}, "
                    f"expected < {threshold:.0%})"
                )

        assert not failures, "\n".join(failures)

    @pytest.mark.slow
    def test_matched_ncdm_velocity_and_shear_match_class(self, fast_mode):
        """``theta_ncdm`` and ``shear_ncdm`` should eventually match CLASS at matched ``(k, tau)`` points."""
        thresholds = {
            "theta_ncdm": 0.002,
            "shear_ncdm": 0.002,
        }
        failures = []
        for k, result in _matched_species_results(bool(fast_mode)).items():
            tau = result["tau"]
            for field, threshold in thresholds.items():
                rel_err = np.abs(result["clax"][field] / result["class"][field] - 1.0)
                worst_idx = int(np.argmax(rel_err))
                max_err = float(rel_err[worst_idx])
                if max_err >= threshold:
                    failures.append(
                        f"{field} at k={k:.3f}: rel err {max_err:.2%} at tau={tau[worst_idx]:.1f} "
                        f"(clax={result['clax'][field][worst_idx]:+.6e}, "
                        f"CLASS={result['class'][field][worst_idx]:+.6e}, "
                        f"expected < {threshold:.0%})"
                    )

        assert not failures, "\n".join(failures)


class TestNcdmMomentSanity:
    """Tests local consistency properties of the integrated ncdm moments."""

    @pytest.mark.slow
    def test_ncdm_observable_projection_matches_integrated_moments(self, fast_mode):
        """The no-fluid ncdm observable helper should match direct integrated-moment projection."""
        prec_base = PrecisionParams.planck_fast() if fast_mode else PERTURBATION_MATCH_PREC
        prec = dataclass_replace(prec_base, ncdm_fluid_approximation="none")
        params = CosmoParams()
        bg = background_solve(params, prec)
        th = thermodynamics_solve(params, prec, bg)
        k = 0.05
        ref_pert = load_perturbation_reference(k)
        tau_samples = perturbation_match_tau_samples(ref_pert["tau_Mpc"], bool(fast_mode))
        solve_result = solve_matched_perturbation_states(
            params,
            prec,
            bg,
            th,
            k,
            tau_samples,
            tau_ini_mode="direct",
        )

        failures = []
        for tau_i, y_i in zip(tau_samples, solve_result["y_samples"], strict=True):
            y_jax = jnp.asarray(y_i)
            tau_jax = jnp.asarray(tau_i)
            delta_obs, theta_obs, shear_obs, _ = _ncdm_observables_from_state(
                y_jax,
                tau_jax,
                k,
                bg,
                solve_result["idx"],
                solve_result["q_ncdm"],
                solve_result["w_ncdm"],
                solve_result["M_ncdm"],
                solve_result["ncdmfa_mode_code"],
                solve_result["ncdmfa_trigger"],
            )
            a_i = float(jnp.exp(bg.loga_of_tau.evaluate(tau_jax)))
            rho_delta, rho_plus_p_theta, rho_plus_p_shear, _, rho_unnorm, p_unnorm = _ncdm_integrated_moments(
                y_jax,
                solve_result["q_ncdm"],
                solve_result["w_ncdm"],
                solve_result["M_ncdm"],
                a_i,
                k,
                solve_result["idx"],
            )
            rho_plus_p = jnp.maximum(rho_unnorm + p_unnorm, 1.0e-30)
            delta_direct = rho_delta / jnp.maximum(rho_unnorm, 1.0e-30)
            theta_direct = rho_plus_p_theta / rho_plus_p
            shear_direct = rho_plus_p_shear / rho_plus_p

            for field, lhs, rhs in (
                ("delta_ncdm", delta_obs, delta_direct),
                ("theta_ncdm", theta_obs, theta_direct),
                ("shear_ncdm", shear_obs, shear_direct),
            ):
                abs_err = abs(float(lhs - rhs))
                if abs_err > 1.0e-10:
                    failures.append(
                        f"{field} at tau={float(tau_i):.1f}: helper={float(lhs):+.6e} "
                        f"direct={float(rhs):+.6e} abs_err={abs_err:.3e}"
                    )

        assert not failures, "\n".join(failures)

    def test_ncdm_shear_vanishes_without_quadrupole(self):
        """If all ``Psi_2`` entries are zero, the integrated ncdm shear should vanish."""
        idx = _build_indices(2, 2, 2, n_q_ncdm=2, l_max_ncdm=2)
        y = jnp.zeros(idx["n_eq"])
        q = jnp.array([1.0, 2.0])
        w = jnp.array([0.6, 0.4])
        M = 0.0
        a = 1.0
        k = 0.1

        y = y.at[idx["psi_ncdm_start"]].set(1.0)
        y = y.at[idx["psi_ncdm_start"] + 3].set(2.0)
        _, _, rho_plus_p_shear, _, _, _ = _ncdm_integrated_moments(y, q, w, M, a, k, idx)

        assert float(rho_plus_p_shear) == pytest.approx(0.0, abs=1.0e-14)

    def test_ncdm_relativistic_pressure_tracks_density(self):
        """In the relativistic limit, the integrated pressure perturbation should be ``delta_p = delta_rho / 3``."""
        idx = _build_indices(2, 2, 2, n_q_ncdm=2, l_max_ncdm=2)
        y = jnp.zeros(idx["n_eq"])
        q = jnp.array([1.0, 2.0])
        w = jnp.array([0.6, 0.4])
        M = 0.0
        a = 1.0
        k = 0.1

        psi0_q0 = idx["psi_ncdm_start"]
        psi0_q1 = idx["psi_ncdm_start"] + 3
        y = y.at[psi0_q0].set(1.5)
        y = y.at[psi0_q1].set(0.5)

        rho_delta, _, _, delta_p, _, _ = _ncdm_integrated_moments(y, q, w, M, a, k, idx)

        assert float(delta_p / rho_delta) == pytest.approx(1.0 / 3.0, rel=1.0e-12, abs=1.0e-12)
