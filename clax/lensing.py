"""Lensing module for clax.

Computes lensed CMB power spectra from unlensed C_l's and the lensing
potential power spectrum C_l^phiphi.

Uses the correlation function method (Gaussian approximation) with the
addback technique for numerical stability:
1. Compute deflection correlation Cgl(mu) and sigma^2(mu) using d^l_{11}
2. Forward transform: build lensed-minus-unlensed correlation functions
   using Wigner d-functions weighted by (lensing_exponential - 1)
3. Inverse transform via Gauss-Legendre quadrature
4. Add back unlensed C_l

Supports TT, TE, EE, BB lensing with proper spin-2 Wigner d-matrices.

References:
    CLASS lensing.c (Kostelec-Rockmore recurrences for Wigner d-functions)
    Challinor & Lewis (2005) Phys.Rev. D71 103010
    Lewis (2005) Phys.Rev. D71 083008
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from clax.background import BackgroundResult
from clax.bessel import spherical_jl
from clax.params import CosmoParams, PrecisionParams
from clax.perturbations import MatterPerturbationResult, PerturbationResult
from clax.primordial import primordial_scalar_pk


def _compute_cl_pp_full_bessel(
    pt: PerturbationResult,
    params: CosmoParams,
    bg: BackgroundResult,
    th,
    l_values: Float[Array, "Nl"],
) -> Float[Array, "Nl"]:
    """Slow oracle: C_l^phiphi via the full Bessel transfer function integral.

    Mirrors CLASS transfer.c:2428 + harmonic.c:1073-1108 exactly:

        source(k, tau) = (phi+psi)(k, tau) * W_lcmb(tau)
        T_l(k) = integral_{tau>tau_rec} dtau * source * j_l(k*chi)
        C_l^pp = 4pi * integral d(lnk) * P_R(k) * T_l(k)^2

    This is the slow, dense-Bessel reference path. The public
    ``compute_cl_pp`` uses the source-Limber kernel (CLASS-accurate at all l
    and ~100x faster). This routine is kept private as an independent
    oracle that tests cross-check against.

    Args:
        pt: perturbation results (must have source_phi_plus_psi)
        params: cosmological parameters
        bg: background results
        th: thermodynamics results (for tau_rec)
        l_values: multipoles at which to evaluate

    Returns:
        C_l^phiphi at each l in l_values
    """
    tau_grid = pt.tau_grid
    k_grid = pt.k_grid
    tau_0 = bg.conformal_age

    # tau_rec: start of lensing integration (CLASS transfer.c:1712)
    loga_rec = jnp.log(1.0 / (1.0 + th.z_rec))
    tau_rec = bg.tau_of_loga.evaluate(loga_rec)

    chi_grid = tau_0 - tau_grid
    chi_rec = tau_0 - tau_rec

    # Geometric lensing kernel: W = (tau_rec - tau) / [(tau_0-tau)(tau_0-tau_rec)]
    # cf. CLASS transfer.c:2428 (flat geometry)
    chi_nonzero = jnp.where(chi_grid > 0.0, chi_grid, 1.0)
    W_lcmb = (tau_rec - tau_grid) / (chi_nonzero * chi_rec)
    W_lcmb = jnp.where(chi_grid > 0.0, W_lcmb, 0.0)

    # Build transfer source: (phi+psi) * W, zeroed for tau <= tau_rec
    # (CLASS transfer.c:1712 discards times before recombination)
    S_transfer = pt.source_phi_plus_psi * W_lcmb[None, :]
    S_transfer = jnp.where(tau_grid[None, :] > tau_rec, S_transfer, 0.0)

    dtau = jnp.diff(tau_grid)
    dtau_mid = jnp.concatenate([dtau[:1], (dtau[:-1] + dtau[1:]) / 2, dtau[-1:]])
    log_k = jnp.log(k_grid)
    P_R = primordial_scalar_pk(k_grid, params)

    def compute_clpp_single_l(l):
        l_int = int(l)
        l_fl = jnp.float64(l)

        def transfer_single_k(ik):
            k = k_grid[ik]
            x = k * chi_grid
            jl = spherical_jl(l_int, x)
            return jnp.sum(S_transfer[ik, :] * jl * dtau_mid)

        T_l = jax.vmap(transfer_single_k)(jnp.arange(len(k_grid)))
        # No [2/(l(l+1))]^2 prefactor — CLASS stores C_l^pp directly
        # cf. CLASS harmonic.c:1073 (factor = 4*PI/k for all spectra)
        integrand = P_R * T_l**2
        dlnk = jnp.diff(log_k)
        return 4.0 * jnp.pi * jnp.sum(
            0.5 * (integrand[:-1] + integrand[1:]) * dlnk)

    cls = []
    for l in l_values:
        cl = compute_clpp_single_l(l)
        cls.append(cl)
    return jnp.array(cls)




def _halofit_modulator(
    pt: PerturbationResult,
    params: CosmoParams,
    bg: BackgroundResult,
    th,
    n_z: int = 100,
    k_max_extend: float = 20.0,
    *,
    pt_mpk: MatterPerturbationResult,
) -> Float[Array, "Nk Ntau"]:
    """Build sqrt(R(k, tau)) modulator for source-Limber NL injection.

    Computes ``R(k, z) = P_NL(k, z) / P_lin(k, z)`` from Halofit on a
    ``(k, z)`` grid via ``vmap(compute_pk_nonlinear)``, then interpolates
    onto the perturbation ``(k_grid, tau_grid)`` via log-linear
    interpolation in z (using ``bg.loga_of_tau`` for τ→z).

    CLASS-aligned defaults:

    - ``n_z = 100`` matches CLASS's default ``z`` array density in
      ``fourier_init``.
    - ``k_max_extend = 20`` Mpc^-1: CLASS computes Halofit on its own
      nonlinear k-grid that extends past the perturbation k-range; we
      do the equivalent by power-law-extending ``pt.k_grid`` in log-log
      space to k_max=20 Mpc^-1 before evaluating ``compute_pk_nonlinear``.
      Empirically, k=20 is the sweet spot — far enough for σ(R) bisection
      to fully converge, not so far that the local power-law slope drifts
      from the true asymptote. Residual vs CLASS Halofit reference is
      <0.8% across all ℓ ≤ 2500 (typically <0.15% at ℓ ≤ 1500). Going
      to k=30+ degrades accuracy because the power-law extrapolation
      diverges from the true transfer-function shape. Set
      ``k_max_extend = 0`` to disable extension (CLASS-no-extrapolation
      behavior, ~3-4% extra residual at ℓ ≥ 1500). The inline
      ``sigma_R`` check (replicating CLASS ``fourier.c:1706-1716``)
      still forces ``R = 1`` when ``sigma(R_min) < 1`` regardless of
      extension setting.

    Args:
        pt: perturbation results (k_grid, tau_grid, delta_m source)
        params: cosmological parameters
        bg: background results
        th: thermodynamics results (for z_rec)
        n_z: number of redshift points spanning [0, z_rec] (default 100,
            log-spaced in 1+z; matches CLASS density)
        k_max_extend: target k_max in Mpc^-1 for the Halofit k-grid.
            Default 20 — log-log power-law extension of ``pt.k_grid`` to
            this value, mirroring CLASS's dedicated nonlinear k-grid.
            Set to 0 to disable extension.
        pt_mpk: tau0-complete matter solve (perturbations_solve_mpk). Every
            P_lin(k, z) is read from it -- the z-grid starts at 0, beyond the
            C_l solve's 0.999*tau0 grid (smsharma/clax#42) -- evaluated at
            ``pt.k_grid``, so the output stays on ``pt``'s lattice.

    Returns:
        ``sqrt(R(k, tau))`` of shape ``(Nk, Ntau)`` on the
        ``pt.k_grid x pt.tau_grid`` lattice. Multiply this into
        ``S_transfer`` to inject NL corrections into the source-Limber
        kernel (CLASS source-multiplication recipe).
    """
    from clax.nonlinear import compute_pk_nonlinear
    from clax.transfer import compute_linear_matter_pk_from_perturbations

    # --- Extended k-grid (power-law extrapolation if pt.k_grid too narrow) ---
    pt_k_max = float(pt.k_grid[-1])
    k_max_use = max(pt_k_max, k_max_extend)
    if pt_k_max < k_max_extend:
        n_extra = max(int(20.0 * (jnp.log10(k_max_use / pt_k_max))), 10)
        k_extra = jnp.logspace(
            jnp.log10(pt_k_max), jnp.log10(k_max_use), n_extra + 1)[1:]
        k_ext = jnp.concatenate([pt.k_grid, k_extra])
    else:
        k_ext = pt.k_grid

    # --- z-grid spanning [0, z_rec], log-spaced in 1+z ---
    z_rec = float(th.z_rec)
    log1pz_grid = jnp.linspace(jnp.log(1.0), jnp.log(1.0 + z_rec), n_z)
    z_grid = jnp.expm1(log1pz_grid)

    # --- P_lin(k_pt, z) on the native pt.k_grid via the public helper ---
    def _pk_lin_on_pt_grid(z):
        # tau0-complete solve, evaluated at pt's k: explicit k, not an array
        # index, so a pt_mpk at another precision cannot misalign.
        return compute_linear_matter_pk_from_perturbations(
            pt_mpk, bg, params, pt.k_grid, z=z)

    pk_lin_pt_z = jax.vmap(_pk_lin_on_pt_grid)(z_grid)  # (n_z, Nk_pt)

    # --- Extend P_lin to k_ext via log-log power-law extrapolation ---
    # Log-linear interp inside pt.k_grid; power-law tail beyond pt_k_max with
    # slope from the last two points. δ_m extrapolation is non-physical past
    # the solved support, so we extend P_lin directly in log-log space here
    # rather than passing k_ext through ``compute_linear_matter_pk_from_perturbations``.
    log_k_pt = jnp.log(pt.k_grid)
    log_k_ext = jnp.log(k_ext)
    log_pk_pt_z = jnp.log(jnp.maximum(pk_lin_pt_z, 1e-300))  # (n_z, Nk_pt)

    n_eff = (log_pk_pt_z[:, -1] - log_pk_pt_z[:, -2]) / (
        log_k_pt[-1] - log_k_pt[-2])  # (n_z,)
    log_k_high = log_k_pt[-1]

    def _extend_one_z(log_pk_one, slope):
        log_pk_in = jnp.interp(log_k_ext, log_k_pt, log_pk_one)
        log_pk_out = log_pk_one[-1] + slope * (log_k_ext - log_k_high)
        return jnp.where(log_k_ext > log_k_high, log_pk_out, log_pk_in)

    log_pk_lin_z = jax.vmap(_extend_one_z)(log_pk_pt_z, n_eff)  # (n_z, Nk_ext)
    pk_lin_z = jnp.exp(log_pk_lin_z)

    # --- Halofit cosmology scalars ---
    Omega_m_0 = bg.Omega_b + bg.Omega_cdm + bg.Omega_ncdm
    Omega_lambda_0 = bg.Omega_lambda + bg.Omega_de
    Omega_r_0 = bg.Omega_g + bg.Omega_ur
    fnu = bg.Omega_ncdm / jnp.maximum(Omega_m_0, 1e-30)

    # --- P_NL(k_ext, z) via vmap over z ---
    # ``compute_pk_nonlinear`` enforces the CLASS sigma(R) >= 1 fallback
    # (cf. fourier.c:1597-1716) using ``jnp.where`` internally, so this
    # remains vmap-safe even at high z where sigma < 1.
    def _halofit_one_z(pk_lin_one, z):
        return compute_pk_nonlinear(
            k_ext, pk_lin_one,
            Omega_m_0=Omega_m_0, Omega_lambda_0=Omega_lambda_0,
            Omega_r_0=Omega_r_0, w0=params.w0, wa=params.wa,
            fnu=fnu, h=params.h, z=z)

    pk_nl_z = jax.vmap(_halofit_one_z)(pk_lin_z, z_grid)  # (n_z, Nk_ext)

    # Ratio R(k, z); replace bad values with 1.0 (no NL correction)
    R_z = jnp.where(
        (pk_lin_z > 0) & jnp.isfinite(pk_nl_z) & (pk_nl_z > 0),
        pk_nl_z / pk_lin_z, 1.0)
    R_z = jnp.clip(R_z, 0.0, 100.0)

    # --- Interpolate R(k, z) onto (pt.k_grid, pt.tau_grid) ---
    loga_at_tau = bg.loga_of_tau.evaluate(pt.tau_grid)
    z_at_tau = jnp.expm1(-loga_at_tau)
    z_at_tau = jnp.clip(z_at_tau, 0.0, z_rec)
    log1pz_at_tau = jnp.log1p(z_at_tau)

    # Interp in z for each k_ext column -> (Nk_ext, Ntau)
    R_at_tau_full = jax.vmap(
        lambda Rk: jnp.interp(log1pz_at_tau, log1pz_grid, Rk),
        in_axes=1, out_axes=0)(R_z)

    # Restrict to pt.k_grid (first Nk_pt rows)
    Nk_pt = pt.k_grid.shape[0]
    R_at_pt = R_at_tau_full[:Nk_pt, :]

    return jnp.sqrt(jnp.maximum(R_at_pt, 0.0))


def _ept_modulator(
    pt: PerturbationResult,
    params: CosmoParams,
    bg: BackgroundResult,
    th,
    *,
    pt_mpk: MatterPerturbationResult,
) -> Float[Array, "Nk Ntau"]:
    """Build sqrt(R(k, tau)) modulator for source-Limber EPT injection.

    Computes ``R(k, z) = P_mm,1-loop(k, z) / P_lin(k, z)`` from the
    one-loop EFTofLSS (clax-pt) at z=0, then growth-rescales to other
    redshifts via the leading-order EFT scaling

        R(k, z) - 1 = (R(k, 0) - 1) * [D(z) / D(0)]^2

    where the loop term scales as ``D^4`` and the linear term as ``D^2``,
    so the ratio minus one scales as ``D^2``.  Sufficient at ~1% for
    ``k <= 0.3 h/Mpc``; subleading EFT counterterm time dependence is
    neglected.

    Args:
        pt: C_l perturbation solve; supplies only the output lattice
            (``k_grid``, ``tau_grid``).
        params: cosmological parameters
        bg: background results
        th: thermodynamics results
        pt_mpk: tau0-complete matter solve (perturbations_solve_mpk). Both
            P_NL (the EPT run at z=0) and P_lin(z=0) are read from it -- the
            z=0 lookup lies beyond the C_l solve's 0.999*tau0 grid
            (smsharma/clax#42) -- evaluated at explicit k, so the output
            stays on ``pt``'s lattice.

    Returns ``sqrt(R(k, tau))`` of shape ``(Nk, Ntau)`` on the
    ``pt.k_grid x pt.tau_grid`` lattice. The CLASS source-multiplication
    recipe applies this directly to the lensing source.
    """
    from clax.ept import compute_ept_from_clax, pk_mm_real, ept_kgrid
    from clax.transfer import compute_linear_matter_pk_from_perturbations
    from clax.interpolation import CubicSpline

    h = float(params.h)

    # --- Run EPT once at z=0 (Mpc/h units inside, convert to Mpc^-1) ---
    # total-matter field: the ratio below divides by the matter P_lin (C0)
    # Numerator AND denominator from pt_mpk: splitting them across solves
    # would turn the old second-order residual into a first-order one.
    ept = compute_ept_from_clax(params, bg, pt_mpk, z=0.0, th=th, field="m")
    pk_mm_h = pk_mm_real(ept)                  # (Mpc/h)^3 on EPT k-grid (h/Mpc)
    k_h = ept_kgrid()                           # h/Mpc
    k_ept_mpc = jnp.array(k_h) * h              # Mpc^-1
    pk_mm_mpc = pk_mm_h / h**3                  # Mpc^3

    # --- Restrict to overlap with pt.k_grid (compute_linear validates) ---
    pt_k_min = float(pt.k_grid[0])
    pt_k_max = float(pt.k_grid[-1])
    in_range = (k_ept_mpc >= pt_k_min) & (k_ept_mpc <= pt_k_max)
    k_eval = k_ept_mpc[in_range]
    pk_nl_eval = pk_mm_mpc[in_range]

    # P_lin(k_eval, z=0) on overlapping grid
    pk_lin_eval = compute_linear_matter_pk_from_perturbations(
        pt_mpk, bg, params, k_eval, z=0.0)

    # R0(k) = P_NL/P_lin at z=0, defaulting to 1.0 outside overlap
    R0_eval = jnp.where(pk_lin_eval > 0,
                         pk_nl_eval / pk_lin_eval, 1.0)
    R0_eval = jnp.clip(R0_eval, 0.0, 100.0)

    # Spline R0(log k) onto pt.k_grid; outside the EPT-evaluated range, R0 = 1.
    log_k_eval = jnp.log(k_eval)
    log_k_pt = jnp.log(pt.k_grid)
    R0_spline = CubicSpline(log_k_eval, R0_eval)
    R0_pt = R0_spline.evaluate(log_k_pt)
    R0_pt = jnp.where(
        (pt.k_grid >= float(k_eval[0])) & (pt.k_grid <= float(k_eval[-1])),
        R0_pt, 1.0)

    # --- Growth-factor rescaling: R(k, tau) - 1 = (R0(k) - 1) * [D(tau)/D(0)]^2 ---
    loga_at_tau = bg.loga_of_tau.evaluate(pt.tau_grid)
    D_at_tau = bg.D_of_loga.evaluate(loga_at_tau)
    D_at_0 = bg.D_of_loga.evaluate(jnp.zeros_like(loga_at_tau[:1]))[0]
    D_ratio_sq = (D_at_tau / D_at_0) ** 2  # (Ntau,)

    R_at_pt = 1.0 + (R0_pt[:, None] - 1.0) * D_ratio_sq[None, :]  # (Nk_pt, Ntau)
    R_at_pt = jnp.clip(R_at_pt, 0.0, 100.0)

    return jnp.sqrt(jnp.maximum(R_at_pt, 0.0))


def compute_cl_pp(
    pt: PerturbationResult,
    params: CosmoParams,
    bg: BackgroundResult,
    th,
    l_max: int,
    *,
    nonlinear: str = "none",
    pt_mpk: MatterPerturbationResult | None = None,
) -> Float[Array, "Nl"]:
    """Compute C_l^phiphi via source-based Limber (CLASS-accurate at all l).

    Mirrors CLASS transfer.c + harmonic.c:

        source(k, tau) = (phi+psi)(k, tau) * W_lcmb(tau)
        T_l(k) = [S*chi]_interp * IPhiFlat / (l+0.5)     (Limber)
        C_l^pp = 4pi * integral d(lnk) * P_R(k) * T_l(k)^2

    where IPhiFlat = sqrt(pi/(2l)) * (1 - 0.25/l + 1/(32l^2)) is the
    extended Limber correction, and S*chi is interpolated parabolically
    (cf. CLASS transfer_limber_interpolate, transfer.c:3763).

    Accuracy vs CLASS v3.3.4 (Planck 2018 LCDM): matches CLASS to <1% for
    ``l <= 2500`` (linear case), confirmed independently by CAMB.

    Nonlinear corrections (``nonlinear="halofit"`` or ``"ept"``) follow
    CLASS's source-multiplication approach: ``S(k, tau) -> S(k, tau) *
    sqrt(R(k, z(tau)))`` where ``R = P_NL/P_lin``. For ``"halofit"`` the
    ratio is computed on a (k, z) grid via the standard fitting formulae;
    for ``"ept"`` the ratio is computed once via clax-pt one-loop EFTofLSS
    at z=0 and growth-rescaled in time. See ``_halofit_modulator`` and
    ``_ept_modulator``.

    Args:
        pt: perturbation results (must have source_phi_plus_psi)
        params: cosmological parameters
        bg: background results
        th: thermodynamics results (for tau_rec)
        l_max: maximum multipole
        nonlinear: nonlinear correction scheme. Accepted values:
            - ``"none"``: linear matter power spectrum (default)
            - ``"halofit"``: Halofit (Smith 2003 + Takahashi 2012 + Bird
              2012), z-aware on-the-fly via 100-point z-grid +
              log-log k-extension to k_max=20 Mpc^-1, mirroring CLASS's
              dedicated nonlinear k-grid. Residual vs CLASS Halofit
              reference: <1% at all ℓ ≤ 2500 with
              ``pt.k_grid[-1] >= 5 Mpc^-1``. Narrower grids
              gracefully degrade to no NL correction (matching CLASS
              ``fourier.c:1706-1716``).
            - ``"ept"``: clax-pt one-loop EFTofLSS at z=0 + leading-order
              D^2 growth scaling. Documented residual of ~1% for
              k <= 0.3 h/Mpc; subleading EFT counterterm time dependence
              neglected.
            Anything else raises ``ValueError``.
        pt_mpk: tau0-complete matter solve (perturbations_solve_mpk for the
            same params/prec). REQUIRED when ``nonlinear`` is "halofit" or
            "ept": their P_lin lookups start at z=0, beyond ``pt``'s grid,
            which stops at 0.999*tau0 (smsharma/clax#42). Ignored for "none".

    Returns:
        C_l^phiphi array of shape (l_max+1,), indexed by l (l=0,1 are zero)
    """
    if nonlinear not in {"none", "halofit", "ept"}:
        raise ValueError(
            f"compute_cl_pp: unknown nonlinear={nonlinear!r}. "
            f"Accepted values: 'none', 'halofit', 'ept'.")
    if nonlinear != "none" and pt_mpk is None:
        raise ValueError(
            f"compute_cl_pp: nonlinear={nonlinear!r} needs pt_mpk="
            "perturbations_solve_mpk(params, prec, bg, th): its P_lin lookups "
            "start at z=0, beyond the C_l solve's 0.999*tau0 grid.")

    # --- Section A: Setup (all JAX, no numpy) ---
    tau_grid = pt.tau_grid
    k_grid = pt.k_grid
    tau_0 = bg.conformal_age
    Nk = k_grid.shape[0]
    Ntau = tau_grid.shape[0]

    loga_rec = jnp.log(1.0 / (1.0 + th.z_rec))
    tau_rec = bg.tau_of_loga.evaluate(loga_rec)
    chi_rec = tau_0 - tau_rec

    # chi = tau_0 - tau, DECREASING (large chi = early times)
    chi_arr = tau_0 - tau_grid

    # Geometric lensing kernel W_lcmb (CLASS transfer.c:2428, flat case)
    chi_safe = jnp.where(chi_arr > 0.0, chi_arr, 1.0)
    W_lcmb = (tau_rec - tau_grid) / (chi_safe * chi_rec)
    W_lcmb = jnp.where(chi_arr > 0.0, W_lcmb, 0.0)

    # Lensing source: S(k,tau) = (phi+psi) * W, zeroed for tau <= tau_rec
    S_transfer = pt.source_phi_plus_psi * W_lcmb[None, :]
    S_transfer = jnp.where(tau_grid[None, :] > tau_rec, S_transfer, 0.0)

    # NL injection: S(k,tau) -> S(k,tau) * sqrt(R(k, z(tau)))
    if nonlinear == "halofit":
        S_transfer = S_transfer * _halofit_modulator(pt, params, bg, th, pt_mpk=pt_mpk)
    elif nonlinear == "ept":
        S_transfer = S_transfer * _ept_modulator(pt, params, bg, th, pt_mpk=pt_mpk)

    log_k = jnp.log(k_grid)
    dlnk = jnp.diff(log_k)
    P_R = primordial_scalar_pk(k_grid, params)

    # --- Section B: Precompute S*chi product (regular at chi->0) ---
    # cf. CLASS transfer.c:3785: interpolate S*(tau0-tau) not bare S
    S_chi_grid = S_transfer * chi_arr[None, :]  # (Nk, Ntau)

    # --- Section C: Precompute l-dependent quantities ---
    Nl = l_max - 1
    l_arr = jnp.arange(2, l_max + 1, dtype=jnp.float64)
    nu_arr = l_arr + 0.5

    # Extended Limber correction (CLASS transfer.c:3664)
    IPhiFlat = jnp.sqrt(jnp.pi / (2.0 * l_arr)) * (
        1.0 - 0.25 / l_arr + 1.0 / (32.0 * l_arr * l_arr))

    # --- Section D: Precompute interpolation indices for all (l, k) ---
    chi_limber_all = nu_arr[:, None] / k_grid[None, :]  # (Nl, Nk)

    # searchsorted on DECREASING chi_arr: negate both sides to get increasing
    neg_chi_arr = -chi_arr
    idx_all = jnp.searchsorted(neg_chi_arr, -chi_limber_all)
    idx_all = jnp.clip(idx_all, 1, Ntau - 2)

    # Validity mask: chi_limber must be within [chi_arr[-1], chi_arr[0]]
    # chi_arr[0] is largest (earliest time), chi_arr[-1] is smallest (today)
    valid_mask = (chi_limber_all <= chi_arr[0]) & (chi_limber_all >= chi_arr[-1])

    # --- Section E: Precompute Lagrange weights ---
    x0 = chi_arr[idx_all - 1]  # (Nl, Nk)
    x1 = chi_arr[idx_all]
    x2 = chi_arr[idx_all + 1]

    L0 = (chi_limber_all - x1) * (chi_limber_all - x2) / ((x0 - x1) * (x0 - x2))
    L1 = (chi_limber_all - x0) * (chi_limber_all - x2) / ((x1 - x0) * (x1 - x2))
    L2 = (chi_limber_all - x0) * (chi_limber_all - x1) / ((x2 - x0) * (x2 - x1))

    # --- Section F: vmap over l ---
    k_indices = jnp.arange(Nk)

    def _compute_cl_single_l(l_idx):
        """C_l^pp for one l value, vectorized over k."""
        idx_k = idx_all[l_idx, :]  # (Nk,)

        # Gather S*chi at 3-point stencil for all k simultaneously
        y0 = S_chi_grid[k_indices, idx_k - 1]      # (Nk,)
        y1 = S_chi_grid[k_indices, idx_k]           # (Nk,)
        y2_raw = S_chi_grid[k_indices, idx_k + 1]   # (Nk,)
        # Boundary: at last tau point, S*chi is approximately constant
        # cf. CLASS transfer.c:3808-3823
        y2 = jnp.where(idx_k >= Ntau - 2, y1, y2_raw)

        # Parabolic interpolation using precomputed Lagrange weights
        S_chi_interp = y0 * L0[l_idx] + y1 * L1[l_idx] + y2 * L2[l_idx]

        # T_l(k) = S_chi * IPhiFlat / nu, zeroed if out of range
        T_l = jnp.where(valid_mask[l_idx],
                         S_chi_interp * IPhiFlat[l_idx] / nu_arr[l_idx],
                         0.0)

        # C_l = 4pi * trapz(P_R * T_l^2, dlnk)
        integrand = P_R * T_l**2
        return 4.0 * jnp.pi * jnp.sum(
            0.5 * (integrand[:-1] + integrand[1:]) * dlnk)

    cl_2_to_lmax = jax.vmap(_compute_cl_single_l)(jnp.arange(Nl))
    return jnp.concatenate([jnp.zeros(2), cl_2_to_lmax])


# =============================================================================
# Wigner d-matrix recurrence coefficients (Kostelec & Rockmore 2003)
# cf. CLASS lensing.c:1256-1964
#
# All recurrences act on rescaled functions: sqrt((2l+1)/2) * d^l_{mm'}
# General form: rescaled[l+1] = fac1[l]*f(mu)*rescaled[l] - fac3[l]*rescaled[l-1]
# Actual d = rescaled * sqrt(2/(2l+1))
# =============================================================================

def _precompute_d_facs(l_max):
    """Precompute recurrence coefficients for all Wigner d-functions.

    Returns dict of (fac1, fac2, fac3) tuples for each d-function type.
    fac2 is None for d-functions that use plain mu (d00, d20, d40).

    cf. CLASS lensing.c:1256-1964 for all recurrence formulas.
    """
    n = l_max + 1
    facs = {}

    # d11/d1m1 (shared facs, sign of fac2 differs in recurrence)
    # cf. lensing.c:1329-1334
    f1, f2, f3 = np.zeros(n), np.zeros(n), np.zeros(n)
    for l in range(2, n):
        ll = float(l)
        f1[l] = np.sqrt((2*ll+3)/(2*ll+1)) * (ll+1)*(2*ll+1) / (ll*(ll+2))
        f2[l] = 1.0 / (ll*(ll+1))
        f3[l] = np.sqrt((2*ll+3)/(2*ll-1)) * (ll-1)*(ll+1) / (ll*(ll+2)) * (ll+1)/ll
    facs['d11'] = (f1.copy(), f2.copy(), f3.copy())

    # d22/d2m2 (shared facs)  cf. lensing.c:1508-1513
    f1[:] = f2[:] = f3[:] = 0
    for l in range(2, n):
        ll = float(l)
        f1[l] = np.sqrt((2*ll+3)/(2*ll+1)) * (ll+1)*(2*ll+1) / ((ll-1)*(ll+3))
        f2[l] = 4.0 / (ll*(ll+1))
        f3[l] = np.sqrt((2*ll+3)/(2*ll-1)) * (ll-2)*(ll+2) / ((ll-1)*(ll+3)) * (ll+1)/ll
    facs['d22'] = (f1.copy(), f2.copy(), f3.copy())

    # d20 (no fac2)  cf. lensing.c:1567-1571
    f1[:] = f3[:] = 0
    for l in range(2, n):
        ll = float(l)
        f1[l] = np.sqrt((2*ll+3)*(2*ll+1) / ((ll-1)*(ll+3)))
        f3[l] = np.sqrt((2*ll+3)*(ll-2)*(ll+2) / ((2*ll-1)*(ll-1)*(ll+3)))
    facs['d20'] = (f1.copy(), None, f3.copy())

    # d31/d3m1 (shared facs, l>=3)  cf. lensing.c:1626-1632
    f1[:] = f2[:] = f3[:] = 0
    for l in range(3, n):
        ll = float(l)
        f1[l] = np.sqrt((2*ll+3)*(2*ll+1)/((ll-2)*(ll+4)*ll*(ll+2))) * (ll+1)
        f2[l] = 3.0 / (ll*(ll+1))
        f3[l] = np.sqrt((2*ll+3)/(2*ll-1)*(ll-3)*(ll+3)*(ll-1)*(ll+1)
                         / ((ll-2)*(ll+4)*ll*(ll+2))) * (ll+1)/ll
    facs['d31'] = (f1.copy(), f2.copy(), f3.copy())

    # d3m3 (l>=3)  cf. lensing.c:1748-1754
    f1[:] = f2[:] = f3[:] = 0
    for l in range(3, n):
        ll = float(l)
        f1[l] = np.sqrt((2*ll+3)*(2*ll+1)) * (ll+1) / ((ll-2)*(ll+4))
        f2[l] = 9.0 / (ll*(ll+1))
        f3[l] = np.sqrt((2*ll+3)/(2*ll-1)) * (ll-3)*(ll+3)*(ll+1) / ((ll-2)*(ll+4)*ll)
    facs['d3m3'] = (f1.copy(), f2.copy(), f3.copy())

    # d40 (no fac2, l>=4)  cf. lensing.c:1808-1813
    f1[:] = f3[:] = 0
    for l in range(4, n):
        ll = float(l)
        f1[l] = np.sqrt((2*ll+3)*(2*ll+1) / ((ll-3)*(ll+5)))
        f3[l] = np.sqrt((2*ll+3)*(ll-4)*(ll+4) / ((2*ll-1)*(ll-3)*(ll+5)))
    facs['d40'] = (f1.copy(), None, f3.copy())

    # d4m2 (l>=4)  cf. lensing.c:1869-1875
    f1[:] = f2[:] = f3[:] = 0
    for l in range(4, n):
        ll = float(l)
        f1[l] = np.sqrt((2*ll+3)*(2*ll+1)/((ll-3)*(ll+5)*(ll-1)*(ll+3))) * (ll+1)
        f2[l] = 8.0 / (ll*(ll+1))
        f3[l] = np.sqrt((2*ll+3)*(ll-4)*(ll+4)*(ll-2)*(ll+2)
                         / ((2*ll-1)*(ll-3)*(ll+5)*(ll-1)*(ll+3))) * (ll+1)/ll
    facs['d4m2'] = (f1.copy(), f2.copy(), f3.copy())

    # d4m4 (l>=4)  cf. lensing.c:1931-1937
    f1[:] = f2[:] = f3[:] = 0
    for l in range(4, n):
        ll = float(l)
        f1[l] = np.sqrt((2*ll+3)*(2*ll+1)) * (ll+1) / ((ll-3)*(ll+5))
        f2[l] = 16.0 / (ll*(ll+1))
        f3[l] = np.sqrt((2*ll+3)/(2*ll-1)) * (ll-4)*(ll+4)*(ll+1) / ((ll-3)*(ll+5)*ll)
    facs['d4m4'] = (f1.copy(), f2.copy(), f3.copy())

    return facs


def lens_cls(
    cl_tt_unlensed: Float[Array, "Nl"],
    cl_ee_unlensed: Float[Array, "Nl"],
    cl_te_unlensed: Float[Array, "Nl"],
    cl_bb_unlensed: Float[Array, "Nl"],
    cl_pp: Float[Array, "Nl"],
    l_max: int = 2500,
    n_gauss: int = 4096,
) -> tuple[Float[Array, "Nl"], Float[Array, "Nl"],
           Float[Array, "Nl"], Float[Array, "Nl"]]:
    """Compute lensed TT, EE, TE, BB using the correlation function method.

    Uses the CLASS addback technique with Cgl2 corrections for accurate
    BB and EE lensing. Implements the full lensing kernels from CLASS
    lensing.c:619-682, including first-order (O(Cgl2)) and second-order
    (O(Cgl2^2)) corrections using 12 Wigner d-functions.

    Args:
        cl_tt_unlensed: unlensed TT, shape (>=l_max+1,), indexed by l
        cl_ee_unlensed: unlensed EE
        cl_te_unlensed: unlensed TE
        cl_bb_unlensed: unlensed BB
        cl_pp: lensing potential C_l^phiphi
        l_max: maximum multipole (default 2500)
        n_gauss: number of GL quadrature points (default 4096)

    Returns:
        (cl_tt_lensed, cl_ee_lensed, cl_te_lensed, cl_bb_lensed),
        each shape (l_max+1,)
    """
    # GL quadrature points and weights
    x_gl_np, w_gl_np = np.polynomial.legendre.leggauss(n_gauss)
    x_gl = jnp.array(x_gl_np)
    w_gl = jnp.array(w_gl_np)

    # Precompute all recurrence coefficients
    facs = _precompute_d_facs(l_max)

    # Convert to JAX arrays
    f1_11, f2_11, f3_11 = [jnp.array(x) for x in facs['d11']]
    f1_22, f2_22, f3_22 = [jnp.array(x) for x in facs['d22']]
    f1_20, _, f3_20 = facs['d20']
    f1_20, f3_20 = jnp.array(f1_20), jnp.array(f3_20)
    f1_31, f2_31, f3_31 = [jnp.array(x) for x in facs['d31']]
    f1_3m3, f2_3m3, f3_3m3 = [jnp.array(x) for x in facs['d3m3']]
    f1_40, _, f3_40 = facs['d40']
    f1_40, f3_40 = jnp.array(f1_40), jnp.array(f3_40)
    f1_4m2, f2_4m2, f3_4m2 = [jnp.array(x) for x in facs['d4m2']]
    f1_4m4, f2_4m4, f3_4m4 = [jnp.array(x) for x in facs['d4m4']]

    # =====================================================================
    # PASS 1: Compute Cgl(mu) and Cgl2(mu) using d11 and d1m1 scans
    # cf. CLASS lensing.c:498-509
    # Cgl  = sum_l (2l+1)/(4pi) * l(l+1) * C_l^pp * d^l_{11}(mu)
    # Cgl2 = sum_l (2l+1)/(4pi) * l(l+1) * C_l^pp * d^l_{1,-1}(mu)
    # =====================================================================

    # d11 initial conditions  cf. lensing.c:1342-1346
    d11_r1 = (1.0 + x_gl) / 2.0 * jnp.sqrt(3.0 / 2.0)
    d11_r2 = (1.0 + x_gl) / 2.0 * (2.0 * x_gl - 1.0) * jnp.sqrt(5.0 / 2.0)
    d11_l2 = d11_r2 * jnp.sqrt(2.0 / 5.0)

    # d1m1 initial conditions  cf. lensing.c:1401-1405
    d1m1_r1 = (1.0 - x_gl) / 2.0 * jnp.sqrt(3.0 / 2.0)
    d1m1_r2 = (1.0 - x_gl) / 2.0 * (2.0 * x_gl + 1.0) * jnp.sqrt(5.0 / 2.0)
    d1m1_l2 = d1m1_r2 * jnp.sqrt(2.0 / 5.0)

    # l=2 contributions
    coeff_l2 = 5.0 / (4.0 * jnp.pi) * 6.0 * cl_pp[2]
    cgl_init = coeff_l2 * d11_l2
    cgl2_init = coeff_l2 * d1m1_l2

    def _cgl_scan(carry, l_idx):
        """Compute d11/d1m1 at l_idx+1, accumulate Cgl/Cgl2."""
        d11_lm1, d11_l, d1m1_lm1, d1m1_l, cgl, cgl2 = carry
        # d11: (mu - fac2)  cf. lensing.c:1349
        d11_lp1 = f1_11[l_idx] * (x_gl - f2_11[l_idx]) * d11_l \
            - f3_11[l_idx] * d11_lm1
        # d1m1: (mu + fac2)  cf. lensing.c:1408
        d1m1_lp1 = f1_11[l_idx] * (x_gl + f2_11[l_idx]) * d1m1_l \
            - f3_11[l_idx] * d1m1_lm1
        l_new = (l_idx + 1).astype(jnp.float64)
        sn = jnp.sqrt(2.0 / (2.0 * l_new + 1.0))
        coeff = (2.0 * l_new + 1.0) / (4.0 * jnp.pi) * l_new * (l_new + 1.0) \
            * cl_pp[l_idx + 1]
        cgl = cgl + coeff * d11_lp1 * sn
        cgl2 = cgl2 + coeff * d1m1_lp1 * sn
        return (d11_l, d11_lp1, d1m1_l, d1m1_lp1, cgl, cgl2), None

    (_, _, _, _, Cgl, Cgl2), _ = jax.lax.scan(
        _cgl_scan,
        (d11_r1, d11_r2, d1m1_r1, d1m1_r2, cgl_init, cgl2_init),
        jnp.arange(2, l_max),
    )

    # sigma2(mu) = Cgl(mu=1) - Cgl(mu)
    l_arr = jnp.arange(l_max + 1, dtype=jnp.float64)
    sigma2_total = jnp.sum(
        (2.0 * l_arr[2:] + 1.0) / (4.0 * jnp.pi)
        * l_arr[2:] * (l_arr[2:] + 1.0) * cl_pp[2:]
    )
    sigma2 = sigma2_total - Cgl

    # =====================================================================
    # PASS 2: Forward transform with full Cgl2-corrected kernels
    #
    # Uses 12 Wigner d-functions: d00, d11, d1m1, d20, d22, d2m2,
    # d31, d3m1, d3m3, d40, d4m2, d4m4
    #
    # X variables (truncated at sigma2^k * Cgl2^m with k+m <= 2):
    #   cf. CLASS lensing.c:588-615
    #
    # Kernels (cf. CLASS lensing.c:619-682):
    #   TT:    X_000^2*d00 + Cgl2*8/(ll1)*X_p000^2*d1m1
    #          + Cgl2^2*(X_p000^2*d00 + X_220^2*d2m2)
    #   TE:    X_022*X_000*d20 + Cgl2*2*X_p000/s5*(X_121*d11+X_132*d3m1)
    #          + Cgl2^2*0.5*((2*X_p022*X_p000+X_220^2)*d20+X_220*X_242*d4m2)
    #   EE+BB: X_022^2*d22 + 2*Cgl2*X_132*X_121*d31
    #          + Cgl2^2*(X_p022^2*d22 + X_242*X_220*d40)
    #   EE-BB: X_022^2*d2m2 + Cgl2*(X_121^2*d1m1 + X_132^2*d3m3)
    #          + Cgl2^2*0.5*(2*X_p022^2*d2m2 + X_220^2*d00 + X_242^2*d4m4)
    #
    # With addback: subtract d00/d20/d22/d2m2 from zeroth-order terms.
    # =====================================================================

    cl_ee_plus_bb = cl_ee_unlensed + cl_bb_unlensed
    cl_ee_minus_bb = cl_ee_unlensed - cl_bb_unlensed

    # --- Initial conditions at l=2 for all d-functions ---
    zeros_gl = jnp.zeros(n_gauss)

    # d00: P_2 = (3x^2-1)/2
    d00_l2 = (3.0 * x_gl**2 - 1.0) / 2.0

    # d20 rescaled  cf. lensing.c:1582
    d20_r1 = zeros_gl
    d20_r2 = jnp.sqrt(15.0) / 4.0 * (1.0 - x_gl**2)
    d20_l2 = d20_r2 * jnp.sqrt(2.0 / 5.0)

    # d22 rescaled  cf. lensing.c:1524
    d22_r1 = zeros_gl
    d22_r2 = (1.0 + x_gl)**2 / 4.0 * jnp.sqrt(5.0 / 2.0)
    d22_l2 = d22_r2 * jnp.sqrt(2.0 / 5.0)

    # d2m2 rescaled  cf. lensing.c:1464
    d2m2_r1 = zeros_gl
    d2m2_r2 = (1.0 - x_gl)**2 / 4.0 * jnp.sqrt(5.0 / 2.0)
    d2m2_l2 = d2m2_r2 * jnp.sqrt(2.0 / 5.0)

    # d31 rescaled: d31[l<=2]=0, d31[l=3]=initial  cf. lensing.c:1639-1644
    d31_r_l3 = jnp.sqrt(105.0 / 2.0) * (1.0 + x_gl)**2 * (1.0 - x_gl) / 8.0

    # d3m1 rescaled: d3m1[l<=2]=0, d3m1[l=3]=initial  cf. lensing.c:1700-1705
    d3m1_r_l3 = jnp.sqrt(105.0 / 2.0) * (1.0 + x_gl) * (1.0 - x_gl)**2 / 8.0

    # d3m3 rescaled: d3m3[l<=2]=0, d3m3[l=3]=initial  cf. lensing.c:1761-1766
    d3m3_r_l3 = jnp.sqrt(7.0 / 2.0) * (1.0 - x_gl)**3 / 8.0

    # d40 rescaled: d40[l<=3]=0, d40[l=4]=initial  cf. lensing.c:1820-1826
    d40_r_l4 = jnp.sqrt(315.0) * (1.0 + x_gl)**2 * (1.0 - x_gl)**2 / 16.0

    # d4m2 rescaled: d4m2[l<=3]=0, d4m2[l=4]=initial  cf. lensing.c:1882-1888
    d4m2_r_l4 = jnp.sqrt(126.0) * (1.0 + x_gl) * (1.0 - x_gl)**3 / 16.0

    # d4m4 rescaled: d4m4[l<=3]=0, d4m4[l=4]=initial  cf. lensing.c:1944-1950
    d4m4_r_l4 = jnp.sqrt(9.0 / 2.0) * (1.0 - x_gl)**4 / 16.0

    # --- Compute l=2 kernel contributions ---
    # At l=2: d31=d3m1=d3m3=d40=d4m2=d4m4=0
    # sqrt values at l=2
    ll2 = 2.0
    s1_2 = jnp.sqrt(4.0 * 3.0 * 2.0 * 1.0)  # sqrt(24)
    s2_2 = jnp.sqrt(4.0 * 1.0)  # 2
    s5_2 = jnp.sqrt(6.0)
    # s3_2 = sqrt(5*0) = 0, s4_2 = sqrt(6*5*0*(-1)) → 0

    fac_c2 = ll2 * (ll2 + 1.0) / 4.0  # 1.5
    X000_2 = jnp.exp(-fac_c2 * sigma2)
    Xp000_2 = -fac_c2 * X000_2
    X022_2 = X000_2 * (1.0 + sigma2 * (1.0 + 0.5 * sigma2))
    Xp022_2 = -(fac_c2 - 1.0) * X022_2
    X220_2 = 0.25 * s1_2 * X000_2
    X121_2 = -0.5 * s2_2 * X000_2 * (1.0 + 2.0 / 3.0 * sigma2)
    # X_132, X_242 = 0 at l=2

    pref_l2 = 5.0 / (4.0 * jnp.pi)
    ll1_2 = ll2 * (ll2 + 1.0)

    # TT kernel at l=2 (addback: subtract d00)
    kern_tt_2 = ((X000_2 * X000_2 - 1.0) * d00_l2
                 + Xp000_2**2 * d1m1_l2 * Cgl2 * 8.0 / ll1_2
                 + (Xp000_2**2 * d00_l2 + X220_2**2 * d2m2_l2) * Cgl2**2)

    # TE kernel at l=2 (addback: subtract d20)
    kern_te_2 = ((X022_2 * X000_2 - 1.0) * d20_l2
                 + Cgl2 * 2.0 * Xp000_2 / s5_2 * X121_2 * d11_l2
                 + 0.5 * Cgl2**2 * (2.0 * Xp022_2 * Xp000_2 + X220_2**2) * d20_l2)

    # lensp kernel at l=2 (addback: subtract d22)
    kern_p_2 = (X022_2**2 - 1.0) * d22_l2 + Cgl2**2 * Xp022_2**2 * d22_l2

    # lensm kernel at l=2 (addback: subtract d2m2)
    kern_m_2 = ((X022_2**2 - 1.0) * d2m2_l2
                + Cgl2 * X121_2**2 * d1m1_l2
                + 0.5 * Cgl2**2 * (2.0 * Xp022_2**2 * d2m2_l2
                                    + X220_2**2 * d00_l2))

    ksi_init = pref_l2 * cl_tt_unlensed[2] * kern_tt_2
    ksiX_init = pref_l2 * cl_te_unlensed[2] * kern_te_2
    ksip_init = pref_l2 * cl_ee_plus_bb[2] * kern_p_2
    ksim_init = pref_l2 * cl_ee_minus_bb[2] * kern_m_2

    def _forward_scan(carry, l_idx):
        """Forward scan: compute all d-functions at l_idx+1, apply kernels."""
        (d00_pm1, d00_p,
         d11_rp, d11_rc, d1m1_rp, d1m1_rc,
         d20_rp, d20_rc,
         d22_rp, d22_rc, d2m2_rp, d2m2_rc,
         d31_rp, d31_rc, d3m1_rp, d3m1_rc, d3m3_rp, d3m3_rc,
         d40_rp, d40_rc, d4m2_rp, d4m2_rc, d4m4_rp, d4m4_rc,
         ksi, ksiX, ksip, ksim) = carry

        l_new = (l_idx + 1).astype(jnp.float64)
        sn = jnp.sqrt(2.0 / (2.0 * l_new + 1.0))

        # ---- Compute d-functions at l_new = l_idx + 1 ----

        # d00 (Legendre)
        d00 = ((2.0 * l_new - 1.0) * x_gl * d00_p
               - (l_new - 1.0) * d00_pm1) / l_new

        # d11: (mu - fac2)
        d11_rn = f1_11[l_idx] * (x_gl - f2_11[l_idx]) * d11_rc \
            - f3_11[l_idx] * d11_rp
        d11 = d11_rn * sn

        # d1m1: (mu + fac2), same facs as d11
        d1m1_rn = f1_11[l_idx] * (x_gl + f2_11[l_idx]) * d1m1_rc \
            - f3_11[l_idx] * d1m1_rp
        d1m1 = d1m1_rn * sn

        # d20: mu only
        d20_rn = f1_20[l_idx] * x_gl * d20_rc - f3_20[l_idx] * d20_rp
        d20 = d20_rn * sn

        # d22: (mu - fac2)
        d22_rn = f1_22[l_idx] * (x_gl - f2_22[l_idx]) * d22_rc \
            - f3_22[l_idx] * d22_rp
        d22 = d22_rn * sn

        # d2m2: (mu + fac2), same facs as d22
        d2m2_rn = f1_22[l_idx] * (x_gl + f2_22[l_idx]) * d2m2_rc \
            - f3_22[l_idx] * d2m2_rp
        d2m2 = d2m2_rn * sn

        # d31: (mu - fac2), jnp.where at l_idx=2 for l=3 initial
        d31_rec = f1_31[l_idx] * (x_gl - f2_31[l_idx]) * d31_rc \
            - f3_31[l_idx] * d31_rp
        d31_rn = jnp.where(l_idx == 2, d31_r_l3, d31_rec)
        d31 = d31_rn * sn

        # d3m1: (mu + fac2), same facs as d31
        d3m1_rec = f1_31[l_idx] * (x_gl + f2_31[l_idx]) * d3m1_rc \
            - f3_31[l_idx] * d3m1_rp
        d3m1_rn = jnp.where(l_idx == 2, d3m1_r_l3, d3m1_rec)
        d3m1 = d3m1_rn * sn

        # d3m3: (mu + fac2)
        d3m3_rec = f1_3m3[l_idx] * (x_gl + f2_3m3[l_idx]) * d3m3_rc \
            - f3_3m3[l_idx] * d3m3_rp
        d3m3_rn = jnp.where(l_idx == 2, d3m3_r_l3, d3m3_rec)
        d3m3 = d3m3_rn * sn

        # d40: mu only, jnp.where at l_idx=3 for l=4 initial
        d40_rec = f1_40[l_idx] * x_gl * d40_rc - f3_40[l_idx] * d40_rp
        d40_rn = jnp.where(l_idx == 3, d40_r_l4, d40_rec)
        d40 = d40_rn * sn

        # d4m2: (mu + fac2)
        d4m2_rec = f1_4m2[l_idx] * (x_gl + f2_4m2[l_idx]) * d4m2_rc \
            - f3_4m2[l_idx] * d4m2_rp
        d4m2_rn = jnp.where(l_idx == 3, d4m2_r_l4, d4m2_rec)
        d4m2 = d4m2_rn * sn

        # d4m4: (mu + fac2)
        d4m4_rec = f1_4m4[l_idx] * (x_gl + f2_4m4[l_idx]) * d4m4_rc \
            - f3_4m4[l_idx] * d4m4_rp
        d4m4_rn = jnp.where(l_idx == 3, d4m4_r_l4, d4m4_rec)
        d4m4 = d4m4_rn * sn

        # ---- X variables  cf. lensing.c:588-615 ----
        fac_c = l_new * (l_new + 1.0) / 4.0
        X_000 = jnp.exp(-fac_c * sigma2)
        X_p000 = -fac_c * X_000

        s1 = jnp.sqrt(jnp.maximum(0.0, (l_new+2)*(l_new+1)*l_new*(l_new-1)))
        s2 = jnp.sqrt(jnp.maximum(0.0, (l_new+2)*(l_new-1)))
        s3 = jnp.sqrt(jnp.maximum(0.0, (l_new+3)*(l_new-2)))
        s4 = jnp.sqrt(jnp.maximum(0.0,
                       (l_new+4)*(l_new+3)*(l_new-2)*(l_new-3)))
        s5 = jnp.sqrt(l_new * (l_new + 1.0))

        X_022 = X_000 * (1.0 + sigma2 * (1.0 + 0.5 * sigma2))
        X_p022 = -(fac_c - 1.0) * X_022
        X_220 = 0.25 * s1 * X_000
        X_242 = 0.25 * s4 * X_000
        X_121 = -0.5 * s2 * X_000 * (1.0 + 2.0/3.0 * sigma2)
        X_132 = -0.5 * s3 * X_000 * (1.0 + 5.0/3.0 * sigma2)

        # ---- Full lensing kernels  cf. lensing.c:619-682 ----
        pref = (2.0 * l_new + 1.0) / (4.0 * jnp.pi)
        ll1 = l_new * (l_new + 1.0)

        # TT (addback: X_000^2 - 1 for zeroth order)
        kern_tt = ((X_000**2 - 1.0) * d00
                   + X_p000**2 * d1m1 * Cgl2 * 8.0 / ll1
                   + (X_p000**2 * d00 + X_220**2 * d2m2) * Cgl2**2)

        # TE (addback: X_022*X_000 - 1 for zeroth order)
        # s5 >= sqrt(12) since l_new >= 3 in scan
        kern_te = ((X_022 * X_000 - 1.0) * d20
                   + Cgl2 * 2.0 * X_p000 / s5
                     * (X_121 * d11 + X_132 * d3m1)
                   + 0.5 * Cgl2**2
                     * ((2.0 * X_p022 * X_p000 + X_220**2) * d20
                        + X_220 * X_242 * d4m2))

        # lensp = EE+BB (addback: X_022^2 - 1)
        kern_p = ((X_022**2 - 1.0) * d22
                  + 2.0 * Cgl2 * X_132 * X_121 * d31
                  + Cgl2**2
                    * (X_p022**2 * d22 + X_242 * X_220 * d40))

        # lensm = EE-BB (addback: X_022^2 - 1)
        kern_m = ((X_022**2 - 1.0) * d2m2
                  + Cgl2 * (X_121**2 * d1m1 + X_132**2 * d3m3)
                  + 0.5 * Cgl2**2
                    * (2.0 * X_p022**2 * d2m2
                       + X_220**2 * d00 + X_242**2 * d4m4))

        # Accumulate
        ksi = ksi + pref * cl_tt_unlensed[l_idx + 1] * kern_tt
        ksiX = ksiX + pref * cl_te_unlensed[l_idx + 1] * kern_te
        ksip = ksip + pref * cl_ee_plus_bb[l_idx + 1] * kern_p
        ksim = ksim + pref * cl_ee_minus_bb[l_idx + 1] * kern_m

        new_carry = (d00_p, d00,
                     d11_rc, d11_rn, d1m1_rc, d1m1_rn,
                     d20_rc, d20_rn,
                     d22_rc, d22_rn, d2m2_rc, d2m2_rn,
                     d31_rc, d31_rn, d3m1_rc, d3m1_rn, d3m3_rc, d3m3_rn,
                     d40_rc, d40_rn, d4m2_rc, d4m2_rn, d4m4_rc, d4m4_rn,
                     ksi, ksiX, ksip, ksim)
        return new_carry, None

    # Initial carry: all d-functions at l=1 and l=2
    forward_init = (
        x_gl, d00_l2,                          # d00: P_1, P_2
        d11_r1, d11_r2, d1m1_r1, d1m1_r2,      # d11, d1m1
        d20_r1, d20_r2,                         # d20
        d22_r1, d22_r2, d2m2_r1, d2m2_r2,       # d22, d2m2
        zeros_gl, zeros_gl, zeros_gl, zeros_gl, zeros_gl, zeros_gl,  # d31, d3m1, d3m3
        zeros_gl, zeros_gl, zeros_gl, zeros_gl, zeros_gl, zeros_gl,  # d40, d4m2, d4m4
        ksi_init, ksiX_init, ksip_init, ksim_init,
    )

    final_carry, _ = jax.lax.scan(
        _forward_scan, forward_init, jnp.arange(2, l_max),
    )
    ksi = final_carry[24]
    ksiX = final_carry[25]
    ksip = final_carry[26]
    ksim = final_carry[27]

    # =====================================================================
    # PASS 3: Inverse transform — extract lensed C_l
    # Only uses d00, d20, d22, d2m2 (same as before)
    # cf. CLASS lensing.c:1049-1214
    # =====================================================================
    w_ksi = w_gl * ksi
    w_ksiX = w_gl * ksiX
    w_ksip = w_gl * ksip
    w_ksim = w_gl * ksim

    def _inverse_scan(carry, l_idx):
        """Inverse scan: extract C_l at l_idx via GL quadrature."""
        (d00_pm1, d00_p,
         d20_rp, d20_rc,
         d22_rp, d22_rc,
         d2m2_rp, d2m2_rc) = carry

        l_fl = l_idx.astype(jnp.float64)
        sn = jnp.sqrt(2.0 / (2.0 * l_fl + 1.0))

        d00_l = ((2.0 * l_fl - 1.0) * x_gl * d00_p
                 - (l_fl - 1.0) * d00_pm1) / l_fl

        l_prev = l_idx - 1
        d20_rn = f1_20[l_prev] * x_gl * d20_rc - f3_20[l_prev] * d20_rp
        d20_l = d20_rn * sn

        d22_rn = f1_22[l_prev] * (x_gl - f2_22[l_prev]) * d22_rc \
            - f3_22[l_prev] * d22_rp
        d22_l = d22_rn * sn

        d2m2_rn = f1_22[l_prev] * (x_gl + f2_22[l_prev]) * d2m2_rc \
            - f3_22[l_prev] * d2m2_rp
        d2m2_l = d2m2_rn * sn

        cl_tt_l = 2.0 * jnp.pi * jnp.sum(w_ksi * d00_l) + cl_tt_unlensed[l_idx]
        cl_te_l = 2.0 * jnp.pi * jnp.sum(w_ksiX * d20_l) + cl_te_unlensed[l_idx]
        cl_p = jnp.sum(w_ksip * d22_l)
        cl_m = jnp.sum(w_ksim * d2m2_l)
        cl_ee_l = jnp.pi * (cl_p + cl_m) + cl_ee_unlensed[l_idx]
        cl_bb_l = jnp.pi * (cl_p - cl_m) + cl_bb_unlensed[l_idx]

        new_carry = (d00_p, d00_l,
                     d20_rc, d20_rn, d22_rc, d22_rn, d2m2_rc, d2m2_rn)
        return new_carry, (cl_tt_l, cl_ee_l, cl_te_l, cl_bb_l)

    # l=2 values
    cl_tt_l2 = 2.0 * jnp.pi * jnp.sum(w_ksi * d00_l2) + cl_tt_unlensed[2]
    cl_te_l2 = 2.0 * jnp.pi * jnp.sum(w_ksiX * d20_l2) + cl_te_unlensed[2]
    cl_p_l2 = jnp.sum(w_ksip * d22_l2)
    cl_m_l2 = jnp.sum(w_ksim * d2m2_l2)
    cl_ee_l2 = jnp.pi * (cl_p_l2 + cl_m_l2) + cl_ee_unlensed[2]
    cl_bb_l2 = jnp.pi * (cl_p_l2 - cl_m_l2) + cl_bb_unlensed[2]

    # Scan l=3 to l_max
    inverse_init = (x_gl, d00_l2,
                    d20_r1, d20_r2, d22_r1, d22_r2, d2m2_r1, d2m2_r2)

    _, (cl_tt_3up, cl_ee_3up, cl_te_3up, cl_bb_3up) = jax.lax.scan(
        _inverse_scan, inverse_init, jnp.arange(3, l_max + 1),
    )

    cl_tt_lensed = jnp.concatenate([
        jnp.zeros(2), jnp.array([cl_tt_l2]), cl_tt_3up])
    cl_ee_lensed = jnp.concatenate([
        jnp.zeros(2), jnp.array([cl_ee_l2]), cl_ee_3up])
    cl_te_lensed = jnp.concatenate([
        jnp.zeros(2), jnp.array([cl_te_l2]), cl_te_3up])
    cl_bb_lensed = jnp.concatenate([
        jnp.zeros(2), jnp.array([cl_bb_l2]), cl_bb_3up])

    return cl_tt_lensed, cl_ee_lensed, cl_te_lensed, cl_bb_lensed


def lens_cl_tt(
    cl_tt_unlensed: Float[Array, "Nl"],
    cl_pp: Float[Array, "Nl"],
    l_max: int = 2500,
    n_gauss: int = 4096,
) -> Float[Array, "Nl"]:
    """Apply lensing to C_l^TT only (backward-compatible wrapper)."""
    n = len(cl_tt_unlensed)
    zeros = jnp.zeros(n)
    cl_tt, _, _, _ = lens_cls(
        cl_tt_unlensed, zeros, zeros, zeros, cl_pp,
        l_max=l_max, n_gauss=n_gauss,
    )
    return cl_tt
