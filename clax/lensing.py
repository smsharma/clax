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
from clax.perturbations import PerturbationResult
from clax.primordial import primordial_scalar_pk


def compute_cl_pp(
    pt: PerturbationResult,
    params: CosmoParams,
    bg: BackgroundResult,
    l_values: Float[Array, "Nl"],
) -> Float[Array, "Nl"]:
    """Compute lensing potential power spectrum C_l^phiphi.

    C_l^pp = [2/(l(l+1))]^2 * 4pi int dlnk P_R |T_l|^2

    where T_l(k) = int dtau source_lens(k,tau) * j_l(k*chi)
    and source_lens = exp(-kappa)*2*Phi.
    cf. CLASS perturbations.c:7686-7690, transfer.c:3144-3160
    """
    tau_grid = pt.tau_grid
    k_grid = pt.k_grid
    tau_0 = bg.conformal_age
    chi_grid = tau_0 - tau_grid

    dtau = jnp.diff(tau_grid)
    dtau_mid = jnp.concatenate([dtau[:1], (dtau[:-1] + dtau[1:]) / 2, dtau[-1:]])
    log_k = jnp.log(k_grid)

    def compute_clpp_single_l(l):
        l_int = int(l)
        l_fl = jnp.float64(l)

        def transfer_single_k(ik):
            k = k_grid[ik]
            x = k * chi_grid
            jl = spherical_jl(l_int, x)
            S_lens = pt.source_lens[ik, :]
            return jnp.sum(S_lens * jl * dtau_mid)

        T_l_coarse = jax.vmap(transfer_single_k)(jnp.arange(len(k_grid)))
        prefactor = (2.0 / (l_fl * (l_fl + 1.0)))**2
        P_R_coarse = primordial_scalar_pk(k_grid, params)
        integrand_k = P_R_coarse * T_l_coarse**2
        dlnk = jnp.diff(log_k)
        cl_pp = prefactor * 4.0 * jnp.pi * jnp.sum(
            0.5 * (integrand_k[:-1] + integrand_k[1:]) * dlnk
        )
        return cl_pp

    cls = []
    for l in l_values:
        cl = compute_clpp_single_l(l)
        cls.append(cl)
    return jnp.array(cls)


def compute_cl_pp_fast(
    pt: PerturbationResult,
    params: CosmoParams,
    bg: BackgroundResult,
    l_max: int,
) -> Float[Array, "Nl"]:
    """Compute C_l^phiphi for l=0..l_max via a single lax.scan over l.

    Fuses the upward Bessel recurrence with the tau-integral: at each l-step,
    advance j_l via ``(2l+1)/x * j_l - j_{l-1}``, contract with
    ``source_lens * dtau`` to get T_l(k), then assemble C_l.  The carry is
    only two (Nk, Ntau) arrays, so memory is O(Nk * Ntau) regardless of l_max.

    This replaces the Python for-loop + per-point ``spherical_jl`` in
    ``compute_cl_pp`` with a single compiled ``lax.scan``.

    Args:
        pt: perturbation results (source functions, k/tau grids)
        params: cosmological parameters
        bg: background results
        l_max: maximum multipole (returns C_l for l=0..l_max)

    Returns:
        C_l^phiphi array of shape (l_max+1,), indexed by l (l=0,1 are zero)
    """
    from clax.bessel import _j0, _j1

    tau_grid = pt.tau_grid
    k_grid = pt.k_grid
    tau_0 = bg.conformal_age
    chi_grid = tau_0 - tau_grid

    dtau = jnp.diff(tau_grid)
    dtau_mid = jnp.concatenate([dtau[:1], (dtau[:-1] + dtau[1:]) / 2, dtau[-1:]])
    log_k = jnp.log(k_grid)
    dlnk = jnp.diff(log_k)
    P_R = primordial_scalar_pk(k_grid, params)

    S_dtau = pt.source_lens * dtau_mid[None, :]  # (Nk, Ntau)

    # x = k * chi for all (k, tau) pairs
    x_grid = k_grid[:, None] * chi_grid[None, :]  # (Nk, Ntau)
    x_safe = jnp.where(jnp.abs(x_grid) < 1e-30, 1e-30, x_grid)

    # j_0 and j_1 at all (k, tau) points
    j0 = _j0(x_grid)
    j1 = _j1(x_grid)

    # C_l from T_l(k) via trapezoidal k-integration
    def _cl_from_Tl(T_l, l_fl):
        prefactor = (2.0 / (l_fl * (l_fl + 1.0)))**2
        integrand = P_R * T_l**2
        return prefactor * 4.0 * jnp.pi * jnp.sum(
            0.5 * (integrand[:-1] + integrand[1:]) * dlnk)

    # Scan: at each step advance j_l -> j_{l+1}, contract with S*dtau -> T
    def scan_fn(carry, l_curr):
        j_prev, j_curr = carry
        l_fl = l_curr.astype(jnp.float64)
        j_next = (2.0 * l_fl + 1.0) / x_safe * j_curr - j_prev
        # Zero classically forbidden region to prevent overflow
        j_next = jnp.where(jnp.abs(x_grid) < 0.7 * (l_fl + 1.0), 0.0, j_next)
        j_next = jnp.clip(j_next, -1.0, 1.0)
        T_next = jnp.sum(S_dtau * j_next, axis=1)
        cl = _cl_from_Tl(T_next, l_fl + 1.0)
        return (j_curr, j_next), cl

    _, cl_2_to_lmax = jax.lax.scan(
        scan_fn, (j0, j1), jnp.arange(1, l_max))

    # l=0,1 have no lensing contribution
    return jnp.concatenate([jnp.zeros(2), cl_2_to_lmax])


def compute_cl_pp_vmap(
    pt: PerturbationResult,
    params: CosmoParams,
    bg: BackgroundResult,
    l_max: int,
    n_x: int = 50000,
    x_max: float = 15000.0,
) -> Float[Array, "Nl"]:
    """Compute C_l^phiphi for l=2..l_max using precomputed Bessel tables + vmap.

    Precomputes j_l(x) and j_l'(x) on a 1-D x-grid for a sparse set of
    l-values (backward + upward recurrence, blended), then evaluates all l
    in parallel via ``jax.vmap`` with cubic Hermite interpolation.
    GPU-optimal: the l-dimension is fully parallel.

    Uses the same ``build_jl_table`` infrastructure as
    ``harmonic.compute_cls_all_fast``, with 4th-order Hermite interpolation
    (matching CLASS's Bessel table lookup) for sub-percent accuracy.

    Args:
        pt: perturbation results (source functions, k/tau grids)
        params: cosmological parameters
        bg: background results
        l_max: maximum multipole (returns C_l for l=0..l_max)
        n_x: Bessel table x-grid points (default 50000)
        x_max: maximum x in table (default 15000)

    Returns:
        C_l^phiphi array of shape (l_max+1,), indexed by l (l=0,1 are zero)
    """
    from clax.bessel import build_jl_table, sparse_l_grid

    tau_grid = pt.tau_grid
    k_grid = pt.k_grid
    tau_0 = bg.conformal_age
    chi_grid = tau_0 - tau_grid
    n_k = k_grid.shape[0]
    n_tau = tau_grid.shape[0]

    dtau = jnp.diff(tau_grid)
    dtau_mid = jnp.concatenate([dtau[:1], (dtau[:-1] + dtau[1:]) / 2, dtau[-1:]])
    log_k = jnp.log(k_grid)
    dlnk = jnp.diff(log_k)
    P_R = primordial_scalar_pk(k_grid, params)

    S_dtau = pt.source_lens * dtau_mid[None, :]  # (Nk, Ntau)

    # 1. Build Bessel table (j_l AND j_l') on sparse l-grid
    x_max_auto = float(jnp.max(k_grid) * tau_0 * 1.05)
    x_max_use = max(x_max, x_max_auto)
    n_x_use = max(n_x, int(x_max_use * 3))
    x_table, jl_table, jlp_table = build_jl_table(
        l_max, n_x=n_x_use, x_max=x_max_use)
    n_x_actual = x_table.shape[0]
    h_table = x_max_use / (n_x_actual - 1)  # uniform spacing

    # 2. Precompute interpolation indices for x_query = k * chi
    x_query = k_grid[:, None] * chi_grid[None, :]  # (Nk, Ntau)
    x_flat = x_query.flatten()

    idx_right = jnp.searchsorted(x_table, x_flat)
    idx_left = jnp.clip(idx_right - 1, 0, n_x_actual - 2)
    idx_right_safe = jnp.clip(idx_right, 1, n_x_actual - 1)
    dx = x_table[idx_right_safe] - x_table[idx_left]
    dx_safe = jnp.where(dx < 1e-30, 1e-30, dx)
    t = jnp.clip((x_flat - x_table[idx_left]) / dx_safe, 0.0, 1.0)

    # Cubic Hermite basis polynomials (4th-order, uses j_l and j_l')
    # H(t) = f_i h00 + f'_i h * h10 + f_{i+1} h01 + f'_{i+1} h * h11
    # cf. CLASS transfer.c Hermite interpolation for Bessel tables
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0   # Hermite basis for f(x_i)
    h10 = t3 - 2.0 * t2 + t             # Hermite basis for f'(x_i) * h
    h01 = -2.0 * t3 + 3.0 * t2          # Hermite basis for f(x_{i+1})
    h11 = t3 - t2                        # Hermite basis for f'(x_{i+1}) * h

    # 3. vmap over sparse l: Hermite table lookup → T_l(k) → C_l
    l_sparse = sparse_l_grid(l_max)
    l_sparse_fl = jnp.array(l_sparse, dtype=jnp.float64)

    def _single_l_cl(l_idx):
        """Compute C_l^pp for one l-index via Hermite table lookup."""
        l_fl = l_sparse_fl[l_idx]
        # Cubic Hermite interpolation from j_l and j_l' tables
        fl = jl_table[l_idx, idx_left]
        fr = jl_table[l_idx, idx_right_safe]
        dfl = jlp_table[l_idx, idx_left]
        dfr = jlp_table[l_idx, idx_right_safe]
        jl_flat = fl * h00 + dfl * h_table * h10 + fr * h01 + dfr * h_table * h11
        jl_2d = jl_flat.reshape(n_k, n_tau)

        # T_l(k) = sum_tau S * j_l * dtau
        T_l = jnp.sum(S_dtau * jl_2d, axis=1)  # (Nk,)

        # C_l = [2/(l(l+1))]^2 * 4pi * integral dlnk P_R T_l^2
        prefactor = (2.0 / (l_fl * (l_fl + 1.0)))**2
        integrand = P_R * T_l**2
        return prefactor * 4.0 * jnp.pi * jnp.sum(
            0.5 * (integrand[:-1] + integrand[1:]) * dlnk)

    cl_sparse = jax.vmap(_single_l_cl)(jnp.arange(len(l_sparse)))

    # 4. Spline-interpolate sparse C_l to all l=0..l_max
    from clax.interpolation import CubicSpline
    l_sparse_log = jnp.log(l_sparse_fl)
    cl_sparse_safe = jnp.maximum(cl_sparse, 1e-100)
    log_cl_spline = CubicSpline(l_sparse_log, jnp.log(cl_sparse_safe))

    l_all = jnp.arange(2, l_max + 1, dtype=jnp.float64)
    log_cl_all = log_cl_spline.evaluate(jnp.log(l_all))
    cl_all = jnp.exp(log_cl_all)

    return jnp.concatenate([jnp.zeros(2), cl_all])



def compute_cl_pp_transfer(
    pt: PerturbationResult,
    params: CosmoParams,
    bg: BackgroundResult,
    th,
    l_values: Float[Array, "Nl"],
) -> Float[Array, "Nl"]:
    """Compute C_l^phiphi via the full Bessel transfer function integral.

    Mirrors CLASS transfer.c:2428 + harmonic.c:1073-1108 exactly:

        source(k, tau) = (phi+psi)(k, tau) * W_lcmb(tau)
        T_l(k) = integral_{tau>tau_rec} dtau * source * j_l(k*chi)
        C_l^pp = 4pi * integral d(lnk) * P_R(k) * T_l(k)^2

    Three bugs in the original ``compute_cl_pp`` are fixed:
    1. Source: uses ``source_phi_plus_psi`` (eta + alpha_prime) with
       the geometric kernel, not ``exp(-kappa) * 2*phi``
    2. No ``[2/(l(l+1))]^2`` prefactor — CLASS stores C_l^pp directly
    3. Integration starts at tau_rec (CLASS transfer.c:1712)

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




def compute_cl_pp_limber(
    pt: PerturbationResult,
    params: CosmoParams,
    bg: BackgroundResult,
    th,
    l_max: int,
    n_chi: int = 500,
    nonlinear: bool = False,
) -> Float[Array, "Nl"]:
    """Compute C_l^phiphi via the Limber approximation and Poisson equation.

    Uses the ABCMB approach (Zhou, Giovanetti & Liu, arXiv:2602.15104, Eq.B.24-B.25):

        P_psi(k, z) = 9/(8 pi^2) * Omega_m(z)^2 * (aH)^4 * P(k,z) / k

        C_l^pp = 8 pi^2 / (l+0.5)^3 * integral d(ln a) * chi/aH * W^2 * P_psi

    where W(chi) = (chi_* - chi) / (chi_* chi) is the geometric lensing kernel,
    ``aH`` is the conformal Hubble rate, and the Limber substitution
    k = (l+0.5)/chi is applied.

    When ``nonlinear=True``, P(k,z) is replaced by the Halofit nonlinear
    P_NL(k,z) at each integration point via ``compute_pk_nonlinear``.
    Requires ``pt.k_grid`` to extend to k >= 5 Mpc^-1 (matching CLASS's
    ``nonlinear_min_k_max``).  At high z where sigma(R) < 1, the NL
    correction is automatically skipped (cf. CLASS fourier.c:1706-1716).

    Args:
        pt: perturbation results (for P(k,z) evaluation via delta_m)
        params: cosmological parameters
        bg: background results
        th: thermodynamics results (for z_rec / tau_rec)
        l_max: maximum multipole
        n_chi: number of integration points (default 500)
        nonlinear: if True, use Halofit P_NL(k,z) instead of P_lin(k,z)

    Returns:
        C_l^phiphi array of shape (l_max+1,), indexed by l (l=0,1 are zero)
    """
    from clax.interpolation import CubicSpline as CS

    tau_0 = bg.conformal_age
    loga_rec = jnp.log(1.0 / (1.0 + th.z_rec))
    tau_rec = bg.tau_of_loga.evaluate(loga_rec)
    chi_star = tau_0 - tau_rec

    Omega_m_0 = bg.Omega_b + bg.Omega_cdm + bg.Omega_ncdm
    H0 = bg.H0  # 1/Mpc

    # Integration grid in ln(a) from recombination to today
    lna_rec = float(loga_rec)
    lna_grid = jnp.linspace(lna_rec, 0.0, n_chi)
    a_grid = jnp.exp(lna_grid)
    z_grid = 1.0 / a_grid - 1.0

    # Background quantities
    tau_grid_int = bg.tau_of_loga.evaluate(lna_grid)
    chi_grid = tau_0 - tau_grid_int
    H_grid = bg.H_of_loga.evaluate(lna_grid)  # cosmic H(z) in 1/Mpc
    aH_grid = a_grid * H_grid                  # conformal Hubble

    # Omega_m(z) = Omega_m * (1+z)^3 / (H/H0)^2
    Om_z_grid = Omega_m_0 * (1.0 + z_grid)**3 / (H_grid / H0)**2

    # Lensing window W(chi) = (chi_* - chi) / (chi_* * chi)
    chi_safe = jnp.where(chi_grid > 1.0, chi_grid, 1.0)
    W_grid = jnp.where(chi_grid > 1.0,
                        (chi_star - chi_grid) / (chi_star * chi_safe), 0.0)

    # l-independent background factor:
    # bg_part = chi / aH * W^2 * 9/(8pi^2) * Om_z^2 * aH^4
    bg_part = (chi_grid / aH_grid * W_grid**2
               * 9.0 / (8.0 * jnp.pi**2) * Om_z_grid**2 * aH_grid**4)

    # Perturbation grid for delta_m interpolation
    log_k_pt = jnp.log(pt.k_grid)
    k_min = float(pt.k_grid[0])
    k_max = float(pt.k_grid[-1])
    delta_m_table = pt.delta_m  # (Nk_pt, Ntau_pt)
    tau_pt = pt.tau_grid

    l_arr = jnp.arange(2, l_max + 1, dtype=jnp.float64)
    n_l = l_max - 1
    dlna = jnp.diff(lna_grid)

    # Precompute NL ratio P_NL/P_lin on the perturbation k-grid at each chi.
    # Requires k_max >= 5 Mpc^-1 for Halofit sigma(R) convergence
    # (cf. CLASS precisions.h: nonlinear_min_k_max = 5.0).
    nl_ratio_spline_at_tau = {}
    if nonlinear:
        from clax.nonlinear import compute_pk_nonlinear, _sigma_convergence_check
        k_max_grid = float(pt.k_grid[-1])
        if k_max_grid < 4.5:
            raise ValueError(
                f"nonlinear=True requires pt_k_max_cl >= 5.0 Mpc^-1 "
                f"(got k_max={k_max_grid:.2f}). Increase pt_k_max_cl "
                f"in PrecisionParams for the lensing perturbation solve.")
        Omega_lambda_0 = bg.Omega_lambda + bg.Omega_de
        Omega_r_0 = bg.Omega_g + bg.Omega_ur
        fnu = bg.Omega_ncdm / jnp.maximum(Omega_m_0, 1e-30)
        ones_ratio = jnp.ones_like(pt.k_grid)
        for i_chi in range(n_chi):
            z_val = float(z_grid[i_chi])
            tau_val = float(tau_grid_int[i_chi])
            i_tau = int(jnp.argmin(jnp.abs(tau_pt - tau_val)))
            dm_at_tau = delta_m_table[:, i_tau]
            P_R_full = primordial_scalar_pk(pt.k_grid, params)
            pk_lin_full = 2.0 * jnp.pi**2 / pt.k_grid**3 * P_R_full * dm_at_tau**2
            # Skip Halofit at high z where sigma(R)<1
            # (cf. CLASS fourier.c:1706-1716: nl_corr_density = 1.0)
            if not _sigma_convergence_check(log_k_pt, pk_lin_full):
                nl_ratio_spline_at_tau[i_chi] = CS(log_k_pt, ones_ratio)
                continue
            pk_nl_full = compute_pk_nonlinear(
                pt.k_grid, pk_lin_full,
                Omega_m_0=Omega_m_0, Omega_lambda_0=Omega_lambda_0,
                Omega_r_0=Omega_r_0, w0=params.w0, wa=params.wa,
                fnu=fnu, h=params.h, z=z_val)
            ratio = jnp.where(pk_lin_full > 0, pk_nl_full / pk_lin_full, 1.0)
            nl_ratio_spline_at_tau[i_chi] = CS(log_k_pt, ratio)

    # Build integrand for all l: loop over chi, vectorize over l
    integrand_all = jnp.zeros((n_l, n_chi))

    for i_chi in range(n_chi):
        chi_val = float(chi_grid[i_chi])
        if chi_val < 1.0:
            continue
        bg_val = float(bg_part[i_chi])
        if abs(bg_val) < 1e-50:
            continue

        # k_limber for all l at this chi
        k_all_l = (l_arr + 0.5) / chi_val
        valid = (k_all_l >= k_min) & (k_all_l <= k_max)

        # Interpolate delta_m(k, tau) at nearest tau
        tau_val = float(tau_grid_int[i_chi])
        i_tau = int(jnp.argmin(jnp.abs(tau_pt - tau_val)))
        dm_at_tau = delta_m_table[:, i_tau]

        dm_spline = CS(log_k_pt, dm_at_tau)
        log_k_eval = jnp.log(jnp.clip(k_all_l, k_min, k_max))
        dm_interp = dm_spline.evaluate(log_k_eval)

        P_R = primordial_scalar_pk(jnp.clip(k_all_l, k_min, k_max), params)
        pk_vals = 2.0 * jnp.pi**2 / k_all_l**3 * P_R * dm_interp**2

        # Apply Halofit nonlinear correction if requested
        if nonlinear:
            log_k_limber = jnp.log(jnp.clip(k_all_l, k_min, k_max))
            nl_ratio_interp = nl_ratio_spline_at_tau[i_chi].evaluate(log_k_limber)
            nl_ratio_vals = jnp.where(valid, jnp.maximum(nl_ratio_interp, 0.0), 1.0)
            pk_vals = pk_vals * nl_ratio_vals

        # Contribution: bg_part * pk / k
        contrib = bg_val * pk_vals / k_all_l
        contrib = jnp.where(valid, contrib, 0.0)
        integrand_all = integrand_all.at[:, i_chi].set(contrib)

    # Integrate over d(ln a) for each l
    cl_arr = jnp.zeros(n_l)
    for i_l in range(n_l):
        l_fl = l_arr[i_l]
        coeff = 8.0 * jnp.pi**2 / (l_fl + 0.5)**3
        integ = integrand_all[i_l, :]
        cl_arr = cl_arr.at[i_l].set(
            coeff * jnp.sum(0.5 * (integ[:-1] + integ[1:]) * dlna))

    return jnp.concatenate([jnp.zeros(2), cl_arr])



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
