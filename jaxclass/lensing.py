"""Lensing module for jaxCLASS.

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

from jaxclass.background import BackgroundResult
from jaxclass.bessel import spherical_jl
from jaxclass.params import CosmoParams, PrecisionParams
from jaxclass.perturbations import PerturbationResult
from jaxclass.primordial import primordial_scalar_pk


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
    tau_0 = float(bg.conformal_age)
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


# =============================================================================
# Wigner d-matrix recurrence coefficients (Kostelec & Rockmore 2003)
# cf. CLASS lensing.c:1256-1597
#
# All recurrences act on rescaled functions: sqrt((2l+1)/2) * d^l_{mm'}
# General form: rescaled[l+1] = fac1[l]*f(mu)*rescaled[l] - fac3[l]*rescaled[l-1]
# Actual d = rescaled * sqrt(2/(2l+1))
# =============================================================================

def _precompute_d11_fac(l_max):
    """d^l_{11} recurrence coefficients. cf. CLASS lensing.c:1329-1334."""
    fac1 = np.zeros(l_max + 1)
    fac2 = np.zeros(l_max + 1)
    fac3 = np.zeros(l_max + 1)
    for l in range(2, l_max + 1):
        ll = float(l)
        fac1[l] = np.sqrt((2*ll+3)/(2*ll+1)) * (ll+1)*(2*ll+1) / (ll*(ll+2))
        fac2[l] = 1.0 / (ll*(ll+1))
        fac3[l] = np.sqrt((2*ll+3)/(2*ll-1)) * (ll-1)*(ll+1) / (ll*(ll+2)) * (ll+1)/ll
    return fac1, fac2, fac3


def _precompute_d22_fac(l_max):
    """d^l_{22} and d^l_{2,-2} recurrence coefficients. cf. CLASS lensing.c:1508-1513."""
    fac1 = np.zeros(l_max + 1)
    fac2 = np.zeros(l_max + 1)
    fac3 = np.zeros(l_max + 1)
    for l in range(2, l_max + 1):
        ll = float(l)
        fac1[l] = np.sqrt((2*ll+3)/(2*ll+1)) * (ll+1)*(2*ll+1) / ((ll-1)*(ll+3))
        fac2[l] = 4.0 / (ll*(ll+1))
        fac3[l] = np.sqrt((2*ll+3)/(2*ll-1)) * (ll-2)*(ll+2) / ((ll-1)*(ll+3)) * (ll+1)/ll
    return fac1, fac2, fac3


def _precompute_d20_fac(l_max):
    """d^l_{20} recurrence coefficients. cf. CLASS lensing.c:1567-1571."""
    fac1 = np.zeros(l_max + 1)
    fac3 = np.zeros(l_max + 1)
    for l in range(2, l_max + 1):
        ll = float(l)
        fac1[l] = np.sqrt((2*ll+3)*(2*ll+1) / ((ll-1)*(ll+3)))
        fac3[l] = np.sqrt((2*ll+3)*(ll-2)*(ll+2) / ((2*ll-1)*(ll-1)*(ll+3)))
    return fac1, fac3


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

    Uses the CLASS addback technique: compute the lensed-minus-unlensed
    difference in correlation function space, then add back unlensed C_l.
    This avoids numerical cancellation at low l where lensing is small.

    The deflection correlation Cgl uses the Wigner d^l_{11} function
    (not Legendre P_l) as required for spin-1 deflection fields.
    cf. CLASS lensing.c:498-509.

    Algorithm (cf. CLASS lensing.c, fast mode):
    1. Compute Cgl(mu) = sum_l (2l+1)/(4pi) l(l+1) C_l^pp d^l_{11}(mu)
    2. sigma2(mu) = Cgl(1) - Cgl(mu)
    3. Forward: build ksi, ksiX, ksip, ksim using addback kernels
    4. Inverse: GL quadrature with appropriate d-functions
    5. Add back unlensed C_l

    Args:
        cl_tt_unlensed: unlensed TT, shape (l_max+1,), indexed by l
        cl_ee_unlensed: unlensed EE, shape (l_max+1,), indexed by l
        cl_te_unlensed: unlensed TE, shape (l_max+1,), indexed by l
        cl_bb_unlensed: unlensed BB, shape (l_max+1,), indexed by l
        cl_pp: lensing potential C_l^phiphi, shape (l_max+1,), indexed by l
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

    # =====================================================================
    # PASS 1: Compute Cgl(mu) using d^l_{11} recurrence
    # cf. CLASS lensing.c:498-509
    # Cgl = sum_l (2l+1)/(4pi) * l*(l+1) * C_l^pp * d^l_{11}(mu)
    # =====================================================================
    fac1_11, fac2_11, fac3_11 = _precompute_d11_fac(l_max)
    fac1_11_j = jnp.array(fac1_11)
    fac2_11_j = jnp.array(fac2_11)
    fac3_11_j = jnp.array(fac3_11)

    # d11 initial conditions (rescaled = sqrt((2l+1)/2) * d11)
    # l=1: rescaled = (1+mu)/2 * sqrt(3/2)  cf. lensing.c:1343
    # l=2: rescaled = (1+mu)/2*(2mu-1) * sqrt(5/2)  cf. lensing.c:1345
    d11_r1 = (1.0 + x_gl) / 2.0 * jnp.sqrt(3.0 / 2.0)
    d11_r2 = (1.0 + x_gl) / 2.0 * (2.0 * x_gl - 1.0) * jnp.sqrt(5.0 / 2.0)
    d11_l2_actual = d11_r2 * jnp.sqrt(2.0 / 5.0)

    # Initialize Cgl with l=2 contribution
    cgl_init = 5.0 / (4.0 * jnp.pi) * 6.0 * cl_pp[2] * d11_l2_actual

    def _d11_scan(carry, l_idx):
        """Scan computing d11 at l_idx+1 and accumulating Cgl."""
        dlm1, dl, cgl_acc = carry
        # Compute rescaled d11 at l_idx+1 using fac at l_idx
        # cf. lensing.c:1349
        dlp1 = fac1_11_j[l_idx] * (x_gl - fac2_11_j[l_idx]) * dl \
            - fac3_11_j[l_idx] * dlm1
        # Actual d11 at l_idx+1
        l_new = (l_idx + 1).astype(jnp.float64)
        d11_val = dlp1 * jnp.sqrt(2.0 / (2.0 * l_new + 1.0))
        # Accumulate Cgl for l = l_idx + 1
        coeff = (2.0 * l_new + 1.0) / (4.0 * jnp.pi) * l_new * (l_new + 1.0) \
            * cl_pp[l_idx + 1]
        cgl_acc = cgl_acc + coeff * d11_val
        return (dl, dlp1, cgl_acc), None

    # Scan l_idx=2,...,l_max-1 → computes d11 at l=3,...,l_max
    (_, _, Cgl), _ = jax.lax.scan(
        _d11_scan,
        (d11_r1, d11_r2, cgl_init),
        jnp.arange(2, l_max),
    )

    # sigma2(mu) = Cgl(mu=1) - Cgl(mu)
    # At mu=1: d11(l,1)=1 for all l, so Cgl(1) = sum_l (2l+1)/(4pi)*l(l+1)*C_l^pp
    l_arr = jnp.arange(l_max + 1, dtype=jnp.float64)
    sigma2_total = jnp.sum(
        (2.0 * l_arr[2:] + 1.0) / (4.0 * jnp.pi) * l_arr[2:] * (l_arr[2:] + 1.0)
        * cl_pp[2:]
    )
    sigma2 = sigma2_total - Cgl

    # =====================================================================
    # PASS 2: Forward transform — build lensed correlation functions
    # Using addback: kernel = (exp_factor - 1) * d_function
    #
    # Lensing exponentials (zeroth order in Cgl2):
    #   TT:    X_000^2     = exp(-l(l+1)/2 * sigma2)
    #   TE:    X_022*X_000 = exp(-(l(l+1)/2 - 1) * sigma2)
    #   EE/BB: X_022^2     = exp(-(l(l+1)/2 - 2) * sigma2)
    # cf. CLASS lensing.c:588-616
    #
    # Correlation functions:
    #   ksi  (TT):  sum_l (2l+1)/(4pi) * C_l^TT * (exp_TT-1) * d00
    #   ksiX (TE):  sum_l (2l+1)/(4pi) * C_l^TE * (exp_TE-1) * d20
    #   ksip (EE+): sum_l (2l+1)/(4pi) * (C_l^EE+C_l^BB) * (exp_EE-1) * d22
    #   ksim (EE-): sum_l (2l+1)/(4pi) * (C_l^EE-C_l^BB) * (exp_EE-1) * d2m2
    # cf. CLASS lensing.c:619-682
    # =====================================================================
    fac1_22, fac2_22, fac3_22 = _precompute_d22_fac(l_max)
    fac1_20, fac3_20 = _precompute_d20_fac(l_max)
    fac1_22_j = jnp.array(fac1_22)
    fac2_22_j = jnp.array(fac2_22)
    fac3_22_j = jnp.array(fac3_22)
    fac1_20_j = jnp.array(fac1_20)
    fac3_20_j = jnp.array(fac3_20)

    # Combined spectra for EE/BB
    cl_ee_plus_bb = cl_ee_unlensed + cl_bb_unlensed
    cl_ee_minus_bb = cl_ee_unlensed - cl_bb_unlensed

    # --- Initial conditions at l=2 ---
    # d00 (Legendre): P_0=1, P_1=x, P_2=(3x^2-1)/2
    d00_l2 = (3.0 * x_gl**2 - 1.0) / 2.0

    # d20 rescaled at l=1: 0; l=2: sqrt(15)/4*(1-mu^2)  cf. lensing.c:1582
    d20_r1 = jnp.zeros(n_gauss)
    d20_r2 = jnp.sqrt(15.0) / 4.0 * (1.0 - x_gl**2)
    d20_l2_actual = d20_r2 * jnp.sqrt(2.0 / 5.0)

    # d22 rescaled at l=1: 0; l=2: (1+mu)^2/4*sqrt(5/2)  cf. lensing.c:1524
    d22_r1 = jnp.zeros(n_gauss)
    d22_r2 = (1.0 + x_gl)**2 / 4.0 * jnp.sqrt(5.0 / 2.0)
    d22_l2_actual = d22_r2 * jnp.sqrt(2.0 / 5.0)

    # d2m2 rescaled at l=1: 0; l=2: (1-mu)^2/4*sqrt(5/2)  cf. lensing.c:1464
    d2m2_r1 = jnp.zeros(n_gauss)
    d2m2_r2 = (1.0 - x_gl)**2 / 4.0 * jnp.sqrt(5.0 / 2.0)
    d2m2_l2_actual = d2m2_r2 * jnp.sqrt(2.0 / 5.0)

    # Lensing factors at l=2
    fac_l2 = 2.0 * 3.0 / 2.0  # l(l+1)/2 = 3
    exp_tt_l2 = jnp.exp(-fac_l2 * sigma2) - 1.0
    exp_te_l2 = jnp.exp(-(fac_l2 - 1.0) * sigma2) - 1.0
    exp_ee_l2 = jnp.exp(-(fac_l2 - 2.0) * sigma2) - 1.0
    pref_l2 = 5.0 / (4.0 * jnp.pi)

    # Initialize correlation functions with l=2 contributions
    ksi_init = pref_l2 * cl_tt_unlensed[2] * exp_tt_l2 * d00_l2
    ksiX_init = pref_l2 * cl_te_unlensed[2] * exp_te_l2 * d20_l2_actual
    ksip_init = pref_l2 * cl_ee_plus_bb[2] * exp_ee_l2 * d22_l2_actual
    ksim_init = pref_l2 * cl_ee_minus_bb[2] * exp_ee_l2 * d2m2_l2_actual

    def _forward_scan(carry, l_idx):
        """Forward scan: compute d-functions at l_idx+1, accumulate correlations."""
        (d00_pm1, d00_p,          # P[l_idx-1], P[l_idx]
         d20_rp, d20_rc,          # rescaled d20 at l_idx-1, l_idx
         d22_rp, d22_rc,          # rescaled d22 at l_idx-1, l_idx
         d2m2_rp, d2m2_rc,        # rescaled d2m2 at l_idx-1, l_idx
         ksi, ksiX, ksip, ksim) = carry

        l_new = (l_idx + 1).astype(jnp.float64)  # target l
        snorm = jnp.sqrt(2.0 / (2.0 * l_new + 1.0))  # rescaled → actual

        # --- Compute d-functions at l_new = l_idx+1 ---

        # d00 (Legendre): P[l_new] from P[l_idx] and P[l_idx-1]
        d00_new = ((2.0 * l_new - 1.0) * x_gl * d00_p
                   - (l_new - 1.0) * d00_pm1) / l_new

        # d20: rescaled[l_new] from rescaled[l_idx] and rescaled[l_idx-1]
        # cf. lensing.c:1586: dlp1 = fac1*mu*dl - fac3*dlm1
        d20_rnew = fac1_20_j[l_idx] * x_gl * d20_rc - fac3_20_j[l_idx] * d20_rp
        d20_new = d20_rnew * snorm

        # d22: rescaled[l_new], recurrence with (mu - fac2)
        # cf. lensing.c:1528
        d22_rnew = fac1_22_j[l_idx] * (x_gl - fac2_22_j[l_idx]) * d22_rc \
            - fac3_22_j[l_idx] * d22_rp
        d22_new = d22_rnew * snorm

        # d2m2: rescaled[l_new], recurrence with (mu + fac2)
        # cf. lensing.c:1468
        d2m2_rnew = fac1_22_j[l_idx] * (x_gl + fac2_22_j[l_idx]) * d2m2_rc \
            - fac3_22_j[l_idx] * d2m2_rp
        d2m2_new = d2m2_rnew * snorm

        # --- Lensing exponentials (addback: subtract 1) ---
        fac = l_new * (l_new + 1.0) / 2.0
        exp_tt = jnp.exp(-fac * sigma2) - 1.0        # X_000^2 - 1
        exp_te = jnp.exp(-(fac - 1.0) * sigma2) - 1.0  # X_022*X_000 - 1
        exp_ee = jnp.exp(-(fac - 2.0) * sigma2) - 1.0  # X_022^2 - 1

        # --- Accumulate correlation functions ---
        pref = (2.0 * l_new + 1.0) / (4.0 * jnp.pi)
        ksi = ksi + pref * cl_tt_unlensed[l_idx + 1] * exp_tt * d00_new
        ksiX = ksiX + pref * cl_te_unlensed[l_idx + 1] * exp_te * d20_new
        ksip = ksip + pref * cl_ee_plus_bb[l_idx + 1] * exp_ee * d22_new
        ksim = ksim + pref * cl_ee_minus_bb[l_idx + 1] * exp_ee * d2m2_new

        new_carry = (d00_p, d00_new,
                     d20_rc, d20_rnew,
                     d22_rc, d22_rnew,
                     d2m2_rc, d2m2_rnew,
                     ksi, ksiX, ksip, ksim)
        return new_carry, None

    # Initial carry: d-functions at l=1 and l=2
    forward_init = (x_gl, d00_l2,          # P_1, P_2
                    d20_r1, d20_r2,         # rescaled d20 at l=1, l=2
                    d22_r1, d22_r2,         # rescaled d22 at l=1, l=2
                    d2m2_r1, d2m2_r2,       # rescaled d2m2 at l=1, l=2
                    ksi_init, ksiX_init, ksip_init, ksim_init)

    # Scan l_idx=2,...,l_max-1 → computes d at l=3,...,l_max, accumulates
    final_carry, _ = jax.lax.scan(
        _forward_scan,
        forward_init,
        jnp.arange(2, l_max),
    )
    _, _, _, _, _, _, _, _, ksi, ksiX, ksip, ksim = final_carry

    # =====================================================================
    # PASS 3: Inverse transform — extract lensed C_l
    #
    # TT: C~_l = 2pi * sum_mu w * ksi * d00(l) + C_l^TT_unlensed
    # TE: C~_l = 2pi * sum_mu w * ksiX * d20(l) + C_l^TE_unlensed
    # EE: C~_l = pi * (sum w ksip d22 + sum w ksim d2m2) + C_l^EE_unlensed
    # BB: C~_l = pi * (sum w ksip d22 - sum w ksim d2m2) + C_l^BB_unlensed
    # cf. CLASS lensing.c:1049-1214, addback: 1090-1240
    # =====================================================================
    w_ksi = w_gl * ksi
    w_ksiX = w_gl * ksiX
    w_ksip = w_gl * ksip
    w_ksim = w_gl * ksim

    def _inverse_scan(carry, l_idx):
        """Inverse scan: compute d-functions at l_idx, extract C_l."""
        (d00_pm1, d00_p,
         d20_rp, d20_rc,
         d22_rp, d22_rc,
         d2m2_rp, d2m2_rc) = carry

        l_fl = l_idx.astype(jnp.float64)
        snorm = jnp.sqrt(2.0 / (2.0 * l_fl + 1.0))

        # d00 at l_idx
        d00_l = ((2.0 * l_fl - 1.0) * x_gl * d00_p
                 - (l_fl - 1.0) * d00_pm1) / l_fl

        # Wigner d at l_idx: compute from rescaled carry
        # rescaled carry holds d at (l_idx-2, l_idx-1)
        # Use fac at l_idx-1 to compute rescaled at l_idx
        l_prev = l_idx - 1
        d20_rnew = fac1_20_j[l_prev] * x_gl * d20_rc - fac3_20_j[l_prev] * d20_rp
        d20_l = d20_rnew * snorm

        d22_rnew = fac1_22_j[l_prev] * (x_gl - fac2_22_j[l_prev]) * d22_rc \
            - fac3_22_j[l_prev] * d22_rp
        d22_l = d22_rnew * snorm

        d2m2_rnew = fac1_22_j[l_prev] * (x_gl + fac2_22_j[l_prev]) * d2m2_rc \
            - fac3_22_j[l_prev] * d2m2_rp
        d2m2_l = d2m2_rnew * snorm

        # GL quadrature + addback
        cl_tt_l = 2.0 * jnp.pi * jnp.sum(w_ksi * d00_l) + cl_tt_unlensed[l_idx]
        cl_te_l = 2.0 * jnp.pi * jnp.sum(w_ksiX * d20_l) + cl_te_unlensed[l_idx]
        cl_p = jnp.sum(w_ksip * d22_l)
        cl_m = jnp.sum(w_ksim * d2m2_l)
        cl_ee_l = jnp.pi * (cl_p + cl_m) + cl_ee_unlensed[l_idx]
        cl_bb_l = jnp.pi * (cl_p - cl_m) + cl_bb_unlensed[l_idx]

        new_carry = (d00_p, d00_l,
                     d20_rc, d20_rnew,
                     d22_rc, d22_rnew,
                     d2m2_rc, d2m2_rnew)
        return new_carry, (cl_tt_l, cl_ee_l, cl_te_l, cl_bb_l)

    # Initial carry for inverse: d-functions at l=1 and l=2
    # For d00: (P_0, P_1). At l_idx=2: P_2 = (3x^2-1)/2 from (P_0, P_1)
    # For Wigner: (rescaled[0], rescaled[1]). At l_idx=2, we need rescaled
    # at l=0 and l=1 in the carry, so the scan computes rescaled at l=2.
    # BUT: the scan uses fac at l_idx-1 = 1. Our fac arrays start at l=2.
    # Solution: handle l=2 separately and scan from l=3.

    # l=2 values (already computed above)
    cl_tt_l2 = 2.0 * jnp.pi * jnp.sum(w_ksi * d00_l2) + cl_tt_unlensed[2]
    cl_te_l2 = 2.0 * jnp.pi * jnp.sum(w_ksiX * d20_l2_actual) + cl_te_unlensed[2]
    cl_p_l2 = jnp.sum(w_ksip * d22_l2_actual)
    cl_m_l2 = jnp.sum(w_ksim * d2m2_l2_actual)
    cl_ee_l2 = jnp.pi * (cl_p_l2 + cl_m_l2) + cl_ee_unlensed[2]
    cl_bb_l2 = jnp.pi * (cl_p_l2 - cl_m_l2) + cl_bb_unlensed[2]

    # Scan from l_idx=3 to l_max
    # Carry: d at (l_idx-2, l_idx-1). For l_idx=3: d at (1, 2).
    # For d00: (P_1, P_2)
    # For Wigner: (rescaled[1], rescaled[2])
    inverse_init = (x_gl, d00_l2,          # P_1, P_2
                    d20_r1, d20_r2,         # rescaled d20 at l=1, l=2
                    d22_r1, d22_r2,         # rescaled d22 at l=1, l=2
                    d2m2_r1, d2m2_r2)       # rescaled d2m2 at l=1, l=2

    _, (cl_tt_3up, cl_ee_3up, cl_te_3up, cl_bb_3up) = jax.lax.scan(
        _inverse_scan,
        inverse_init,
        jnp.arange(3, l_max + 1),
    )

    # Assemble: l=0,1 zero, l=2 from direct, l=3..l_max from scan
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
    """Apply lensing to C_l^TT only (backward-compatible wrapper).

    Uses lens_cls internally with zero EE/TE/BB.
    """
    n = len(cl_tt_unlensed)
    zeros = jnp.zeros(n)
    cl_tt, _, _, _ = lens_cls(
        cl_tt_unlensed, zeros, zeros, zeros, cl_pp,
        l_max=l_max, n_gauss=n_gauss,
    )
    return cl_tt
