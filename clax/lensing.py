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
