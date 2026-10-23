"""
One-loop EFT power spectra via FFTLog decomposition (CLASS-PT algorithm in JAX).

Implements the EFTofLSS 1-loop matter power spectrum and biased tracer spectra,
following the CLASS-PT algorithm (Chudaykin, Ivanov, Philcox & Simonovic 2020).

Physics pipeline:
  linear P(k)
    → FFTLog decomposition into complex power-law basis c_m k^{η_m}
    → Matrix products with precomputed kernels M13, M22, M22basic, IFG2
    → P13(k), P22(k) one-loop corrections
    → IR resummation: P_tree = P_nw + P_w exp(-Σ² k²)
    → Bias combination: P_gg, P_gm, P_mm
    → RSD multipoles: P_ℓ(k) for ℓ=0,2,4

All JAX functions are differentiable w.r.t. linear P(k) input, enabling
autodiff through the full perturbation theory pipeline.

References:
  Ivanov, Simonovic & Zaldarriaga (2020) arXiv:1909.05273
  Chudaykin, Ivanov, Philcox & Simonovic (2020) arXiv:2012.04636
  Schmittfull, Feng, Harikane & Zaldarriaga (2016) arXiv:1603.04405
  CLASS-PT source: nonlinear_pt.c, especially nonlinear_pt_loop() lines 4914+

Mirrors CLASS-PT: source/nonlinear_pt.c::nonlinear_pt_loop()
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Complex

from clax.interpolation import _compute_natural_spline_coeffs


# ---------------------------------------------------------------------------
# Constants and paths
# ---------------------------------------------------------------------------

# Default FFTLog parameters (match CLASS-PT nonlinear_pt.c)
NMAX_EPT: int = 256
B_MATTER: float = -0.3     # FFTLog bias for matter power spectrum (M22/M13)
B_TRANSFER: float = -0.8   # FFTLog bias for transfer functions
B_BASIC: float = -1.6      # FFTLog bias for M22basic (bias spectra; b2=-1.6 in CLASS-PT line 11789)
KMIN_H: float = 0.00005   # k_min in h/Mpc  — CLASS-PT nonlinear_pt.c:5066: kmin = 0.00005*h
KMAX_H: float = 100.0     # k_max in h/Mpc  — CLASS-PT nonlinear_pt.c:5067: kmax = 100.*h
CUTOFF: float = 3.0        # UV cutoff k_cut [h/Mpc] — CLASS-PT nonlinear_pt.c:5976: cutoff = 3.*h

# Path to CLASS-PT matrix files.
# Try ../CLASS-PT relative to the repo root first (HPC layout where CLASS-PT
# is cloned alongside clax), then ~/CLASS-PT (local developer layout).
_PKG_DIR   = os.path.dirname(__file__)           # clax/clax/
_REPO_DIR  = os.path.dirname(_PKG_DIR)           # clax/
_CLASSPT_DIR = (
    os.path.join(_REPO_DIR, "..", "CLASS-PT")    # ../CLASS-PT (HPC)
    if os.path.isdir(os.path.join(_REPO_DIR, "..", "CLASS-PT", "pt_matrices"))
    else os.path.expanduser("~/CLASS-PT")         # ~/CLASS-PT (local)
)
_MATRIX_DIR = os.path.join(_CLASSPT_DIR, "pt_matrices")

# 40-pt Gauss-Legendre nodes and weights from CLASS-PT gauss_tab.dat.
# First 40 lines are nodes on [-1,1], next 40 are weights (sum = 2).
_gauss_tab_path = os.path.join(_MATRIX_DIR, "gauss_tab.dat")
if os.path.exists(_gauss_tab_path):
    _gauss_tab = np.loadtxt(_gauss_tab_path)
    _GAUSS_NODES   = _gauss_tab[:40]   # mu nodes
    _GAUSS_WEIGHTS = _gauss_tab[40:]   # weights
else:
    # Fallback: 10-pt Gauss-Legendre (scipy)
    from numpy.polynomial.legendre import leggauss as _leggauss
    _GAUSS_NODES, _GAUSS_WEIGHTS = _leggauss(10)


# ---------------------------------------------------------------------------
# Precision parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EPTPrecisionParams:
    """Static precision settings for EPT computation.

    These control array shapes; changing them requires recompilation.
    """
    nmax: int = NMAX_EPT          # Number of FFTLog modes (128/256/512)
    kmin_h: float = KMIN_H        # k_min in h/Mpc
    kmax_h: float = KMAX_H        # k_max in h/Mpc
    b_matter: float = B_MATTER    # FFTLog bias parameter
    cutoff_h: float = CUTOFF      # P22 UV cutoff k_cut [h/Mpc]
    ir_resummation: bool = True   # Apply IR (BAO) resummation


# ---------------------------------------------------------------------------
# Matrix loading (run once at import time, cached)
# ---------------------------------------------------------------------------

def _load_complex_vector(filepath: str, n: int) -> np.ndarray:
    """Load n complex numbers from CLASS-PT ASCII file.

    Format: first n lines are real parts, next n lines are imaginary parts.
    Mirrors: nonlinear_pt.c load_M13, load_IFG2 sections.
    """
    data = np.loadtxt(filepath)
    assert len(data) == 2 * n, f"{filepath}: expected {2*n} values, got {len(data)}"
    return data[:n] + 1j * data[n:]


def _load_complex_triangular_lapack_l(filepath: str, n: int) -> np.ndarray:
    """Load LAPACK-packed lower-triangular matrix from CLASS-PT ASCII file.

    Then unpack to full n×n symmetric matrix.
    Format: first n(n+1)/2 lines real, then n(n+1)/2 lines imaginary.

    The file uses LAPACK-style column-major lower-triangular packing (UPLO='L'),
    matching the CLASS-PT zspmv_ call. Column j stores rows i = j, j+1, ..., n-1.
    The index of element M[i,j] (i >= j) in the packed array is:
        k = j*n - j*(j-1)//2 + (i-j)  (equivalently: i + j*(2n-1-j)//2)
    This is confirmed by the matrix generation script (matrix_gen-matter.py line:
        mout_red[i + (2*(Nmax+1)-1-j)*j//2] = m12mat[i][j])

    Mirrors: nonlinear_pt.c load_M22 section + LAPACK zspmv_ UPLO='L' convention.
    """
    n_tri = n * (n + 1) // 2
    data = np.loadtxt(filepath)
    assert len(data) == 2 * n_tri, (
        f"{filepath}: expected {2*n_tri} values, got {len(data)}"
    )
    tri = data[:n_tri] + 1j * data[n_tri:]

    # Unpack LAPACK UPLO='L' column-major lower-triangular packed to full symmetric matrix.
    # Column j stores rows i = j, j+1, ..., n-1; index k = j*n - j*(j-1)//2 + (i-j).
    # Confirmed by diag_m22_packing.py: zspmv_ UPLO='L' matches this packing exactly.
    M = np.zeros((n, n), dtype=complex)
    idx = 0
    for j in range(n):
        for i in range(j, n):
            M[i, j] = tri[idx]
            M[j, i] = tri[idx]  # symmetric: M22 uses zdotu (bilinear), not zdotc
            idx += 1
    return M


@lru_cache(maxsize=4)
def _load_matrices(nmax: int) -> dict:
    """Load all CLASS-PT kernel matrices for the given N_max.

    CLASS-PT stores M22/M22basic in LAPACK 'L' column-major lower-triangular
    packed format for EVERY N; only the file name varies (Nstr_M22 is
    "N128" / "N512" / "N256_packed", nonlinear_pt.c:847-864, all read through
    the same CONVERT_REAL_TO_COMPLEX_M22 macro at :895 into one P22 consumer).

    Returns a dict with keys:
      M13   : (nmax+1,) complex  — P13 kernel (matter)
      M22   : (nmax+1, nmax+1) complex — P22 kernel (matter, includes F2)
      M22b  : (nmax+1, nmax+1) complex — Basic mode coupling I(ν₁,ν₂)
      IFG2  : (nmax+1,) complex  — F·G2 coupling
    """
    n = nmax + 1  # matrix dimension

    def mat_path(name):
        return os.path.join(_MATRIX_DIR, name)

    # One ordering for every N (see the docstring): choose the packed file
    # when CLASS-PT ships one, otherwise the plain name, and read both with the
    # LAPACK 'L' unpacker. Reading N128/N512 row-major instead gives a silently
    # wrong M22 -- ~300% error with a negative monopole. M22oneline_N256.dat,
    # the non-packed N=256 file, is legacy and CLASS-PT never reads it.
    packed_m22 = mat_path(f"M22oneline_N{nmax}_packed.dat")
    packed_m22b = mat_path(f"M22basiconeline_N{nmax}_packed.dat")
    m22_file = (packed_m22 if os.path.exists(packed_m22)
                else mat_path(f"M22oneline_N{nmax}.dat"))
    m22b_file = (packed_m22b if os.path.exists(packed_m22b)
                 else mat_path(f"M22basiconeline_N{nmax}.dat"))
    m22 = _load_complex_triangular_lapack_l(m22_file, n)
    m22b = _load_complex_triangular_lapack_l(m22b_file, n)

    return {
        "M13":  _load_complex_vector(
            mat_path(f"M13oneline_N{nmax}.dat"), n),
        "IFG2": _load_complex_vector(
            mat_path(f"IFG2oneline_N{nmax}.dat"), n),
        "M22":  m22,
        "M22b": m22b,
    }


# ---------------------------------------------------------------------------
# EPT spectral components (JAX pytree)
# ---------------------------------------------------------------------------

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class EPTComponents:
    """All spectral components for galaxy bias combination.

    All power spectra are on the EPT k-grid in h-units:
      k in h/Mpc,  P in (Mpc/h)³.

    Bias combination formulas:
      pk_mm_real(cs0) = Pk_tree + Pk_loop + 2*cs0*Pk_ctr/h²
      pk_gg_real(b1,b2,bG2,bGamma3,cs,cs0,Pshot):
        = (b1²*(Pk_tree+Pk_loop) + b1*b2*Pk_Id2 + b2²/4*Pk_Id2d2
           + 2*b1*bG2*Pk_IG2 + b1*(2*bG2+0.8*bGamma3)*Pk_IFG2
           + bG2²*Pk_IG2G2 + b2*bG2*Pk_Id2G2
           + 2*(cs*b1²+cs0*b1)*Pk_ctr/h²) * h³ + Pshot

    For RSD multipoles, the vv/vd/dd decomposition is stored separately.
    """

    # k-grid in h/Mpc
    kh: Float[Array, "Nk"]         # k values [h/Mpc]
    h:  Float[Array, ""]            # Hubble parameter (for unit conversion)
    f:  Float[Array, ""]            # growth rate d ln D/d ln a

    # Core matter components
    Pk_tree:  Float[Array, "Nk"]   # P_tree = IR-resummed linear P (index 14)
    Pk_loop:  Float[Array, "Nk"]   # P_1loop = P13+P22 (index 0)
    Pk_ctr:   Float[Array, "Nk"]   # counterterm = -k² P_lin (index 10)

    # Bias cross-spectra (real space, h-units, indices 1–9)
    Pk_Id2d2:  Float[Array, "Nk"]  # δ²δ² power (index 1)
    Pk_Id2:    Float[Array, "Nk"]  # b1·δ² cross (index 2)
    Pk_IG2:    Float[Array, "Nk"]  # b1·G2 cross (index 3)
    Pk_Id2G2:  Float[Array, "Nk"] # δ²·G2 cross (index 4)
    Pk_IG2G2:  Float[Array, "Nk"] # G2·G2 (index 5)
    Pk_IFG2:   Float[Array, "Nk"] # F·G2 term (index 6)
    Pk_IFG2_0b1: Float[Array, "Nk"] # FG2 monopole b1-weighted (index 7)
    Pk_IFG2_0:   Float[Array, "Nk"] # FG2 monopole (index 8)
    Pk_IFG2_2:   Float[Array, "Nk"] # FG2 quadrupole (index 9)

    # Counterterm multipoles (indices 11-13)
    Pk_ctr0:  Float[Array, "Nk"]   # monopole counterterm (index 11)
    Pk_ctr2:  Float[Array, "Nk"]   # quadrupole counterterm (index 12)
    Pk_ctr4:  Float[Array, "Nk"]   # hexadecapole counterterm (index 13)

    # RSD tree-level multipoles (indices 15–20 + 49–51 for aniso corrections)
    Pk_0_vv:  Float[Array, "Nk"]   # monopole tree, vv (index 15)
    Pk_0_vd:  Float[Array, "Nk"]   # monopole tree, vd (index 16)
    Pk_0_dd:  Float[Array, "Nk"]   # monopole tree, dd (index 17)
    Pk_2_vv:  Float[Array, "Nk"]   # quadrupole tree, vv (index 18)
    Pk_2_vd:  Float[Array, "Nk"]   # quadrupole tree, vd (index 19)
    Pk_4_vv:  Float[Array, "Nk"]   # hexadecapole tree, vv (index 20)
    # Anisotropic tree corrections (zero in isotropic approx; non-zero with Sigmatot(mu))
    Pk_2_dd:  Float[Array, "Nk"]   # quadrupole tree, dd (index 49)
    Pk_4_vd:  Float[Array, "Nk"]   # hexadecapole tree, vd (index 50)
    Pk_4_dd:  Float[Array, "Nk"]   # hexadecapole tree, dd (index 51)

    # RSD 1-loop multipoles (indices 21–29): vv, vd, dd for each multipole
    Pk_0_vv1: Float[Array, "Nk"]   # monopole 1-loop, vv (index 21)
    Pk_0_vd1: Float[Array, "Nk"]   # monopole 1-loop, vd (index 22)
    Pk_0_dd1: Float[Array, "Nk"]   # monopole 1-loop, dd (index 23)
    Pk_2_vv1: Float[Array, "Nk"]   # quadrupole 1-loop, vv (index 24)
    Pk_2_vd1: Float[Array, "Nk"]   # quadrupole 1-loop, vd (index 25)
    Pk_2_dd1: Float[Array, "Nk"]   # quadrupole 1-loop, dd (index 26)
    Pk_4_vv1: Float[Array, "Nk"]   # hexadecapole 1-loop, vv (index 27)
    Pk_4_vd1: Float[Array, "Nk"]   # hexadecapole 1-loop, vd (index 28)
    Pk_4_dd1: Float[Array, "Nk"]   # hexadecapole 1-loop, dd (index 29)

    # RSD bias cross terms monopole (indices 30–33)
    Pk_0_b1b2:  Float[Array, "Nk"] # monopole b1b2 (index 30)
    Pk_0_b2:    Float[Array, "Nk"] # monopole b2 (index 31)
    Pk_0_b1bG2: Float[Array, "Nk"] # monopole b1bG2 (index 32)
    Pk_0_bG2:   Float[Array, "Nk"] # monopole bG2 (index 33)

    # RSD bias cross terms quadrupole (indices 34–37)
    Pk_2_b1b2:  Float[Array, "Nk"]
    Pk_2_b2:    Float[Array, "Nk"]
    Pk_2_b1bG2: Float[Array, "Nk"]
    Pk_2_bG2:   Float[Array, "Nk"]

    # RSD bias cross terms hexadecapole (indices 38–41)
    Pk_4_b2:    Float[Array, "Nk"]
    Pk_4_bG2:   Float[Array, "Nk"]
    Pk_4_b1b2:  Float[Array, "Nk"]
    Pk_4_b1bG2: Float[Array, "Nk"]

    # Anisotropic IR resummation + higher-order RSD (added for Gauss quadrature)
    pk_nw: Float[Array, "Nk"]           # no-wiggle P(k) in (Mpc/h)³
    pk_w:  Float[Array, "Nk"]           # wiggle P(k) = pk_lin - pk_nw
    sigma2_bao: Float[Array, ""]         # isotropic BAO damping Σ²_BAO in (Mpc/h)²
    delta_sigma2_bao: Float[Array, ""]   # anisotropic correction δΣ²_BAO in (Mpc/h)²
    Pd2d2_0: Float[Array, ""]            # zero-lag P_δ²δ²(k→0) in (Mpc/h)³
    P22_mu6_vv: Float[Array, "Nk"]      # P22 bare mu^6 vv coefficient
    P22_mu6_vd: Float[Array, "Nk"]      # P22 bare mu^6 vd coefficient
    P22_mu8:    Float[Array, "Nk"]      # P22 bare mu^8 coefficient
    P13_mu6:    Float[Array, "Nk"]      # P13 bare mu^6 coefficient

    def tree_flatten(self):
        # Separate arrays (leaves) from scalars (aux)
        arrays = [
            self.kh, self.Pk_tree, self.Pk_loop, self.Pk_ctr,
            self.Pk_Id2d2, self.Pk_Id2, self.Pk_IG2, self.Pk_Id2G2,
            self.Pk_IG2G2, self.Pk_IFG2,
            self.Pk_IFG2_0b1, self.Pk_IFG2_0, self.Pk_IFG2_2,
            self.Pk_ctr0, self.Pk_ctr2, self.Pk_ctr4,
            self.Pk_0_vv, self.Pk_0_vd, self.Pk_0_dd,
            self.Pk_2_vv, self.Pk_2_vd, self.Pk_4_vv,
            self.Pk_0_vv1, self.Pk_0_vd1, self.Pk_0_dd1,
            self.Pk_2_vv1, self.Pk_2_vd1, self.Pk_2_dd1,
            self.Pk_4_vv1, self.Pk_4_vd1, self.Pk_4_dd1,
            self.Pk_0_b1b2, self.Pk_0_b2, self.Pk_0_b1bG2, self.Pk_0_bG2,
            self.Pk_2_b1b2, self.Pk_2_b2, self.Pk_2_b1bG2, self.Pk_2_bG2,
            self.Pk_4_b2, self.Pk_4_bG2, self.Pk_4_b1b2, self.Pk_4_b1bG2,
            self.pk_nw, self.pk_w,                               # 43, 44
            self.P22_mu6_vv, self.P22_mu6_vd, self.P22_mu8, self.P13_mu6,  # 45-48
            self.Pk_2_dd, self.Pk_4_vd, self.Pk_4_dd,          # 49, 50, 51
            # Scalars as leaves (tracer-safety for traced h/f/sigma2_bao/
            # delta_sigma2_bao/Pd2d2_0): these are the LAST FIVE leaves, in
            # order h, f, sigma2_bao, delta_sigma2_bao, Pd2d2_0 -- see
            # tree_unflatten below.
            # Emitted bare (no jnp.asarray): the constructor already coerces,
            # and transforming during flatten breaks eval_shape/vmap-sentinel
            # round-trips.
            self.h, self.f, self.sigma2_bao, self.delta_sigma2_bao,
            self.Pd2d2_0,
        ]
        aux = None
        return arrays, aux

    @classmethod
    def tree_unflatten(cls, aux, arrays):
        # The five scalars are the LAST FIVE leaves, in order h, f,
        # sigma2_bao, delta_sigma2_bao, Pd2d2_0 (matches tree_flatten above).
        # Negative indexing so the tail stays correct if the array-leaf
        # count above it changes.
        h, f, sigma2_bao, delta_sigma2_bao, Pd2d2_0 = (
            arrays[-5], arrays[-4], arrays[-3], arrays[-2], arrays[-1])
        return cls(
            kh=arrays[0], h=h, f=f,
            Pk_tree=arrays[1], Pk_loop=arrays[2], Pk_ctr=arrays[3],
            Pk_Id2d2=arrays[4], Pk_Id2=arrays[5], Pk_IG2=arrays[6],
            Pk_Id2G2=arrays[7], Pk_IG2G2=arrays[8], Pk_IFG2=arrays[9],
            Pk_IFG2_0b1=arrays[10], Pk_IFG2_0=arrays[11], Pk_IFG2_2=arrays[12],
            Pk_ctr0=arrays[13], Pk_ctr2=arrays[14], Pk_ctr4=arrays[15],
            Pk_0_vv=arrays[16], Pk_0_vd=arrays[17], Pk_0_dd=arrays[18],
            Pk_2_vv=arrays[19], Pk_2_vd=arrays[20], Pk_4_vv=arrays[21],
            Pk_0_vv1=arrays[22], Pk_0_vd1=arrays[23], Pk_0_dd1=arrays[24],
            Pk_2_vv1=arrays[25], Pk_2_vd1=arrays[26], Pk_2_dd1=arrays[27],
            Pk_4_vv1=arrays[28], Pk_4_vd1=arrays[29], Pk_4_dd1=arrays[30],
            Pk_0_b1b2=arrays[31], Pk_0_b2=arrays[32],
            Pk_0_b1bG2=arrays[33], Pk_0_bG2=arrays[34],
            Pk_2_b1b2=arrays[35], Pk_2_b2=arrays[36],
            Pk_2_b1bG2=arrays[37], Pk_2_bG2=arrays[38],
            Pk_4_b2=arrays[39], Pk_4_bG2=arrays[40],
            Pk_4_b1b2=arrays[41], Pk_4_b1bG2=arrays[42],
            pk_nw=arrays[43], pk_w=arrays[44],
            sigma2_bao=sigma2_bao, delta_sigma2_bao=delta_sigma2_bao,
            Pd2d2_0=Pd2d2_0,
            P22_mu6_vv=arrays[45], P22_mu6_vd=arrays[46],
            P22_mu8=arrays[47], P13_mu6=arrays[48],
            Pk_2_dd=arrays[49], Pk_4_vd=arrays[50], Pk_4_dd=arrays[51],
        )


# ---------------------------------------------------------------------------
# FFTLog decomposition (JAX-native, differentiable)
# ---------------------------------------------------------------------------

def _fftlog_decompose(
    pk_disc: Float[Array, "Nmax"],
    kmin: float,
    kmax: float,
    nmax: int,
    b: float,
) -> tuple[Complex[Array, "Nmax+1"], Complex[Array, "Nmax+1"]]:
    """FFTLog decomposition of P(k) into complex power-law basis.

    Decomposes P(k) ≈ Σ_m c_m k^{η_m} where η_m are complex frequencies.

    Algorithm:
      1. Multiply P(k_j) by (k_j/k_min)^{-b} weight
      2. DFT to get raw coefficients
      3. Symmetrize for double-sided spectrum
      4. Apply k_min^{-η_m} normalization
      5. Half-weight endpoints m=0, m=N_max

    Mirrors: nonlinear_pt.c lines 5859–5938.

    Args:
        pk_disc: P(k) on N_max log-spaced points, shape (nmax,)
        kmin, kmax: k range [h/Mpc]
        nmax: number of FFTLog modes
        b: bias parameter (typically -0.3 for matter)

    Returns:
        cmsym: complex coefficients, shape (nmax+1,)
        etam:  complex frequencies, shape (nmax+1,)
    """
    step = jnp.log(kmax / kmin) / (nmax - 1)  # log-k step between adjacent grid points

    # Complex frequencies η_m = b + 2πi j_m / (N_max × step)
    # j_m = m - N_max/2  (signed mode index, -N_max/2 to +N_max/2)
    m = jnp.arange(nmax + 1)
    j_m = m - nmax // 2
    etam = b + 2j * jnp.pi * j_m / (nmax * step)  # shape (nmax+1,) complex

    # Step 1: weight P(k_j) by k_j^{-b} relative to k_min^{-b}
    #   input[j] = P(k_j) × exp(-j × b × step) = P(k_j) × (k_j/k_min)^{-b}
    j_idx = jnp.arange(nmax)
    input_arr = pk_disc * jnp.exp(-j_idx * b * step)  # real, shape (nmax,)

    # Step 2: DFT  (standard convention: cm[m] = sum_j input[j] exp(-2πi jm/N))
    cm_fft = jnp.fft.fft(input_arr)  # complex, shape (nmax,)

    # Step 3: symmetrize to get nmax+1 coefficients
    #   m < nmax/2:   c_sym[m] = conj(cm_fft[nmax/2 - m])
    #   m >= nmax/2:  c_sym[m] = cm_fft[m - nmax/2]
    nmax2 = nmax // 2
    # Low half (m = 0 .. nmax2-1): take conjugate of cm_fft at positive frequencies
    idx_low = nmax2 - jnp.arange(nmax2)      # [nmax2, nmax2-1, ..., 1]
    cm_sym_low = jnp.conj(cm_fft[idx_low])   # shape (nmax2,)
    # High half (m = nmax2 .. nmax): direct from cm_fft
    idx_high = jnp.arange(nmax2 + 1)          # [0, 1, ..., nmax2]
    cm_sym_high = cm_fft[idx_high]            # shape (nmax2+1,)

    cm_raw = jnp.concatenate([cm_sym_low, cm_sym_high]) / nmax  # shape (nmax+1,)

    # Step 4: apply k_min^{-η_m} factor
    # k_min^{-η_m} = exp(-η_m × log(k_min))
    cmsym = cm_raw * jnp.exp(-etam * jnp.log(kmin))  # complex, shape (nmax+1,)

    # Step 5: half-weight endpoints (avoid double-counting m=0 and m=nmax)
    cmsym = cmsym.at[0].multiply(0.5)
    cmsym = cmsym.at[-1].multiply(0.5)

    return cmsym, etam


def _x_at_k(
    cmsym: Complex[Array, "Nmax+1"],
    etam: Complex[Array, "Nmax+1"],
    k: Float[Array, "Nk"],
) -> Complex[Array, "Nk Nmax+1"]:
    """Evaluate FFTLog basis at k values: x[j,m] = c_m × k_j^{η_m}.

    Args:
        cmsym: FFTLog coefficients, shape (nmax+1,)
        etam:  complex frequencies, shape (nmax+1,)
        k:     output k values [h/Mpc], shape (Nk,)

    Returns:
        x: shape (Nk, nmax+1) complex
    """
    # x[j,m] = cmsym[m] × exp(η_m × log(k_j))
    log_k = jnp.log(k)  # shape (Nk,)
    # Broadcast: (Nk,1) × (1, nmax+1) → (Nk, nmax+1)
    return cmsym[None, :] * jnp.exp(etam[None, :] * log_k[:, None])


# ---------------------------------------------------------------------------
# One-loop P13 and P22 (JAX-native)
# ---------------------------------------------------------------------------

def _compute_p22(
    x: Complex[Array, "Nk Nmax+1"],
    k: Float[Array, "Nk"],
    M22: Complex[Array, "Nmax+1 Nmax+1"],
    cutoff_h: float,
) -> Float[Array, "Nk"]:
    """Compute P22(k) via matrix quadratic form.

    P22(k_j) = Re{ k_j³ × x_j^T M22 x_j } × exp[-(k_j/k_cut)⁶]

    The exp damping removes UV artifacts; k_cut ≈ 10 h/Mpc.
    Note: uses bilinear form (no conjugate), matching CLASS-PT zdotu.

    Mirrors: nonlinear_pt.c lines 6082–6102 (zspmv + zdotu).
    """
    # y[j,:] = M22 @ x[j,:]  via batched matmul:  x @ M22^T = x @ M22 (symmetric)
    y = x @ M22       # (Nk, nmax+1), complex
    # f22[j] = sum_m x[j,m] * y[j,m]  (bilinear, no conjugate)
    f22 = jnp.sum(x * y, axis=-1)  # (Nk,) complex
    # UV damping
    uv_damp = jnp.exp(-(k / cutoff_h) ** 6)
    return jnp.real(k ** 3 * f22) * uv_damp


def _compute_p13(
    x: Complex[Array, "Nk Nmax+1"],
    k: Float[Array, "Nk"],
    pk_disc: Float[Array, "Nk"],
    M13: Complex[Array, "Nmax+1"],
    lnk: Float[Array, "Nk"],
    cutoff_h: float = CUTOFF,
) -> Float[Array, "Nk"]:
    """Compute P13(k) via vector dot product + UV renormalization.

    P13_raw(k_j) = Re{ k_j³ × (x_j · M13) × P_lin(k_j) }
    P13_UV(k_j)  = -(61/105) σ_v² k_j² P_lin(k_j)   [UV subtraction]
    P13(k_j)     = (P13_raw + P13_UV) × exp[-(k_j/k_cut)⁶]

    where σ_v² = (1/6π²) ∫ dk P_lin(k) is the 1D velocity dispersion.
    UV damping matches P22 (k_cut = 3 h/Mpc).

    Mirrors: nonlinear_pt.c lines 6068–6079 (zdotu + sigma_v UV term + exp damping).
    """
    # f13[j] = sum_m x[j,m] * M13[m]  (bilinear dot product)
    f13 = jnp.sum(x * M13[None, :], axis=-1)  # (Nk,) complex
    P13_raw = jnp.real(k ** 3 * f13 * pk_disc)

    # UV counterterm: σ_v² = ∫ P(k) dk / (6π²)
    # Integrate via trapezoidal rule over ln(k)
    integrand = pk_disc * k  # = P(k) × k, for d(ln k) integration: ∫ P k d(ln k)
    sigma2_v = jnp.trapezoid(integrand, lnk) / (6.0 * jnp.pi ** 2)
    P13_UV = -(61.0 / 105.0) * sigma2_v * k ** 2 * pk_disc

    # UV damping (same cutoff as P22; cf. nonlinear_pt.c line 6068-6078)
    uv_damp = jnp.exp(-(k / cutoff_h) ** 6)
    return (P13_raw + P13_UV) * uv_damp


# ---------------------------------------------------------------------------
# IR resummation (BAO damping via DST-II decomposition)
# ---------------------------------------------------------------------------

def _ir_resummation_numpy(
    pk_lin_h: np.ndarray,
    k_h: np.ndarray,
    rs_h: float = 99.0,
    h: float = 0.6736,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Separate linear P(k) into no-wiggle and wiggle components.

    Uses Discrete Sine Transform II (DST-II) to identify and remove the
    BAO oscillation band, then reconstructs P_nw. The Σ_BAO² damping
    scale is computed from P_nw using the CLASS-PT filter formula.

    Mirrors: `nonlinear_pt_ir_resummation()`, nonlinear_pt.c:2711-3015 at
    CLASS-PT `09d5531a` (every anchor below re-verified at that commit; the
    5xxx line numbers this docstring used to cite are from a layout that no
    longer exists).

    CRITICAL: CLASS-PT uses a LINEAR k-grid (not log-spaced) for the DST.
    Modes 120–240 of a linear-k DST target k-space oscillation periods
    of (k_max-k_min)/120 to (k_max-k_min)/240 ≈ 0.08–0.04 h/Mpc, which
    brackets the BAO period 2π/r_s ≈ 0.06 h/Mpc. On a log-k grid those
    modes map to entirely different scales, giving wrong P_nw.

    CLASS-PT splits DST modes into odd and even sub-arrays (cmodd/cmeven)
    and applies natural cubic spline interpolation to each separately when
    removing modes 120–240. This gives ~3× more accurate P_nw than simple
    linear interpolation across the full DST. cf. nonlinear_pt.c:2815-2860.

    The linear spectrum is put onto the DST grid by a NATURAL CUBIC SPLINE of
    ln P against ln k (nonlinear_pt.c:2782), in CLASS-PT's own 1/Mpc, Mpc^3
    units (:2784) — both details matter at the 1 % level in the RSD one-loop
    rows; see the block comment on the DST input below.

    Outside the SOURCE TABLE (``k_h``), ln P is continued as a power law, as
    CLASS-PT does at nonlinear_pt.c:2779 (below the table) and :2788 (above),
    rather than left to the interpolator's out-of-range convention — scipy's
    CubicSpline continues the end cubic while `clax.interpolation.CubicSpline`
    clamps, so relying on either would put the two twins on different footing
    there. ONE DELIBERATE DEPARTURE: CLASS-PT's low-k branch uses ``ppm->n_s``,
    which is not an input to this function and cannot be made one without a
    signature change, so the first-interval ln-slope stands in. The substitution
    is INERT — neither branch is reachable for any grid clax feeds this
    function. The DST grid is [7e-5/h, 7/h] = [1.039e-4, 10.39] h/Mpc, while
    ``ept_kgrid`` spans [5e-5, 100] h/Mpc and CLASS-PT's own ``lnk_l`` spans
    [1.05e-5, 105] h/Mpc at each of the three campaign cosmologies: both source
    tables strictly bracket the DST grid on both sides (measured, not assumed).

    For k outside [k_min2, k_max2] — the DST grid, not the source table —
    CLASS-PT sets P_nw = P_lin and P_w = 0. cf. nonlinear_pt.c:2991-2997.
    Keeping that branch is not cosmetic: replacing it with a clamped
    extrapolation of P_nw injects up to 452× the linear power at k > 10.4 h/Mpc,
    and a correspondingly bogus P_w where CLASS-PT has exactly zero, into both
    RSD FFTLogs (ept.py:1393-1394) — measured; it is what made D1 §5.9 read the
    cubic-spline fix as a 3× regression.

    The Σ_BAO² formula uses a spherical Bessel j_2 filter to suppress
    the contribution of modes below the BAO scale:
      Σ_BAO² = (1/6π²) ∫₀^{ks} P_nw(q) × [1 - 3j₁(qr)/qr + ...] × q d(ln q)
    cf. nonlinear_pt.c:2919-2947 (IntegrandBAO).

    Args:
        pk_lin_h: P_lin(k) in (Mpc/h)³, shape (N,)
        k_h:      k in h/Mpc, shape (N,)
        rs_h:     Sound horizon at drag epoch times h, in Mpc (i.e. r_s(z_d) * h,
                  NOT r_s/h). Default 99.0 ≈ r_s_drag * h at Planck 2018 fiducial
                  (147.05 Mpc * 0.6736 = 99.05). For cosmology-varying values,
                  use ``clax.background.sound_horizon_drag(params) * params.h``.
                  This product enters the j_2-filter argument
                  ``x = k_int [h/Mpc] * rs_h [Mpc*h] = k [1/Mpc] * r_s [Mpc]``,
                  matching CLASS-PT nonlinear_pt.c:2936 ``qint2[index_ir] * rbao``.
        h:        Hubble parameter h = H₀/100 (needed to match CLASS-PT DST grid
                  which is hardcoded in 1/Mpc: kmin2=0.00007 1/Mpc, kmax2=7 1/Mpc)

    Returns:
        pk_nw:   no-wiggle (broadband) P(k), same shape as input
        pk_w:    wiggle P(k) = pk_lin_h - pk_nw
        sigma2_bao: BAO damping scale Σ_BAO² in (Mpc/h)²
    """
    try:
        from scipy.fft import dst, idst
        from scipy.interpolate import CubicSpline
    except ImportError:
        return _ir_resummation_gaussian(pk_lin_h, k_h)

    # LINEAR k-grid matching CLASS-PT nonlinear_pt.c:2748-2749 which hardcodes
    # kmin2=0.00007 1/Mpc and kmax2=7 1/Mpc (in physical 1/Mpc units), sampled
    # at Nir=65536 points (:2744) with kstep=(kmax2-kmin2)/(Nir-1) (:2764).
    # Converting to h-units: kmin2_h = 0.00007/h h/Mpc, kmax2_h = 7/h h/Mpc.
    # Using the exact CLASS-PT values is critical: the DST mode number for the
    # BAO oscillation is n_BAO ≈ kmax2/BAO_period, so a 4% kmax2 difference
    # shifts the BAO mode by 4%, causing different Pnw extraction.
    N_IR = 65536
    k_min2 = 7e-5 / h   # 0.00007 1/Mpc in h/Mpc (CLASS-PT kmin2)
    k_max2 = 7.0 / h    # 7 1/Mpc in h/Mpc (CLASS-PT kmax2)
    k_ir = np.linspace(k_min2, k_max2, N_IR)

    # --- DST input: ln(k P) on the linear grid ---------------------------------
    # cf. nonlinear_pt.c:2775-2795, which builds logkPdiscr in three pieces:
    #   :2779  k below the table   -> power law (CLASS-PT uses ppm->n_s)
    #   :2782  inside the table    -> array_interpolate_spline(lnk_l, lnpk_l,
    #                                 myddlnpk), a NATURAL cubic spline of ln P
    #                                 against ln k (SPLINE_SETUP, :3212)
    #   :2788  k above the table   -> power law with the last local ln-slope
    # clax used a piecewise-LINEAR np.interp here, which alone accounted for
    # ~2.5 pp of the 6.85 % of the BAO wiggle D1 measured as retained in pk_nw.
    #
    # UNITS (nonlinear_pt.c:2784, `log(_kbin2) + _logPbin2`): CLASS-PT forms
    # ln(k P) with k in 1/Mpc and P in Mpc^3, while clax carries h-units and
    # ln(k_h P_h) = ln(k P) + 2 ln h. That constant offset is NOT harmless. The
    # band removal below is linear, but it does not reproduce a constant: the
    # DST-II of a constant c is 2c / sin(pi (m+1) / 2N) on the even modes, a 1/m
    # tail that the natural spline across the removed [120, 240) band cannot
    # follow, so the offset comes back as a BAO-period ringing of ~3e-3 in
    # ln P_nw (measured; ~2.7 pp of the retained wiggle). Transform in CLASS-PT's
    # units and put the offset back after the inverse transform.
    ln_h2 = 2.0 * np.log(h)
    lnk_src = np.log(k_h)
    lnpk_src = np.log(np.clip(pk_lin_h, 1e-300, None))
    lnk_ir = np.log(k_ir)
    lnpk_ir = CubicSpline(lnk_src, lnpk_src, bc_type="natural")(lnk_ir)
    # Power-law extrapolation outside the source table, so the result never
    # depends on the interpolator's out-of-range behaviour (scipy continues the
    # end cubic; clax.interpolation.CubicSpline clamps -- the twins must agree).
    # CLASS-PT uses ppm->n_s below the table (:2779); n_s is not an input here,
    # so the first-interval slope stands in. Both branches are unreachable for
    # every grid clax feeds this function (ept_kgrid and CLASS's own lnk_l both
    # start below kmin2 and end above kmax2), so the substitution is inert.
    slope_lo = (lnpk_src[1] - lnpk_src[0]) / (lnk_src[1] - lnk_src[0])
    slope_hi = (lnpk_src[-1] - lnpk_src[-2]) / (lnk_src[-1] - lnk_src[-2])
    lnpk_ir = np.where(lnk_ir < lnk_src[0],
                       lnpk_src[0] + slope_lo * (lnk_ir - lnk_src[0]), lnpk_ir)
    lnpk_ir = np.where(lnk_ir > lnk_src[-1],
                       lnpk_src[-1] + slope_hi * (lnk_ir - lnk_src[-1]), lnpk_ir)

    # DST-II of log(k P(k)) — forward transform.
    # CLASS-PT packs (-1)^i logkPdiscr[i] into the odd slots of a length-4N real
    # FFT (:2792-2794) and reads out_ir[j] = fft_or[N-1-j] (:2805); expanding the
    # sum gives out_ir[j] = 2 sum_i f_i sin(pi (j+1)(2i+1) / 2N), i.e. exactly
    # scipy's unnormalised dst(type=2) in the same mode ordering (D1 §4). The
    # norm="ortho" constant cancels between dst and idst below.
    f_ir = lnk_ir + lnpk_ir - ln_h2
    f_dst = dst(f_ir, type=2, norm="ortho")

    # Remove BAO modes 120–240 by cubic spline on odd and even modes separately.
    # CLASS-PT splits cmnew[2i] (odd) and cmnew[2i+1] (even), removes Nleft:Nright
    # from each, then spline-interpolates back. This matches nonlinear_pt.c:2815-2860.
    # cf. nonlinear_pt.c:2817-2818: cmodd[i] = out_ir[2*i], cmeven[i] = out_ir[2*i+1];
    # Nleft/Nright at :2823-2824.
    N_left  = 120
    N_right = 240
    N_IR_half = N_IR // 2  # = 32768

    # Split full DST into odd and even indexed modes (over the half-spectrum).
    # out_ir is scipy's dst(type=2) mode for mode (see the forward transform
    # above), so this split is CLASS-PT's :2817-2818 exactly, not an
    # approximation. norm="ortho" scales every mode by sqrt(1/2N) except the
    # last (sqrt(1/4N)); that last mode is a KEPT knot of the cmeven spline and
    # so is reproduced exactly, and every mode the spline actually mixes carries
    # the same constant -- the normalisation therefore cancels with idst below.
    cmodd  = f_dst[0:N_IR:2]   # even indices of DST (odd "mode" in CLASS-PT sense)
    cmeven = f_dst[1:N_IR:2]   # odd indices of DST

    # For each of odd and even, remove modes [N_left, N_right) via cubic spline.
    # Indices run from 0..N_IR_half-1, mapping to DST indices 0,2,4,...
    # The "throw" region is [N_left, N_right) in the sub-arrays.

    def _remove_bao_modes(cm, n_left, n_right):
        """Cubic spline interpolation across [n_left, n_right) in cm array."""
        n = len(cm)
        n_throw = n_right - n_left
        n_new = n - n_throw
        # Build compact array: indices 0..n_left-1, n_right..n-1 → indices 0..n_new-1
        idx_keep = np.concatenate([np.arange(n_left), np.arange(n_right, n)])
        val_keep = cm[idx_keep]
        # Original indices (1-based as in CLASS-PT)
        i_orig = np.concatenate([np.arange(1, n_left + 1),
                                  np.arange(n_right + 1, n + 1)])
        # Spline on compact grid, evaluate at full 1-based indices
        cs = CubicSpline(i_orig, val_keep, bc_type='natural')
        cm_nw = cs(np.arange(1, n + 1))
        return cm_nw

    cmodd_nw  = _remove_bao_modes(cmodd,  N_left, N_right)
    cmeven_nw = _remove_bao_modes(cmeven, N_left, N_right)

    # Reconstruct no-wiggle DST coefficients: interleave odd and even
    f_dst_nw = np.empty(N_IR)
    f_dst_nw[0:N_IR:2] = cmodd_nw
    f_dst_nw[1:N_IR:2] = cmeven_nw

    # Inverse DST-II to recover log(k P_nw) on linear grid.
    # CLASS-PT's inverse packing (:2867-2893) expands to
    # out_2[i] = (-1)^i cmnew[N-1] + 2 sum_{p<N-1} cmnew[p] sin(pi (p+1)(2i+1)/2N),
    # i.e. an unnormalised DST-III, and it divides by 2N at :2906 -- exactly
    # scipy's idst(type=2). The +ln_h2 puts back the units offset removed before
    # the forward transform (see the UNITS note above); :2906 divides by k in
    # 1/Mpc, so P_nw comes out in Mpc^3 -> (Mpc/h)^3 = h^3 P_nw and k_h = k/h
    # give exp(f_nw + 2 ln h) / k_h.
    f_nw_ir = idst(f_dst_nw, type=2, norm="ortho")
    pk_nw_ir = np.exp(np.clip(f_nw_ir + ln_h2, -700, 700)) / k_ir

    # Map back to input k-grid; outside [k_min2, k_max2] set Pnw = Plin
    pk_nw = pk_lin_h.copy()
    in_range = (k_h >= k_min2) & (k_h <= k_max2)
    if in_range.sum() > 1:
        pk_nw[in_range] = np.exp(
            np.interp(
                np.log(k_h[in_range]),
                np.log(k_ir),
                np.log(np.clip(pk_nw_ir, 1e-300, None)),
            )
        )
    pk_w = pk_lin_h - pk_nw

    # Σ_BAO² with CLASS-PT j₂-filter formula, integrating up to ks=0.2 h/Mpc.
    # IntegrandBAO = P_nw × [1 - 3sin(qr)/(qr) + 6(sin(qr)/(qr)³ - cos(qr)/(qr)²)]
    # cf. nonlinear_pt.c:2926: ks = 0.2 * pba->h (1/Mpc) = 0.2 h/Mpc
    k_s = 0.2  # h/Mpc
    k_int = np.geomspace(k_min2, k_s, 1000)
    pk_nw_int = np.exp(
        np.interp(np.log(k_int), np.log(k_ir), np.log(np.clip(pk_nw_ir, 1e-300, None)))
    )
    x_bao = k_int * rs_h
    bao_filter = (
        1.0
        - 3.0 * np.sin(x_bao) / x_bao
        + 6.0 * (np.sin(x_bao) / x_bao ** 3 - np.cos(x_bao) / x_bao ** 2)
    )
    sigma2_bao = (
        np.trapezoid(pk_nw_int * bao_filter * k_int, np.log(k_int)) / (6.0 * np.pi ** 2)
    )

    # δΣ_BAO² (anisotropic second damping scale):
    # IntegrandBAO2 = -Pnw * (3*cos(qr)*r*q + (-3+(qr)²)*sin(qr)) / (qr)³
    # cf. nonlinear_pt.c:2940; normalization 1/(2π²) vs 1/(6π²) for Σ_BAO²
    bao_filter2 = -1.0 * (
        3.0 * np.cos(x_bao) * x_bao + (-3.0 + x_bao ** 2) * np.sin(x_bao)
    ) / x_bao ** 3
    delta_sigma2_bao = (
        np.trapezoid(pk_nw_int * bao_filter2 * k_int, np.log(k_int)) / (2.0 * np.pi ** 2)
    )

    return pk_nw, pk_w, float(sigma2_bao), float(delta_sigma2_bao)


def _ir_resummation_jax(
    pk_lin_h: Float[Array, "N"],
    k_h_static: np.ndarray,
    rs_h: Float[Array, ""],
    h_conc: float,
) -> tuple[Float[Array, "N"], Float[Array, "N"], Float[Array, ""], Float[Array, ""]]:
    """Traced JAX counterpart of :func:`_ir_resummation_numpy`.

    Same DST-based BAO extraction (see :func:`_ir_resummation_numpy` for the
    full physics writeup and CLASS-PT anchors), but differentiable in
    ``pk_lin_h`` (the linear power spectrum values) AND ``rs_h`` (the sound
    horizon times h that sets the BAO filter scale). Everything that is a
    *shape* of the computation -- the DST grid, the odd/even mode-removal
    index arithmetic, the k-integration grids -- stays static numpy/Python,
    exactly mirroring the numpy splitter's control flow; everything that is
    *data* flowing from ``pk_lin_h``/``rs_h`` is jnp and stays traced through
    ``dst2_ortho``/``idst2_ortho`` (Task 1) and
    :class:`clax.interpolation.CubicSpline` (natural-BC, same math as the
    scipy `CubicSpline(bc_type='natural')` used in the numpy version).

    Mirrors: `nonlinear_pt_ir_resummation()`, nonlinear_pt.c:2711-3015 at
    CLASS-PT `09d5531a`, same as `_ir_resummation_numpy`.

    Outside the SOURCE TABLE (``k_h_static``), ln P is continued as a power law
    via ``jnp.where`` — CLASS-PT's :2779 (below) and :2788 (above) — rather than
    left to the interpolator, precisely so the twins agree there: scipy's
    CubicSpline continues the end cubic while
    :class:`clax.interpolation.CubicSpline` clamps. Same deliberate departure as
    the numpy twin: CLASS-PT's low-k branch uses ``ppm->n_s``, not an input
    here, so the first-interval ln-slope stands in. INERT — neither branch is
    reachable for any grid clax feeds this function (DST grid [1.039e-4, 10.39]
    h/Mpc against ``ept_kgrid``'s [5e-5, 100] and CLASS-PT's ``lnk_l``
    [1.05e-5, 105] at all three campaign cosmologies), and both are ``jnp.where``
    on statically-shaped arrays, so nothing branches on a traced value.
    The separate out-of-DST-grid rule (P_nw = P_lin, P_w = 0, :2991-2997) is
    kept as in the numpy twin; see there for why dropping it is not cosmetic.

    Wired into `compute_ept_from_clax` (Task 3), this retires two of the
    three stop_gradient channels its "stop_gradient rationale" note used to
    document: the numpy IR-resummation input snapshot of pk_h, and the
    rs_h channel feeding compute_ept -- both flow through here untouched
    by stop_gradient. This also closes the previously-documented 1.39%
    ln10A_s-class residual (job 13132): pk_nw structurally dropping
    d(pk_nw)/d(pk_lin_h) is exactly the derivative this function restores;
    `test_pk_nw_gradient_exists_and_matches_fd` in
    tests/test_ir_resummation_jax.py checks it directly against FD. TWO
    frozen channels survive this closure, consistent with
    `compute_ept_from_clax`'s docstring (~2221-2231): (1) THIS function's
    own DST grid endpoints (k_min2/k_max2 above, built from a concrete
    h_conc) and the static `in_range` mask derived from them -- now the
    leading suspect for the measured h end-to-end residual (see the CAVEAT
    on the ENDPOINTS DELIBERATELY STATIC comment below, and the h FINDING
    in tests/test_ept_gradients.py); and (2) the RSD FFTLog basis (phase 2,
    pinned explicitly at compute_ept's `_ir_precomputed` RSD consumption) --
    unaffected by this function, which sits one level upstream of that
    split.

    Args:
        pk_lin_h:   P_lin(k) in (Mpc/h)^3, shape (N,), TRACED
        k_h_static: k in h/Mpc, shape (N,), static numpy (grid, not traced)
        rs_h:       r_s(z_d) * h in Mpc*h (see `_ir_resummation_numpy`
                    docstring for the exact convention), TRACED
        h_conc:     Hubble parameter h = H0/100, Python float, STATIC
                    (fixes the DST grid endpoints only -- see the freeze
                    comment below)

    Returns:
        pk_nw:        no-wiggle (broadband) P(k), same shape as input, jnp
        pk_w:         wiggle P(k) = pk_lin_h - pk_nw, jnp
        sigma2_bao:   BAO damping scale Sigma_BAO^2 in (Mpc/h)^2, jnp scalar
        delta_sigma2_bao: anisotropic damping scale, jnp scalar
    """
    from clax.interpolation import dst2_ortho, idst2_ortho, CubicSpline

    # LINEAR k-grid matching CLASS-PT nonlinear_pt.c:2748-2749 which hardcodes
    # kmin2=0.00007 1/Mpc and kmax2=7 1/Mpc (in physical 1/Mpc units), sampled
    # at Nir=65536 points (:2744) with kstep=(kmax2-kmin2)/(Nir-1) (:2764).
    # Converting to h-units: kmin2_h = 0.00007/h h/Mpc, kmax2_h = 7/h h/Mpc.
    # Using the exact CLASS-PT values is critical: the DST mode number for the
    # BAO oscillation is n_BAO ≈ kmax2/BAO_period, so a 4% kmax2 difference
    # shifts the BAO mode by 4%, causing different Pnw extraction.
    N_IR = 65536
    k_min2 = 7e-5 / h_conc   # 0.00007 1/Mpc in h/Mpc (CLASS-PT kmin2)
    k_max2 = 7.0 / h_conc    # 7 1/Mpc in h/Mpc (CLASS-PT kmax2)
    # ENDPOINTS DELIBERATELY STATIC: CLASS-PT-hardcoded cuts (nonlinear_pt.c:2748);
    # their h-derivative is a boundary term with negligible content (P·k weight
    # ~0 at both cuts). Everything BETWEEN the cuts is traced.
    # CAVEAT (measured, not closed by the above): this "negligible content"
    # argument covers only the boundary VALUES. The h-derivative of the GRID
    # ITSELF -- not just its endpoints -- shifts the whole interior
    # discretization that pk_nw is extracted on under central FD (h_conc is a
    # genuinely different float at h0+eps vs h0-eps), while under AD at fixed
    # h this snapshot is a constant with zero gradient by construction. That
    # mismatch measured a 1.3831e-02 residual on the h end-to-end gradient
    # (job 14146); this is the concrete phase-2 item -- see the h FINDING in
    # tests/test_ept_gradients.py.
    k_ir = np.linspace(k_min2, k_max2, N_IR)

    # --- DST input: ln(k P) on the linear grid ---------------------------------
    # Line for line with `_ir_resummation_numpy`; see the long block comment
    # there for the CLASS-PT anchors (:2775-2795) and, in particular, for why
    # the natural cubic spline and CLASS-PT's 1/Mpc, Mpc^3 units are both
    # required. The knots (`lnk_src`) are static grid arithmetic; the values
    # (`lnpk_src`) are traced, so clax.interpolation.CubicSpline carries the
    # derivative through, as it already does for the mode-removal spline below.
    ln_h2 = 2.0 * np.log(h_conc)
    lnk_src = np.log(k_h_static)
    lnpk_src = jnp.log(jnp.clip(pk_lin_h, 1e-300, None))
    lnk_ir = np.log(k_ir)
    lnpk_ir = CubicSpline(jnp.asarray(lnk_src), lnpk_src).evaluate(jnp.asarray(lnk_ir))
    # Power-law extrapolation outside the source table (nonlinear_pt.c:2779 with
    # ppm->n_s, :2788 with the last local slope); unreachable for every clax
    # grid, and stated explicitly so the two twins cannot drift on the
    # interpolators' differing out-of-range conventions (scipy continues the end
    # cubic, clax.interpolation.CubicSpline clamps).
    slope_lo = (lnpk_src[1] - lnpk_src[0]) / (lnk_src[1] - lnk_src[0])
    slope_hi = (lnpk_src[-1] - lnpk_src[-2]) / (lnk_src[-1] - lnk_src[-2])
    lnpk_ir = jnp.where(lnk_ir < lnk_src[0],
                        lnpk_src[0] + slope_lo * (lnk_ir - lnk_src[0]), lnpk_ir)
    lnpk_ir = jnp.where(lnk_ir > lnk_src[-1],
                        lnpk_src[-1] + slope_hi * (lnk_ir - lnk_src[-1]), lnpk_ir)

    # DST-II of log(k P(k)) — forward transform.
    # CLASS-PT packs (-1)^i logkPdiscr[i] into the odd slots of a length-4N real
    # FFT (:2792-2794) and reads out_ir[j] = fft_or[N-1-j] (:2805), which expands
    # to the unnormalised dst(type=2) in the same mode ordering (D1 §4).
    f_ir = lnk_ir + lnpk_ir - ln_h2
    f_dst = dst2_ortho(f_ir)

    # Remove BAO modes 120–240 by cubic spline on odd and even modes separately.
    # CLASS-PT splits cmnew[2i] (odd) and cmnew[2i+1] (even), removes Nleft:Nright
    # from each, then spline-interpolates back. This matches nonlinear_pt.c:2815-2860.
    # cf. nonlinear_pt.c:2817-2818: cmodd[i] = out_ir[2*i], cmeven[i] = out_ir[2*i+1];
    # Nleft/Nright at :2823-2824.
    N_left  = 120
    N_right = 240
    N_IR_half = N_IR // 2  # = 32768

    # Split full DST into odd and even indexed modes (over the half-spectrum).
    # out_ir is scipy's dst(type=2) mode for mode (see the forward transform
    # above), so this split is CLASS-PT's :2817-2818 exactly, not an
    # approximation; the ortho normalisation cancels against idst2_ortho below.
    cmodd  = f_dst[0:N_IR:2]   # even indices of DST (odd "mode" in CLASS-PT sense)
    cmeven = f_dst[1:N_IR:2]   # odd indices of DST

    # For each of odd and even, remove modes [N_left, N_right) via cubic spline.
    # Indices run from 0..N_IR_half-1, mapping to DST indices 0,2,4,...
    # The "throw" region is [N_left, N_right) in the sub-arrays.

    def _remove_bao_modes_jax(cm, n_left, n_right):
        """Cubic spline interpolation across [n_left, n_right) in cm array.

        jnp counterpart of the numpy splitter's `_remove_bao_modes`: knots
        (`i_orig`) and evaluation points (`np.arange(1, n + 1)`) are static
        index arithmetic; the spline VALUES (`val_keep`, drawn from the
        traced DST coefficients `cm`) flow through
        `clax.interpolation.CubicSpline`, which implements the same
        natural-BC math as scipy's `CubicSpline(bc_type='natural')`.
        """
        n = len(cm)
        n_throw = n_right - n_left
        n_new = n - n_throw
        # Build compact array: indices 0..n_left-1, n_right..n-1 → indices 0..n_new-1
        idx_keep = np.concatenate([np.arange(n_left), np.arange(n_right, n)])
        val_keep = cm[idx_keep]
        # Original indices (1-based as in CLASS-PT)
        i_orig = np.concatenate([np.arange(1, n_left + 1),
                                  np.arange(n_right + 1, n + 1)])
        # Spline on compact grid, evaluate at full 1-based indices
        cs = CubicSpline(jnp.asarray(i_orig, dtype=float), val_keep)
        cm_nw = cs.evaluate(jnp.arange(1, n + 1, dtype=float))
        return cm_nw

    cmodd_nw  = _remove_bao_modes_jax(cmodd,  N_left, N_right)
    cmeven_nw = _remove_bao_modes_jax(cmeven, N_left, N_right)

    # Reconstruct no-wiggle DST coefficients: interleave odd and even
    f_dst_nw = jnp.zeros(N_IR).at[0::2].set(cmodd_nw).at[1::2].set(cmeven_nw)

    # Inverse DST-II to recover log(k P_nw) on linear grid. CLASS-PT's inverse
    # packing (:2867-2893) plus the 1/(2N) at :2906 is scipy's idst(type=2);
    # +ln_h2 restores the units offset removed before the forward transform.
    f_nw_ir = idst2_ortho(f_dst_nw)
    pk_nw_ir = jnp.exp(jnp.clip(f_nw_ir + ln_h2, -700, 700)) / k_ir

    # Map back to input k-grid; outside [k_min2, k_max2] set Pnw = Plin
    in_range = (k_h_static >= k_min2) & (k_h_static <= k_max2)
    assert in_range.sum() > 1
    pk_nw = jnp.where(
        in_range,
        jnp.exp(
            jnp.interp(
                jnp.log(k_h_static),
                jnp.log(k_ir),
                jnp.log(jnp.clip(pk_nw_ir, 1e-300, None)),
            )
        ),
        pk_lin_h,
    )
    pk_w = pk_lin_h - pk_nw

    # Σ_BAO² with CLASS-PT j₂-filter formula, integrating up to ks=0.2 h/Mpc.
    # IntegrandBAO = P_nw × [1 - 3sin(qr)/(qr) + 6(sin(qr)/(qr)³ - cos(qr)/(qr)²)]
    # cf. nonlinear_pt.c:2926: ks = 0.2 * pba->h (1/Mpc) = 0.2 h/Mpc
    k_s = 0.2  # h/Mpc
    k_int = np.geomspace(k_min2, k_s, 1000)
    pk_nw_int = jnp.exp(
        jnp.interp(jnp.log(k_int), jnp.log(k_ir), jnp.log(jnp.clip(pk_nw_ir, 1e-300, None)))
    )
    x_bao = jnp.asarray(k_int) * rs_h
    bao_filter = (
        1.0
        - 3.0 * jnp.sin(x_bao) / x_bao
        + 6.0 * (jnp.sin(x_bao) / x_bao ** 3 - jnp.cos(x_bao) / x_bao ** 2)
    )
    sigma2_bao = (
        jnp.trapezoid(pk_nw_int * bao_filter * k_int, x=np.log(k_int)) / (6.0 * jnp.pi ** 2)
    )

    # δΣ_BAO² (anisotropic second damping scale):
    # IntegrandBAO2 = -Pnw * (3*cos(qr)*r*q + (-3+(qr)²)*sin(qr)) / (qr)³
    # cf. nonlinear_pt.c:2940; normalization 1/(2π²) vs 1/(6π²) for Σ_BAO²
    bao_filter2 = -1.0 * (
        3.0 * jnp.cos(x_bao) * x_bao + (-3.0 + x_bao ** 2) * jnp.sin(x_bao)
    ) / x_bao ** 3
    delta_sigma2_bao = (
        jnp.trapezoid(pk_nw_int * bao_filter2 * k_int, x=np.log(k_int)) / (2.0 * jnp.pi ** 2)
    )

    return pk_nw, pk_w, sigma2_bao, delta_sigma2_bao


def _ir_resummation_gaussian(
    pk_lin_h: np.ndarray,
    k_h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Gaussian-smoothing BAO separation (fallback, less accurate than DST).

    Smooths log P(k) with a Gaussian in log-k space to isolate broadband shape.
    ~0.3% error at BAO peak vs DST method. Used when scipy is unavailable.
    """
    # Gaussian smoothing in log-k space
    sigma_lnk = 0.25  # smoothing scale
    log_k = np.log(k_h)
    log_pk = np.log(np.clip(pk_lin_h, 1e-300, None))

    # Build smoothed P_nw via Gaussian kernel convolution
    log_pk_nw = np.zeros_like(log_pk)
    for i, lk0 in enumerate(log_k):
        w = np.exp(-0.5 * ((log_k - lk0) / sigma_lnk) ** 2)
        w /= w.sum()
        log_pk_nw[i] = np.dot(w, log_pk)

    pk_nw = np.exp(log_pk_nw)
    pk_w  = pk_lin_h - pk_nw
    sigma2_bao = np.trapezoid(pk_nw * k_h, np.log(k_h)) / (6.0 * np.pi ** 2)
    return pk_nw, pk_w, float(sigma2_bao), 0.0


# ---------------------------------------------------------------------------
# Vectorized Gauss-Legendre mu-integration (the CLASS-PT in-loop block)
# ---------------------------------------------------------------------------
# Channel names consumed by _gl_multipoles. Each LOOP name is present in `chan`
# twice, as "<name>_nw" and "<name>_w" (no-wiggle / wiggle FFTLog parts).
# The 15 names are exactly the RSD loop channels CLASS-PT re-interpolates at
# ktrue inside the AP loop (nonlinear_pt.c:4411-4440, local 09d5531a -- 30
# AP_INTERP_FAST lines, the un-suffixed name = clax's "_nw", `_w` = "_w").
_GL_CHANNELS_LOOP = (
    "P22_mu0_dd", "P13_mu0_dd",
    "P22_mu2_vd", "P22_mu2_dd", "P13_mu2_vd", "P13_mu2_dd",
    "P22_mu4_vv", "P22_mu4_vd", "P22_mu4_dd", "P13_mu4_vv", "P13_mu4_vd",
    "P22_mu6_vv", "P22_mu6_vd", "P13_mu6", "P22_mu8",
)
# The 14 bias channels CLASS-PT re-interpolates in its bias loop
# (nonlinear_pt.c:5301-5314, AP_INTERP_EX_FAST on the `_new` copies).
_GL_CHANNELS_BIAS = (
    "Pk_0_b1b2", "Pk_2_b1b2", "Pk_0_b1bG2", "Pk_2_b1bG2",
    "Pk_0_b2", "Pk_2_b2", "Pk_4_b2", "Pk_0_bG2", "Pk_2_bG2", "Pk_4_bG2",
    "Pk_Id2d2", "Pk_Id2G2", "Pk_IG2G2", "Pk_IFG2",
)
# Linear spectra interpolated at ktrue in both loops (nonlinear_pt.c:4403-4404
# RSD loop, 5279-5281 bias loop): Pnw, Pw and -- bias loop only -- Pbin
# (= clax's `pk_disc`). Read by _channels_at through `chan`, which carries these
# three plus _GL_CHANNELS_LOOP x2 and _GL_CHANNELS_BIAS = 47 keys in total; the
# three are remapped to ktrue exactly like the other 44.
_GL_CHANNELS_LIN = ("pk_nw", "pk_w", "pk_disc")


def _channels_at(k, chan: dict, kq) -> dict:
    """Every channel in `chan` (knots `k`, shape (Nk,)) evaluated at `kq`
    (any shape, here (40, Nk) = ktrue) with a NATURAL cubic spline in LINEAR
    k over the whole grid -- CLASS-PT's AP_SPLINE_SETUP
    (nonlinear_pt.c:2356-2362, `array_spline_table_columns(kdisc, ...,
    _SPLINE_NATURAL_)`). Outside [k[0], k[-1]] the end interval's cubic is
    continued, as AP_BSEARCH_SETUP/AP_INTERP_FAST do (2372-2394: the interval
    is clamped, the weights a, b are not). At kq == k the knot values are
    returned exactly (a = 1, b = 0), which keeps alpha = 1 bit-identical.
    Differentiable in kq (and through it in hratio, Dratio) and in the channel
    values.
    """
    names = tuple(chan)
    Y = jnp.stack([chan[n] for n in names])                                   # (C, Nk)
    d2 = jax.vmap(_compute_natural_spline_coeffs, in_axes=(None, 0))(k, Y)    # (C, Nk)
    n = k.shape[0]
    idx = jnp.clip(jnp.searchsorted(k, kq, side="right") - 1, 0, n - 2)       # 2384-2388 bisection
    hh = k[idx + 1] - k[idx]                                                  # 2389
    b = (kq - k[idx]) / hh                                                    # 2390, unclamped
    a = 1.0 - b                                                               # 2391
    a3a, b3b, h2_6 = a ** 3 - a, b ** 3 - b, hh ** 2 / 6.0                    # 2392-2394

    def one(y, dd):                                                           # 2373-2374 AP_INTERP_FAST
        return a * y[idx] + b * y[idx + 1] + (a3a * dd[idx] + b3b * dd[idx + 1]) * h2_6

    vals = jax.vmap(one)(Y, d2)                                               # (C, *kq.shape)
    return dict(zip(names, vals))


def _gl_multipoles(chan: dict, k, f, sigma2_bao, delta_sigma2_bao,
                   hratio=1.0, Dratio=1.0) -> dict:
    """40-node Gauss-Legendre mu-integration of every RSD channel, vectorized.

    Mirrors the per-node body of CLASS-PT's AP/IR loop (local checkout
    09d5531a): tree, 1-loop and counterterms `nonlinear_pt.c:4386-4562`,
    bias and IFG2 `5225-5366`, with the AP remap of `4382-4394`
    (hratio = H_true/H_fid, Dratio = D_A,true/D_A,fid; both 1 => CLASS-PT's
    alpha=1 branch `4396-4397`, because
    ap_fac = sqrt(1/Dr^2 + (hr^2 - 1/Dr^2) mu^2) is exactly 1.0 in IEEE double
    there, so ktrue == k on the knots and `_channels_at` returns the channel
    values unchanged). Axis 0 is the mu node (40), axis 1 is k.

    This function is LINEAR in every entry of `chan`, so it is agnostic to the
    leaf sign convention: feed it clax's leaves (the sign `classy.pyx`'s
    `get_pk_mult` hands out) and every output comes back in the same
    convention. The one exception is `Pk_4_b1bG2`, whose two parents
    (`Pk_0_b1bG2` = pm[32], `Pk_2_b1bG2` = pm[36]) ARE negated by
    classy.pyx:4721-4722 while its own row pm[41] (classy.pyx:4735) is NOT; the
    caller flips it back. See `_compute_bias_spectra`.

    Projections use the FIDUCIAL-mu Legendre weights
    W_l = (2l+1)/2 * w * L_l(mu) (`4470-4471` fiducial L2/L4, `2565-2568`
    LEGENDRE_PROJECT macro); every kernel mu-power and the IR damping
    Sigma_tot(mu) (`4480`) use the "true" mu, and every channel is splined at
    the "true" k (`_channels_at`, `4403-4440`, `5279-5314`).

    `V = hratio/Dratio**2` (`4384`, written inline on every in-loop term,
    `4494-4500`, `4503`/`4512`/`4521`, `5320-5328`) is folded into W0/W2/W4:
    every output of this function goes through `proj(...)`, so one factor on
    the projection weights is one factor on each accumulated term. An output
    that ever bypasses `proj` must carry `V` explicitly.

    chan: 47 arrays on the k grid -- `<name>_nw`/`<name>_w` for every
          _GL_CHANNELS_LOOP name, the _GL_CHANNELS_BIAS spectra, and
          pk_nw, pk_w (= 0 without IR), pk_disc (the FFTLog input spectrum,
          CLASS-PT's `Pbin`, `nonlinear_pt.c:2989`).
    Returns 39 multipole arrays (tree x9, loop x9, ctr x3, IFG2 x3,
    Id2d2/Id2G2/IG2G2, bias x12).

    Known deviation from CLASS-PT, inherited from clax's leaf layout: CLASS-PT
    folds the tree into the mu^0/mu^2 brackets of `P1loopdd_ap_ir` (`4529`) /
    `P1loopvd_ap_ir` (`4531`) before projecting l=2,4 (`4535` 2_dd, `4536`
    4_dd, `4539` 4_vd), so its rows 26/28/29 carry tree+loop; clax projects
    tree and loop separately into `Pk_l_dd` / `Pk_l_dd1`, whose sum is
    identical (the projection is linear). The b1-weighted galaxy accessors
    consume exactly that sum (`pk_gg_l2` / `pk_gg_l4` below).
    """
    mu = jnp.asarray(_GAUSS_NODES)                       # (40,)   gauss_x, 4388/5258
    w = jnp.asarray(_GAUSS_WEIGHTS)                      # (40,)   gauss_w
    hratio = jnp.asarray(hratio, dtype=k.dtype)
    Dratio = jnp.asarray(Dratio, dtype=k.dtype)
    inv_D2 = 1.0 / Dratio ** 2                                        # 4382 ap_inv_Dr2_
    ap_fac = jnp.sqrt(inv_D2 + (hratio ** 2 - inv_D2) * mu ** 2)[:, None]   # 4392  (40, 1)
    mu_k = mu[:, None] * hratio / ap_fac                 # 4393 mutrue  (40, 1)
    kk = k[None, :] * ap_fac                             # 4394 ktrue   (40, Nk)
    V = hratio / Dratio ** 2                             # 4384 ap_w_hr_Dr2_
    mu2t = mu_k ** 2                                     # 4474-4477 mu2t..mu8t from mutrue
    mu4t, mu6t, mu8t = mu2t ** 2, mu2t ** 3, mu2t ** 4
    L2 = 0.5 * (3.0 * mu ** 2 - 1.0)                     # 4470 LegendreP2, FIDUCIAL mu
    L4 = (35.0 * mu ** 4 - 30.0 * mu ** 2 + 3.0) / 8.0   # 4471 LegendreP4, FIDUCIAL mu
    # 2565-2568 LEGENDRE_PROJECT weights x V (4494-4500, 4503/4512/4521, 5320-5328).
    # Every leaf below is built by proj(), so V rides on the weights; the one
    # AP-free real-space output of the module, `_pd2d2_0` (classy.pyx:4805-4813),
    # is computed outside this function and deliberately takes no V and no proj.
    W0, W2, W4 = 0.5 * w * V, 2.5 * w * L2 * V, 4.5 * w * L4 * V

    def proj(val, W):
        return jnp.einsum("m,mk->k", W, val)

    at = _channels_at(k, chan, kk)       # 4403-4440, 5279-5314: all 47 channels at ktrue

    def c(name):
        return at[name]                                  # (40, Nk)

    def nw(name):
        return c(name + "_nw")

    def wg(name):
        return c(name + "_w")

    # 4480 Sigmatot = SigmaBAO*(1 + f*mu2t*(2+f)) + f*f*mu2t*(mu2t-1)*deltaSigmaBAO
    Sig = (sigma2_bao * (1.0 + f * mu2t * (2.0 + f))
           + delta_sigma2_bao * f ** 2 * mu2t * (mu2t - 1.0))
    Exp = jnp.exp(-Sig * kk ** 2)                                     # 4481
    Pnw, Pw = c("pk_nw"), c("pk_w")
    Pnw_safe = jnp.where(Pnw > 1e-100, Pnw, 1.0)
    # 4485 P13ratio = 1 + (Pw/Pnw)*Exp (clax guards the Pnw -> 0 tail)
    P13ratio = jnp.where(Pnw > 1e-100, 1.0 + (Pw / Pnw_safe) * Exp, 1.0)
    p_tree = Pnw + (1.0 + Sig * kk ** 2) * Pw * Exp                   # 4483 p_tree
    p_lin = Pnw + Pw * Exp                                            # 4498-4500, 5318 bracket

    # 4503 P1loopvv: mu^4 + mu^6 + mu^8
    Pvv = ((nw("P13_mu4_vv") * P13ratio + nw("P22_mu4_vv") + (wg("P22_mu4_vv") + wg("P13_mu4_vv")) * Exp) * mu4t
           + (nw("P13_mu6") * P13ratio + nw("P22_mu6_vv") + (wg("P22_mu6_vv") + wg("P13_mu6")) * Exp) * mu6t
           + (nw("P22_mu8") + wg("P22_mu8") * Exp) * mu8t)
    # 4512 P1loopdd: mu^0 + mu^2 + mu^4
    Pdd = ((nw("P22_mu0_dd") + nw("P13_mu0_dd") * P13ratio + (wg("P13_mu0_dd") + wg("P22_mu0_dd")) * Exp)
           + (nw("P22_mu2_dd") + nw("P13_mu2_dd") * P13ratio + (wg("P22_mu2_dd") + wg("P13_mu2_dd")) * Exp) * mu2t
           + (nw("P22_mu4_dd") + wg("P22_mu4_dd") * Exp) * mu4t)
    # 4521 P1loopvd: mu^2 + mu^4 + mu^6
    Pvd = ((nw("P13_mu2_vd") * P13ratio + nw("P22_mu2_vd") + (wg("P22_mu2_vd") + wg("P13_mu2_vd")) * Exp) * mu2t
           + (nw("P13_mu4_vd") * P13ratio + nw("P22_mu4_vd") + (wg("P22_mu4_vd") + wg("P13_mu4_vd")) * Exp) * mu4t
           + (nw("P22_mu6_vd") + wg("P22_mu6_vd") * Exp) * mu6t)
    # 4494-4496 p_tree_vv/vd/dd = p_tree * {f^2 mu4t, 2 f mu2t, 1}
    tree = {"vv": f ** 2 * mu4t * p_tree, "vd": 2.0 * f * mu2t * p_tree, "dd": p_tree}
    loop = {"vv": Pvv, "vd": Pvd, "dd": Pdd}

    out = {}
    # 4533-4539 loop projections; 4554-4557 tree projections. CLASS-PT stores
    # no Tree_2_dd / Tree_4_vd / Tree_4_dd row (they live inside the _ap_ir
    # loop rows); clax keeps all nine tree multipoles as their own leaves.
    for ell, W in ((0, W0), (2, W2), (4, W4)):
        for ch in ("vv", "vd", "dd"):
            out[f"Pk_{ell}_{ch}"] = proj(tree[ch], W)
            out[f"Pk_{ell}_{ch}1"] = proj(loop[ch], W)

    # 4498-4500 Pctr0/2/4 = ktrue^2 (Pnw + Pw Exp) * {1, f mu2t, f^2 mu4t},
    # projected at 4550-4552. clax stores -P_CTR_l (= pm[11..13],
    # classy.pyx:4694-4697). The closed forms this replaces -- -k^2 Pbin,
    # -k^2 Pbin f 2/3, -k^2 Pbin f^2 8/35 -- are the isotropic Legendre moments,
    # exact only if Exp were mu-independent; Sigmatot(mu) (4480) makes it not.
    out["Pk_ctr0"] = -proj(kk ** 2 * p_lin, W0)
    out["Pk_ctr2"] = -proj(kk ** 2 * p_lin * f * mu2t, W2)
    out["Pk_ctr4"] = -proj(kk ** 2 * p_lin * f ** 2 * mu4t, W4)

    # 5225-5366 bias block. 5318 p_lo = (Pnw + Pw Exp)/Pbin rescales P_IFG2
    # (which carries a factor Pbin, `4987` / clax's pk_disc) from the isotropic
    # FFTLog input to the anisotropic resummed linear spectrum.
    Pbin = c("pk_disc")
    p_lo = p_lin / jnp.where(Pbin > 1e-100, Pbin, 1.0)
    IFG2_in = p_lo * c("Pk_IFG2")                                     # 5320
    out["Pk_IFG2_0b1"] = proj(IFG2_in, W0)                            # 5344
    out["Pk_IFG2_0"] = proj(IFG2_in * f * mu2t, W0)                   # 5345
    out["Pk_IFG2_2"] = proj(IFG2_in * f * mu2t, W2)                   # 5346
    # 5322-5324 Pd2d2_in/Pd2G2_in/PG2G2_in, projected at 5347-5349; clax keeps
    # the l=0 moment only (CLASS-PT rows 42-47, the l=2/4 moments, are unused).
    for name in ("Pk_Id2d2", "Pk_Id2G2", "Pk_IG2G2"):
        out[name] = proj(c(name), W0)
    L2t = 0.5 * (3.0 * mu2t - 1.0)                                    # 5273 LegendreP2true
    L4t = (35.0 * mu4t - 30.0 * mu2t + 3.0) / 8.0                     # 5274 LegendreP4true
    # 5325-5326 Pb1b2_in / Pb1bG2_in rebuild mu-dependence from the l=0,2
    # inputs only; projecting at 5350-5351 GENERATES P_4_b1b2 / P_4_b1bG2
    # (zero at alpha=1 by <L2 L4> = 0, non-zero once mutrue != mu).
    for b in ("b1b2", "b1bG2"):
        val = c(f"Pk_0_{b}") + L2t * c(f"Pk_2_{b}")
        for ell, W in ((0, W0), (2, W2), (4, W4)):
            out[f"Pk_{ell}_{b}"] = proj(val, W)
    # 5327-5328 Pb2_in / PbG2_in use l=0,2,4 inputs; projected at 5352-5353.
    for b in ("b2", "bG2"):
        val = c(f"Pk_0_{b}") + L2t * c(f"Pk_2_{b}") + L4t * c(f"Pk_4_{b}")
        for ell, W in ((0, W0), (2, W2), (4, W4)):
            out[f"Pk_{ell}_{b}"] = proj(val, W)
    return out


# ---------------------------------------------------------------------------
# Bias spectral cross-terms from M22basic (JAX-native)
# ---------------------------------------------------------------------------

def _simpson(y: Float[Array, "N"], x: Float[Array, "N"]) -> Float[Array, ""]:
    """scipy.integrate.simpson (>= 1.11) for one 1-D sample set, in JAX.

    Odd N: composite Simpson over (N-1)/2 parabolic segments. Even N: the same
    over the first N-2 intervals plus Cartwright's (2017, eq. 8) correction for
    the last interval, which is the branch scipy takes, so the two agree to
    round-off. N is static; N >= 3.
    """
    n = y.shape[0]
    if n < 3:
        raise ValueError("_simpson needs at least 3 samples")
    h = jnp.diff(x)

    def basic(stop):
        h0, h1 = h[0:stop:2], h[1:stop + 1:2]
        y0, y1, y2 = y[0:stop:2], y[1:stop + 1:2], y[2:stop + 2:2]
        hsum = h0 + h1
        return jnp.sum(hsum / 6.0 * (y0 * (2.0 - h1 / h0)
                                     + y1 * hsum ** 2 / (h0 * h1)
                                     + y2 * (2.0 - h0 / h1)))

    if n % 2 == 1:
        return basic(n - 2)
    h0, h1 = h[-2], h[-1]
    alpha = (2.0 * h1 ** 2 + 3.0 * h0 * h1) / (6.0 * (h0 + h1))
    beta = (h1 ** 2 + 3.0 * h0 * h1) / (6.0 * h0)
    eta = h1 ** 3 / (6.0 * h0 * (h0 + h1))
    return basic(n - 3) + alpha * y[-1] + beta * y[-2] - eta * y[-3]


def _pd2d2_0(pk_tree: Float[Array, "Nk"], kh: Float[Array, "Nk"]) -> Float[Array, ""]:
    """Zero-lag P_{delta^2 delta^2}(k -> 0) that pk_gg_l0 adds back.

    CLASS-PT computes it in `initialize_output` (classy.pyx:4805-4813) as
    simpson(Ptree^2 * kh^3, x=ln kh) / pi^2, where Ptree = pm[14]*h^3 is the
    IR-resummed tree (nonlinear_pt.c:2999) -- not the FFTLog-discretised
    spectrum, and not a trapezoid. Units (Mpc/h)^3.
    """
    return _simpson(pk_tree ** 2 * kh ** 3, jnp.log(kh)) / jnp.pi ** 2


def _compute_bias_spectra(
    x: Complex[Array, "Nk Nmax+1"],
    k: Float[Array, "Nk"],
    pk_disc: Float[Array, "Nk"],
    lnk: Float[Array, "Nk"],
    M22b: Complex[Array, "Nmax+1 Nmax+1"],
    IFG2: Complex[Array, "Nmax+1"],
    M13: Complex[Array, "Nmax+1"],
    M22: Complex[Array, "Nmax+1 Nmax+1"],
    etam: Complex[Array, "Nmax+1"],
    cmsym2: Complex[Array, "Nmax+1"],
    etam2: Complex[Array, "Nmax+1"],
    f: float,
    cutoff_h: float,
    cmsym_nw: Optional[Complex[Array, "Nmax+1"]] = None,
    cmsym_w: Optional[Complex[Array, "Nmax+1"]] = None,
    pk_nw: Optional[Float[Array, "Nk"]] = None,
    sigma2_bao: Optional[float] = None,
    delta_sigma2_bao: Optional[float] = None,
    pk_w: Optional[Float[Array, "Nk"]] = None,
    hratio: float | Float[Array, ""] = 1.0,
    Dratio: float | Float[Array, ""] = 1.0,
) -> dict:
    """Compute bias cross-spectra and RSD multipole components.

    Exact implementation matching CLASS-PT nonlinear_pt.c.

    Bias spectra use a second FFTLog decomposition with bias b=-1.6
    (etam2/cmsym2), matching CLASS-PT's etam2 with b2=-1.6000001.
    Modified M22basic matrices are built by elementwise multiplication
    with rational kernels in etam2 (lines 11919-11925 in nonlinear_pt.c).

    RSD 1-loop multipoles use the matter basis (etam, b=-0.3) with
    M22 × rational kernels in nu1,nu2,f (lines 6647, 6928, 7159, 7275).

    Every μ-dependent output (tree, 1-loop, counterterm, IFG2 and bias
    multipoles) is produced by the 40-node Gauss–Legendre loop in
    `_gl_multipoles`, exactly as CLASS-PT does. hratio/Dratio are CLASS-PT's
    Alcock–Paczynski ratios, forwarded to that loop; `compute_ept` does not
    expose them yet and always leaves them at (1, 1) = no remap.

    Mirrors:
      nonlinear_pt.c lines 11880–12518 (bias spectra)
      nonlinear_pt.c lines 6600–7300 (RSD multipole matrices)
    """
    uv_damp = jnp.exp(-(k / cutoff_h) ** 6)

    # ===========================================================
    # BIAS SPECTRA: use x2 basis (b=-1.6 FFTLog decomposition)
    # ===========================================================

    # Build x2: shape (Nk, Nmax+1), using b=-1.6 basis
    # x2[j,m] = cmsym2[m] × k_j^{etam2[m]}
    log_k = jnp.log(k)
    x2 = cmsym2[None, :] * jnp.exp(etam2[None, :] * log_k[:, None])

    # Kernel matrices (Nmax+1, Nmax+1) for bias spectra:
    # All kernels are symmetric in (eta_i, eta_l) because they depend only
    # on the sum s = eta_i + eta_l.
    # Ref: nonlinear_pt.c lines 11919-11925 (etam2 = bias FFTLog frequencies)
    eta_i = etam2[jnp.newaxis, :]  # (1, Nmax+1)
    eta_l = etam2[:, jnp.newaxis]  # (Nmax+1, 1)
    s = eta_i + eta_l  # sum of exponents
    # nu1, nu2 for RSD bias kernels: nu = -0.5 * eta
    # From nonlinear_pt.c: nu1 = -0.5*etam2[index_i], nu2 = -0.5*etam2[index_l]
    nu1 = -0.5 * eta_i  # (1, Nmax+1)
    nu2 = -0.5 * eta_l  # (Nmax+1, 1)

    # M_IG2G2 kernel: (3+s)(1+s) / ((-0.5η_i)(-0.5η_l)(1-0.5η_i)(1-0.5η_l))
    # From nonlinear_pt.c line 11919
    k_IG2G2 = (3 + s) * (1 + s) / (
        (-0.5 * eta_i) * (-0.5 * eta_l)
        * (1 - 0.5 * eta_i) * (1 - 0.5 * eta_l)
    )

    # M_Id2 kernel: (3+s)(4+3.5s) / (14(-0.5η_i)(-0.5η_l))
    # From nonlinear_pt.c line 11921
    k_Id2 = (3 + s) * (4 + 3.5 * s) / (14 * (-0.5 * eta_i) * (-0.5 * eta_l))

    # M_IG2 kernel: -(3+s)(1+s)(6-3.5s) / (28(1-0.5η_i)(1-0.5η_l)(-0.5η_i)(-0.5η_l))
    # From nonlinear_pt.c line 11925
    k_IG2 = -(3 + s) * (1 + s) * (6 - 3.5 * s) / (
        28 * (1 - 0.5 * eta_i) * (1 - 0.5 * eta_l)
        * (-0.5 * eta_i) * (-0.5 * eta_l)
    )

    # M_Id2G2 kernel: (3+s) / ((-0.5η_i)(-0.5η_l))
    # From nonlinear_pt.c line 11923
    k_Id2G2 = (3 + s) / ((-0.5 * eta_i) * (-0.5 * eta_l))

    # Build modified matrices (element-wise product with kernel)
    M_IG2G2 = M22b * k_IG2G2
    M_Id2   = M22b * k_Id2
    M_IG2   = M22b * k_IG2
    M_Id2G2 = M22b * k_Id2G2

    # Quadratic form helper for x2 basis: Re{ k³ x2^T M x2 } × UV_damp
    def qf2(M):
        y = x2 @ M
        return jnp.real(k ** 3 * jnp.sum(x2 * y, axis=-1)) * uv_damp

    # P_Id2d2 = |2 × k³ Re{x2^T M22b x2} − (value at kmin)| + ε
    # IR renormalization: subtract k=kmin value to remove IR divergence.
    # From nonlinear_pt.c line 11907:
    #   P_Id2d2[j] = fabs(Re{k_j³ f22[j]} - Re{k_0³ f22[0]} + epsilon_for_logs)
    # where k_0 = kdisc[0] = kmin.
    y2_base = x2 @ M22b
    raw_Id2d2 = 2.0 * jnp.real(k ** 3 * jnp.sum(x2 * y2_base, axis=-1))
    # No abs() here.  CLASS-PT's fabs() at this point belongs to its
    # add-large-constant / log-spline / subtract machinery, not to the physical
    # spectrum: the P_Id2d2 it returns is negative (P_d2d2(k->0) is the maximum,
    # so P_d2d2(k) - P_d2d2(0) <= 0).  Taking the absolute value flipped the sign
    # of the 0.25*b2^2*Pk_Id2d2 term in every galaxy spectrum, real space
    # included; the ratio to CLASS-PT was exactly -1.000 at every k.
    Pk_Id2d2 = (raw_Id2d2 - raw_Id2d2[0]) * uv_damp

    # Zero-lag value that the subtraction above removes:
    #     P_d2d2(k->0) = 2 * int P_lin(q)^2 d^3q/(2pi)^3
    #                  = (1/pi^2) * int P_lin(q)^2 q^3 dln q
    # CLASS-PT adds this back explicitly in pk_gg_l0 (classy.pyx:4911), so we
    # must carry it.  Note CLASS-PT evaluates the integral on the USER'S OUTPUT
    # GRID (classy.pyx:4813, `self.kh`), which leaves it under-converged for a
    # narrow output range -- 2253 vs 2840 (Mpc/h)^3 on a 0.01-0.3 h/Mpc grid at
    # the AbacusSummit c000 z=0.5 cosmology.  We integrate over the full EPT
    # grid instead, which is converged; the residual difference is
    # 0.25*b2^2*(2840-2253) ~ 10 (Mpc/h)^3 for |b2| ~ 0.3.
    # Exact bias spectra from modified M22basic matrices
    # From nonlinear_pt.c lines 12095-12493 (all use x2 with b=-1.6 basis)
    Pk_Id2   = qf2(M_Id2)
    Pk_IG2   = qf2(M_IG2)    # sign: negative (anti-correlated with δ on large scales)
    Pk_Id2G2 = qf2(M_Id2G2)  # sign: negative
    Pk_IG2G2 = qf2(M_IG2G2)  # sign: positive (G2² > 0)

    # P_IFG2: 13-type using IFG2 vector and x2 basis (b=-1.6)
    # From nonlinear_pt.c line 12515: f13_IFG2 = zdotu(x2, IFG2) × P_lin
    f13_IFG2 = jnp.sum(x2 * IFG2[None, :], axis=-1)
    Pk_IFG2 = jnp.real(k ** 3 * f13_IFG2 * pk_disc)  # sign: negative

    # IFG2 RSD multipoles, counterterms and every tree/1-loop multipole are
    # computed inside the 40-node GL μ-loop -- see `_gl_multipoles` below.
    # The closed forms that used to live here (P_IFG2_0 = P_IFG2 f/3,
    # Pk_ctr0 = -k² Pbin, ...) are the isotropic Legendre moments, exact only
    # while the IR damping is μ-independent; nonlinear_pt.c:4480 makes it not,
    # and :5318 additionally rescales P_IFG2 by (Pnw + Pw Exp)/Pbin.

    # ===========================================================
    # RSD 1-LOOP MULTIPOLES (from M22 × kernel and M13)
    # Kernels: ratio of RSD polynomial to F2² denominator.
    # nu1=-0.5*etam[i], nu2=-0.5*etam[l], nu12=nu1+nu2 (b=-0.3 matter basis)
    # Common denominator D = 98 nu1 nu2 nu12² - 91 nu12² + 36 nu1 nu2
    #                         - 14 nu1 nu2 nu12 + 3 nu12 + 58
    # From CLASS-PT nonlinear_pt.c lines 6647, 6928, 7054, 7159, 7275,
    #   7395, 7506, 7618, 7739 (M22 kernels) and 6820, 6960, 7083,
    #   7201, 7313, 7433, 7544, 7657 (M13 kernels).
    # ===========================================================
    nu_i  = -0.5 * etam[jnp.newaxis, :]   # (1, Nmax+1) complex
    nu_l  = -0.5 * etam[:, jnp.newaxis]   # (Nmax+1, 1) complex
    nu1   = nu_i
    nu2   = nu_l
    nu12  = nu1 + nu2

    D_matter = (98 * nu1 * nu2 * nu12 ** 2 - 91 * nu12 ** 2
                + 36 * nu1 * nu2 - 14 * nu1 * nu2 * nu12
                + 3 * nu12 + 58)
    D_inv = 196.0 / D_matter  # 196 / D  (combines with M22's own factor)

    # Shared polynomials used in multiple kernels
    N_dd = (50 - 9*nu2 + 98*nu1**3*nu2 - 35*nu2**2
            + 7*nu1**2*(-5 - 18*nu2 + 28*nu2**2)
            + nu1*(-9 - 66*nu2 - 126*nu2**2 + 98*nu2**3))
    N_vd = (36 - 8*nu2 + 70*nu1**3*nu2 - 23*nu2**2
            + nu1**2*(-23 - 94*nu2 + 140*nu2**2)
            + nu1*(-8 - 42*nu2 - 94*nu2**2 + 70*nu2**3))
    N_vv = (24 - 8*nu2 - 15*nu2**2 + 5*nu2**3
            + 5*nu1**3*(1 + 7*nu2)
            + 5*nu1**2*(-3 - 10*nu2 + 14*nu2**2)
            + nu1*(-8 - 24*nu2 - 50*nu2**2 + 35*nu2**3))

    # --- M22 monopole kernels (lines 6647, 6928, 7054) ---
    k_0_vv = D_inv * f**2 * (14*f**2*N_vv + 18*f*N_vd + 9*N_dd) / 8820.0
    M22_0_vv = M22 * k_0_vv

    N_vd3 = (6 + 3*nu2 - 10*nu2**2 + 2*nu2**3
             + 2*nu1**3*(1 + 5*nu2)
             + 2*nu1**2*(-5 - 2*nu2 + 10*nu2**2)
             + nu1*(3 - 24*nu2 - 4*nu2**2 + 10*nu2**3))
    N_vd2_0 = (18 + 11*nu2 + 42*nu1**3*nu2 - 31*nu2**2
               + nu1**2*(-31 - 22*nu2 + 84*nu2**2)
               + nu1*(11 - 74*nu2 - 22*nu2**2 + 42*nu2**3))
    # M22_0_vd: cf. nonlinear_pt.c line 6928 — note 5*N_dd2 where N_dd2 is different!
    N_dd2_0 = (46 + 13*nu2 + 98*nu1**3*nu2 - 63*nu2**2
               + 7*nu1**2*(-9 - 10*nu2 + 28*nu2**2)
               + nu1*(13 - 138*nu2 - 70*nu2**2 + 98*nu2**3))
    k_0_vd = D_inv * f * (21*f**2*N_vd3 + 14*f*N_vd2_0 + 5*N_dd2_0) / 1470.0
    M22_0_vd = M22 * k_0_vd

    # M22_0_dd: cf. nonlinear_pt.c line 7054
    N_dd2_0dd = (4 - 2*nu2 - 5*nu2**2 + nu2**3
                 + nu1**3*(1 + 3*nu2)
                 + nu1**2*(-5 + 2*nu2 + 6*nu2**2)
                 + nu1*(-2 - 4*nu2 + 2*nu2**2 + 3*nu2**3))
    N_dd3_0dd = (10 - nu2 + 14*nu1**3*nu2 - 17*nu2**2
                 + nu1**2*(-17 + 6*nu2 + 28*nu2**2)
                 + nu1*(-1 - 22*nu2 + 6*nu2**2 + 14*nu2**3))
    N_dd4_0dd = (58 + 3*nu2 + 98*nu1**3*nu2 - 91*nu2**2
                 + 7*nu1**2*(-13 - 2*nu2 + 28*nu2**2)
                 + nu1*(3 - 146*nu2 - 14*nu2**2 + 98*nu2**3))
    k_0_dd = D_inv * (98*f**2*N_dd2_0dd + 70*f*N_dd3_0dd + 15*N_dd4_0dd) / 2940.0
    M22_0_dd = M22 * k_0_dd

    # --- M22 quadrupole kernels (lines 7159, 7275, 7395) ---
    N1_2vv = (142 - 21*nu2 + 280*nu1**3*nu2 - 106*nu2**2
              + 2*nu1**2*(-53 - 174*nu2 + 280*nu2**2)
              + nu1*(-21 - 204*nu2 - 348*nu2**2 + 280*nu2**3))
    N2_2vv = (336 - 62*nu2 - 255*nu2**2 + 50*nu2**3
              + 10*nu1**3*(5 + 56*nu2)
              + 5*nu1**2*(-51 - 142*nu2 + 224*nu2**2)
              + nu1*(-62 - 486*nu2 - 710*nu2**2 + 560*nu2**3))
    k_2_vv = D_inv * f**2 * (396*N_dd + 231*f*N1_2vv + 49*f**2*N2_2vv) / 135828.0
    M22_2_vv = M22 * k_2_vv

    N_2vd3 = (22 + 11*nu2 - 40*nu2**2 + 4*nu2**3
              + nu1**3*(4 + 40*nu2)
              + 8*nu1**2*(-5 - nu2 + 10*nu2**2)
              + nu1*(11 - 88*nu2 - 8*nu2**2 + 40*nu2**3))
    N_2vd2 = (306 + 161*nu2 + 672*nu1**3*nu2 - 538*nu2**2
              + 2*nu1**2*(-269 - 134*nu2 + 672*nu2**2)
              + nu1*(161 - 1196*nu2 - 268*nu2**2 + 672*nu2**3))
    # cf. nonlinear_pt.c line 7275: 7f²*N_2vd3 + f*N_2vd2 + 4*N_dd2_2vd
    N_dd2_2vd = (46 + 13*nu2 + 98*nu1**3*nu2 - 63*nu2**2
                 + 7*nu1**2*(-9 - 10*nu2 + 28*nu2**2)
                 + nu1*(13 - 138*nu2 - 70*nu2**2 + 98*nu2**3))
    k_2_vd = D_inv * f * (7*f**2*N_2vd3 + f*N_2vd2 + 4*N_dd2_2vd) / 588.0
    M22_2_vd = M22 * k_2_vd

    # M22_2_dd: cf. nonlinear_pt.c line 7395
    N_2dd_a = (10 - nu2 + 14*nu1**3*nu2 - 17*nu2**2
               + nu1**2*(-17 + 6*nu2 + 28*nu2**2)
               + nu1*(-1 - 22*nu2 + 6*nu2**2 + 14*nu2**3))
    N_2dd_b = (26 - 13*nu2 - 37*nu2**2 + 2*nu2**3
               + nu1**3*(2 + 24*nu2)
               + nu1**2*(-37 + 22*nu2 + 48*nu2**2)
               + nu1*(-13 - 26*nu2 + 22*nu2**2 + 24*nu2**3))
    k_2_dd = D_inv * f * (4*N_2dd_a + f*N_2dd_b) / 84.0
    M22_2_dd = M22 * k_2_dd

    # --- M22 hexadecapole kernels (lines 7506, 7618, 7739) ---
    N1_4vv = (50 + 98*nu1**3*nu2 - nu2*(9 + 35*nu2)
              + 7*nu1**2*(-5 + 2*nu2*(-9 + 14*nu2))
              + nu1*(-9 + 2*nu2*(-33 + 7*nu2*(-9 + 7*nu2))))
    N2_4vv = (206 + 420*nu1**3*nu2 + nu2*(7 - 208*nu2)
              + 8*nu1**2*(-26 + nu2*(-53 + 105*nu2))
              + nu1*(7 + 4*nu2*(-108 + nu2*(-106 + 105*nu2))))
    N3_4vv = (483 + 40*nu1**3*(-1 + 28*nu2) - 2*nu2*(-57 + 10*nu2*(29 + 2*nu2))
              + 20*nu1**2*(-29 + 2*nu2*(-25 + 56*nu2))
              + 2*nu1*(57 + 2*nu2*(-327 + 10*nu2*(-25 + 28*nu2))))
    k_4_vv = D_inv * f**2 * (1144*N1_4vv + 728*f*N2_4vv + 147*f**2*N3_4vv) / 980980.0
    M22_4_vv = M22 * k_4_vv

    N_4vd_a = (58 + 21*nu2 + 112*nu1**3*nu2 - 106*nu2**2
               + 2*nu1**2*(-53 - 6*nu2 + 112*nu2**2)
               + nu1*(21 - 204*nu2 - 12*nu2**2 + 112*nu2**3))
    N_4vd_b = (26 + 13*nu2 - 60*nu2**2 - 8*nu2**3
               + nu1**3*(-8 + 60*nu2)
               + 4*nu1**2*(-15 + 4*nu2 + 30*nu2**2)
               + nu1*(13 - 104*nu2 + 16*nu2**2 + 60*nu2**3))
    k_4_vd = D_inv * f**2 * (11*N_4vd_a + 14*f*N_4vd_b) / 2695.0
    M22_4_vd = M22 * k_4_vd

    # M22_4_dd: cf. nonlinear_pt.c line 7739 — (2nu1-1)(2nu2-1)(1+nu12)(2+nu12)*f²/35
    k_4_dd = D_inv * f**2 * (2*nu1 - 1) * (2*nu2 - 1) * (1 + nu12) * (2 + nu12) / 35.0
    M22_4_dd = M22 * k_4_dd

    # IR resummation setup for RSD 1-loop terms.
    # CLASS-PT (nonlinear_pt.c lines 8215, 8246, 8562, 8586) uses:
    #   x_nw = cmsym_nw × k^etam  (FFTLog of no-wiggle Pnw)
    #   x_w  = cmsym_w  × k^etam  (FFTLog of wiggle Pw, no factor 2 for P13)
    #   x_w2 = 2 × cmsym_w × k^etam  (factor-2 version for P22 cross-term)
    # P22_IR = P22(x_nw) + Re{k³ x_nw^T M22 x_w2} × Exp
    # P13_IR = (Re{k³ f13_nw × Pbin} + UV × k² × Pbin) × exp_cut  [= P13_nw × P13ratio]
    #        + Re{k³ f13_w × Pnw} × exp_cut × Exp  [wiggle correction, no UV]
    use_ir_rsd = (cmsym_nw is not None)
    log_k = jnp.log(k)
    if use_ir_rsd:
        x_nw = cmsym_nw[None, :] * jnp.exp(etam[None, :] * log_k[:, None])
        x_w  = cmsym_w[None,  :] * jnp.exp(etam[None, :] * log_k[:, None])
        x_w2 = 2.0 * x_w
        Exp  = jnp.exp(-sigma2_bao * k ** 2)
        pk_nw_arr = jnp.asarray(pk_nw)
    else:
        # No IR resummation: use x from pk_resummed (original behaviour)
        x_nw = x
        x_w  = jnp.zeros_like(x)
        x_w2 = x_w
        Exp  = jnp.ones_like(k)
        pk_nw_arr = pk_disc

    def qf_rsd(M):
        """P22_IR = P22(x_nw) + cross-term(x_nw, x_w2) × Exp."""
        y_nw = x_nw @ M
        p22_nw = jnp.real(k ** 3 * jnp.sum(x_nw * y_nw, axis=-1)) * uv_damp
        if use_ir_rsd:
            y_w = x_w2 @ M
            p22_w = jnp.real(k ** 3 * jnp.sum(x_nw * y_w, axis=-1)) * uv_damp
            return p22_nw + p22_w * Exp
        return p22_nw

    P22_0_vv = qf_rsd(M22_0_vv)
    P22_0_vd = qf_rsd(M22_0_vd)
    P22_0_dd = qf_rsd(M22_0_dd)
    P22_2_vv = qf_rsd(M22_2_vv)
    P22_2_vd = qf_rsd(M22_2_vd)
    P22_2_dd = qf_rsd(M22_2_dd)
    P22_4_vv = qf_rsd(M22_4_vv)
    P22_4_vd = qf_rsd(M22_4_vd)
    P22_4_dd = qf_rsd(M22_4_dd)

    # M13 RSD kernels — applied to the standard M13 vector element-wise.
    # nu1_m = -0.5 * etam (1D, shape (Nmax+1,))
    # cf. nonlinear_pt.c lines 6820, 6960, 7083, 7201, 7313, 7433, 7544, 7657
    nu1_m = -0.5 * etam  # (Nmax+1,) complex

    M13_0_vv = M13 * (112.0 / (1 + 9*nu1_m)
                      * 3*f**2*(7*(-5 + 3*nu1_m) + 6*f*(-7 + 5*nu1_m)) / 3920.0)
    M13_0_vd = M13 * (112.0 / (1 + 9*nu1_m)
                      * f*(-35 - 18*f + 45*nu1_m + 54*f*nu1_m) / 840.0)
    M13_0_dd = M13 / (1 + 9*nu1_m) * (1 + 9*nu1_m + 6*f*(1 + nu1_m))

    M13_2_vv = M13 * (112.0 / (1 + 9*nu1_m)
                      * 3*f**2*(-5 + 3*nu1_m + f*(-6 + 5*nu1_m)) / 196.0)
    M13_2_vd = M13 * (112.0 / (1 + 9*nu1_m)
                      * f*(-49 - 9*f + 63*nu1_m + 108*f*nu1_m) / 588.0)
    M13_2_dd = M13 * (112.0 / (1 + 9*nu1_m) * 3*f*(1 + nu1_m) / 28.0)

    M13_4_vv = M13 * (112.0 / (1 + 9*nu1_m)
                      * 3*f**2*(-55 + 33*nu1_m + f*(-66 + 90*nu1_m)) / 5390.0)
    M13_4_vd = M13 * (112.0 / (1 + 9*nu1_m) * 9*f**2*(1 + 2*nu1_m) / 245.0)

    sigma2_v = jnp.trapezoid(pk_disc * k, lnk) / (6.0 * jnp.pi**2)

    # UV counterterm coefficients (from CLASS-PT lines 6832, 6980, 7101, 7211, 7323, 7443, 7554, 7667)
    UV_0_vv = -sigma2_v * f**2 * (441 + 566*f + 175*f**2) / 1225.0
    UV_0_vd = -sigma2_v * 2*f * (625 + 558*f + 315*f**2) / 1575.0
    UV_0_dd = -sigma2_v * (61 - 2*f + 35*f**2) / 105.0
    UV_2_vv = -sigma2_v * 2*f**2 * (54 + 74*f + 25*f**2) / 105.0
    UV_2_vd = -sigma2_v * 4*f * (175 + 180*f + 126*f**2) / 441.0
    UV_2_dd = -sigma2_v * 2*f * (35*f - 2) / 105.0
    UV_4_vv = -sigma2_v * 24*f**2 * (33 + 58*f + 25*f**2) / 1925.0
    UV_4_vd = -sigma2_v * 16*f**2 * (22 + 35*f) / 1225.0

    def p13_rsd(M13_kernel, sigma2_UV_coeff):
        """P13 with proper IR resummation.

        IR path (use_ir_rsd=True):
          P13_nw × P13ratio = (Re{k³ f13_nw Pbin} + UV k² Pbin) × exp_cut
          P13_w correction  = Re{k³ f13_w Pnw} × exp_cut × Exp  (no UV)
        Non-IR path:
          P13 = (Re{k³ f13 Pbin} + UV k² Pbin) × exp_cut  (same as before)
        """
        f13_nw_ch = jnp.sum(x_nw * M13_kernel[None, :], axis=-1)
        p13_main = (jnp.real(k**3 * f13_nw_ch * pk_disc) + sigma2_UV_coeff * k**2 * pk_disc) * uv_damp
        if use_ir_rsd:
            f13_w_ch = jnp.sum(x_w * M13_kernel[None, :], axis=-1)
            p13_w = jnp.real(k**3 * f13_w_ch * pk_nw_arr) * uv_damp
            return p13_main + p13_w * Exp
        return p13_main

    P13_0_vv = p13_rsd(M13_0_vv, UV_0_vv)
    P13_0_vd = p13_rsd(M13_0_vd, UV_0_vd)
    P13_0_dd = p13_rsd(M13_0_dd, UV_0_dd)
    P13_2_vv = p13_rsd(M13_2_vv, UV_2_vv)
    P13_2_vd = p13_rsd(M13_2_vd, UV_2_vd)
    P13_2_dd = p13_rsd(M13_2_dd, UV_2_dd)
    P13_4_vv = p13_rsd(M13_4_vv, UV_4_vv)
    P13_4_vd = p13_rsd(M13_4_vd, UV_4_vd)

    # --- Higher-order mu^6/mu^8 bare kernels (nonlinear_pt.c lines 8069-8159) ---
    # These give bare mu-power coefficients for Gauss quadrature integration.
    N_mu6_vv = (7*f*(1 + 4*nu1**3 + nu1**2*(2 - 12*nu2) + 2*nu2 + 2*nu2**2 + 4*nu2**3
                      - 2*nu1*(-1 + 4*nu2 + 6*nu2**2))
                 + 2*(26 + 9*nu2 + 56*nu1**3*nu2 - 38*nu2**2
                      + 2*nu1**2*(-19 - 18*nu2 + 56*nu2**2)
                      + nu1*(9 - 84*nu2 - 36*nu2**2 + 56*nu2**3)))
    k_mu6_vv = D_inv * f**3 * N_mu6_vv / 112.0
    M22_mu6_vv_mat = M22 * k_mu6_vv

    k_mu6_vd = (D_inv * f**3
                * (2*nu1 - 1)*(2*nu2 - 1)
                * (2 + 2*nu1**2 + 5*nu2 + 2*nu2**2 + nu1*(5 + 4*nu2)) / 8.0)
    M22_mu6_vd_mat = M22 * k_mu6_vd

    k_mu8 = (D_inv * f**4
             * (2*nu1 - 1)*(2*nu2 - 1)
             * (3 + 4*nu1**2 + 8*nu2 + 4*nu2**2 + 8*nu1*(1 + nu2)) / 32.0)
    M22_mu8_mat = M22 * k_mu8

    # M13_mu6: nonlinear_pt.c line 8159 — M13_mu6 = M13 * 18*f³*nu1 / (1+9*nu1)
    M13_mu6_mat = M13 * (112.0 / (1 + 9*nu1_m)) * (9.0 * 2.0 * f**3 * nu1_m / 112.0)
    # UV counterterm for mu^6 P13: nonlinear_pt.c line 8226
    UV_mu6 = -sigma2_v * f**3 * (46.0 + 35.0 * f) / 35.0

    # Bare mu-power M13 kernels (CLASS-PT nonlinear_pt.c lines 8155-8158)
    M13_mu2_dd_bare = M13 * (2.0 / (1 + 9*nu1_m)) * 9*f*(1+nu1_m)
    M13_mu2_vd_bare = M13 * (2.0 / (1 + 9*nu1_m)) * (-f*(7+9*f-9*nu1_m))
    M13_mu4_vv_bare = M13 * (1.0 / (1 + 9*nu1_m)) * (-3*f**2*(5+6*f-3*nu1_m))
    M13_mu4_vd_bare = M13 * (2.0 / (1 + 9*nu1_m)) * (9*f**2*(1+2*nu1_m))

    # UV counterterms for bare channels (CLASS-PT lines 8219-8225)
    UV_mu0_dd = -sigma2_v * 61.0 / 105.0
    UV_mu2_dd = -sigma2_v * f * (105*f - 6) / 105.0
    UV_mu2_vd = -sigma2_v * f * (250 + 144*f) / 105.0
    UV_mu4_vv = -sigma2_v * f**2 * (63 + 48*f) / 35.0
    UV_mu4_vd = -sigma2_v * f**2 * (44 + 70*f) / 35.0

    # Bare mu-power M22 kernels (CLASS-PT lines 8059-8067)
    k_mu2_vd_bare = D_inv * (-f) * (
        7*f*(-1+2*nu1)*(-1+2*nu2)*(6+7*nu12)
        - 4*(46 + 13*nu2 + 98*nu1**3*nu2 - 63*nu2**2
             + 7*nu1**2*(-9-10*nu2+28*nu2**2)
             + nu1*(13-138*nu2-70*nu2**2+98*nu2**3))
    ) / 392.0
    k_mu2_dd_bare = D_inv * f * (
        7*f*(2+2*nu1**3-nu2-nu2**2+2*nu2**3-nu1**2*(1+2*nu2)-nu1*(1+2*nu2+2*nu2**2))
        + 4*(10-nu2+14*nu1**3*nu2-17*nu2**2
             +nu1**2*(-17+6*nu2+28*nu2**2)
             +nu1*(-1-22*nu2+6*nu2**2+14*nu2**3))
    ) / 56.0
    k_mu4_vv_bare = D_inv * f**2 * (
        147*f**2*(-1+2*nu1)*(-1+2*nu2)
        - 28*f*(-1+2*nu1)*(-1+2*nu2)*(-2+7*nu12)
        + 8*(50-9*nu2+98*nu1**3*nu2-35*nu2**2
             +7*nu1**2*(-5-18*nu2+28*nu2**2)
             +nu1*(-9-66*nu2-126*nu2**2+98*nu2**3))
    ) / 1568.0
    k_mu4_vd_bare = D_inv * f**2 * (
        58+21*nu2+112*nu1**3*nu2-106*nu2**2
        + 2*nu1**2*(-53-6*nu2+112*nu2**2)
        + 7*f*(2+nu1+4*nu1**3+nu2-8*nu1*nu2-8*nu1**2*nu2-8*nu1*nu2**2+4*nu2**3)
        + nu1*(21-204*nu2-12*nu2**2+112*nu2**3)
    ) / 56.0
    k_mu4_dd_bare = D_inv * f**2 * (2*nu1-1)*(2*nu2-1)*(2+nu1**2+3*nu2+nu2**2+nu1*(3+2*nu2)) / 8.0

    M22_mu2_vd_bare = M22 * k_mu2_vd_bare
    M22_mu2_dd_bare = M22 * k_mu2_dd_bare
    M22_mu4_vv_bare = M22 * k_mu4_vv_bare
    M22_mu4_vd_bare = M22 * k_mu4_vd_bare
    M22_mu4_dd_bare = M22 * k_mu4_dd_bare

    def qf_split(M):
        """Returns (P22_nw, P22_w) separately for anisotropic GL integration."""
        y_nw = x_nw @ M
        p22_nw = jnp.real(k**3 * jnp.sum(x_nw * y_nw, axis=-1)) * uv_damp
        if use_ir_rsd:
            y_w2 = x_w2 @ M
            p22_w = jnp.real(k**3 * jnp.sum(x_nw * y_w2, axis=-1)) * uv_damp
            return p22_nw, p22_w
        return p22_nw, jnp.zeros_like(p22_nw)

    def p13_split(M13_kernel, UV_coeff):
        """Returns (P13_nw, P13_w) for anisotropic GL integration."""
        f13_nw = jnp.sum(x_nw * M13_kernel[None, :], axis=-1)
        p13_nw = (jnp.real(k**3 * f13_nw * pk_nw_arr) + UV_coeff * k**2 * pk_nw_arr) * uv_damp
        if use_ir_rsd:
            f13_w = jnp.sum(x_w * M13_kernel[None, :], axis=-1)
            p13_w = jnp.real(k**3 * f13_w * pk_nw_arr) * uv_damp
            return p13_nw, p13_w
        return p13_nw, jnp.zeros_like(p13_nw)

    # nw/w split of every μ-power channel, keyed for _gl_multipoles
    split = {}
    for name, M in (("P22_mu0_dd", M22), ("P22_mu2_vd", M22_mu2_vd_bare), ("P22_mu2_dd", M22_mu2_dd_bare),
                    ("P22_mu4_vv", M22_mu4_vv_bare), ("P22_mu4_vd", M22_mu4_vd_bare), ("P22_mu4_dd", M22_mu4_dd_bare),
                    ("P22_mu6_vv", M22_mu6_vv_mat), ("P22_mu6_vd", M22_mu6_vd_mat), ("P22_mu8", M22_mu8_mat)):
        split[name + "_nw"], split[name + "_w"] = qf_split(M)
    for name, M, UV in (("P13_mu0_dd", M13, UV_mu0_dd), ("P13_mu2_vd", M13_mu2_vd_bare, UV_mu2_vd),
                        ("P13_mu2_dd", M13_mu2_dd_bare, UV_mu2_dd), ("P13_mu4_vv", M13_mu4_vv_bare, UV_mu4_vv),
                        ("P13_mu4_vd", M13_mu4_vd_bare, UV_mu4_vd), ("P13_mu6", M13_mu6_mat, UV_mu6)):
        split[name + "_nw"], split[name + "_w"] = p13_split(M, UV)

    # EPTComponents still needs these combined (for _pk_mm_tree_mu68_at_mu).
    # These four keep the ISOTROPIC Exp (`Exp` above): they are consumed by the
    # μ-resolved matter accessor, not by the GL loop.
    P22_mu6_vv = split["P22_mu6_vv_nw"] + split["P22_mu6_vv_w"] * Exp
    P22_mu6_vd = split["P22_mu6_vd_nw"] + split["P22_mu6_vd_w"] * Exp
    P22_mu8    = split["P22_mu8_nw"] + split["P22_mu8_w"] * Exp
    P13_mu6    = split["P13_mu6_nw"] + split["P13_mu6_w"] * Exp

    _pk_w_for_ratio = (jnp.asarray(pk_w) if pk_w is not None else jnp.zeros_like(k)) if use_ir_rsd else jnp.zeros_like(k)
    _delta_sig2 = delta_sigma2_bao if delta_sigma2_bao is not None else 0.0
    _sig2_bao = sigma2_bao if sigma2_bao is not None else 0.0

    # ===========================================================
    # RSD BIAS CROSS-TERMS
    # All built from M22basic × rational kernel in nu1, nu2, f.
    # nu1 = -0.5*etam2[i], nu2 = -0.5*etam2[l]  (b=-1.6 bias basis)
    # Exact kernels from nonlinear_pt.c lines 12871–13339.
    # ===========================================================

    # These kernels require the b = -1.6 BIAS FFTLog basis: they are contracted
    # with x2, which is the b = -1.6 decomposition.  nu1/nu2 were bound to that
    # basis near the top of this function, but the RSD 1-loop multipole block
    # above rebinds them to the b = -0.3 MATTER basis (etam).  Rebind them here.
    #
    # The real-space bias channels (Pk_Id2, Pk_IG2, Pk_Id2G2, Pk_IG2G2) are
    # computed before that rebinding, which is why only the RSD bias channels
    # were affected -- and why comparisons against a reference with b2 = bG2 = 0
    # could never see it.
    nu1 = -0.5 * eta_i
    nu2 = -0.5 * eta_l

    # Monopole bias kernels (ℓ=0)
    # M22_0_b1b2: (-3+2ν1+2ν2)(-12+7(3+f)(ν1+ν2)) / (42 ν1 ν2)
    # Ref: nonlinear_pt.c line 12871
    k_0_b1b2 = (
        (-3.0 + 2.0*nu1 + 2.0*nu2)
        * (-12.0 + 7.0*(3.0 + f)*(nu1 + nu2))
        / (42.0 * nu1 * nu2)
    )
    M_0_b1b2 = M22b * k_0_b1b2

    # M22_0_b2: [7f²(12+6ν1²-17ν2+6ν2²+ν1(-17+12ν2)) + 5f(24+14ν1²-37ν2+14ν2²+ν1(-37+28ν2))]
    #           / (210 ν1 ν2)
    # Ref: nonlinear_pt.c line 12927
    k_0_b2 = (
        7.0*f**2 * (12.0 + 6.0*nu1**2 - 17.0*nu2 + 6.0*nu2**2 + nu1*(-17.0 + 12.0*nu2))
        + 5.0*f  * (24.0 + 14.0*nu1**2 - 37.0*nu2 + 14.0*nu2**2 + nu1*(-37.0 + 28.0*nu2))
    ) / (210.0 * nu1 * nu2)
    M_0_b2 = M22b * k_0_b2

    # M22_0_b1bG2: (-3+2ν1+2ν2)(-1+2ν1+2ν2)(7f(2+ν1+ν2)+3(6+7ν1+7ν2))
    #              / (42 ν1(1+ν1) ν2(1+ν2))
    # Ref: nonlinear_pt.c line 12961
    k_0_b1bG2 = (
        (-3.0 + 2.0*nu1 + 2.0*nu2)
        * (-1.0 + 2.0*nu1 + 2.0*nu2)
        * (7.0*f*(2.0 + nu1 + nu2) + 3.0*(6.0 + 7.0*nu1 + 7.0*nu2))
        / (42.0 * nu1 * (1.0 + nu1) * nu2 * (1.0 + nu2))
    )
    M_0_b1bG2 = M22b * k_0_b1bG2

    # M22_0_bG2: (-3+2ν1+2ν2)(-1+2ν1+2ν2)(-10f+7f(5(ν1+ν2)+f(-2+3ν1+3ν2)))
    #            / (210 ν1(1+ν1) ν2(1+ν2))
    # Ref: nonlinear_pt.c line 12995
    k_0_bG2 = (
        (-3.0 + 2.0*nu1 + 2.0*nu2)
        * (-1.0 + 2.0*nu1 + 2.0*nu2)
        * (-10.0*f + 7.0*f*(5.0*(nu1+nu2) + f*(-2.0 + 3.0*nu1 + 3.0*nu2)))
        / (210.0 * nu1 * (1.0 + nu1) * nu2 * (1.0 + nu2))
    )
    M_0_bG2 = M22b * k_0_bG2

    # Quadrupole bias kernels (ℓ=2)
    # M22_2_b1b2: (-3+2ν1+2ν2) f (ν1+ν2) / (3 ν1 ν2)
    # Ref: nonlinear_pt.c line 13173
    k_2_b1b2 = (
        (-3.0 + 2.0*nu1 + 2.0*nu2) * f * (nu1 + nu2)
        / (3.0 * nu1 * nu2)
    )
    M_2_b1b2 = M22b * k_2_b1b2

    # M22_2_b2: (-3+2ν1+2ν2) f (-16+14(ν1+ν2)+f(-13+12(ν1+ν2))) / (42 ν1 ν2)
    # Ref: nonlinear_pt.c line 13207
    k_2_b2 = (
        (-3.0 + 2.0*nu1 + 2.0*nu2)
        * f * (-16.0 + 14.0*(nu1+nu2) + f*(-13.0 + 12.0*(nu1+nu2)))
        / (42.0 * nu1 * nu2)
    )
    M_2_b2 = M22b * k_2_b2

    # M22_2_b1bG2: (-3+2ν1+2ν2)(-1+2ν1+2ν2) f (2+ν1+ν2)
    #              / (3 ν1(1+ν1) ν2(1+ν2))
    # Ref: nonlinear_pt.c line 13241
    k_2_b1bG2 = (
        (-3.0 + 2.0*nu1 + 2.0*nu2)
        * (-1.0 + 2.0*nu1 + 2.0*nu2)
        * f * (2.0 + nu1 + nu2)
        / (3.0 * nu1 * (1.0 + nu1) * nu2 * (1.0 + nu2))
    )
    M_2_b1bG2 = M22b * k_2_b1bG2

    # M22_2_bG2: (-3+2ν1+2ν2)(-1+2ν1+2ν2) f (-2-f+7(ν1+ν2)+6f(ν1+ν2))
    #            / (21 ν1(1+ν1) ν2(1+ν2))
    # Ref: nonlinear_pt.c line 13271
    k_2_bG2 = (
        (-3.0 + 2.0*nu1 + 2.0*nu2)
        * (-1.0 + 2.0*nu1 + 2.0*nu2)
        * f * (-2.0 - f + 7.0*(nu1+nu2) + 6.0*f*(nu1+nu2))
        / (21.0 * nu1 * (1.0 + nu1) * nu2 * (1.0 + nu2))
    )
    M_2_bG2 = M22b * k_2_bG2

    # Hexadecapole bias kernels (ℓ=4)
    # M22_4_b2: (-3+2ν1+2ν2)(-1+2ν1+2ν2) 2f² / (35 ν1 ν2)
    # Ref: nonlinear_pt.c line 13305
    k_4_b2 = (
        (-3.0 + 2.0*nu1 + 2.0*nu2)
        * (-1.0 + 2.0*nu1 + 2.0*nu2)
        * 2.0 * f**2
        / (35.0 * nu1 * nu2)
    )
    M_4_b2 = M22b * k_4_b2

    # M22_4_bG2: (-3+2ν1+2ν2)(-1+2ν1+2ν2) 4f²(1+ν1+ν2)
    #            / (35 ν1(1+ν1) ν2(1+ν2))
    # Ref: nonlinear_pt.c line 13339
    k_4_bG2 = (
        (-3.0 + 2.0*nu1 + 2.0*nu2)
        * (-1.0 + 2.0*nu1 + 2.0*nu2)
        * 4.0 * f**2 * (1.0 + nu1 + nu2)
        / (35.0 * nu1 * (1.0 + nu1) * nu2 * (1.0 + nu2))
    )
    M_4_bG2 = M22b * k_4_bG2

    # Compute RSD bias spectra via quadratic form.
    # CLASS-PT negates the bG2 RSD channels when it unpacks them --
    # classy.pyx:4721-4722, 4727-4728, 4732, indices 32,33 (l=0 b1bG2,bG2),
    # 36,37 (l=2), 39 (l=4):
    #     pk_mult[32] = -raw_pk[32] + large_b   etc.
    # Both codes then use "+ b1*bG2*<channel> + bG2*<channel>" in pk_gg_l*, so we
    # must carry the same sign.  Index 41 (Pk_4_b1bG2) is NOT negated upstream;
    # see the flip after the _gl_multipoles call below.
    Pk_0_b1b2  = qf2(M_0_b1b2)
    Pk_0_b2    = qf2(M_0_b2)
    Pk_0_b1bG2 = -qf2(M_0_b1bG2)
    Pk_0_bG2   = -qf2(M_0_bG2)
    Pk_2_b1b2  = qf2(M_2_b1b2)
    Pk_2_b2    = qf2(M_2_b2)
    Pk_2_b1bG2 = -qf2(M_2_b1bG2)
    Pk_2_bG2   = -qf2(M_2_bG2)
    Pk_4_b2    = qf2(M_4_b2)
    Pk_4_bG2   = -qf2(M_4_bG2)

    # The 47 channels the GL μ-loop consumes; _gl_multipoles remaps every one of
    # them to ktrue before projecting (nonlinear_pt.c:4403-4440, 5279-5314).
    chan = {
        **split,
        "pk_nw": pk_nw_arr, "pk_w": _pk_w_for_ratio, "pk_disc": pk_disc,
        "Pk_IFG2": Pk_IFG2, "Pk_Id2d2": Pk_Id2d2, "Pk_Id2G2": Pk_Id2G2, "Pk_IG2G2": Pk_IG2G2,
        "Pk_0_b1b2": Pk_0_b1b2, "Pk_2_b1b2": Pk_2_b1b2, "Pk_0_b1bG2": Pk_0_b1bG2, "Pk_2_b1bG2": Pk_2_b1bG2,
        "Pk_0_b2": Pk_0_b2, "Pk_2_b2": Pk_2_b2, "Pk_4_b2": Pk_4_b2,
        "Pk_0_bG2": Pk_0_bG2, "Pk_2_bG2": Pk_2_bG2, "Pk_4_bG2": Pk_4_bG2,
    }
    gl = _gl_multipoles(chan, k, f, _sig2_bao, _delta_sig2,
                        hratio=hratio, Dratio=Dratio)
    # Sign bookkeeping. `_gl_multipoles` is linear in `chan`, so the six leaves
    # clax stores negated relative to the raw C arrays -- Pk_Id2d2 (pm[1]),
    # Pk_0_b1bG2/Pk_0_bG2 (pm[32,33]), Pk_2_b1bG2/Pk_2_bG2 (pm[36,37]),
    # Pk_4_bG2 (pm[39]), all negated by classy.pyx:4676/4721-4722/4727-4728/4732
    # -- come back out already in clax's convention. The single exception is
    # Pk_4_b1bG2: the C loop builds it (nonlinear_pt.c:5326, 5351) from the raw
    # P_0_b1bG2 and P_2_b1bG2, and classy.pyx:4735 does NOT negate row 41, so
    # feeding it the two negated parents flips it. Flip it back.
    gl["Pk_4_b1bG2"] = -gl["Pk_4_b1bG2"]
    return {
        "Pk_Id2": Pk_Id2, "Pk_IG2": Pk_IG2, "Pk_IFG2": Pk_IFG2,
        "P22_mu6_vv": P22_mu6_vv, "P22_mu6_vd": P22_mu6_vd,
        "P22_mu8": P22_mu8, "P13_mu6": P13_mu6,
        **gl,
    }


# ---------------------------------------------------------------------------
# Main EPT computation
# ---------------------------------------------------------------------------

def compute_ept(
    pk_lin_h: Float[Array, "Nk"],
    k_h: Float[Array, "Nk"],
    h: float | Float[Array, ""],
    f: float | Float[Array, ""],
    prec: EPTPrecisionParams = EPTPrecisionParams(),
    _ir_precomputed: Optional[tuple] = None,
    rs_h: float = 99.0,
    hratio: float | Float[Array, ""] = 1.0,
    Dratio: float | Float[Array, ""] = 1.0,
) -> EPTComponents:
    """Compute all EPT spectral components from linear P(k).

    This is the main entry point for the CLASS-PT algorithm in JAX.
    Differentiable w.r.t. pk_lin_h (for autodiff through P(k)).

    Args:
        pk_lin_h: linear P(k) in (Mpc/h)³ on the EPT k-grid, shape (Nmax,)
                  Must be evaluated at k_h = EPT k-grid (prec.kmin_h to prec.kmax_h)
        k_h:      EPT k-grid [h/Mpc], shape (Nmax,)
                  Should match EPT_kgrid(prec) for best accuracy
        h:        Hubble parameter h = H₀/100 (float or scalar jnp array;
                  may be traced -- stored as an EPTComponents pytree leaf)
        f:        growth rate f = d ln D / d ln a at the target redshift
                  (float or scalar jnp array; may be traced -- stored as an
                  EPTComponents pytree leaf)
        prec:     EPT precision parameters (static)
        _ir_precomputed: optional tuple (pk_nw, pk_w, sigma2_bao[, delta_sigma2_bao])
                  precomputed outside this function's own IR-resummation branch.
                  Two kinds of caller supply this tuple, and both are supported:
                  (1) ``compute_ept_from_clax`` now builds it from the TRACED
                  ``_ir_resummation_jax`` splitter, so pk_nw/pk_w/sigma2_bao are
                  jnp arrays and d(pk_nw)/d(params), d(sigma2_bao)/d(params) both
                  flow through; (2) a caller may still pass the legacy tuple from
                  ``_ir_resummation_numpy()`` (concrete numpy, computed outside
                  the JAX trace) -- see the example below. Either way, only
                  pk_nw and sigma2_bao are used as-is; the wiggle component is
                  recomputed as pk_w = pk_lin_h - pk_nw (JAX-traced through
                  pk_lin_h regardless of which kind of pk_nw was supplied), so
                  that gradients flow:
                    pk_resummed = pk_nw + (pk_lin_h - pk_nw) * exp(-Σ²k²)
                                = pk_lin_h × exp(-Σ²k²) + pk_nw × (1 - exp(-Σ²k²))
                  Use this to enable jax.grad() through the IR resummation path:

                      pk_nw_np, pk_w_np, sigma2, delta_sigma2 = _ir_resummation_numpy(pk_lin_np, k_np)
                      def f(pk_lin):
                          return compute_ept(pk_lin, k_ept, h=h, f=f,
                                             _ir_precomputed=(pk_nw_np, pk_w_np, sigma2, delta_sigma2))
                      grad = jax.grad(f)(pk_lin_ept)  # works!

                  Physically-correct-freeze justification (applies ONLY to the
                  concrete-numpy caller, kind (2) above): the no-wiggle template
                  pk_nw is a property of the broadband shape at fixed cosmology,
                  so for a perturbation δpk around the fiducial, the wiggle
                  component changes as δpk_w = δpk_lin, and the resummed
                  spectrum damps it by exp(-Σ²k²) -- this is what justifies
                  freezing pk_nw when the caller never re-derives it under AD.
                  For the traced caller (kind (1), ``compute_ept_from_clax``),
                  no such freeze is needed or applied: d(pk_nw)/d(params) flows
                  from ``_ir_resummation_jax`` itself.
        rs_h:     Sound horizon at drag epoch times h, in Mpc (i.e. r_s(z_d) * h,
                  NOT r_s/h).  Default 99.0 ≈ Planck 2018 fiducial; for
                  cosmology-varying values, prefer ``compute_ept_from_clax`` which
                  computes ``sound_horizon_drag(params) * params.h`` automatically.
                  Used by IR resummation in the Σ²_BAO j₂-filter integral.
                  Ignored if ``_ir_precomputed`` is provided (the caller already
                  computed Σ²_BAO) or ``ir_resummation=False``.

    No Alcock–Paczynski remap is reachable from here: the μ-loop this calls
    (`_gl_multipoles`) carries CLASS-PT's `hratio`/`Dratio` because its
    structure needs them, but they are left at their (1, 1) defaults, which is
    exactly CLASS-PT's alpha=1 branch (nonlinear_pt.c:4396-4397). Threading a
    real geometry through is a separate change (it also needs `omfid` and the
    fiducial background) and is deliberately not part of this API yet.

    Returns:
        EPTComponents with all spectral arrays on the EPT k-grid

    Example:
        >>> prec = EPTPrecisionParams()
        >>> k_h = ept_kgrid(prec)
        >>> pk_h = my_pk_func(k_h * h) * h**3  # convert to h-units
        >>> ept = compute_ept(pk_h, k_h, h=0.67, f=0.78)
        >>> P_mm = pk_mm_real(ept, cs0=10.0)  # in (Mpc/h)^3
    """
    nmax     = prec.nmax
    kmin     = prec.kmin_h
    kmax     = prec.kmax_h
    b        = prec.b_matter
    cutoff_h = prec.cutoff_h

    # Load matrices (cached, numpy arrays)
    mats = _load_matrices(nmax)
    M13  = jnp.array(mats["M13"])
    M22  = jnp.array(mats["M22"])
    M22b = jnp.array(mats["M22b"])
    IFG2 = jnp.array(mats["IFG2"])

    lnk = jnp.log(k_h)

    # --- IR resummation ---
    if _ir_precomputed is not None:
        # Use caller-supplied precomputed decomposition (enables jax.grad).
        # pk_nw_np may be TRACED jnp (compute_ept_from_clax's _ir_resummation_jax
        # tuple, gradients flow through it too) or concrete numpy (the legacy
        # _ir_resummation_numpy() tuple, effectively fixed for this branch); pk_w
        # is always derived from pk_lin_h so that gradients flow through
        # pk_lin_h → pk_w → pk_resummed → P13/P22 either way.
        #
        # KEY: pk_w = pk_lin_h - pk_nw  (JAX-traced, depends on pk_lin_h)
        # pk_resummed = pk_nw + pk_w × exp(-Σ²k²)
        #             = pk_nw × (1-exp(-Σ²k²)) + pk_lin_h × exp(-Σ²k²)
        # This is linear in pk_lin_h, so d(pk_resummed)/d(pk_lin_h) = exp(-Σ²k²) ≠ 0.
        _ir_pre = _ir_precomputed
        if len(_ir_pre) == 3:
            pk_nw_np, _pk_w_np_unused, sigma2_bao = _ir_pre
            delta_sigma2_bao = 0.0
        else:
            pk_nw_np, _pk_w_np_unused, sigma2_bao, delta_sigma2_bao = _ir_pre
        pk_nw = jnp.array(pk_nw_np)
        # pk_w is traced: wiggle component = full spectrum minus no-wiggle
        pk_w  = pk_lin_h - pk_nw
        damp = jnp.exp(-sigma2_bao * k_h ** 2)
        # IR-resummed linear spectrum (input to FFTLog)
        pk_resummed = pk_nw + pk_w * damp
        # Tree-level spectrum = CLASS-PT's Ptree, pm[14] (nonlinear_pt.c:2999):
        # the IR-resummed tree carries the isotropic damping AND the (1 + Sigma k^2)
        # factor. The raw pk_lin_h used before is the Sigma -> 0 limit of this.
        Pk_tree = pk_nw + pk_w * damp * (1.0 + sigma2_bao * k_h ** 2)
    elif prec.ir_resummation:
        # Default path: call NumPy IR resummation (NOT differentiable through pk_lin_h).
        # Use _ir_precomputed to enable gradients.
        pk_nw_np, pk_w_np, sigma2_bao, delta_sigma2_bao = _ir_resummation_numpy(
            np.array(pk_lin_h), np.array(k_h), rs_h=rs_h, h=h
        )
        pk_nw = jnp.array(pk_nw_np)
        pk_w  = jnp.array(pk_w_np)
        damp = jnp.exp(-sigma2_bao * k_h ** 2)
        # IR-resummed linear spectrum (input to FFTLog)
        pk_resummed = pk_nw + pk_w * damp
        # Tree-level spectrum = CLASS-PT's Ptree, pm[14] (nonlinear_pt.c:2999).
        Pk_tree = pk_nw + pk_w * damp * (1.0 + sigma2_bao * k_h ** 2)
    else:
        # IR off: CLASS-PT has Pw = 0, so Ptree = Pbin (nonlinear_pt.c:3294).
        pk_resummed = pk_lin_h
        Pk_tree = pk_lin_h

    # --- FFTLog decomposition of resummed P(k): matter basis (b=-0.3) ---
    cmsym, etam = _fftlog_decompose(pk_resummed, kmin, kmax, nmax, b)

    # --- Second FFTLog decomposition: bias basis (b=-1.6 for M22basic) ---
    # Matches CLASS-PT nonlinear_pt.c line 11789: b2 = -1.6000001
    cmsym2, etam2 = _fftlog_decompose(pk_resummed, kmin, kmax, nmax, B_BASIC)

    # --- FFTLog decompositions for IR resummation (RSD 1-loop) ---
    # CLASS-PT uses x_nw = cmsym_nw × k^etam for all RSD loop quadratic forms.
    # Wiggle correction uses x_w = cmsym_w × k^etam (×2 for P22, ×1 for P13).
    # cf. nonlinear_pt.c lines 8215, 8246, 8562, 8586.
    # NOTE: _fftlog_decompose is jnp.fft internally (jnp.fft.fft, see its
    # definition above), NOT numpy -- it accepts and can differentiate through
    # traced inputs just fine. The `_pk_nw_np_rsd`/`_pk_w_np_rsd` naming below
    # is legacy; what actually happens is that the RSD path deliberately reads
    # the PINNED (stop_gradient) snapshot of _ir_precomputed[0]/[1] rather than
    # the traced pk_nw/pk_w computed above, per the PHASE-2 FREEZE noted just
    # below -- a deliberate gradient freeze, not a numpy/jax type requirement.
    if prec.ir_resummation:
        if _ir_precomputed is not None:
            # PHASE-2 FREEZE: the FFTLog cmsym basis is built from a pinned
            # snapshot (the pre-existing deliberate freeze, now explicit).
            # Gradient flows through the resummed spectra and damping; the
            # basis-response term is deferred -- residual after this PR
            # quantified in tests/test_ept_gradients.py docstrings.
            _pk_nw_np_rsd = jax.lax.stop_gradient(jnp.asarray(_ir_precomputed[0]))
            _pk_w_np_rsd  = jax.lax.stop_gradient(jnp.asarray(_ir_precomputed[1]))
        else:
            _pk_nw_np_rsd = np.array(pk_nw)
            _pk_w_np_rsd  = np.array(pk_w)
        _cmsym_nw_np, _ = _fftlog_decompose(_pk_nw_np_rsd, kmin, kmax, nmax, b)
        _cmsym_w_np, _  = _fftlog_decompose(_pk_w_np_rsd,  kmin, kmax, nmax, b)
        cmsym_nw_jnp = jnp.array(_cmsym_nw_np)
        cmsym_w_jnp  = jnp.array(_cmsym_w_np)
        pk_nw_jnp    = jnp.array(_pk_nw_np_rsd)
    else:
        cmsym_nw_jnp = None
        cmsym_w_jnp  = None
        pk_nw_jnp    = None
        sigma2_bao   = None
        delta_sigma2_bao = 0.0
        pk_nw = pk_lin_h
        pk_w  = jnp.zeros_like(pk_lin_h)

    # --- Evaluate matter basis at k-grid ---
    x = _x_at_k(cmsym, etam, k_h)  # (nmax, nmax+1) complex

    # --- P22 and P13 ---
    Pk_P22 = _compute_p22(x, k_h, M22, cutoff_h)
    Pk_P13 = _compute_p13(x, k_h, pk_resummed, M13, lnk, cutoff_h)
    Pk_loop = Pk_P13 + Pk_P22

    # --- Counterterm basis: P_CTR = -k² P_lin ---
    # User multiplies by cs0/h² to get the EFT counterterm contribution
    Pk_ctr = -k_h ** 2 * pk_resummed

    # --- Bias cross-spectra and RSD components ---
    bias = _compute_bias_spectra(
        x, k_h, pk_resummed, lnk,
        M22b, IFG2, M13, M22,
        etam, cmsym2, etam2,
        f, cutoff_h,
        cmsym_nw=cmsym_nw_jnp,
        cmsym_w=cmsym_w_jnp,
        pk_nw=pk_nw_jnp,
        sigma2_bao=sigma2_bao if prec.ir_resummation else None,
        delta_sigma2_bao=delta_sigma2_bao if prec.ir_resummation else None,
        pk_w=pk_w if prec.ir_resummation else None,
        hratio=hratio,
        Dratio=Dratio,
    )

    _sigma2 = sigma2_bao if sigma2_bao is not None else 0.0
    _delta_sigma2 = delta_sigma2_bao if delta_sigma2_bao is not None else 0.0

    return EPTComponents(
        kh=k_h, h=jnp.asarray(h), f=jnp.asarray(f),
        Pk_tree=Pk_tree,
        Pk_loop=Pk_loop,
        Pk_ctr=Pk_ctr,
        pk_nw=pk_nw,
        pk_w=pk_w,
        sigma2_bao=jnp.asarray(_sigma2),
        delta_sigma2_bao=jnp.asarray(_delta_sigma2),
        # classy.pyx:4813 builds this from Ptree on the output grid, so it is
        # computed here rather than inside _compute_bias_spectra, which only
        # ever sees the FFTLog-discretised spectrum.
        Pd2d2_0=_pd2d2_0(Pk_tree, k_h),
        **bias,
    )


def ept_kgrid(prec: EPTPrecisionParams = EPTPrecisionParams()) -> np.ndarray:
    """Return the EPT k-grid in h/Mpc (N_max log-spaced points).

    Args:
        prec: EPT precision parameters

    Returns:
        k_h: array of shape (prec.nmax,), k values in h/Mpc
    """
    return np.exp(
        np.linspace(np.log(prec.kmin_h), np.log(prec.kmax_h), prec.nmax)
    )


# ---------------------------------------------------------------------------
# Galaxy power spectrum functions (linear combinations of EPTComponents)
# ---------------------------------------------------------------------------

def pk_mm_real(
    ept: EPTComponents,
    cs0: float = 0.0,
) -> Float[Array, "Nk"]:
    """Real-space matter-matter power spectrum.

    P_mm(k) = P_tree + P_1loop + 2 cs0 P_CTR / h²

    cf. CLASS-PT classy.pyx::pk_mm_real()

    Args:
        ept:  EPTComponents from compute_ept()
        cs0:  EFT sound speed² in (Mpc/h)²

    Returns:
        P_mm in (Mpc/h)³
    """
    # Note: Pk_ctr = -k_h² × P_resummed is in Mpc/h (not (Mpc/h)³).
    # cs0 in (Mpc/h)² × Pk_ctr → (Mpc/h)³.  No extra 1/h² factor needed.
    # cf. CLASS-PT: 2*cs*pk_mult[10]*h where pk_mult[10] = -k²P/h (1/Mpc units internally)
    return ept.Pk_tree + ept.Pk_loop + 2.0 * cs0 * ept.Pk_ctr


def pk_gm_real(
    ept: EPTComponents,
    b1: float,
    b2: float,
    bG2: float,
    bGamma3: float,
    cs0: float = 0.0,
    cs: float = 0.0,
) -> Float[Array, "Nk"]:
    """Real-space galaxy-matter cross-power spectrum.

    P_gm(k) = b1*(P_tree + P_loop) + (2*cs*b1 + cs0)*P_CTR/h²
              + b2/2 P_Id2 + bG2 P_IG2
              + (bG2 + 0.4 bΓ3) P_IFG2

    EPTComponents are in (Mpc/h)³, so no h³ conversion needed.
    cf. CLASS-PT classy.pyx::pk_gm_real(), return at classy.pyx:4834:
    the matter counterterm enters as (2*cs*b1 + cs0), not (cs*b1 + cs0).

    Args:
        ept:       EPTComponents from compute_ept()
        b1, b2:    linear and quadratic bias
        bG2:       Galileon (tidal) bias
        bGamma3:   third-order operator coefficient
        cs0, cs:   EFT counterterm coefficients in (Mpc/h)²

    Returns:
        P_gm in (Mpc/h)³
    """
    return (
        b1 * (ept.Pk_tree + ept.Pk_loop)
        + (2.0 * cs * b1 + cs0) * ept.Pk_ctr   # classy.pyx:4834
        + (b2 / 2.0) * ept.Pk_Id2
        + bG2 * ept.Pk_IG2
        + (bG2 + 0.4 * bGamma3) * ept.Pk_IFG2
    )


def pk_gg_real(
    ept: EPTComponents,
    b1: float,
    b2: float,
    bG2: float,
    bGamma3: float,
    cs: float = 0.0,
    cs0: float = 0.0,
    Pshot: float = 0.0,
) -> Float[Array, "Nk"]:
    """Real-space galaxy-galaxy power spectrum.

    P_gg(k) = b1²(P_tree + P_loop) + 2(cs b1² + cs0 b1) P_CTR/h²
              + b1 b2 P_Id2 + b2²/4 P_Id2d2
              + 2 b1 bG2 P_IG2 + b1(2bG2 + 0.8 bΓ3) P_IFG2
              + bG2² P_IG2G2 + b2 bG2 P_Id2G2 + Pshot

    EPTComponents are in (Mpc/h)³, so no h³ conversion needed.
    cf. CLASS-PT classy.pyx::pk_gg_real()
    """
    return (
        b1 ** 2 * (ept.Pk_tree + ept.Pk_loop)
        + 2.0 * (cs * b1 ** 2 + cs0 * b1) * ept.Pk_ctr
        + b1 * b2 * ept.Pk_Id2
        + 0.25 * b2 ** 2 * ept.Pk_Id2d2
        + 2.0 * b1 * bG2 * ept.Pk_IG2
        + b1 * (2.0 * bG2 + 0.8 * bGamma3) * ept.Pk_IFG2
        + bG2 ** 2 * ept.Pk_IG2G2
        + b2 * bG2 * ept.Pk_Id2G2
    ) + Pshot


def _pk_mm_tree_mu68_at_mu(
    mu: float,
    k: Float[Array, "Nk"],
    ept: EPTComponents,
    f: float,
) -> Float[Array, "Nk"]:
    """Anisotropic tree + new mu^6/mu^8 1-loop contributions at a single mu.

    Computes only the parts that need Gauss quadrature:
      (1) Anisotropic IR-resummed tree term with Sigmatot(mu)
      (2) New mu^6 and mu^8 1-loop terms with P13ratio(mu)

    The existing mu^0/mu^2/mu^4 1-loop multipoles are added analytically
    in the calling functions (preserves existing CLASS-PT accuracy).

    Anisotropic Sigmatot formula: nonlinear_pt.c line 9385:
      Sigmatot = Σ²(1 + f μ²(2+f)) + δΣ² f² μ²(μ²-1)
    """
    k2 = k ** 2
    # Anisotropic BAO damping scale (nonlinear_pt.c line 9385)
    Sigmatot = (ept.sigma2_bao * (1.0 + f * mu**2 * (2.0 + f))
                + ept.delta_sigma2_bao * f**2 * mu**2 * (mu**2 - 1.0))
    Exp = jnp.exp(-Sigmatot * k2)

    # Tree: Pbin(mu) = Pnw + Pw × Exp(mu)  (anisotropic pk_disc, no enhancement factor)
    # CLASS-PT uses Pbin (not Ptree) for RSD multipoles: nonlinear_pt.c line 6903
    pk_disc_aniso = ept.pk_nw + ept.pk_w * Exp
    Ptree = pk_disc_aniso * (1.0 + f * mu**2) ** 2

    # P13ratio for mu^6/mu^8 wiggle correction
    pk_nw_safe = jnp.where(ept.pk_nw > 1e-100, ept.pk_nw, jnp.ones_like(ept.pk_nw))
    wiggle_ratio = jnp.where(ept.pk_nw > 1e-100, ept.pk_w / pk_nw_safe, jnp.zeros_like(ept.pk_nw))
    P13ratio = 1.0 + wiggle_ratio * Exp

    # P_mu68 is handled analytically by the Pk_*_vv1 multipole kernels (Fix 1).
    return Ptree


def pk_mm_l0(
    ept: EPTComponents,
    cs0: float = 0.0,
) -> Float[Array, "Nk"]:
    """Redshift-space matter-matter monopole (ℓ=0).

    Tree via isotropic IR-resummed Kaiser formula (matches CLASS-PT non-AP).
    1-loop added analytically.
    cf. CLASS-PT nonlinear_pt.c lines 6901–6910.
    """
    # Tree: isotropic Pbin × (1 + 2f/3 + f²/5)  [CLASS-PT non-AP]
    tree_l0 = ept.Pk_0_dd + ept.Pk_0_vd + ept.Pk_0_vv
    P1loop_l0 = ept.Pk_0_dd1 + ept.Pk_0_vd1 + ept.Pk_0_vv1
    return tree_l0 + P1loop_l0 + 2.0 * cs0 * ept.Pk_ctr0


def pk_mm_l2(
    ept: EPTComponents,
    cs2: float = 0.0,
) -> Float[Array, "Nk"]:
    """Redshift-space matter-matter quadrupole (ℓ=2).

    Tree via isotropic IR-resummed Kaiser formula (matches CLASS-PT non-AP).
    1-loop added analytically.
    cf. CLASS-PT nonlinear_pt.c lines 7020–7030.
    """
    # Tree: anisotropic GL integral; Pk_2_dd is non-zero with Sigmatot(mu)
    tree_l2 = ept.Pk_2_vv + ept.Pk_2_vd + ept.Pk_2_dd
    P1loop_l2 = ept.Pk_2_vv1 + ept.Pk_2_vd1 + ept.Pk_2_dd1
    return tree_l2 + P1loop_l2 + 2.0 * cs2 * ept.Pk_ctr2


def pk_mm_l4(
    ept: EPTComponents,
    cs4: float = 0.0,
) -> Float[Array, "Nk"]:
    """Redshift-space matter-matter hexadecapole (ℓ=4).

    Tree via isotropic IR-resummed Kaiser formula (matches CLASS-PT non-AP).
    1-loop added analytically.
    cf. CLASS-PT nonlinear_pt.c lines 7127–7140.
    """
    # Tree: anisotropic GL integral; Pk_4_vd, Pk_4_dd non-zero with Sigmatot(mu)
    tree_l4 = ept.Pk_4_vv + ept.Pk_4_vd + ept.Pk_4_dd
    P1loop_l4 = ept.Pk_4_vv1 + ept.Pk_4_vd1 + ept.Pk_4_dd1
    return tree_l4 + P1loop_l4 + 2.0 * cs4 * ept.Pk_ctr4


def pk_gg_l0(
    ept: EPTComponents,
    b1: float,
    b2: float,
    bG2: float,
    bGamma3: float,
    cs0: float = 0.0,
    Pshot: float = 0.0,
    b4: float = 0.0,
    a0: float = 0.0,
    a2: float = 0.0,
) -> Float[Array, "Nk"]:
    """Redshift-space galaxy-galaxy monopole (ℓ=0).

    Full formula from CLASS-PT classy.pyx::pk_gg_l0(), lines 1193–1199.

    Includes: tree-level Kaiser, 1-loop vv/vd/dd, bias cross-terms,
    EFT counterterm, shot noise, and higher-order stochastic (b4).
    """
    f = ept.f
    h = ept.h
    kh = ept.kh

    # Tree: isotropic Pbin × (b1² + 2f·b1/3 + f²/5)  [CLASS-PT non-AP]
    new_l0_tree = b1 ** 2 * ept.Pk_0_dd + b1 * ept.Pk_0_vd + ept.Pk_0_vv

    # 1-loop contributions (isotropic IR, analytical)
    P_loop_l0 = (
        ept.Pk_0_vv1
        + b1 * ept.Pk_0_vd1
        + b1 ** 2 * ept.Pk_0_dd1
    )

    # Bias cross-terms
    P_bias_l0 = (
        0.25 * b2 ** 2 * ept.Pk_Id2d2
        + b1 * b2 * ept.Pk_0_b1b2
        + b2 * ept.Pk_0_b2
        + b1 * bG2 * ept.Pk_0_b1bG2
        + bG2 * ept.Pk_0_bG2
        + b2 * bG2 * ept.Pk_Id2G2
        + bG2 ** 2 * ept.Pk_IG2G2
        + (2.0 * bG2 + 0.8 * bGamma3) * (b1 * ept.Pk_IFG2_0b1 + ept.Pk_IFG2_0)
    )

    P_b4 = (
        f ** 2 * b4 * kh ** 2
        * (f ** 2 / 9.0 + 2.0 * f * b1 / 7.0 + b1 ** 2 / 5.0)
        * (35.0 / 8.0)
        * ept.Pk_ctr4
    )

    # CLASS-PT adds the zero-lag P_d2d2(k->0) back here -- classy.pyx:4911,
    # "+ 0.25*b2**2.*self.Pd2d2_0" -- so that this term uses the UNSUBTRACTED
    # P_d2d2.  It is NOT degenerate with Pshot once b2 floats, because it scales
    # as b2^2.  NB pk_gg_real deliberately does NOT carry it (CLASS-PT
    # classy.pyx:4827 omits it there too); the constant is absorbed into Pshot.
    return ((new_l0_tree + P_loop_l0 + P_bias_l0 + 2.0 * cs0 * ept.Pk_ctr0)
            + Pshot + P_b4 + 0.25 * b2 ** 2 * ept.Pd2d2_0
            # classy.pyx:4910: the scale-dependent stochastic terms, already
            # divided by nbar. a2 enters the monopole with weight 1/3 and the
            # quadrupole with 2/3 -- the Legendre moments of its mu^2 shape.
            + a0 * (ept.kh / 0.45) ** 2
            + a2 * (1.0 / 3.0) * (ept.kh / 0.45) ** 2)


def pk_gg_l2(
    ept: EPTComponents,
    b1: float,
    b2: float,
    bG2: float,
    bGamma3: float,
    cs2: float = 0.0,
    b4: float = 0.0,
    a2: float = 0.0,
) -> Float[Array, "Nk"]:
    """Redshift-space galaxy-galaxy quadrupole (ℓ=2).

    cf. CLASS-PT classy.pyx::pk_gg_l2(), line 1201–1206.
    """
    f = ept.f
    h = ept.h
    kh = ept.kh

    # Tree: (b1 + f μ²)² p_tree projected on L2.  classy.pyx:4921 is
    # pm[18] + b1·pm[19] + b1²·pm[26], and pm[26] is P1loopdd_ap_ir, which
    # nonlinear_pt.c:4529 OPENS with p_tree -- so the b1² dd tree belongs here,
    # carrying b1² like every other dd term.
    new_l2_tree = ept.Pk_2_vv + b1 * ept.Pk_2_vd + b1 ** 2 * ept.Pk_2_dd

    P_loop_l2 = (
        ept.Pk_2_vv1
        + b1 * ept.Pk_2_vd1
        + b1 ** 2 * ept.Pk_2_dd1
    )

    P_bias_l2 = (
        b1 * b2 * ept.Pk_2_b1b2
        + b2 * ept.Pk_2_b2
        + b1 * bG2 * ept.Pk_2_b1bG2
        + bG2 * ept.Pk_2_bG2
        + (2.0 * bG2 + 0.8 * bGamma3) * ept.Pk_IFG2_2
    )

    P_b4 = (
        f ** 2 * b4 * kh ** 2
        * (f ** 2 * 70.0 + 165.0 * f * b1 + 99.0 * b1 ** 2)
        * (4.0 / 693.0) * (35.0 / 8.0)
        * ept.Pk_ctr4
    )

    return ((new_l2_tree + P_loop_l2 + P_bias_l2 + 2.0 * cs2 * ept.Pk_ctr2) + P_b4
            # classy.pyx:4921: a2 enters the quadrupole with weight 2/3.
            + a2 * (2.0 / 3.0) * (ept.kh / 0.45) ** 2)


def pk_gg_l4(
    ept: EPTComponents,
    b1: float,
    b2: float,
    bG2: float,
    bGamma3: float,
    cs4: float = 0.0,
    b4: float = 0.0,
) -> Float[Array, "Nk"]:
    """Redshift-space galaxy-galaxy hexadecapole (ℓ=4).

    cf. CLASS-PT classy.pyx::pk_gg_l4(), line 1208–1213.
    """
    f = ept.f
    h = ept.h
    kh = ept.kh

    # Tree: classy.pyx:4931 is pm[20] + pm[27] + b1·pm[28] + b1²·pm[29], and
    # pm[28]/pm[29] are P1loopvd_ap_ir / P1loopdd_ap_ir, which open with the
    # vd/dd tree (nonlinear_pt.c:4529-4539).  The tree therefore carries the
    # same b1 powers as the loop: vv, b1·vd, b1²·dd.
    new_l4_tree = ept.Pk_4_vv + b1 * ept.Pk_4_vd + b1 ** 2 * ept.Pk_4_dd

    P_loop_l4 = (
        ept.Pk_4_vv1
        + b1 * ept.Pk_4_vd1
        + b1 ** 2 * ept.Pk_4_dd1
    )

    # classy.pyx:4925-4931 has no pm[40]/pm[41] (P_4_b1b2 / P_4_b1bG2) term,
    # although the C loop fills both rows; mirror the accessor.  The leaves stay
    # in EPTComponents.
    P_bias_l4 = (
        b2 * ept.Pk_4_b2
        + bG2 * ept.Pk_4_bG2
    )

    P_b4 = (
        f ** 2 * b4 * kh ** 2
        * (f ** 2 * 210.0 + 390.0 * f * b1 + 143.0 * b1 ** 2)
        * (8.0 / 5005.0) * (35.0 / 8.0)
        * ept.Pk_ctr4
    )

    return (new_l4_tree + P_loop_l4 + P_bias_l4 + 2.0 * cs4 * ept.Pk_ctr4) + P_b4


# ---------------------------------------------------------------------------
# Integration with clax pipeline
# ---------------------------------------------------------------------------

def ept_inputs_from_clax(
    params,           # CosmoParams
    bg,               # BackgroundResult
    pt,               # MatterPerturbationResult (perturbations_solve_mpk)
    z: float = 0.0,
    prec: EPTPrecisionParams = EPTPrecisionParams(),
    *,
    field: str = "cb",
) -> tuple[Float[Array, "Nk"], Float[Array, ""]]:
    """Linear P(k, z) on the EPT k-grid in (Mpc/h)^3 and the growth rate f(z).

    field="cb" (default) samples ``pt.delta_cb`` -- CLASS-PT's ``cb: Yes``
    (input.c:3952 switches the PT input from delta_m to delta_cb; spec §4.5).
    field="m" samples ``pt.delta_m`` (``cb: No``); clax.lensing uses it
    because the CMB-lensing nonlinear ratio is a total-matter quantity.

    Args:
        params: CosmoParams (h, primordial spectrum); h is traced throughout
        bg:     BackgroundResult (tau(z), f(z))
        pt:     MatterPerturbationResult from perturbations_solve_mpk (reaches
                today; carries delta_cb and delta_m). A PerturbationResult
                from perturbations_solve works only for z >~ 0.0032: its
                grid stops at 0.999*tau0, and a z~0 lookup raises.
        z:      target redshift (may be traced)
        prec:   EPT precision (sets the k-grid)
        field:  "cb" or "m"

    Returns:
        (pk_h, f): pk_h shape (prec.nmax,) in (Mpc/h)^3 on ept_kgrid(prec);
        f = bg.f_of_loga at z (0-d).
    """
    from clax.primordial import primordial_scalar_pk
    from clax.interpolation import CubicSpline as CS

    if field == "cb":
        delta = getattr(pt, "delta_cb", None)
        if delta is None:
            raise ValueError(
                "field='cb' needs pt.delta_cb (perturbations_solve_mpk provides it); "
                f"{type(pt).__name__} has none -- use field='m'")
    elif field == "m":
        delta = pt.delta_m
    else:
        raise ValueError(f"field must be 'cb' or 'm', got {field!r}")

    h = params.h  # traced: carries d(pk_h)/dh AND the k-resampling channel

    # EPT k-grid in h/Mpc (static shape source) -> Mpc^-1, TRACED in h so
    # the delta/primordial sampling points move with h under AD exactly
    # as they do under finite differences (issue #30 item 4).
    k_h = ept_kgrid(prec)              # static numpy array
    k_mpc = jnp.asarray(k_h) * h       # traced jnp array

    lnk_pt  = jnp.log(pt.k_grid)
    lnk_out = jnp.log(jnp.array(k_mpc))

    # Interpolate delta to tau(z) along the tau axis (vmap-safe; no Python
    # branch on z), then spline along log-k onto the EPT k-grid. Beyond
    # pt.k_grid[-1] the spline clamps to the last value (constant delta):
    # solve with pt_k_max_cl >= 3 Mpc^-1 (the P22/P13 UV cutoff CUTOFF = 3
    # h/Mpc) when the loop integrals matter, cf. tests/test_ept_e2e_multicosmo.py.
    loga_z = jnp.log(1.0 / (1.0 + z))
    # Raise, rather than let the tau spline clamp, if tau(z) is outside the
    # solved grid: perturbations_solve stops at 0.999*tau0, so a z~0 lookup
    # on it silently returned -0.33% in P (smsharma/clax#42).
    from clax.perturbations import _raise_if_tau_outside_grid
    tau_z = _raise_if_tau_outside_grid(
        bg.tau_of_loga.evaluate(loga_z), pt.tau_grid, "ept_inputs_from_clax")
    delta_at_z = jax.vmap(
        lambda d_k: CS(pt.tau_grid, d_k).evaluate(tau_z))(delta)
    delta_ept = CS(lnk_pt, delta_at_z).evaluate(lnk_out)

    # Linear P(k) in Mpc^3: P(k) = 2 pi^2 / k^3 * P_R(k) * delta^2(k)
    # (matches clax/transfer.py::compute_linear_matter_pk_from_perturbations)
    k_arr = jnp.array(k_mpc)
    prim = primordial_scalar_pk(k_arr, params)  # dimensionless P_R(k)
    pk_mpc3 = 2.0 * jnp.pi**2 / k_arr ** 3 * prim * delta_ept ** 2

    # Convert to h-units: P_h = P * h^3, k_h = k / h. This h**3 cancels
    # exactly against 1/k_arr**3 (k_arr = k_h * h) and contributes nothing
    # to d(pk_h)/dh; the true h-derivative is the traced k-resampling above.
    pk_h = pk_mpc3 * h ** 3  # (Mpc/h)^3

    # Growth rate from the background solve (f = dlnD/dlna spline),
    # z-consistent and differentiable. cf. background.py:681 (f_of_loga).
    f = bg.f_of_loga.evaluate(loga_z)
    return pk_h, f


def compute_ept_from_clax(
    params,           # CosmoParams
    bg,               # BackgroundResult
    pt,               # MatterPerturbationResult (perturbations_solve_mpk)
    z: float = 0.0,
    prec: EPTPrecisionParams = EPTPrecisionParams(),
    *,
    th=None,          # ThermoResult; supplies the integrated r_s(z_d) (smsharma/clax#44)
    omfid: Optional[float] = None,
    field: str = "cb",
) -> EPTComponents:
    """Compute EPT components from a full clax perturbation run.

    ``th``: the ThermoResult of the same solve. With it, the IR-resummation
    damping scale is its integrated ``rs_d`` (CLASS's ``pth->rs_d``, the
    quantity CLASS-PT uses at nonlinear_pt.c:5596); without it, the Aubourg
    et al. 2015 fit of ``clax.background.sound_horizon_drag``, which is
    0.12% accurate at best and 0.150% off at omega_b +20% (smsharma/clax#44).
    Every pipeline call site passes ``th``.

    Converts clax's linear P(k) (in Mpc³) to h-units, evaluates on
    the EPT k-grid, and runs compute_ept().

    Args:
        params: CosmoParams (for h, primordial spectrum)
        bg:     BackgroundResult (for growth factor, distances)
        pt:     MatterPerturbationResult from perturbations_solve_mpk. Required
                near z=0: perturbations_solve's grid stops at 0.999*tau0 and
                the lookup raises (smsharma/clax#42).
        z:      target redshift
        prec:   EPT precision

    Returns:
        EPTComponents in h-units

    Note (stop_gradient rationale): `h` is traced throughout, including the
    k_mpc = k_h * h resampling of delta_m/the primordial spectrum onto the
    EPT k-grid, so d(pk_h)/dh carries the k-resampling channel exactly as
    finite differences do (issue #30 item 4; previously frozen at -9.48e4
    of the stage h-gradient, GPU job 13313). The IR resummation itself now
    runs through the traced `_ir_resummation_jax` splitter below, so
    d(pk_nw)/d(params) and d(Sigma^2)/d(params) both flow -- this closes
    the two channels that used to be stop_gradient'd here: the 1.39%-class
    pk_nw/ln10A_s residual (job 13132) and the rs_h (sound-horizon) channel
    (previously bounded below -1.0e2 of the stage h-gradient, GPU job
    13313). Two channels remain deliberately frozen:
      - the DST grid endpoints inside `_ir_resummation_jax` itself, built
        from a concrete h_conc = stop_gradient(h) (np.linspace(7e-5/h,
        7/h, ...) needs a fixed shape, not a differentiability choice --
        see that function's docstring);
      - the RSD FFTLog basis (`_pk_nw_np_rsd`/`_pk_w_np_rsd` at
        compute_ept's `_ir_precomputed` RSD consumption): PHASE-2 FREEZE,
        deferred by this PR -- see the inline comment there and
        tests/test_ept_gradients.py docstrings for the residual this
        leaves. pk_mm_real is unaffected -- it never reads those FFTLog
        bases.
    """
    from clax.transfer import compute_pk_from_perturbations
    from clax.primordial import primordial_scalar_pk
    from clax.interpolation import CubicSpline

    from clax.background import sound_horizon_drag

    h = params.h  # traced: carries d(pk_h)/dh AND the k-resampling channel
    # Concrete copy EXISTS ONLY for the DST grid endpoints inside
    # _ir_resummation_jax below (7e-5/h .. 7/h feeding np.linspace, which
    # must stay concrete -- see that function's docstring). Do NOT reuse
    # h_conc anywhere else: reusing it for k_mpc was the frozen resampling
    # channel measured at -9.48e4 of the stage h-gradient (GPU job 13313,
    # issue #30 item 4).
    h_conc = float(jax.lax.stop_gradient(h))

    # Alcock-Paczynski ratios (nonlinear_pt.c:1245-1296). The branch is on the
    # STATIC omfid, never on a traced value; ap_ratios itself is differentiable
    # in bg. omfid=None reproduces the pre-AP behaviour exactly.
    if omfid is None:
        hratio, Dratio = 1.0, 1.0
    else:
        from clax.ap import ap_ratios
        hratio, Dratio = ap_ratios(bg, z, omfid)

    # EPT k-grid in h/Mpc (static shape source) -> Mpc^-1, TRACED in h so
    # the delta_m/primordial sampling points move with h under AD exactly
    # as they do under finite differences.
    k_h = ept_kgrid(prec)
    pk_h, f = ept_inputs_from_clax(params, bg, pt, z, prec, field=field)

    # Traced IR resummation: d(pk_nw)/d(params) and d(Sigma^2)/d(params)
    # now flow (closes the 1.39% ln10A_s / 1.19% h structural residual,
    # issue #30). Only the DST grid endpoints stay pinned (see
    # _ir_resummation_jax) and the RSD FFTLog basis (phase 2, below).
    rs_d = th.rs_d if th is not None else sound_horizon_drag(params)
    rs_h_traced = rs_d * params.h
    ir_pre = _ir_resummation_jax(pk_h, k_h, rs_h_traced, h_conc)
    return compute_ept(pk_h, jnp.array(k_h), h=h, f=f, prec=prec,
                        # rs_h is a dead argument on the _ir_precomputed path
                        # (audit 2026-08-31: only reaches the untaken numpy
                        # branch); stop_gradient is a no-op here but keeps
                        # this call site's own gradient graph minimal.
                        rs_h=jax.lax.stop_gradient(rs_h_traced),
                        _ir_precomputed=ir_pre,
                        hratio=hratio, Dratio=Dratio)
