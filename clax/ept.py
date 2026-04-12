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

    The _packed.dat files store the matrix in LAPACK 'L' (lower triangular,
    column-major) packed format, as used by zspmv_ in nonlinear_pt.c line 6099.

    LAPACK 'L' column-major packed storage: column j (0-indexed) stores
    A(j,j), A(j+1,j), ..., A(n-1,j). Position of A(i,j) for i >= j is:
      idx = j*n - j*(j-1)//2 + (i-j)

    Confirmed by nonlinear_pt.c line 2412 (sequential count over packed storage)
    and by matrix_gen-matter.py:
      mout_red[i + (2*(Nmax+1)-1-j)*j//2] = m12mat[i][j]

    Mirrors: nonlinear_pt.c load_M22 section + LAPACK zspmv_ UPLO='L' convention.
    """
    n_tri = n * (n + 1) // 2
    data = np.loadtxt(filepath)
    assert len(data) == 2 * n_tri, (
        f"{filepath}: expected {2*n_tri} values, got {len(data)}"
    )
    tri = data[:n_tri] + 1j * data[n_tri:]

    # Unpack LAPACK 'L' column-major lower-triangular packed to full symmetric matrix.
    # Column j stores rows i = j, j+1, ..., n-1; sequential index = j*n-j*(j-1)//2 + (i-j).
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

    For N=256, CLASS-PT uses _packed.dat files in LAPACK 'L' column-major
    lower-triangular packed format (cf. nonlinear_pt.c line 980, 982).
    For other N, non-packed files use row-major lower-triangular packing.

    Returns a dict with keys:
      M13   : (nmax+1,) complex  — P13 kernel (matter)
      M22   : (nmax+1, nmax+1) complex — P22 kernel (matter, includes F2)
      M22b  : (nmax+1, nmax+1) complex — Basic mode coupling I(ν₁,ν₂)
      IFG2  : (nmax+1,) complex  — F·G2 coupling
    """
    n = nmax + 1  # matrix dimension

    def mat_path(name):
        return os.path.join(_MATRIX_DIR, name)

    # Check for _packed.dat files (N=256 in CLASS-PT uses these)
    packed_m22 = mat_path(f"M22oneline_N{nmax}_packed.dat")
    packed_m22b = mat_path(f"M22basiconeline_N{nmax}_packed.dat")
    use_packed = os.path.exists(packed_m22)

    if use_packed:
        m22 = _load_complex_triangular_lapack_l(packed_m22, n)
        m22b = _load_complex_triangular_lapack_l(packed_m22b, n)
    else:
        # Non-packed files: row-major lower triangular
        def _load_rm(path):
            n_tri = n * (n + 1) // 2
            data = np.loadtxt(path)
            tri = data[:n_tri] + 1j * data[n_tri:]
            M = np.zeros((n, n), dtype=complex)
            idx = 0
            for i in range(n):
                for j in range(i + 1):
                    M[i, j] = tri[idx]; M[j, i] = tri[idx]; idx += 1
            return M
        m22 = _load_rm(mat_path(f"M22oneline_N{nmax}.dat"))
        m22b = _load_rm(mat_path(f"M22basiconeline_N{nmax}.dat"))

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
    h:  float                       # Hubble parameter (for unit conversion)
    f:  float                       # growth rate d ln D/d ln a

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
    sigma2_bao: float                    # isotropic BAO damping Σ²_BAO in (Mpc/h)²
    delta_sigma2_bao: float              # anisotropic correction δΣ²_BAO in (Mpc/h)²
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
        ]
        aux = (self.h, self.f, self.sigma2_bao, self.delta_sigma2_bao)
        return arrays, aux

    @classmethod
    def tree_unflatten(cls, aux, arrays):
        h, f, sigma2_bao, delta_sigma2_bao = aux
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

    Mirrors: nonlinear_pt.c lines 5315–5776 (DST-based BAO extraction).

    CRITICAL: CLASS-PT uses a LINEAR k-grid (not log-spaced) for the DST.
    Modes 120–240 of a linear-k DST target k-space oscillation periods
    of (k_max-k_min)/120 to (k_max-k_min)/240 ≈ 0.08–0.04 h/Mpc, which
    brackets the BAO period 2π/r_s ≈ 0.06 h/Mpc. On a log-k grid those
    modes map to entirely different scales, giving wrong P_nw.

    CLASS-PT splits DST modes into odd and even sub-arrays (cmodd/cmeven)
    and applies natural cubic spline interpolation to each separately when
    removing modes 120–240. This gives ~3× more accurate P_nw than simple
    linear interpolation across the full DST. cf. nonlinear_pt.c:5404–5520.

    For k outside [k_min2, k_max2], CLASS-PT sets P_nw = P_lin and P_w = 0.
    cf. nonlinear_pt.c lines 5739–5748.

    The Σ_BAO² formula uses a spherical Bessel j_2 filter to suppress
    the contribution of modes below the BAO scale:
      Σ_BAO² = (1/6π²) ∫₀^{ks} P_nw(q) × [1 - 3j₁(qr)/qr + ...] × q d(ln q)
    cf. nonlinear_pt.c lines 5605–5640 (IntegrandBAO).

    Args:
        pk_lin_h: P_lin(k) in (Mpc/h)³, shape (N,)
        k_h:      k in h/Mpc, shape (N,)
        rs_h:     BAO sound horizon at drag epoch in Mpc/h (default 99.0 Mpc/h)
        h:        Hubble parameter h = H₀/100 (needed to match CLASS-PT DST grid
                  which is hardcoded in 1/Mpc: kmin2=0.00007 1/Mpc, kmax2=7 1/Mpc)

    Returns:
        pk_nw:   no-wiggle (broadband) P(k), same shape as input
        pk_w:    wiggle P(k) = pk_lin_h - pk_nw
        sigma2_bao: BAO damping scale Σ_BAO² in (Mpc/h)²
    """
    try:
        from scipy.fft import dst, idst
    except ImportError:
        return _ir_resummation_gaussian(pk_lin_h, k_h)

    # LINEAR k-grid matching CLASS-PT nonlinear_pt.c:5322-5323 which hardcodes
    # kmin2=0.00007 1/Mpc and kmax2=7 1/Mpc (in physical 1/Mpc units).
    # Converting to h-units: kmin2_h = 0.00007/h h/Mpc, kmax2_h = 7/h h/Mpc.
    # Using the exact CLASS-PT values is critical: the DST mode number for the
    # BAO oscillation is n_BAO ≈ kmax2/BAO_period, so a 4% kmax2 difference
    # shifts the BAO mode by 4%, causing different Pnw extraction.
    N_IR = 65536
    k_min2 = 7e-5 / h   # 0.00007 1/Mpc in h/Mpc (CLASS-PT kmin2)
    k_max2 = 7.0 / h    # 7 1/Mpc in h/Mpc (CLASS-PT kmax2)
    k_ir = np.linspace(k_min2, k_max2, N_IR)

    # Interpolate P_lin to the linear grid (log-log interpolation)
    log_k_in = np.log(k_h)
    log_pk_in = np.log(np.clip(pk_lin_h, 1e-300, None))
    pk_ir = np.exp(np.interp(np.log(k_ir), log_k_in, log_pk_in))

    # DST-II of log(k P(k)) — forward transform.
    # CLASS-PT uses a custom FFT-based DST via DCT-like trick.
    # cf. nonlinear_pt.c:5355-5398 (input_realv2 construction + FFT)
    f_ir = np.log(k_ir * pk_ir)
    f_dst = dst(f_ir, type=2, norm="ortho")

    # Remove BAO modes 120–240 by cubic spline on odd and even modes separately.
    # CLASS-PT splits cmnew[2i] (odd) and cmnew[2i+1] (even), removes Nleft:Nright
    # from each, then spline-interpolates back. This matches nonlinear_pt.c:5420-5520.
    # cf. nonlinear_pt.c:5404-5413: cmodd[i] = out_ir[2*i], cmeven[i] = out_ir[2*i+1]
    N_left  = 120
    N_right = 240
    N_IR_half = N_IR // 2  # = 32768

    # Split full DST into odd and even indexed modes (over the half-spectrum)
    # The actual CLASS-PT split works on the FFT output of their custom DST,
    # which differs from scipy's DST-II by a sign/ordering. We approximate
    # by splitting scipy DST modes directly into odd/even index.
    cmodd  = f_dst[0:N_IR:2]   # even indices of DST (odd "mode" in CLASS-PT sense)
    cmeven = f_dst[1:N_IR:2]   # odd indices of DST

    # For each of odd and even, remove modes [N_left, N_right) via cubic spline.
    # Indices run from 0..N_IR_half-1, mapping to DST indices 0,2,4,...
    # The "throw" region is [N_left, N_right) in the sub-arrays.
    from scipy.interpolate import CubicSpline

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

    # Inverse DST-II to recover log(k P_nw) on linear grid
    f_nw_ir = idst(f_dst_nw, type=2, norm="ortho")
    pk_nw_ir = np.exp(np.clip(f_nw_ir, -700, 700)) / k_ir

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
    # cf. nonlinear_pt.c:5614: ks = 0.2 * pba->h = 0.2 h/Mpc
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
        np.trapz(pk_nw_int * bao_filter * k_int, np.log(k_int)) / (6.0 * np.pi ** 2)
    )

    # δΣ_BAO² (anisotropic second damping scale):
    # IntegrandBAO2 = -Pnw * (3*cos(qr)*r*q + (-3+(qr)²)*sin(qr)) / (qr)³
    # cf. nonlinear_pt.c line 5639; normalization 1/(2π²) vs 1/(6π²) for Σ_BAO²
    bao_filter2 = -1.0 * (
        3.0 * np.cos(x_bao) * x_bao + (-3.0 + x_bao ** 2) * np.sin(x_bao)
    ) / x_bao ** 3
    delta_sigma2_bao = (
        np.trapz(pk_nw_int * bao_filter2 * k_int, np.log(k_int)) / (2.0 * np.pi ** 2)
    )

    return pk_nw, pk_w, float(sigma2_bao), float(delta_sigma2_bao)


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
    sigma2_bao = np.trapz(pk_nw * k_h, np.log(k_h)) / (6.0 * np.pi ** 2)
    return pk_nw, pk_w, float(sigma2_bao), 0.0


# ---------------------------------------------------------------------------
# Bias spectral cross-terms from M22basic (JAX-native)
# ---------------------------------------------------------------------------

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
    Pk_tree: Float[Array, "Nk"],
    cutoff_h: float,
    cmsym_nw: Optional[Complex[Array, "Nmax+1"]] = None,
    cmsym_w: Optional[Complex[Array, "Nmax+1"]] = None,
    pk_nw: Optional[Float[Array, "Nk"]] = None,
    sigma2_bao: Optional[float] = None,
    delta_sigma2_bao: Optional[float] = None,
    pk_w: Optional[Float[Array, "Nk"]] = None,
) -> dict:
    """Compute bias cross-spectra and RSD multipole components.

    Exact implementation matching CLASS-PT nonlinear_pt.c.

    Bias spectra use a second FFTLog decomposition with bias b=-1.6
    (etam2/cmsym2), matching CLASS-PT's etam2 with b2=-1.6000001.
    Modified M22basic matrices are built by elementwise multiplication
    with rational kernels in etam2 (lines 11919-11925 in nonlinear_pt.c).

    RSD 1-loop multipoles use the matter basis (etam, b=-0.3) with
    M22 × rational kernels in nu1,nu2,f (lines 6647, 6928, 7159, 7275).

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
    Pk_Id2d2 = (jnp.abs(raw_Id2d2 - raw_Id2d2[0]) + 1e-6) * uv_damp

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

    # IFG2 RSD multipoles (no-AP path):
    # From nonlinear_pt.c lines 12511–12535:
    #   P_IFG2_0b1 = P_IFG2              (b1 × G2 cross, no f factor)
    #   P_IFG2_0   = P_IFG2 × f/3        (monopole: f × <μ²> = f/3)
    #   P_IFG2_2   = P_IFG2 × 2f/3       (quadrupole)
    Pk_IFG2_0b1 = Pk_IFG2
    Pk_IFG2_0   = Pk_IFG2 * f / 3.0
    Pk_IFG2_2   = Pk_IFG2 * 2.0 * f / 3.0

    # ===========================================================
    # COUNTERTERM MULTIPOLES
    # Pk_ctr_ℓ encodes the shape of the EFT counterterm for each multipole.
    # Convention: P_ℓ^EFT = 2 cs_ℓ Pk_ctr_ℓ / h² (caller supplies cs_ℓ)
    # From CLASS-PT line 7239: P_CTR_2 = k² Pbin × f × 2/3
    # ===========================================================
    Pk_ctr0 = -k ** 2 * pk_disc                    # monopole  (= -pk_CTR_0; pm[11] = -P_CTR_0)
    Pk_ctr2 = -k ** 2 * pk_disc * f * (2.0 / 3.0) # quadrupole (pm[12] = -P_CTR_2; P_CTR_2 = k²Pbin f 2/3)
    Pk_ctr4 = -k ** 2 * pk_disc * (8.0 / 35.0) * f ** 2  # hexadecapole (pm[13] = -P_CTR_4)

    # ===========================================================
    # RSD TREE-LEVEL MULTIPOLES (anisotropic IR resummation via GL quadrature)
    # Following CLASS-PT AP path: compute the full P_tree(k,mu)
    # with anisotropic Sigmatot(mu) at each GL node, then project onto Legendre
    # multipoles.  The tree integrand at each GL node is:
    #   p_tree(k,mu) = Pnw + Pw * exp(-Sigmatot(mu)*k²) * (1 + Sigmatot(mu)*k²)
    # This matches the CLASS-PT AP path (nonlinear_pt.c line 9388).
    # No empirical alpha factor needed.
    #
    # Storage convention: Pk_0_vv/vd/dd store the f-weighted components
    # so that galaxy combination is:
    #   P_0^gal = Pk_0_vv + b1 Pk_0_vd + b1² Pk_0_dd + 1-loop
    # Accumulated in the GL loop below.
    # ===========================================================
    Pk_0_vv = jnp.zeros_like(k)
    Pk_0_vd = jnp.zeros_like(k)
    Pk_0_dd = jnp.zeros_like(k)
    Pk_2_vv = jnp.zeros_like(k)
    Pk_2_vd = jnp.zeros_like(k)
    Pk_2_dd = jnp.zeros_like(k)
    Pk_4_vv = jnp.zeros_like(k)
    Pk_4_vd = jnp.zeros_like(k)
    Pk_4_dd = jnp.zeros_like(k)

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

    # Compute nw/w split for all mu-power channels
    P22_mu0_dd_nw, P22_mu0_dd_w = qf_split(M22)
    P13_mu0_dd_nw, P13_mu0_dd_w = p13_split(M13, UV_mu0_dd)
    P22_mu2_vd_nw, P22_mu2_vd_w = qf_split(M22_mu2_vd_bare)
    P22_mu2_dd_nw, P22_mu2_dd_w = qf_split(M22_mu2_dd_bare)
    P13_mu2_vd_nw, P13_mu2_vd_w = p13_split(M13_mu2_vd_bare, UV_mu2_vd)
    P13_mu2_dd_nw, P13_mu2_dd_w = p13_split(M13_mu2_dd_bare, UV_mu2_dd)
    P22_mu4_vv_nw, P22_mu4_vv_w = qf_split(M22_mu4_vv_bare)
    P22_mu4_vd_nw, P22_mu4_vd_w = qf_split(M22_mu4_vd_bare)
    P22_mu4_dd_nw, P22_mu4_dd_w = qf_split(M22_mu4_dd_bare)
    P13_mu4_vv_nw, P13_mu4_vv_w = p13_split(M13_mu4_vv_bare, UV_mu4_vv)
    P13_mu4_vd_nw, P13_mu4_vd_w = p13_split(M13_mu4_vd_bare, UV_mu4_vd)
    P22_mu6_vv_nw, P22_mu6_vv_w = qf_split(M22_mu6_vv_mat)
    P22_mu6_vd_nw, P22_mu6_vd_w = qf_split(M22_mu6_vd_mat)
    P13_mu6_nw,    P13_mu6_w    = p13_split(M13_mu6_mat, UV_mu6)
    P22_mu8_nw,    P22_mu8_w    = qf_split(M22_mu8_mat)

    # EPTComponents still needs these combined (for _pk_mm_tree_mu68_at_mu)
    P22_mu6_vv = P22_mu6_vv_nw + P22_mu6_vv_w * Exp
    P22_mu6_vd = P22_mu6_vd_nw + P22_mu6_vd_w * Exp
    P22_mu8    = P22_mu8_nw + P22_mu8_w * Exp
    P13_mu6    = P13_mu6_nw + P13_mu6_w * Exp

    # GL loop with anisotropic Sigmatot for 1-loop multipoles
    # Replaces old analytic Pk_0/2/4_vv1 = P13_0/2/4_vv + P22_0/2/4_vv
    Pk_0_vv1 = jnp.zeros_like(k)
    Pk_2_vv1 = jnp.zeros_like(k)
    Pk_4_vv1 = jnp.zeros_like(k)
    Pk_0_vd1 = jnp.zeros_like(k)
    Pk_2_vd1 = jnp.zeros_like(k)
    Pk_4_vd1 = jnp.zeros_like(k)
    Pk_0_dd1 = jnp.zeros_like(k)
    Pk_2_dd1 = jnp.zeros_like(k)
    Pk_4_dd1 = jnp.zeros_like(k)

    _pk_w_for_ratio = (jnp.asarray(pk_w) if pk_w is not None else jnp.zeros_like(k)) if use_ir_rsd else jnp.zeros_like(k)
    _pk_nw_safe = jnp.where(pk_nw_arr > 1e-100, pk_nw_arr, jnp.ones_like(pk_nw_arr))
    _delta_sig2 = delta_sigma2_bao if delta_sigma2_bao is not None else 0.0
    _sig2_bao = sigma2_bao if sigma2_bao is not None else 0.0

    for _mu_g, _w_g in zip(_GAUSS_NODES, _GAUSS_WEIGHTS):
        _mu2 = float(_mu_g)**2
        _Sig = _sig2_bao*(1 + f*_mu2*(2+f)) + _delta_sig2*f**2*_mu2*(_mu2-1)
        _Eg  = jnp.exp(-_Sig * k**2)
        _r13 = jnp.where(pk_nw_arr > 1e-100,
                         1.0 + (_pk_w_for_ratio / _pk_nw_safe) * _Eg,
                         jnp.ones_like(k))

        # vv: mu^4 + mu^6 + mu^8
        _Pvv = (
            (P13_mu4_vv_nw * _r13 + P22_mu4_vv_nw + (P22_mu4_vv_w + P13_mu4_vv_w) * _Eg) * _mu2**2
          + (P13_mu6_nw * _r13 + P22_mu6_vv_nw + (P22_mu6_vv_w + P13_mu6_w) * _Eg) * _mu2**3
          + (P22_mu8_nw + P22_mu8_w * _Eg) * _mu2**4
        )
        # dd: mu^0 + mu^2 + mu^4
        _Pdd = (
            (P22_mu0_dd_nw + P13_mu0_dd_nw * _r13 + (P13_mu0_dd_w + P22_mu0_dd_w) * _Eg)
          + (P22_mu2_dd_nw + P13_mu2_dd_nw * _r13 + (P22_mu2_dd_w + P13_mu2_dd_w) * _Eg) * _mu2
          + (P22_mu4_dd_nw + P22_mu4_dd_w * _Eg) * _mu2**2
        )
        # vd: mu^2 + mu^4 + mu^6
        _Pvd = (
            (P13_mu2_vd_nw * _r13 + P22_mu2_vd_nw + (P22_mu2_vd_w + P13_mu2_vd_w) * _Eg) * _mu2
          + (P13_mu4_vd_nw * _r13 + P22_mu4_vd_nw + (P22_mu4_vd_w + P13_mu4_vd_w) * _Eg) * _mu2**2
          + (P22_mu6_vd_nw + P22_mu6_vd_w * _Eg) * _mu2**3
        )

        _L0 = 1.0
        _L2 = 0.5 * (3*_mu2 - 1)
        _L4 = (35*_mu2**2 - 30*_mu2 + 3) / 8.0

        # Anisotropic tree: CLASS-PT AP path (nonlinear_pt.c line 9388)
        # p_tree(k,mu) = Pnw + Pw * exp(-Sigmatot(mu)*k²) * (1 + Sigmatot(mu)*k²)
        _p_tree = pk_nw_arr + _pk_w_for_ratio * _Eg * (1.0 + _Sig * k**2)
        _tree_vv = f**2 * _mu2**2 * _p_tree
        _tree_vd = 2.0 * f * _mu2 * _p_tree
        _tree_dd = _p_tree

        Pk_0_vv = Pk_0_vv + _w_g * 0.5 * _L0 * _tree_vv
        Pk_2_vv = Pk_2_vv + _w_g * 2.5 * _L2 * _tree_vv
        Pk_4_vv = Pk_4_vv + _w_g * 4.5 * _L4 * _tree_vv
        Pk_0_vd = Pk_0_vd + _w_g * 0.5 * _L0 * _tree_vd
        Pk_2_vd = Pk_2_vd + _w_g * 2.5 * _L2 * _tree_vd
        Pk_4_vd = Pk_4_vd + _w_g * 4.5 * _L4 * _tree_vd
        Pk_0_dd = Pk_0_dd + _w_g * 0.5 * _L0 * _tree_dd
        Pk_2_dd = Pk_2_dd + _w_g * 2.5 * _L2 * _tree_dd
        Pk_4_dd = Pk_4_dd + _w_g * 4.5 * _L4 * _tree_dd

        Pk_0_vv1 = Pk_0_vv1 + _w_g * 0.5 * _L0 * _Pvv
        Pk_2_vv1 = Pk_2_vv1 + _w_g * 2.5 * _L2 * _Pvv
        Pk_4_vv1 = Pk_4_vv1 + _w_g * 4.5 * _L4 * _Pvv
        Pk_0_dd1 = Pk_0_dd1 + _w_g * 0.5 * _L0 * _Pdd
        Pk_2_dd1 = Pk_2_dd1 + _w_g * 2.5 * _L2 * _Pdd
        Pk_4_dd1 = Pk_4_dd1 + _w_g * 4.5 * _L4 * _Pdd
        Pk_0_vd1 = Pk_0_vd1 + _w_g * 0.5 * _L0 * _Pvd
        Pk_2_vd1 = Pk_2_vd1 + _w_g * 2.5 * _L2 * _Pvd
        Pk_4_vd1 = Pk_4_vd1 + _w_g * 4.5 * _L4 * _Pvd

    # ===========================================================
    # RSD BIAS CROSS-TERMS
    # All built from M22basic × rational kernel in nu1, nu2, f.
    # nu1 = -0.5*etam2[i], nu2 = -0.5*etam2[l]  (b=-1.6 bias basis)
    # Exact kernels from nonlinear_pt.c lines 12871–13339.
    # ===========================================================

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

    # Compute RSD bias spectra via quadratic form
    Pk_0_b1b2  = qf2(M_0_b1b2)
    Pk_0_b2    = qf2(M_0_b2)
    Pk_0_b1bG2 = qf2(M_0_b1bG2)
    Pk_0_bG2   = qf2(M_0_bG2)
    Pk_2_b1b2  = qf2(M_2_b1b2)
    Pk_2_b2    = qf2(M_2_b2)
    Pk_2_b1bG2 = qf2(M_2_b1bG2)
    Pk_2_bG2   = qf2(M_2_bG2)
    Pk_4_b2    = qf2(M_4_b2)
    Pk_4_bG2   = qf2(M_4_bG2)
    # Pk_4_b1b2 and Pk_4_b1bG2: populated via AP integration in CLASS-PT;
    # zero in no-AP case (our reference was generated without AP).
    Pk_4_b1b2  = jnp.zeros_like(k)
    Pk_4_b1bG2 = jnp.zeros_like(k)

    return {
        "Pk_Id2d2": Pk_Id2d2, "Pk_Id2": Pk_Id2, "Pk_IG2": Pk_IG2,
        "Pk_Id2G2": Pk_Id2G2, "Pk_IG2G2": Pk_IG2G2,
        "Pk_IFG2": Pk_IFG2, "Pk_IFG2_0b1": Pk_IFG2_0b1,
        "Pk_IFG2_0": Pk_IFG2_0, "Pk_IFG2_2": Pk_IFG2_2,
        "Pk_ctr0": Pk_ctr0, "Pk_ctr2": Pk_ctr2, "Pk_ctr4": Pk_ctr4,
        "Pk_0_vv": Pk_0_vv, "Pk_0_vd": Pk_0_vd, "Pk_0_dd": Pk_0_dd,
        "Pk_2_vv": Pk_2_vv, "Pk_2_vd": Pk_2_vd, "Pk_4_vv": Pk_4_vv,
        "Pk_2_dd": Pk_2_dd, "Pk_4_vd": Pk_4_vd, "Pk_4_dd": Pk_4_dd,
        "Pk_0_vv1": Pk_0_vv1, "Pk_0_vd1": Pk_0_vd1, "Pk_0_dd1": Pk_0_dd1,
        "Pk_2_vv1": Pk_2_vv1, "Pk_2_vd1": Pk_2_vd1, "Pk_2_dd1": Pk_2_dd1,
        "Pk_4_vv1": Pk_4_vv1, "Pk_4_vd1": Pk_4_vd1, "Pk_4_dd1": Pk_4_dd1,
        "Pk_0_b1b2": Pk_0_b1b2, "Pk_0_b2": Pk_0_b2,
        "Pk_0_b1bG2": Pk_0_b1bG2, "Pk_0_bG2": Pk_0_bG2,
        "Pk_2_b1b2": Pk_2_b1b2, "Pk_2_b2": Pk_2_b2,
        "Pk_2_b1bG2": Pk_2_b1bG2, "Pk_2_bG2": Pk_2_bG2,
        "Pk_4_b2": Pk_4_b2, "Pk_4_bG2": Pk_4_bG2,
        "Pk_4_b1b2": Pk_4_b1b2, "Pk_4_b1bG2": Pk_4_b1bG2,
        "P22_mu6_vv": P22_mu6_vv, "P22_mu6_vd": P22_mu6_vd,
        "P22_mu8": P22_mu8, "P13_mu6": P13_mu6,
    }


# ---------------------------------------------------------------------------
# Main EPT computation
# ---------------------------------------------------------------------------

def compute_ept(
    pk_lin_h: Float[Array, "Nk"],
    k_h: Float[Array, "Nk"],
    h: float,
    f: float,
    prec: EPTPrecisionParams = EPTPrecisionParams(),
    _ir_precomputed: Optional[tuple] = None,
    rs_h: float = 99.0,
) -> EPTComponents:
    """Compute all EPT spectral components from linear P(k).

    This is the main entry point for the CLASS-PT algorithm in JAX.
    Differentiable w.r.t. pk_lin_h (for autodiff through P(k)).

    Args:
        pk_lin_h: linear P(k) in (Mpc/h)³ on the EPT k-grid, shape (Nmax,)
                  Must be evaluated at k_h = EPT k-grid (prec.kmin_h to prec.kmax_h)
        k_h:      EPT k-grid [h/Mpc], shape (Nmax,)
                  Should match EPT_kgrid(prec) for best accuracy
        h:        Hubble parameter h = H₀/100
        f:        growth rate f = d ln D / d ln a at the target redshift
        prec:     EPT precision parameters (static)
        _ir_precomputed: optional tuple (pk_nw_np, pk_w_np, sigma2_bao) from
                  _ir_resummation_numpy(), pre-computed outside the JAX trace.
                  When provided, only pk_nw_np and sigma2_bao are used; the
                  wiggle component is recomputed as pk_w = pk_lin_h - pk_nw
                  (JAX-traced), so that gradients flow:
                    pk_resummed = pk_nw + (pk_lin_h - pk_nw) * exp(-Σ²k²)
                                = pk_lin_h × exp(-Σ²k²) + pk_nw × (1 - exp(-Σ²k²))
                  Use this to enable jax.grad() through the IR resummation path:

                      pk_nw_np, pk_w_np, sigma2 = _ir_resummation_numpy(pk_lin_np, k_np)
                      def f(pk_lin):
                          return compute_ept(pk_lin, k_ept, h=h, f=f,
                                             _ir_precomputed=(pk_nw_np, pk_w_np, sigma2))
                      grad = jax.grad(f)(pk_lin_ept)  # works!

                  Physically correct: the no-wiggle template pk_nw is a property
                  of the broadband shape at fixed cosmology. For a perturbation
                  δpk around the fiducial, the wiggle component changes as
                  δpk_w = δpk_lin, and the resummed spectrum damps it by exp(-Σ²k²).
        rs_h:     Sound horizon at drag epoch in Mpc/h (~99 for Planck 2018 LCDM).
                  Used by IR resummation to set the BAO oscillation scale k_osc = 1/rs_h
                  for the Σ²_BAO j₂-filter integral.  Ignored if _ir_precomputed is
                  provided (the caller already computed Σ²_BAO) or ir_resummation=False.

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
        # pk_nw_np is fixed (not traced); pk_w is derived from pk_lin_h so that
        # gradients flow through pk_lin_h → pk_w → pk_resummed → P13/P22.
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
        # Tree-level spectrum for real-space: use raw pk_lin_h (no IR damping).
        # The real-space tree has no RSD, so no anisotropic damping applies.
        # Using raw pk_lin avoids sensitivity to our DST-derived
        # sigma2_bao, which may differ slightly from CLASS-PT's.
        # RSD tree multipoles are computed via anisotropic GL quadrature in
        # _compute_bias_spectra (see Pk_0/2/4_vv/vd/dd accumulated there).
        Pk_tree = pk_lin_h
    elif prec.ir_resummation:
        # Default path: call NumPy IR resummation (NOT differentiable through pk_lin_h).
        # Use _ir_precomputed to enable gradients.
        pk_nw_np, pk_w_np, sigma2_bao, delta_sigma2_bao = _ir_resummation_numpy(
            np.array(pk_lin_h), np.array(k_h), rs_h=rs_h, h=h
        )
        pk_nw = jnp.array(pk_nw_np)
        pk_w  = jnp.array(pk_w_np)
        # IR-resummed linear spectrum (input to FFTLog)
        pk_resummed = pk_nw + pk_w * jnp.exp(-sigma2_bao * k_h ** 2)
        # Tree-level spectrum for real-space: use raw pk_lin_h (no IR damping).
        # RSD tree multipoles are computed via anisotropic GL quadrature.
        Pk_tree = pk_lin_h
    else:
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
    # Always use NUMPY arrays for _fftlog_decompose (it uses np.fft internally).
    # In the _ir_precomputed path, pk_w may be JAX-traced, so use _ir_precomputed[1]
    # (the original numpy wiggle) rather than the traced pk_w = pk_lin_h - pk_nw.
    if prec.ir_resummation:
        if _ir_precomputed is not None:
            _pk_nw_np_rsd = _ir_precomputed[0]   # numpy, not traced
            _pk_w_np_rsd  = _ir_precomputed[1]   # numpy, not traced
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
        f, Pk_tree, cutoff_h,
        cmsym_nw=cmsym_nw_jnp,
        cmsym_w=cmsym_w_jnp,
        pk_nw=pk_nw_jnp,
        sigma2_bao=sigma2_bao if prec.ir_resummation else None,
        delta_sigma2_bao=delta_sigma2_bao if prec.ir_resummation else None,
        pk_w=pk_w if prec.ir_resummation else None,
    )

    _sigma2 = float(sigma2_bao) if sigma2_bao is not None else 0.0
    _delta_sigma2 = float(delta_sigma2_bao) if delta_sigma2_bao is not None else 0.0

    return EPTComponents(
        kh=k_h, h=h, f=f,
        Pk_tree=Pk_tree,
        Pk_loop=Pk_loop,
        Pk_ctr=Pk_ctr,
        pk_nw=pk_nw,
        pk_w=pk_w,
        sigma2_bao=_sigma2,
        delta_sigma2_bao=_delta_sigma2,
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

    P_gm(k) = b1*(P_tree + P_loop) + (cs*b1 + cs0)*P_CTR/h²
              + b2/2 P_Id2 + bG2 P_IG2
              + (bG2 + 0.4 bΓ3) P_IFG2

    EPTComponents are in (Mpc/h)³, so no h³ conversion needed.
    cf. CLASS-PT classy.pyx::pk_gm_real()

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
        + (cs * b1 + cs0) * ept.Pk_ctr
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

    return (new_l0_tree + P_loop_l0 + P_bias_l0 + 2.0 * cs0 * ept.Pk_ctr0) + Pshot + P_b4


def pk_gg_l2(
    ept: EPTComponents,
    b1: float,
    b2: float,
    bG2: float,
    bGamma3: float,
    cs2: float = 0.0,
    b4: float = 0.0,
) -> Float[Array, "Nk"]:
    """Redshift-space galaxy-galaxy quadrupole (ℓ=2).

    cf. CLASS-PT classy.pyx::pk_gg_l2(), line 1201–1206.
    """
    f = ept.f
    h = ept.h
    kh = ept.kh

    # Tree: Kaiser formula (b1+fμ²)² projected to L2 via GL quadrature.
    # With anisotropic Σ_tot(μ), the dd tree term is nonzero (the integral
    # ∫L2(μ)·p_tree(k,μ) dμ ≠ 0 because p_tree depends on μ through the
    # damping).  cf. pk_mm_l2 which includes Pk_2_dd.
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

    return (new_l2_tree + P_loop_l2 + P_bias_l2 + 2.0 * cs2 * ept.Pk_ctr2) + P_b4


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

    # Tree: CLASS-PT uses pm[20] (the matter anisotropic tree) for both matter
    # and galaxy l=4; b1 factors appear only on the 1-loop terms pm[28]/pm[29].
    # cf. CLASS-PT classy.pyx line 1213: pm[20]+pm[27]+b1*pm[28]+b1²*pm[29].
    new_l4_tree = ept.Pk_4_vv + ept.Pk_4_vd + ept.Pk_4_dd

    P_loop_l4 = (
        ept.Pk_4_vv1
        + b1 * ept.Pk_4_vd1
        + b1 ** 2 * ept.Pk_4_dd1
    )

    P_bias_l4 = (
        b2 * ept.Pk_4_b2
        + bG2 * ept.Pk_4_bG2
        + b1 * b2 * ept.Pk_4_b1b2
        + b1 * bG2 * ept.Pk_4_b1bG2
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

def compute_ept_from_clax(
    params,           # CosmoParams
    bg,               # BackgroundResult
    pt,               # PerturbationResult
    z: float = 0.0,
    prec: EPTPrecisionParams = EPTPrecisionParams(),
) -> EPTComponents:
    """Compute EPT components from a full clax perturbation run.

    Converts clax's linear P(k) (in Mpc³) to h-units, evaluates on
    the EPT k-grid, and runs compute_ept().

    Args:
        params: CosmoParams (for h, primordial spectrum)
        bg:     BackgroundResult (for growth factor, distances)
        pt:     PerturbationResult (for δ_m(k,τ))
        z:      target redshift
        prec:   EPT precision

    Returns:
        EPTComponents in h-units
    """
    from clax.transfer import compute_pk_from_perturbations
    from clax.primordial import primordial_scalar_pk
    from clax.interpolation import CubicSpline

    h = float(params.h)

    # EPT k-grid in h/Mpc → convert to Mpc⁻¹ for clax
    k_h = ept_kgrid(prec)  # h/Mpc
    k_mpc = k_h * h        # Mpc⁻¹ (clax internal units)

    # Get δ_m at each k, at redshift z, from perturbation result
    # Then P_lin(k) = 2π²/k³ × A_s × (k/k_pivot)^{n_s-1} × δ_m²
    A_s = jnp.exp(params.ln10A_s) * 1e-10
    lnk_pt  = jnp.log(pt.k_grid)
    lnk_out = jnp.log(jnp.array(k_mpc))

    # Interpolate δ_m to EPT k-grid at τ corresponding to redshift z
    # (simple: use z=0 δ_m for now)
    # delta_m_interp = ...  # TODO: proper z interpolation
    # For now, use δ_m at last τ in pt.tau_grid
    delta_m_0 = pt.delta_m[:, -1]  # shape (Nk_pt,), at z≈0
    from clax.interpolation import CubicSpline as CS
    spline = CS(lnk_pt, delta_m_0)
    delta_m_ept = spline(lnk_out)

    # Linear P(k) in Mpc³
    k_arr = jnp.array(k_mpc)
    prim = primordial_scalar_pk(params, k_arr)  # primordial power spectrum
    pk_mpc3 = prim * delta_m_ept ** 2 / k_arr ** 3

    # Convert to h-units: P_h = P * h³,  k_h = k / h
    pk_h = pk_mpc3 * h ** 3  # (Mpc/h)³

    # Growth rate f ≈ Ω_m(z)^0.55 (approximation; improve later)
    f = float(bg.Omega_m_of_z(z)) ** 0.55 if hasattr(bg, "Omega_m_of_z") else 0.8

    return compute_ept(jnp.array(pk_h), jnp.array(k_h), h=h, f=f, prec=prec)
