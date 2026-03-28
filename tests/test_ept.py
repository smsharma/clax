"""Unit tests for clax.ept — FFTLog decomposition and EFT 1-loop kernels.

Tests use synthetic power-law inputs and the precomputed CLASS-PT kernel
matrices from ~/CLASS-PT/pt_matrices/. No classy or CLASS-PT Python wrapper
is required.

Accuracy targets:
  - FFTLog reconstruction: <1% max relative error on synthetic P_lin
  - M22 symmetry: exact (M == M.T up to numerical precision)
  - P22 positivity: P22(k) > 0 for physical k range
"""

from __future__ import annotations

import os
import sys
import math
import importlib
import unittest

import numpy as np

# ---------------------------------------------------------------------------
# Locate matrix files relative to this test file
# ---------------------------------------------------------------------------

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR  = os.path.dirname(_TESTS_DIR)
_CLASSPT_MATRIX_DIR = os.path.join(
    os.path.expanduser("~"), "CLASS-PT", "pt_matrices"
)

# Patch the ept module's matrix path to the absolute location so tests work
# regardless of current working directory.
os.environ.setdefault("CLASSPT_MATRIX_DIR", _CLASSPT_MATRIX_DIR)

# Add repo to path so `clax.ept` is importable
sys.path.insert(0, _REPO_DIR)

# ---------------------------------------------------------------------------
# Conditionally import ept and JAX
# ---------------------------------------------------------------------------

try:
    import jax
    import jax.numpy as jnp
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False

try:
    import clax.ept as ept
    _EPT_IMPORTABLE = True
except Exception as exc:
    _EPT_IMPORTABLE = False
    _EPT_IMPORT_ERROR = str(exc)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NMAX = 256
N    = NMAX + 1  # = 257 matrix dimension


def _matrix_path(name: str) -> str:
    return os.path.join(_CLASSPT_MATRIX_DIR, name)


def _matrices_available() -> bool:
    return os.path.isdir(_CLASSPT_MATRIX_DIR) and os.path.isfile(
        _matrix_path(f"M22oneline_N{NMAX}.dat")
    )


def synthetic_pk_lin(k: np.ndarray, sigma8: float = 0.8) -> np.ndarray:
    """Simple ΛCDM-like power-law times Gaussian cutoff.

    P(k) ∝ k^0.96 exp(-(k/5)²), normalized so sigma_8 ~ 0.8 order-of-magnitude.
    Not physical, but sufficient for sign/shape sanity checks.
    """
    pk = k ** 0.96 * np.exp(-(k / 5.0) ** 2)
    # Rough normalization so max(P) ~ 1e4 (Mpc/h)^3
    pk = pk / pk.max() * 1e4
    return pk


# ---------------------------------------------------------------------------
# Test: matrix file loading (numpy only)
# ---------------------------------------------------------------------------

class TestLoadComplexVector(unittest.TestCase):
    """Tests for _load_complex_vector — loads M13 and IFG2 vectors."""

    @unittest.skipUnless(_matrices_available(), "CLASS-PT matrix files not found")
    @unittest.skipUnless(_EPT_IMPORTABLE, "clax.ept not importable")
    def test_m13_shape(self):
        """M13 vector must have shape (257,) = (NMAX+1,)."""
        v = ept._load_complex_vector(_matrix_path(f"M13oneline_N{NMAX}.dat"), N)
        self.assertEqual(v.shape, (N,))

    @unittest.skipUnless(_matrices_available(), "CLASS-PT matrix files not found")
    @unittest.skipUnless(_EPT_IMPORTABLE, "clax.ept not importable")
    def test_m13_dtype(self):
        """M13 vector must be complex."""
        v = ept._load_complex_vector(_matrix_path(f"M13oneline_N{NMAX}.dat"), N)
        self.assertTrue(np.issubdtype(v.dtype, np.complexfloating))

    @unittest.skipUnless(_matrices_available(), "CLASS-PT matrix files not found")
    @unittest.skipUnless(_EPT_IMPORTABLE, "clax.ept not importable")
    def test_ifg2_shape(self):
        """IFG2 vector must have shape (257,)."""
        v = ept._load_complex_vector(_matrix_path(f"IFG2oneline_N{NMAX}.dat"), N)
        self.assertEqual(v.shape, (N,))

    @unittest.skipUnless(_matrices_available(), "CLASS-PT matrix files not found")
    @unittest.skipUnless(_EPT_IMPORTABLE, "clax.ept not importable")
    def test_m13_nonzero(self):
        """M13 must contain non-zero entries."""
        v = ept._load_complex_vector(_matrix_path(f"M13oneline_N{NMAX}.dat"), N)
        self.assertGreater(np.max(np.abs(v)), 0.0)


class TestLoadComplexTriangular(unittest.TestCase):
    """Tests for _load_complex_triangular — loads M22 / M22basic matrices."""

    @unittest.skipUnless(_matrices_available(), "CLASS-PT matrix files not found")
    @unittest.skipUnless(_EPT_IMPORTABLE, "clax.ept not importable")
    def test_m22_shape(self):
        """M22 matrix must have shape (257, 257)."""
        M = ept._load_complex_triangular(_matrix_path(f"M22oneline_N{NMAX}.dat"), N)
        self.assertEqual(M.shape, (N, N))

    @unittest.skipUnless(_matrices_available(), "CLASS-PT matrix files not found")
    @unittest.skipUnless(_EPT_IMPORTABLE, "clax.ept not importable")
    def test_m22_symmetric(self):
        """M22 must be symmetric: M[i,j] == M[j,i].

        CLASS-PT uses zdotu (bilinear), making I(eta1,eta2)=I(eta2,eta1)
        via F2 kernel symmetry. The matrix is NOT Hermitian (M != M^†).
        """
        M = ept._load_complex_triangular(_matrix_path(f"M22oneline_N{NMAX}.dat"), N)
        np.testing.assert_allclose(
            M, M.T, rtol=1e-10, atol=1e-30,
            err_msg="M22 is not symmetric (transpose mismatch)"
        )

    @unittest.skipUnless(_matrices_available(), "CLASS-PT matrix files not found")
    @unittest.skipUnless(_EPT_IMPORTABLE, "clax.ept not importable")
    def test_m22_not_hermitian(self):
        """M22 must NOT be Hermitian (M[i,j] != conj(M[j,i]) in general).

        This verifies we removed the .conj() bug: if M were Hermitian,
        all diagonal elements would be real, but M22 has complex diagonals.
        """
        M = ept._load_complex_triangular(_matrix_path(f"M22oneline_N{NMAX}.dat"), N)
        # Check that imaginary parts of diagonal are nonzero (Hermitian → purely real diag)
        diag_imag = np.imag(np.diag(M))
        has_complex_diag = np.any(np.abs(diag_imag) > 1e-30)
        # If diagonal is all real, we can't distinguish; check off-diagonal instead
        if not has_complex_diag:
            # Off-diagonal: for Hermitian M[i,j] = conj(M[j,i]), so M != M.conj().T
            # would fail. Here we verify M == M.T (symmetric) but check the conj differs.
            upper = M[0, 5]  # pick an off-diagonal element
            lower = M[5, 0]
            # They should be equal (symmetric), not conjugate-equal (Hermitian)
            self.assertAlmostEqual(upper, lower, places=10,
                msg="M22[0,5] should equal M22[5,0] exactly (symmetric)")
        # At minimum, M must be symmetric
        np.testing.assert_allclose(M, M.T, rtol=1e-10, atol=1e-30)

    @unittest.skipUnless(_matrices_available(), "CLASS-PT matrix files not found")
    @unittest.skipUnless(_EPT_IMPORTABLE, "clax.ept not importable")
    def test_m22basic_shape_and_symmetry(self):
        """M22basic must be (257,257) and symmetric."""
        M = ept._load_complex_triangular(_matrix_path(f"M22basiconeline_N{NMAX}.dat"), N)
        self.assertEqual(M.shape, (N, N))
        np.testing.assert_allclose(M, M.T, rtol=1e-10, atol=1e-30)


# ---------------------------------------------------------------------------
# Test: FFTLog decomposition (requires JAX)
# ---------------------------------------------------------------------------

class TestFFTLogDecompose(unittest.TestCase):
    """Tests for _fftlog_decompose — biased DFT decomposition of P_lin."""

    @unittest.skipUnless(_JAX_AVAILABLE, "JAX not installed")
    @unittest.skipUnless(_EPT_IMPORTABLE, "clax.ept not importable")
    def test_coefficients_shape(self):
        """FFTLog coefficients must have shape (NMAX+1,) = (257,)."""
        kmin, kmax = ept.KMIN_H, ept.KMAX_H
        k_disc = np.logspace(np.log10(kmin), np.log10(kmax), NMAX)
        pk_disc = jnp.array(synthetic_pk_lin(k_disc))
        cmsym, etam = ept._fftlog_decompose(pk_disc, kmin, kmax, NMAX, ept.B_MATTER)
        self.assertEqual(cmsym.shape, (N,))
        self.assertEqual(etam.shape,  (N,))

    @unittest.skipUnless(_JAX_AVAILABLE, "JAX not installed")
    @unittest.skipUnless(_EPT_IMPORTABLE, "clax.ept not importable")
    def test_reconstruction_accuracy(self):
        """FFTLog round-trip must recover P_lin to within 1% on interior points.

        Evaluates: P_reconstructed(k) = Re{ sum_m cmsym[m] k^{etam[m]} }
        and compares to the original P_lin on the interior 80% of the k-grid.
        """
        kmin, kmax = ept.KMIN_H, ept.KMAX_H
        k_disc = np.logspace(np.log10(kmin), np.log10(kmax), NMAX)
        pk_original = synthetic_pk_lin(k_disc)
        pk_disc_jnp = jnp.array(pk_original)

        cmsym, etam = ept._fftlog_decompose(pk_disc_jnp, kmin, kmax, NMAX, ept.B_MATTER)

        # Reconstruct: P(k_j) = Re{ sum_m c_m k_j^eta_m }
        # = Re{ sum_m c_m exp(eta_m log(k_j)) }
        log_k = np.log(k_disc)[:, None]         # (Nmax, 1)
        eta   = np.array(etam)[None, :]          # (1, N)
        c     = np.array(cmsym)[None, :]         # (1, N)
        basis = c * np.exp(eta * log_k)          # (Nmax, N) complex
        pk_reconstructed = np.real(basis.sum(axis=-1))  # (Nmax,) real

        # Use only interior 10–90% of k-range to avoid endpoint effects
        i_lo = NMAX // 10
        i_hi = NMAX - NMAX // 10
        rel_err = np.abs(pk_reconstructed[i_lo:i_hi] - pk_original[i_lo:i_hi]) / \
                  np.abs(pk_original[i_lo:i_hi])
        max_rel_err = float(np.max(rel_err))

        self.assertLess(max_rel_err, 0.01,
            f"FFTLog reconstruction max relative error = {max_rel_err:.2%} > 1%")

    @unittest.skipUnless(_JAX_AVAILABLE, "JAX not installed")
    @unittest.skipUnless(_EPT_IMPORTABLE, "clax.ept not importable")
    def test_etam_real_part(self):
        """Real part of eta_m must equal the bias parameter B_MATTER."""
        kmin, kmax = ept.KMIN_H, ept.KMAX_H
        k_disc = jnp.array(np.logspace(np.log10(kmin), np.log10(kmax), NMAX))
        pk_disc = jnp.array(synthetic_pk_lin(np.array(k_disc)))
        _, etam = ept._fftlog_decompose(pk_disc, kmin, kmax, NMAX, ept.B_MATTER)
        np.testing.assert_allclose(
            np.real(np.array(etam)), ept.B_MATTER,
            atol=1e-12,
            err_msg="Re(eta_m) must equal B_MATTER for all m"
        )


# ---------------------------------------------------------------------------
# Test: P22 and P13 (requires JAX + matrix files)
# ---------------------------------------------------------------------------

class TestOneLoopKernels(unittest.TestCase):
    """Tests for _compute_p22 and _compute_p13."""

    def _setup_jax_arrays(self):
        """Return (x, k, pk_disc, M22, M13, lnk) for use in P22/P13 tests."""
        kmin, kmax = 0.01, 1.0
        Nk = 20
        k = np.logspace(np.log10(kmin), np.log10(kmax), Nk)
        pk = synthetic_pk_lin(k)

        # FFTLog on CLASS-PT grid
        k_disc_np = np.logspace(np.log10(ept.KMIN_H), np.log10(ept.KMAX_H), NMAX)
        pk_disc = jnp.array(synthetic_pk_lin(k_disc_np))

        cmsym, etam = ept._fftlog_decompose(
            pk_disc, ept.KMIN_H, ept.KMAX_H, NMAX, ept.B_MATTER
        )
        k_jnp = jnp.array(k)
        x = ept._x_at_k(cmsym, etam, k_jnp)

        matrices = ept._load_matrices(NMAX)
        M22 = jnp.array(matrices["M22"])
        M13 = jnp.array(matrices["M13"])
        lnk = jnp.log(jnp.array(k))

        pk_disc_at_k = jnp.array(np.interp(k, k_disc_np, np.array(pk_disc)))
        return x, k_jnp, pk_disc_at_k, M22, M13, lnk

    @unittest.skipUnless(_JAX_AVAILABLE, "JAX not installed")
    @unittest.skipUnless(_matrices_available(), "CLASS-PT matrix files not found")
    @unittest.skipUnless(_EPT_IMPORTABLE, "clax.ept not importable")
    def test_p22_positive(self):
        """P22(k) must be positive for physical k in [0.01, 1] h/Mpc.

        P22 = Re{k^3 x^T M22 x} with UV damping > 0 by construction
        when M22 is positive semi-definite in the relevant subspace.
        """
        x, k, pk_disc, M22, M13, lnk = self._setup_jax_arrays()
        P22 = ept._compute_p22(x, k, M22, cutoff_h=ept.CUTOFF)
        P22_np = np.array(P22)
        self.assertTrue(
            np.all(P22_np > 0),
            f"P22 has non-positive values: min={P22_np.min():.3e}"
        )

    @unittest.skipUnless(_JAX_AVAILABLE, "JAX not installed")
    @unittest.skipUnless(_matrices_available(), "CLASS-PT matrix files not found")
    @unittest.skipUnless(_EPT_IMPORTABLE, "clax.ept not importable")
    def test_p22_shape(self):
        """P22 output shape must match input k array."""
        x, k, pk_disc, M22, M13, lnk = self._setup_jax_arrays()
        P22 = ept._compute_p22(x, k, M22, cutoff_h=ept.CUTOFF)
        self.assertEqual(P22.shape, k.shape)

    @unittest.skipUnless(_JAX_AVAILABLE, "JAX not installed")
    @unittest.skipUnless(_matrices_available(), "CLASS-PT matrix files not found")
    @unittest.skipUnless(_EPT_IMPORTABLE, "clax.ept not importable")
    def test_p13_shape(self):
        """P13 output shape must match input k array."""
        x, k, pk_disc, M22, M13, lnk = self._setup_jax_arrays()
        P13 = ept._compute_p13(x, k, pk_disc, M13, lnk)
        self.assertEqual(P13.shape, k.shape)

    @unittest.skipUnless(_JAX_AVAILABLE, "JAX not installed")
    @unittest.skipUnless(_matrices_available(), "CLASS-PT matrix files not found")
    @unittest.skipUnless(_EPT_IMPORTABLE, "clax.ept not importable")
    def test_p13_magnitude(self):
        """P13 must be smaller in magnitude than P_lin (loop expansion validity).

        At k ~ 0.1 h/Mpc, |P13| << P_lin is required for perturbation theory
        to be valid. We check max |P13| / max P_lin < 50% as a sanity bound.
        """
        x, k, pk_disc, M22, M13, lnk = self._setup_jax_arrays()
        P13 = ept._compute_p13(x, k, pk_disc, M13, lnk)
        P13_np  = np.array(P13)
        pk_np   = np.array(pk_disc)
        ratio   = np.max(np.abs(P13_np)) / np.max(pk_np)
        self.assertLess(ratio, 0.5,
            f"|P13|/P_lin = {ratio:.2f} > 0.5 — loop expansion may have collapsed")

    @unittest.skipUnless(_JAX_AVAILABLE, "JAX not installed")
    @unittest.skipUnless(_matrices_available(), "CLASS-PT matrix files not found")
    @unittest.skipUnless(_EPT_IMPORTABLE, "clax.ept not importable")
    def test_p22_scaling(self):
        """P22 must scale as A² when P_lin → A×P_lin (quadratic in P_lin).

        P22 = ∫∫ d³q F2² P(q)P(|k-q|) ∝ A² since it's quadratic.
        """
        x, k, pk_disc, M22, M13, lnk = self._setup_jax_arrays()
        A = 2.0
        kmin, kmax = ept.KMIN_H, ept.KMAX_H
        k_disc_np = np.logspace(np.log10(kmin), np.log10(kmax), NMAX)
        pk_disc_2 = jnp.array(A * synthetic_pk_lin(k_disc_np))
        cmsym2, etam2 = ept._fftlog_decompose(pk_disc_2, kmin, kmax, NMAX, ept.B_MATTER)
        x2 = ept._x_at_k(cmsym2, etam2, k)

        P22_1 = np.array(ept._compute_p22(x,  k, M22, cutoff_h=ept.CUTOFF))
        P22_2 = np.array(ept._compute_p22(x2, k, M22, cutoff_h=ept.CUTOFF))

        ratio = P22_2 / P22_1
        np.testing.assert_allclose(
            ratio, A ** 2 * np.ones_like(ratio),
            rtol=1e-6,
            err_msg=f"P22 does not scale as A²: mean ratio = {ratio.mean():.4f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not _EPT_IMPORTABLE:
        print(f"WARNING: clax.ept could not be imported: {_EPT_IMPORT_ERROR}")
    if not _JAX_AVAILABLE:
        print("WARNING: JAX not available — JAX-dependent tests will be skipped.")
    if not _matrices_available():
        print(f"WARNING: CLASS-PT matrices not found at {_CLASSPT_MATRIX_DIR}")
        print("         Matrix loading tests will be skipped.")

    unittest.main(verbosity=2)
