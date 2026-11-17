"""Tests nonlinear ``P(k)`` forward and local-gradient behavior.

Contract:
- HaloFit-derived nonlinear spectra are structurally sane, approximately CLASS-consistent, and differentiable with respect to linear inputs.

Scope:
- Covers ``sigma(R)``, HaloFit parameters, nonlinear ``P(k)`` values, and local differentiability checks.
- Excludes linear-spectrum contracts owned by ``test_pk_accuracy.py`` and ``test_pk_gradients.py``.

Notes:
- These tests consume CLASS nonlinear reference data and do not depend on the perturbation solver.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import os
import pytest

from clax.nonlinear import (
    sigma_R,
    halofit_parameters,
    halofit_nl_pk,
    compute_pk_nonlinear,
)

REFERENCE_DIR = os.path.join(os.path.dirname(__file__), '..', 'reference_data')


@pytest.fixture(scope="module")
def pk_nl_ref():
    """Load nonlinear ``P(k)`` CLASS reference data once for this module."""
    path = os.path.join(REFERENCE_DIR, 'lcdm_fiducial', 'pk_nonlinear.npz')
    return dict(np.load(path))


# Fiducial cosmology parameters (Planck 2018)
OMEGA_B = 0.02237
OMEGA_CDM = 0.1200
H = 0.6736
M_NCDM = 0.06

# Derived
OMEGA_B_FRAC = OMEGA_B / H**2
OMEGA_CDM_FRAC = OMEGA_CDM / H**2
OMEGA_M_0 = OMEGA_B_FRAC + OMEGA_CDM_FRAC + 0.00138  # approximate ncdm contribution
OMEGA_LAMBDA_0 = 1.0 - OMEGA_M_0 - 5.4e-5 - 3.7e-5  # 1 - Om - Og - Our (approx)
OMEGA_R_0 = 5.4e-5 + 3.7e-5  # Og + Our (approx)
FNU = 0.00138 / OMEGA_M_0  # Omega_ncdm / Omega_m


class TestSigmaR:
    """Tests ``sigma(R)`` behavior."""

    def test_sigma_decreasing(self, pk_nl_ref):
        """``sigma(R)`` decreases with ``R``; expects monotonic ordering on the probe grid."""
        k = pk_nl_ref['k']
        pk_lin = pk_nl_ref['pk_lin_z0']
        lnk = jnp.log(k)
        pk = jnp.array(pk_lin)

        s1 = float(sigma_R(1.0, lnk, pk))
        s5 = float(sigma_R(5.0, lnk, pk))
        s10 = float(sigma_R(10.0, lnk, pk))

        assert s1 > s5 > s10, f"sigma not decreasing: {s1:.3f}, {s5:.3f}, {s10:.3f}"

    def test_sigma_positive(self, pk_nl_ref):
        """``sigma(R)`` is positive on the probe grid; expects positive values."""
        k = pk_nl_ref['k']
        pk_lin = pk_nl_ref['pk_lin_z0']
        lnk = jnp.log(k)
        pk = jnp.array(pk_lin)

        for R in [0.5, 1.0, 5.0, 10.0, 50.0]:
            s = float(sigma_R(R, lnk, pk))
            assert s > 0, f"sigma(R={R}) = {s}, expected > 0"


class TestHaloFitParameters:
    """Tests HaloFit parameter extraction."""

    def test_k_sigma_range(self, pk_nl_ref):
        """``k_sigma`` and ``n_eff`` have reasonable values; expects both in the documented ranges."""
        k = pk_nl_ref['k']
        pk_lin = pk_nl_ref['pk_lin_z0']
        lnk = jnp.log(k)
        pk = jnp.array(pk_lin)

        k_sigma, n_eff, C = halofit_parameters(lnk, pk)
        k_sigma = float(k_sigma)
        n_eff = float(n_eff)

        assert 0.05 < k_sigma < 1.0, f"k_sigma = {k_sigma:.4f} out of range"
        assert -3.0 < n_eff < 0.0, f"n_eff = {n_eff:.4f} out of range"

    def test_sigma_at_k_sigma_is_one(self, pk_nl_ref):
        """``sigma(1/k_sigma)`` equals unity by construction; expects <1e-4 absolute error."""
        k = pk_nl_ref['k']
        pk_lin = pk_nl_ref['pk_lin_z0']
        lnk = jnp.log(k)
        pk = jnp.array(pk_lin)

        k_sigma, _, _ = halofit_parameters(lnk, pk)
        R_nl = 1.0 / k_sigma
        s = float(sigma_R(R_nl, lnk, pk))

        assert abs(s - 1.0) < 1e-4, f"sigma(1/k_sigma) = {s:.6f}, expected 1.0"


class TestHaloFitPk:
    """Tests nonlinear ``P(k)`` value behavior."""

    def test_pk_nl_z0_low_k(self, pk_nl_ref):
        """Nonlinear ``P(k)`` matches linear ``P(k)`` at low ``k``; expects exact agreement on the low-``k`` mask."""
        k = pk_nl_ref['k']
        pk_lin = pk_nl_ref['pk_lin_z0']

        pk_nl = compute_pk_nonlinear(
            jnp.array(k), jnp.array(pk_lin),
            Omega_m_0=OMEGA_M_0, Omega_lambda_0=OMEGA_LAMBDA_0,
            Omega_r_0=OMEGA_R_0, fnu=FNU, h=H, z=0.0,
        )

        # At k < 0.001, should be identical to linear
        mask = k < 0.001
        np.testing.assert_allclose(
            np.array(pk_nl[mask]), pk_lin[mask], rtol=1e-10,
            err_msg="Non-linear P(k) should equal linear P(k) at very low k"
        )

    def test_pk_nl_z0_enhancement(self, pk_nl_ref):
        """Nonlinear ``P(k)`` exceeds linear ``P(k)`` at high ``k``; expects enhancement at ``k≈1``."""
        k = pk_nl_ref['k']
        pk_lin = pk_nl_ref['pk_lin_z0']

        pk_nl = compute_pk_nonlinear(
            jnp.array(k), jnp.array(pk_lin),
            Omega_m_0=OMEGA_M_0, Omega_lambda_0=OMEGA_LAMBDA_0,
            Omega_r_0=OMEGA_R_0, fnu=FNU, h=H, z=0.0,
        )

        # At k ~ 1 Mpc^-1, should have significant enhancement
        idx_k1 = np.argmin(np.abs(k - 1.0))
        ratio = float(pk_nl[idx_k1]) / pk_lin[idx_k1]
        assert ratio > 2.0, f"P_NL/P_lin at k=1: {ratio:.2f}, expected > 2"

    def test_pk_nl_z0_vs_class(self, pk_nl_ref):
        """Nonlinear ``P(k)`` at ``z=0`` matches CLASS; expects <10% max relative error for ``0.01 < k < 10``."""
        k = pk_nl_ref['k']
        pk_lin = pk_nl_ref['pk_lin_z0']
        pk_class_nl = pk_nl_ref['pk_nl_z0']

        pk_nl = compute_pk_nonlinear(
            jnp.array(k), jnp.array(pk_lin),
            Omega_m_0=OMEGA_M_0, Omega_lambda_0=OMEGA_LAMBDA_0,
            Omega_r_0=OMEGA_R_0, fnu=FNU, h=H, z=0.0,
        )

        # Compare where non-linear effects matter: k > 0.01 and k < 10
        mask = (k > 0.01) & (k < 10.0)
        ratio = np.array(pk_nl[mask]) / pk_class_nl[mask]
        max_err = np.max(np.abs(ratio - 1.0))
        mean_err = np.mean(np.abs(ratio - 1.0))

        # Find worst k
        worst_idx = np.argmax(np.abs(ratio - 1.0))
        k_worst = k[mask][worst_idx]

        assert max_err < 0.10, (
            f"P_NL(k) max error {max_err:.2%} at k={k_worst:.3f} Mpc^-1 "
            f"(mean {mean_err:.2%}), expected < 10%"
        )

    def test_pk_nl_z0_vs_class_quasi_linear(self, pk_nl_ref):
        """Nonlinear ``P(k)`` matches CLASS in the quasi-linear regime; expects <5% max relative error."""
        k = pk_nl_ref['k']
        pk_lin = pk_nl_ref['pk_lin_z0']
        pk_class_nl = pk_nl_ref['pk_nl_z0']

        pk_nl = compute_pk_nonlinear(
            jnp.array(k), jnp.array(pk_lin),
            Omega_m_0=OMEGA_M_0, Omega_lambda_0=OMEGA_LAMBDA_0,
            Omega_r_0=OMEGA_R_0, fnu=FNU, h=H, z=0.0,
        )

        # Quasi-linear regime: 0.01 < k < 0.3
        mask = (k > 0.01) & (k < 0.3)
        ratio = np.array(pk_nl[mask]) / pk_class_nl[mask]
        max_err = np.max(np.abs(ratio - 1.0))

        assert max_err < 0.05, (
            f"Quasi-linear P_NL(k) max error {max_err:.2%}, expected < 5%"
        )


class TestHaloFitDifferentiable:
    """Tests HaloFit differentiability and JIT behavior."""

    def test_grad_pk_nl_wrt_pk_lin(self, pk_nl_ref):
        """The HaloFit map is differentiable in ``P_lin``; expects finite non-zero gradients."""
        k_test = jnp.array([0.01, 0.1, 1.0, 10.0])
        # Create a simple power-law linear P(k)
        A_s = 2.1e-9
        n_s = 0.9649
        pk_test = A_s * (k_test / 0.05) ** (n_s - 1) * 2 * jnp.pi**2 / k_test**3 * 1e3

        def pk_nl_sum(pk_lin):
            return jnp.sum(compute_pk_nonlinear(
                k_test, pk_lin,
                Omega_m_0=0.31, Omega_lambda_0=0.69,
                Omega_r_0=9e-5, fnu=0.004, h=0.6736, z=0.0,
            ))

        grad = jax.grad(pk_nl_sum)(pk_test)
        assert jnp.all(jnp.isfinite(grad)), "Gradient contains NaN/Inf"
        assert jnp.any(grad != 0.0), "Gradient is all zeros"

    def test_jit_compatible(self, pk_nl_ref):
        """HaloFit is JIT-compatible; expects finite outputs under ``jax.jit``."""
        k = jnp.array(pk_nl_ref['k'])
        pk_lin = jnp.array(pk_nl_ref['pk_lin_z0'])

        @jax.jit
        def compute_nl(pk_lin):
            return compute_pk_nonlinear(
                k, pk_lin,
                Omega_m_0=OMEGA_M_0, Omega_lambda_0=OMEGA_LAMBDA_0,
                Omega_r_0=OMEGA_R_0, fnu=FNU, h=H, z=0.0,
            )

        pk_nl = compute_nl(pk_lin)
        assert jnp.all(jnp.isfinite(pk_nl)), "JIT result contains NaN/Inf"


# halofit_parameters(ln k, P_lin) = (k_sigma [Mpc^-1], n_eff, C) on the CLASS linear P(k) at
# z = 0, reference_data/<name>/pk.npz["pk_lin_z0"], from the bisection as it was before the
# implicit-function tangent (clax/nonlinear.py before #54, main 5ac3df3; CPU, eager, job 28763). The
# tangent leaves these values bit-identical on CPU (smsharma/clax#54).
HALOFIT_PARAMS_REF = {
    "lcdm_fiducial": (0.24570584575996515, -1.7411794478559754, 0.3411749812387166),
    "h_high": (0.26385384279187535, -1.7651881724348482, 0.3329749020911661),
    "omega_b_high": (0.26040864431932365, -1.7657158999064142, 0.33064248735323254),
    "omega_cdm_low": (0.32687007871620355, -1.9042101163246437, 0.282806437856729),
    "ns_high": (0.23655701493402806, -1.6908435744495975, 0.3462992005355152),
}
HALOFIT_QUANTITIES = ("k_sigma", "n_eff", "C")


def _class_pk_lin_z0(name):
    """(ln k, P_lin(k, z=0)) from reference_data/<name>/pk.npz (CLASS; k in Mpc^-1, P in Mpc^3)."""
    d = np.load(os.path.join(REFERENCE_DIR, name, "pk.npz"))
    return jnp.log(jnp.asarray(d["k"])), jnp.asarray(d["pk_lin_z0"])


class TestHaloFitTangent:
    """The nonlinear scale's derivative (smsharma/clax#54).

    halofit_parameters finds R_nl, sigma(R_nl) = 1, by a 60-step bisection whose
    jnp.where updates carry no derivative, so AD through it returned
    d k_sigma / d theta = 0 exactly (and n_eff, C to 1e-16), and every Halofit gradient
    missed the response of the nonlinear scale: d sum ln P_NL / dA was 44% low for
    P_lin -> A P_lin (job 28762). The root now carries the implicit-function tangent
    (clax.nonlinear._log10_r_nl). Checked on the CLASS P_lin at each COSMOLOGY_GRID_LCDM
    point (lcdm_cosmology; fiducial only under --fast).

    The tangent is checked along both inputs of the root: the amplitude A of P_lin
    at A = 1 (i.e. A_s d/dA_s) and a shift s of ln k at s = 0. The finite
    difference is central with step 1e-2 (in A, or in ln k). The bisection stops
    once |sigma - 1| <= 1e-6, so k_sigma carries a ~1e-6 relative floor, and the FD
    inherits ~2e-6 / step of it (4.5e-4 at step 1e-3, 8.5e-5 at 1e-2, job 28762).
    The FD's truncation error at 1e-2 is ~1e-4."""

    def test_primal_unchanged(self, lcdm_cosmology):
        """(k_sigma, n_eff, C) equal the stored pre-tangent bisection values to 1e-12
        (bit-identical on CPU; the bound leaves room for a few ulp of reordered
        trapezoid sums on another backend)."""
        name, _ = lcdm_cosmology
        lnk, pk = _class_pk_lin_z0(name)
        got = np.array([float(x) for x in halofit_parameters(lnk, pk)])
        ref = np.array(HALOFIT_PARAMS_REF[name])
        rel = np.abs(got / ref - 1.0)
        assert np.all(rel <= 1e-12), (
            f"ERROR halofit_parameters changed at {name}: max rel {rel.max():.2e} on "
            f"{HALOFIT_QUANTITIES[int(np.argmax(rel))]} (got {got.tolist()}, stored {ref.tolist()})")

    # Tangent channels of the root's inputs (ln k, P_lin), each as (t0, move(lnk, pk, t)):
    # "amplitude" P_lin -> A P_lin at A = 1; "lnk_shift" ln k -> ln k + s at s = 0, the
    # channel a caller whose k-grid depends on cosmology (k = k_h h) would exercise.
    CHANNELS = {
        "amplitude": (1.0, lambda lnk, pk, t: (lnk, t * pk)),
        "lnk_shift": (0.0, lambda lnk, pk, t: (lnk + t, pk)),
    }

    @pytest.mark.parametrize("channel", list(CHANNELS))
    @pytest.mark.parametrize("quantity", HALOFIT_QUANTITIES)
    def test_tangent_matches_fd(self, lcdm_cosmology, quantity, channel):
        """d quantity / dt along each channel by jax.jvp and by jax.grad (both jitted)
        within 1e-3 of the central FD (step 1e-2 in A, or in ln k)."""
        name, _ = lcdm_cosmology
        lnk, pk = _class_pk_lin_z0(name)
        i = HALOFIT_QUANTITIES.index(quantity)
        t0, move = self.CHANNELS[channel]

        def F(t):
            return halofit_parameters(*move(lnk, pk, t))[i]

        x0, one = jnp.asarray(t0), jnp.asarray(1.0)
        jv = float(jax.jit(lambda t: jax.jvp(F, (t,), (one,))[1])(x0))
        gr = float(jax.jit(jax.grad(F))(x0))
        Fj, eps = jax.jit(F), 1e-2
        fd = (float(Fj(x0 + eps)) - float(Fj(x0 - eps))) / (2.0 * eps)
        rel_jvp, rel_grad = abs(jv / fd - 1.0), abs(gr / fd - 1.0)
        print(f"halofit tangent {name} {channel} d{quantity}/dt: jvp {jv:.8e} grad {gr:.8e} "
              f"fd {fd:.8e} jvp/fd-1 {rel_jvp:.2e} grad/fd-1 {rel_grad:.2e}")
        assert np.isfinite(jv) and np.isfinite(gr), (
            f"ERROR non-finite d{quantity}/dt ({channel}) at {name}")
        assert rel_jvp <= 1e-3, (f"ERROR jvp d{quantity}/dt ({channel}) {jv:.6e} vs FD "
                                 f"{fd:.6e}: rel {rel_jvp:.2e} > 1e-3 at {name}")
        assert rel_grad <= 1e-3, (f"ERROR grad d{quantity}/dt ({channel}) {gr:.6e} vs FD "
                                  f"{fd:.6e}: rel {rel_grad:.2e} > 1e-3 at {name}")
