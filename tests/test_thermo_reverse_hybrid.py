"""Reverse-vs-forward consistency for ``thermodynamics_solve`` (issue #30).

Contract:
- With the default ``th_grad_mode="stable"`` (the hybrid custom VJP:
  CosmoParams cotangent via a batched forward jacfwd basis, BackgroundResult
  cotangent via the native reverse rule), ``jax.grad`` of scalar functionals
  of ``thermodynamics_solve`` agrees with forward-mode ``jax.jvp`` --
  the six-way-verified exact reference for this solver (issue #30).
- Primal values are bit-identical between the "stable" and "native" modes.
- ``jax.jvp`` through the "stable" mode raises ``TypeError`` (JAX blocks
  forward mode across custom_vjp), and works through "native".

Why these functionals/directions (CPU-measured on the pre-fix native rule,
th_n_points=3000, bg_n_points=400):
- ``omega_b`` and ``h``-with-live-bg have healthy-sized true derivatives
  (+7.018e3 and -5.770e2 for sum(xe.y^2)); the stable rule matches jvp to
  2e-15 and 5e-11 respectively -> asserted at <1e-6.
- ``h`` with bg PINNED is a near-null direction for the ionization tables
  (n_H_0 ~ omega_b exactly: the h in rho_crit cancels against Omega_b =
  omega_b/h^2), so the ones-cotangent functional's true d/dh is only
  +5.71e-9 -- and there the NATIVE reverse rule returns -2.72e-7: 48x too
  large with the WRONG SIGN even on CPU (the issue #30 ULP-cancellation
  disease; on V100 the same mechanism produced exact ULP quanta like 2^-9
  against ~1e13-scale intermediates). The stable rule lands on the jvp
  value to 1.3e-3 relative, which is the forward-mode vmap-vs-single
  rounding floor of this near-null direction, not a defect -> asserted at
  <1e-2 (native fails at ~48).

The slow test locks the pipeline-level acceptance number: grad-vs-jvp of
sum(pt.delta_m[:, -1]**2) at fast_cl(pt_k_max_cl=5, chunk 20), previously
+4.1e-3, must be <1e-4 (GPU-scale run; measured target <1e-5).
"""

import dataclasses

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve
from clax.params import CosmoParams, PrecisionParams


BASE = CosmoParams()

# ode_adjoint="direct" in BOTH arms so the background discretization is
# identical and any grad-vs-jvp gap isolates thermodynamics_solve's reverse
# rule (reverse mode through DirectAdjoint is supported; forward mode
# through RecursiveCheckpointAdjoint is not).
PREC_STABLE = PrecisionParams(
    bg_n_points=400, ncdm_bg_n_points=200, bg_tol=1e-8,
    th_n_points=3000, th_z_max=5e4,  # 5e4 floor: see PrecisionParams.th_z_max
    ode_adjoint="direct",
)  # th_grad_mode="stable" is the production default
PREC_NATIVE = dataclasses.replace(PREC_STABLE, th_grad_mode="native")


@pytest.fixture(scope="module")
def bg0():
    """Fiducial background, computed once and treated as a pinned constant."""
    bg = background_solve(BASE, PREC_NATIVE)
    jax.block_until_ready(bg.conformal_age)
    return bg


def _f_xe2(th):
    """Quadratic in the ionization table: the cotangent is the table itself,
    concentrating weight in the recombination era where the issue #30
    reverse-mode disease was localized."""
    return jnp.sum(th.xe_of_loga.y ** 2)


def _f_all(th):
    """Ones-cotangent over every ThermoResult leaf (the aggregate of the
    per-leaf diagnostic probes from the issue #30 investigation)."""
    return sum(jnp.sum(leaf) for leaf in jtu.tree_leaves(th))


def _make_f(prec, pname, functional, bg_pinned=None):
    def f(v):
        p = dataclasses.replace(BASE, **{pname: v})
        b = bg_pinned if bg_pinned is not None else background_solve(p, prec)
        return functional(thermodynamics_solve(p, prec, b))
    return f


def _grad_stable_vs_jvp(pname, functional, bg_pinned=None):
    """(grad under stable rule, jvp under native rule) for d/d pname."""
    v0 = jnp.asarray(float(getattr(BASE, pname)))
    _, tan = jax.jvp(_make_f(PREC_NATIVE, pname, functional, bg_pinned),
                     (v0,), (jnp.asarray(1.0),))
    g = jax.grad(_make_f(PREC_STABLE, pname, functional, bg_pinned))(v0)
    return float(g), float(tan)


class TestStablePrimalUnchanged:
    def test_primal_bit_identical_stable_vs_native(self, bg0):
        """Both modes call the identical solver body; every output leaf must
        agree exactly (the custom rule may not perturb the primal)."""
        th_s = thermodynamics_solve(BASE, PREC_STABLE, bg0)
        th_n = thermodynamics_solve(BASE, PREC_NATIVE, bg0)
        for (path, a), b in zip(jtu.tree_flatten_with_path(th_s)[0],
                                jtu.tree_leaves(th_n)):
            diff = float(jnp.max(jnp.abs(a - b)))
            assert diff == 0.0, (
                f"primal leaf {jtu.keystr(path)} differs stable-vs-native: "
                f"max abs diff {diff:.3e}")


class TestGradMatchesJvp:
    """Reverse (stable) vs forward (native, exact) consistency."""

    @pytest.mark.parametrize("pname", ["h", "omega_b"])
    def test_live_bg_xe2(self, pname):
        """Full chain params -> (bg, th): grad must match jvp to <1e-6.

        RED on the native rule for the pipeline (issue #30: +4.1e-3 at the
        delta_m level); at this thermo-level functional the CPU-visible
        defect sits in the pinned-bg arm below, while this arm locks the
        end-to-end contract the acceptance criteria name.
        """
        g, tan = _grad_stable_vs_jvp(pname, _f_xe2)
        rel = abs(g - tan) / max(abs(tan), 1e-30)
        assert rel < 1e-6, (
            f"d(sum xe^2)/d{pname} grad-vs-jvp rel={rel:.2e} "
            f"(grad={g:.10e}, jvp={tan:.10e}, expected <1e-6)")

    def test_pinned_bg_omega_b_xe2(self, bg0):
        """Thermo-only omega_b direction (the n_H_0 funnel): <1e-6.

        Measured on CPU: stable rel 2.3e-15 against jvp +7.0177520480e3.
        """
        g, tan = _grad_stable_vs_jvp("omega_b", _f_xe2, bg0)
        rel = abs(g - tan) / max(abs(tan), 1e-30)
        assert rel < 1e-6, (
            f"d(sum xe^2)/d omega_b (bg pinned) grad-vs-jvp rel={rel:.2e} "
            f"(grad={g:.10e}, jvp={tan:.10e}, expected <1e-6)")

    def test_pinned_bg_h_ones_cotangent(self, bg0):
        """The CPU-reproducible issue #30 disease case.

        h with bg pinned is a near-null direction (true d/dh = +5.71e-9 for
        the ones-cotangent functional); the NATIVE reverse rule returns
        -2.72e-7 here -- 48x too large, wrong sign (RED before the fix).
        The stable rule must land on the jvp value to <1e-2, the measured
        forward-mode rounding floor of this near-null direction (1.3e-3).
        """
        g, tan = _grad_stable_vs_jvp("h", _f_all, bg0)
        rel = abs(g - tan) / max(abs(tan), 1e-30)
        assert rel < 1e-2, (
            f"d(sum all leaves)/dh (bg pinned) grad-vs-jvp rel={rel:.2e} "
            f"(grad={g:.10e}, jvp={tan:.10e}, expected <1e-2; the native "
            f"rule fails this at ~48 with the wrong sign)")


class TestModeWiring:
    def test_jvp_through_stable_raises(self, bg0):
        """custom_vjp blocks forward mode by JAX design; the flag is the
        documented escape hatch (see PrecisionParams.th_grad_mode)."""
        with pytest.raises(TypeError):
            jax.jvp(
                lambda v: _f_xe2(thermodynamics_solve(
                    dataclasses.replace(BASE, h=v), PREC_STABLE, bg0)),
                (jnp.asarray(float(BASE.h)),), (jnp.asarray(1.0),))

    def test_jvp_through_native_works(self, bg0):
        _, tan = jax.jvp(
            lambda v: _f_xe2(thermodynamics_solve(
                dataclasses.replace(BASE, h=v), PREC_NATIVE, bg0)),
            (jnp.asarray(float(BASE.h)),), (jnp.asarray(1.0),))
        assert jnp.isfinite(tan)

    def test_invalid_mode_raises(self, bg0):
        with pytest.raises(ValueError, match="th_grad_mode"):
            thermodynamics_solve(
                BASE, dataclasses.replace(PREC_STABLE, th_grad_mode="bogus"),
                bg0)


@pytest.mark.slow
def test_pipeline_delta_m_grad_matches_jvp():
    """Pipeline-level acceptance (GPU-scale): grad-vs-jvp of
    sum(pt.delta_m[:, -1]**2) at fast_cl(pt_k_max_cl=5, chunk 20) < 1e-4.

    Was +4.1e-3 on the native rule (jvp=-7.96748395e10 vs buggy
    grad=-7.93444874e10, issue #30); the stable rule must close it (the
    freeze-th diagnostic bound the reachable gap at +6.8e-7).
    """
    from clax.perturbations import perturbations_solve

    prec_grad = dataclasses.replace(
        PrecisionParams.fast_cl(), pt_k_max_cl=5.0, pt_k_chunk_size=20)
    prec_jvp = dataclasses.replace(
        prec_grad, th_grad_mode="native", ode_adjoint="direct",
        ode_max_steps=16384)

    def make_g(prec):
        def g(h_val):
            p = dataclasses.replace(BASE, h=h_val)
            bg = background_solve(p, prec)
            th = thermodynamics_solve(p, prec, bg)
            pt = perturbations_solve(p, prec, bg, th)
            return jnp.sum(pt.delta_m[:, -1] ** 2)
        return g

    h0 = jnp.asarray(float(BASE.h))
    _, tan = jax.jvp(make_g(prec_jvp), (h0,), (jnp.asarray(1.0),))
    g = jax.grad(make_g(prec_grad))(h0)
    g, tan = float(g), float(tan)
    rel = abs(g - tan) / max(abs(tan), 1e-30)
    print(f"\ndelta_m pipeline: grad_stable={g:.8e} jvp={tan:.8e} rel={rel:.3e}")
    assert rel < 1e-4, (
        f"pipeline grad-vs-jvp gap {rel:.3e} (grad={g:.8e}, jvp={tan:.8e}, "
        f"expected <1e-4; native rule sat at +4.1e-3)")
