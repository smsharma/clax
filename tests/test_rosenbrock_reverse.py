"""Reverse-mode (jax.grad) through Rodas5 / Rodas5Batched solves run under jax.vmap.

Physics consequence of the bug (found while making C_l differentiable, smsharma/clax#54;
GPU job 26946): at ``PrecisionParams.fit_cl()`` (Rodas5) ``jax.grad`` of any C_l or
source-function functional was NaN in every cosmological parameter, so gradient-based
samplers and fitters could not use the production preset.  Forward mode (``jax.jvp``)
was fine.

Mechanism.  ``perturbations_solve`` integrates the k-modes with ``jax.vmap`` over one
single-mode Rodas5 solve.  Under vmap, diffrax's step loop keeps running until the
slowest k-mode reaches the end time; a mode that has already finished keeps executing
masked steps with step size dt = 0.  Rodas5 used to form W = I/(dt*gamma) - J and stage
terms C_ij/dt, which are infinite at dt = 0.  The forward pass throws those steps away
(the batched step loop discards them: a select on the per-member loop predicate), but
the backward pass multiplies the zero cotangent of a discarded step by inf, and
0 * inf = NaN spreads to the whole gradient.  Ruled out by job 26946: not max_steps, not
the tolerance, not the adjoint; Kvaerno5 and
``pt_k_chunk_size=1`` (no vmap) are finite.  The fix divides by
``dt_div = where(dt == 0, 1, dt)`` instead of dt (clax/rosenbrock.py module docstring):
the same number on every real step, finite on a masked one.

Toy tests (CPU, fast): vmapped solves of a stiff oscillator whose slow component is
exactly cos(omega t), so the derivative has a closed form, and a single masked dt = 0
step on a stiff linear system.  Exempt from the multi-cosmology rule: no cosmology
enters.

Pipeline tests (slow, GPU preferred; CPU works but takes much longer): d/dh of the
log matter power at fit_cl by reverse vs forward mode, at the 3
``recomb_downstream_cosmology`` points (tests/conftest.py).  The production reverse
configuration is tested twice: with diffrax's direct adjoint (hard gate) and with the
production recursive-checkpoint adjoint (strict xfail at omega_b_high, a pre-existing
adjoint defect the NaN hid; see that test).
"""
import functools
from dataclasses import replace as dc_replace

import diffrax
import jax
import jax.numpy as jnp
import pytest

from clax import PrecisionParams
from clax.perturbations import perturbations_solve
from clax.rosenbrock import Rodas5, Rodas5Batched
from clax.thermodynamics import solve_background_and_thermo

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Toy problem: the stiff oscillator of tests/test_rosenbrock.py
# ---------------------------------------------------------------------------

def _stiff_oscillator_rhs(t, y, p):
    """u' = omega v, v' = -omega u, w' = -lam (w - u); u = cos(omega t) exactly
    for u(0) = 1, v(0) = 0, so sum_i u(t_i) has d/d omega = -sum_i t_i sin(omega t_i)."""
    omega, lam = p[0], p[1]
    return jnp.array([omega * y[1], -omega * y[0], -lam * (y[2] - y[0])])


def _y_init(omega, lam):
    # w starts one unit off its slow manifold: a genuine stiff transient
    return jnp.array([1.0, 0.0, lam * lam / (lam**2 + omega**2) + 1.0])


T1 = 3.0
TS = jnp.linspace(0.0, T1, 301)
RTOL, ATOL = 1e-6, 1e-9


def _exact_d_ds(omega_base):
    """d/ds of sum_i cos(s * omega_base * t_i) at s = 1, summed over the members."""
    return float(jnp.sum(-omega_base[..., None] * TS * jnp.sin(omega_base[..., None] * TS)))


class TestVmappedReverseMode:
    """jax.grad through jax.vmap of Rodas5 / Rodas5Batched solves whose members
    finish after different numbers of steps (so the early finishers run masked
    dt = 0 steps).  Before the 1/dt guard the gradient was NaN."""

    # (omega, lam) per member; the existing vmap carry test in test_rosenbrock.py
    # checks that this set takes different step counts per member.
    OMEGAS = jnp.array([2.0 * jnp.pi, 3.0 * jnp.pi, 1.0 * jnp.pi])
    LAMS = jnp.array([1000.0, 300.0, 3000.0])

    def _solve_single(self, omega, lam):
        p = jnp.array([omega, lam])
        return diffrax.diffeqsolve(
            diffrax.ODETerm(_stiff_oscillator_rhs), Rodas5(), t0=0.0, t1=T1, dt0=1e-3,
            y0=_y_init(omega, lam), args=p,
            stepsize_controller=diffrax.PIDController(rtol=RTOL, atol=ATOL),
            max_steps=100000, saveat=diffrax.SaveAt(ts=TS),
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
        )

    def test_vmapped_rodas5_grad_finite_and_exact(self):
        """Rodas5 vmapped over 3 members: d/ds of sum_members sum_i u(t_i) with
        omega = s * omega_member, against the closed form.  Bar 1e-3, as for the
        unvmapped dense-output AD test (measured 4.9e-6 there)."""
        n_steps = jax.vmap(lambda om, lam: self._solve_single(om, lam).stats["num_steps"])(
            self.OMEGAS, self.LAMS)
        assert len(set(int(n) for n in n_steps)) > 1, (
            f"members must finish after different step counts to exercise the "
            f"masked dt = 0 steps; got {n_steps}")

        def objective(s):
            ys = jax.vmap(lambda om, lam: self._solve_single(s * om, lam).ys)(self.OMEGAS, self.LAMS)
            return jnp.sum(ys[:, :, 0])

        g = float(jax.grad(objective)(1.0))
        exact = _exact_d_ds(self.OMEGAS)
        rel = abs(g / exact - 1.0)
        print(f"vmapped Rodas5 (steps {[int(n) for n in n_steps]}): d/ds AD {g:.8e} "
              f"exact {exact:.8e} rel {rel:.2e}")
        assert jnp.isfinite(g), f"ERROR vmapped Rodas5 reverse-mode gradient is {g}"
        assert rel < 1e-3, f"ERROR vmapped Rodas5 d/ds rel err {rel:.3e}"

    def test_vmapped_rodas5_batched_grad_finite_and_exact(self):
        """Rodas5Batched (shared step size within a batch) vmapped over 2 batches
        of 2 modes, e.g. a vmap over cosmologies of the batched matter solve.
        The batches take different step counts; same closed-form check."""
        omegas = jnp.array([[2.0 * jnp.pi, 3.0 * jnp.pi], [1.0 * jnp.pi, 0.5 * jnp.pi]])
        lams = jnp.array([[1000.0, 300.0], [3000.0, 100.0]])

        def vf_batched(t, y, args):
            # diffrax's term check calls vf once with the full (f, data) tuple;
            # stepping passes only the per-mode data (cf. tests/test_rosenbrock.py).
            if isinstance(args, tuple) and len(args) == 2:
                return jnp.zeros_like(y)
            return jax.vmap(_stiff_oscillator_rhs, in_axes=(None, 0, 0))(t, y, args)

        def solve_batch(om, la):
            p = jnp.stack([om, la], axis=-1)            # (2 modes, 2)
            y0 = jax.vmap(_y_init)(om, la)               # (2 modes, 3)
            return diffrax.diffeqsolve(
                diffrax.ODETerm(vf_batched), Rodas5Batched(), t0=0.0, t1=T1, dt0=1e-3,
                y0=y0, args=(_stiff_oscillator_rhs, p),
                stepsize_controller=diffrax.PIDController(rtol=RTOL, atol=ATOL),
                max_steps=100000, saveat=diffrax.SaveAt(ts=TS),
                adjoint=diffrax.RecursiveCheckpointAdjoint(),
            )

        n_steps = jax.vmap(lambda om, lam: solve_batch(om, lam).stats["num_steps"])(omegas, lams)
        assert int(n_steps[0]) != int(n_steps[1]), (
            f"batches must finish after different step counts; got {n_steps}")

        def objective(s):
            ys = jax.vmap(lambda om, lam: solve_batch(s * om, lam).ys)(omegas, lams)
            return jnp.sum(ys[..., 0])                   # (batch, ts, mode, comp)

        g = float(jax.grad(objective)(1.0))
        exact = _exact_d_ds(omegas)
        rel = abs(g / exact - 1.0)
        print(f"vmapped Rodas5Batched (steps {[int(n) for n in n_steps]}): d/ds AD {g:.8e} "
              f"exact {exact:.8e} rel {rel:.2e}")
        assert jnp.isfinite(g), f"ERROR vmapped Rodas5Batched reverse-mode gradient is {g}"
        assert rel < 1e-3, f"ERROR vmapped Rodas5Batched d/ds rel err {rel:.3e}"


def _stiff_linear_rhs(t, y, p):
    """y' = p * A y with eigenvalues of A at -1e3, -1e4, -1e5, -1e6 (the stiffness
    range of the perturbation hierarchy) plus weak couplings."""
    A = (jnp.diag(jnp.array([-1e3, -1e4, -1e5, -1e6]))
         + jnp.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]]))
    return p * (A @ y)


@pytest.mark.parametrize("batched", [False, True], ids=["Rodas5", "Rodas5Batched"])
def test_masked_zero_step_stiff_jacobian_finite_cotangent(batched):
    """One solver step with t0 == t1 (dt = 0), as a vmapped member that has
    already finished executes, on a stiff linear system.  The batched step loop
    discards the step (a select on the per-member loop predicate, modelled here as
    ``jnp.where(False, y1, y0)``), so the exact gradient of the kept state w.r.t. the
    RHS parameter is 0; reverse mode must return exactly 0, not NaN (0 * inf before the
    fix)."""
    y_init = jnp.array([1.0, -0.5, 0.25, 2.0])
    t = jnp.asarray(7.0)    # an array, as diffrax passes times (a float would raise in Python)

    def vf_batched(t, y, args):
        # diffrax-style batched field (cf. the vmapped Rodas5Batched test above)
        if isinstance(args, tuple) and len(args) == 2:
            return jnp.zeros_like(y)
        return jax.vmap(_stiff_linear_rhs, in_axes=(None, 0, 0))(t, y, args)

    def kept_state(p):
        if batched:
            solver, term = Rodas5Batched(), diffrax.ODETerm(vf_batched)
            y0 = jnp.stack([y_init, 2.0 * y_init])
            args = (_stiff_linear_rhs, jnp.stack([p, p]))
        else:
            solver, term = Rodas5(), diffrax.ODETerm(_stiff_linear_rhs)
            y0, args = y_init, p
        state = solver.init(term, t, t, y0, args)
        y1 = solver.step(term, t, t, y0, args, state, jnp.array(False))[0]
        return jnp.sum(jnp.where(jnp.array(False), y1, y0))

    g = float(jax.grad(kept_state)(jnp.asarray(1.0)))
    assert g == 0.0, f"ERROR masked dt = 0 step, stiff J (|lambda| 1e3-1e6): gradient {g}, exact 0"


# ---------------------------------------------------------------------------
# Pipeline: fit_cl log matter power, reverse vs forward mode
# ---------------------------------------------------------------------------
#
# Functional: F = sum_k ln delta_m(k, tau_end)^2, delta_m from perturbations_solve at
# the last time of its tau grid (0.999 tau_0): F ~ 828, dF/dh ~ -136, so d ln F/dh is
# O(1).  The first version used sum(source_T0**2), F ~ 19.7 with dF/dh ~ 5e-5, so its
# logarithmic derivative d ln F/dh ~ 3e-6 is tiny: a relative bar on it turns a
# 1e-6-of-F absolute error into an O(1) miss.
# The O(1) misses seen with it (V100 job 27644: reverse 7.39e-5 / 4.02e-5 / 5.17e-5
# against forward 5.41e-5 / 6.58e-5 / 5.05e-5 at the 3 points, |reverse - forward| / F =
# 1e-6, 1.2e-6, 6e-8) were the recursive-checkpoint adjoint defect of the production test
# below, amplified by the small derivative, not path noise: on the same functional at
# fiducial the direct adjoint matched forward mode to 7.4e-5 while the recursive one was
# 22% off (job 27675, division-free form: reverse 5.41292e-5 direct, 6.58e-5 recursive,
# forward 5.41332e-5).

_FIT = PrecisionParams.fit_cl()
# fwd(FIT): jax.jvp needs the native (non-custom_vjp) thermodynamics and diffrax's
# direct adjoint.
_FIT_FWD = dc_replace(_FIT, th_grad_mode="native", ode_adjoint="direct")
# The same with the ESDIRK solver: Kvaerno5's Newton iterations converge each stage, so
# the primals of the jvp and grad programs agree to ~1e-9 (job 27726).
_FIT_FWD_K5 = dc_replace(_FIT_FWD, pt_ode_solver="kvaerno5")
# Production thermodynamics gradient mode with diffrax's direct adjoint: the production
# reverse configuration except for the ODE adjoint (the two production tests below).
_FIT_DIRECT = dc_replace(_FIT, ode_adjoint="direct")
# Kvaerno5 same-settings bar: 2x the worst |grad/jvp - 1| over the 3 points, rounded up
# to 2 significant figures.  Measured floor (V100): 1.610e-6 / 6.230e-6 / 7.229e-6
# (fiducial / omega_b_high / YHe_028; job 27726, the same on main, job 27725), with the
# reverse-run and forward-run primals 1.5e-9 to 3.1e-9 apart.  With the perturbation
# tolerances 100x tighter (rtol 1e-6, atol 1e-8): 1.558e-6 / 2.739e-6 / 1.492e-6, primals
# 1.1e-9 to 1.8e-8 apart (job 27786).  The floor moves by a factor of 1 to 5, not ~100,
# so it does not follow the ODE tolerance; its origin is not established.  With the
# background and thermodynamics held fixed, Kvaerno5's perturbation solve alone is a
# transpose to 6e-15 with bitwise-equal primals (d/dw0, d/dwa, d/dcs2_fld at w0 = -0.9,
# wa = 0.1; job 27848), which points to the stages upstream of it.  Candidates not yet
# separated: AD through the background and thermodynamics stages (whose settings job
# 27786 did not change), and any accept/reject decisions or iteration counts there that
# differ between the two compiled programs.  Still 20x tighter than the 3e-4
# reverse-vs-forward bar used for #54.
_BAR_ADJOINT_TRANSPOSE_K5 = 1.5e-5


def _ln_power_functional(p, prec):
    bg, th = solve_background_and_thermo(p, prec)
    delta_m = perturbations_solve(p, prec, bg, th).delta_m[:, -1]
    return jnp.sum(jnp.log(delta_m ** 2))


# One compilation per precision preset, shared by the cosmologies (traced values).
@functools.partial(jax.jit, static_argnums=1)
def _value_and_grad_h(p, prec):
    return jax.value_and_grad(lambda h: _ln_power_functional(p.replace(h=h), prec))(p.h)


@functools.partial(jax.jit, static_argnums=1)
def _jvp_h(p, prec):
    return jax.jvp(lambda h: _ln_power_functional(p.replace(h=h), prec),
                   (p.h,), (jnp.ones_like(p.h),))


def _reverse_vs_forward(cosmology, prec_reverse, prec_forward):
    """(name, reverse dF/dh, forward dF/dh, |rev/fwd - 1|); prints both primals too."""
    name, params = cosmology
    p = jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float64), params)
    F_rev, g = (float(x) for x in _value_and_grad_h(p, prec_reverse))
    F_fwd, t = (float(x) for x in _jvp_h(p, prec_forward))
    rel = abs(g - t) / max(abs(t), 1e-300)
    print(f"{name}: F reverse-run {F_rev:.12e} forward-run {F_fwd:.12e} (rel diff "
          f"{abs(F_rev - F_fwd) / abs(F_fwd):.3e}); dF/dh reverse {g:.12e} forward {t:.12e}; "
          f"|rev-fwd| {abs(g - t):.3e}, rel {rel:.3e}")
    return name, g, t, rel


@pytest.mark.slow
def test_fit_cl_grad_equals_jvp_same_settings_kvaerno5(recomb_downstream_cosmology):
    """The adjoint is the transpose of the forward pass: jax.grad and jax.jvp of F
    both under fwd(FIT) with Kvaerno5, to the measured floor (bar comment; its origin
    is not established).
    A control, not red-able: Kvaerno5 has no 1/dt, so it passes on main by
    construction (job 27725 measured the same values there as job 27726 here)."""
    name, g, t, rel = _reverse_vs_forward(recomb_downstream_cosmology, _FIT_FWD_K5, _FIT_FWD_K5)
    assert jnp.isfinite(g), f"ERROR {name}: Kvaerno5 same-settings reverse d/dh is {g}"
    assert rel < _BAR_ADJOINT_TRANSPOSE_K5, (
        f"ERROR {name}: Kvaerno5 same-settings reverse {g:.10e} vs forward {t:.10e}, rel {rel:.3e}")


@pytest.mark.slow
def test_fit_cl_grad_equals_jvp_same_settings(recomb_downstream_cosmology):
    """jax.grad and jax.jvp of F both under fwd(FIT) with Rodas5: finite (NaN before
    the fix, jobs 26946, 27632) and equal to 3e-4 (the reverse-vs-forward bar used for
    #54).

    Not a transpose test at the 1e-6 level.  Rodas5 is linearly implicit (one LU solve
    per step, no Newton iteration to contract roundoff), so the different instruction
    order of the compiled jvp and grad programs reaches the step accept/reject
    decisions and the two programs integrate on different step sequences.  Job 27675
    (on the division-free form of the stage system; old functional sum(source_T0**2),
    fiducial, these settings): the two primals are 4.7e-6 apart (F 19.743101871 vs
    19.743195578) and dF/d omega_b differs by 0.36%; an exact algebraic rewrite of the
    stage system changed the step sequence of 31 of 100 k-modes (job 27641); the
    Kvaerno5 control's primals agree to 2.5e-11 (job 27676).  On this tree (the 1/dt
    guard) and the log matter power the primals are 2e-8 to 5e-8 apart (job 27788).
    This is a solver property (follow-up issue), not an adjoint defect."""
    name, g, t, rel = _reverse_vs_forward(recomb_downstream_cosmology, _FIT_FWD, _FIT_FWD)
    assert jnp.isfinite(g), f"ERROR {name}: same-settings reverse d/dh is {g} (forward {t:.6e})"
    assert rel < 3e-4, (
        f"ERROR {name}: same-settings reverse {g:.10e} vs forward {t:.10e}, rel {rel:.3e}")


@pytest.mark.slow
def test_fit_cl_production_thermo_direct_adjoint_matches_forward(recomb_downstream_cosmology):
    """jax.grad of F under fit_cl with the production thermodynamics gradient mode
    ("stable") and diffrax's direct adjoint vs jax.jvp under fwd(FIT); bar 3e-4
    (the reverse-vs-forward bar used for #54).  The production reverse configuration
    except for the ODE adjoint (see the next test).  On the guard tree before #59 it
    measured 1.146e-5 (fiducial) and 3.595e-6 (omega_b_high), job 27787."""
    name, g, t, rel = _reverse_vs_forward(recomb_downstream_cosmology, _FIT_DIRECT, _FIT_FWD)
    assert jnp.isfinite(g), f"ERROR {name}: direct-adjoint reverse d/dh is {g} (forward {t:.6e})"
    assert rel < 3e-4, (
        f"ERROR {name}: direct-adjoint reverse {g:.10e} vs forward {t:.10e}, rel {rel:.3e}")


_XFAIL_RECURSIVE_ADJOINT = (
    "RecursiveCheckpointAdjoint through Rodas5 at fit_cl differs from the direct adjoint "
    "and from forward mode by 2.2e-4 (fiducial) to 1.3e-2 (omega_b_high), job 27787; "
    "pre-existing, hidden by the NaN; issue to be filed")


@pytest.mark.slow
def test_fit_cl_production_reverse_matches_forward(recomb_downstream_cosmology, request):
    """jax.grad of F under production fit_cl (stable thermodynamics gradient mode,
    recursive-checkpoint adjoint) vs jax.jvp under fwd(FIT); bar 3e-4
    (the reverse-vs-forward bar used for #54).

    Strict xfail at omega_b_high only.  There the recursive-checkpoint adjoint is
    1.278e-2 off forward mode under either thermodynamics gradient mode, while the
    direct adjoint is 3.6e-6 off (job 27787); #59 left it unchanged (job 27788).  At
    fiducial and YHe_028 the same defect is 2.191e-4 / 2.404e-4, inside the bar, so
    those two stay hard passes (a strict xfail there would turn their passes into
    failures).  Once the adjoint is fixed, omega_b_high passes and the strict marker
    fails the run, so that the marker gets removed."""
    if recomb_downstream_cosmology[0] == "omega_b_high":
        request.applymarker(pytest.mark.xfail(strict=True, reason=_XFAIL_RECURSIVE_ADJOINT))
    name, g, t, rel = _reverse_vs_forward(recomb_downstream_cosmology, _FIT, _FIT_FWD)
    assert jnp.isfinite(g), f"ERROR {name}: production reverse d/dh is {g} (forward {t:.6e})"
    assert rel < 3e-4, f"ERROR {name}: production reverse {g:.10e} vs forward {t:.10e}, rel {rel:.3e}"
