"""Shared helpers for the recombination-staging tests (smsharma/clax#48).
Presets and metrics copy the design study's probe script (not in the repo:
/lustre/work/n2minh/clax-probes/helium/scripts/thermo_probe.py) so the in-repo
numbers are comparable."""
import functools
import os
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from clax import PrecisionParams
from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve

REF_DIR = os.path.join(os.path.dirname(__file__), "..", "reference_data")
PREC_RB = replace(PrecisionParams.fast_cl(), pt_ode_solver="rodas5",
                  pt_ode_rtol=1e-4, pt_ode_atol=1e-4, ode_max_steps=16384)   # matter-solve preset of the #48 gradient tests
THERMO_GRIDS = (20000, 3000)          # default grid; fit_cl's thermodynamics grid
THETAS = ("h", "omega_b", "omega_cdm", "Y_He", "N_ur")
QS = ("xe", "cs2", "tb")
# The five opacity tables perturbations.py reads (smsharma/clax#55).
QS_OPACITY = ("kappa_dot", "exp_m_kappa", "g", "dkd_dloga", "g_prime")


def thermo_prec(n_points):
    """Thermodynamics preset on an n_points grid; jax.jvp through
    background_solve needs the direct adjoint."""
    return replace(PREC_RB, th_n_points=n_points, ode_adjoint="direct")


def as_traced(p):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float64), p)


def tables(p, prec):
    bg = background_solve(p, prec)
    th = thermodynamics_solve(p, prec, bg)
    return {"loga": th.xe_of_loga.x, "xe": th.xe_of_loga.y, "cs2": th.cs2_of_loga.y,
            "tb": th.Tb_of_loga.y, "g": th.g_of_loga.y}


def tables_opacity(p, prec):
    """Knots and values of the opacity tables, in the layout `tables` uses, so
    `_fd_check` can re-spline them. The knots are kappa_dot_of_loga.x (the same
    ln a values as the x_e table; before #55 a stop_gradient copy of them)."""
    bg = background_solve(p, prec)
    th = thermodynamics_solve(p, prec, bg)
    return {"loga": th.kappa_dot_of_loga.x, "kappa_dot": th.kappa_dot_of_loga.y,
            "exp_m_kappa": th.exp_m_kappa_of_loga.y, "g": th.g_of_loga.y,
            "dkd_dloga": th.dkappa_dot_dloga_of_loga.y, "g_prime": th.g_prime_of_loga.y}


@functools.partial(jax.jit, static_argnums=(2,))
def _tables_jvp(p, t, prec):
    return jax.jvp(lambda q: tables(q, prec), (p,), (t,))


def tangent_tables(p, prec, theta):
    """(primal tables, forward tangents d/d theta) as numpy arrays. One compilation
    per prec (static) and device: the cosmology and theta enter only as traced
    values (the tangent t), so repeat calls do not recompile. jax.default_device
    still selects where each call runs (checked for jax 0.9.2), which the CPU-vs-GPU
    test relies on."""
    p = as_traced(p)
    t = jax.tree_util.tree_map(jnp.zeros_like, p).replace(**{theta: jnp.asarray(1.0)})
    prim, tan = _tables_jvp(p, t, prec)
    return (jax.tree_util.tree_map(np.asarray, prim), jax.tree_util.tree_map(np.asarray, tan))


def z_of(loga):
    return 1.0 / np.exp(np.asarray(loga)) - 1.0


def spike_ratio(dln, z, zlo, zhi, half=25, floor=1e-8):
    """max over grid points in [zlo, zhi] of |dln| / median(|dln| over +-half points)."""
    a = np.abs(np.asarray(dln))
    pad = np.pad(a, half, mode="edge")
    med = np.median(np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1), axis=1)
    sel = (z >= zlo) & (z <= zhi)
    r = np.where(sel, a / np.maximum(med, floor), 0.0)
    i = int(np.argmax(r))
    return float(r[i]), float(z[i])


def visibility_peak(z, g, zlo=800.0, zhi=1400.0):
    s = (z >= zlo) & (z <= zhi)
    zz, gg = np.asarray(z)[s], np.asarray(g)[s]
    i = int(np.argmax(gg))
    c = np.polyfit(zz[i - 1:i + 2], gg[i - 1:i + 2], 2)
    return float(-c[1] / (2.0 * c[0]))


def load_ref(name):
    return np.load(os.path.join(REF_DIR, name, "thermo_recfast_hesw1.npz"))


def load_drag(name):
    """CLASS RECFAST (Heswitch = 1) derived epochs for the cosmology `name`
    (scripts/generate_class_reference.py --recfast-hesw1-drag): z_d, tau_d, rs_d
    (baryon drag, kappa_b = 1) and z_rec, tau_rec, rs_rec (CLASS's visibility peak,
    a three-point parabola on its table; clax calls that epoch z_star)."""
    import json
    with open(os.path.join(REF_DIR, name, "drag_recfast_hesw1.json")) as fh:
        return json.load(fh)


def xe_on_ref_grid(prim, ref_z):
    """clax x_e (CubicSpline in log a) evaluated at the reference redshifts."""
    return table_on_ref_grid(prim, "xe", ref_z)


def table_on_ref_grid(prim, q, ref_z):
    """clax table q of `tables` (CubicSpline in log a) at the reference redshifts."""
    from clax.interpolation import CubicSpline
    sp = CubicSpline(jnp.asarray(prim["loga"]), jnp.asarray(prim[q]))
    return np.asarray(sp.evaluate(jnp.log(1.0 / (1.0 + jnp.asarray(ref_z)))))


def dm_functional(prec, base):
    """f(h) = sum_k delta_m(k, tau0)^2 through the matter solve at PREC_RB."""
    from clax.perturbations import perturbations_solve_mpk

    def f(h_val):
        p = replace(as_traced(base), h=h_val)
        bg = background_solve(p, prec)
        th = thermodynamics_solve(p, prec, bg)
        pt = perturbations_solve_mpk(p, prec, bg, th)
        return jnp.sum(pt.delta_m[:, -1] ** 2)
    return f


def dm_functional_production(prec, base):
    """Same functional through solve_background_and_thermo (ADR 0001's stable
    custom_vjp), the path clax.compute and NUTS use. Reverse mode only."""
    from clax.perturbations import perturbations_solve_mpk
    from clax.thermodynamics import solve_background_and_thermo

    def f(h_val):
        p = replace(as_traced(base), h=h_val)
        bg, th = solve_background_and_thermo(p, replace(prec, th_grad_mode="stable"))
        pt = perturbations_solve_mpk(p, prec, bg, th)
        return jnp.sum(pt.delta_m[:, -1] ** 2)
    return f
