"""Massive-neutrino fluid approximation: the handover from the hierarchy (smsharma/clax#46).

CLASS evolves the full Psi_l(q) hierarchy of a massive-neutrino species until
k*tau exceeds ``ncdm_fluid_trigger_tau_over_tau_k`` (31; perturbations.c:6274-6281),
then starts three fluid variables (delta, theta, shear) from the hierarchy's
integrated moments at that instant (perturbations.c:4920-4987) and evolves them
with the fluid equations (perturbations.c:9501-9570). clax cannot restart its
ODE at a per-mode switch time, so it carries the fluid variables in the state
vector from the start and blends the two descriptions around k*tau = 31.

These tests hold that branch-free handover to CLASS's:

1. Before the switch the fluid variables hold the hierarchy's moments, so the
   fluid starts where CLASS's starts. Fails on any lag or relaxation between
   the fluid slots and the hierarchy projection.
2. P_m(z=0) in the "class" mode matches CLASS's P_m, which CLASS computes with
   its own fluid approximation on, across k = 1.5e-3 to 2e-2 Mpc^-1: the band
   where k*tau crosses 31 between z ~ 2 and today. Fails on any handover defect
   that reaches the matter power.

Multi-cosmology: ``nulcdm_cosmology`` (fiducial 0.06, 0.15 and 0.30 eV;
``massive_nu_006`` duplicates the fiducial and has no reference directory).
Under ``--fast`` only the fiducial point runs, on a reduced k set.
"""

import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import clax
from clax.background import background_solve
from clax.perturbations import _ncdm_hierarchy_observables
from clax.thermodynamics import thermodynamics_solve
from tests.conftest import cosmology_reference_dir
from tests.pk_test_utils import (
    PK_TABLE_SOLVE_PREC,
    pk_reference_values,
    solve_matched_perturbation_states,
)

import pytest

# k at which the switch (k*tau = 31) falls at tau ~ 8400 Mpc, z ~ 0.6.
HANDOVER_K = 3.7e-3
# k*tau samples before the switch.
PRE_SWITCH_K_TAU = (5.0, 10.0, 20.0)

# Switch band: k*tau_0 runs from 21 (switch never reached today) to 280.
BAND_K_FULL = (1.5e-3, 2.5e-3, 3.7e-3, 5.0e-3, 7.0e-3, 1.0e-2, 2.0e-2)
BAND_K_FAST = (2.5e-3, 3.7e-3, 1.0e-2)

# Bars: 2x the worst measured case, rounded up to two significant figures
# (Oct 25, 2026, /lustre/work/n2minh/clax-probes/nu46; CPU jobs 30118/30120 and
# V100 job 30124 agree to 1e-8 in P_m).
# Pre-switch: worst 3.30e-6 (theta at k*tau = 5, 0.15 eV, V100); the solver's
# integration error, as fluid and hierarchy carry the same derivative. Before the
# fix the fluid lagged by 4-12% (relaxation at 20 aH; jobs 30117, 30121).
PRE_SWITCH_BAR = 6.6e-6
# P_m across k <= 1e-2: worst 6.93e-5 (2.5e-3 Mpc^-1, 0.30 eV); the full
# hierarchy reads up to 3.6e-5 against CLASS there. Before the fix: 2.5e-3
# (fiducial), 4.8e-3 (0.15 eV), 9.9e-3 (0.30 eV).
PM_BAND_BAR = 1.4e-4
# At k = 2e-2 (worst 1.35e-4, 0.15 eV) the checked-in reference carries its own
# floor, CLASS's spline in k at the onset of the BAO region: a CLASS run at
# exactly this k puts clax within 2e-5. Before the fix: 5.1e-3 / 1.2e-2.
PM_BAO_EDGE_K = 1.5e-2
PM_BAO_EDGE_BAR = 2.7e-4


def _reference_pk(name):
    path = cosmology_reference_dir(name)
    if path is None or not os.path.exists(os.path.join(path, "pk.npz")):
        pytest.skip(f"no CLASS P(k) reference for {name}")
    return dict(np.load(os.path.join(path, "pk.npz"), allow_pickle=True))


def test_fluid_variables_hold_hierarchy_moments_before_switch(nulcdm_cosmology):
    """Before k*tau = 31 the fluid delta, theta, shear equal the hierarchy's moments."""
    name, params = nulcdm_cosmology
    if cosmology_reference_dir(name) is None:
        pytest.skip(f"{name} duplicates the fiducial point")
    prec = PK_TABLE_SOLVE_PREC
    assert prec.ncdm_fluid_approximation == "class"
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    k = HANDOVER_K
    tau_samples = np.array(PRE_SWITCH_K_TAU) / k
    st = solve_matched_perturbation_states(params, prec, bg, th, k, tau_samples)
    idx = st["idx"]

    worst = (0.0, None, None)
    for k_tau, tau, y in zip(PRE_SWITCH_K_TAU, tau_samples, st["y_samples"], strict=True):
        a = jnp.exp(bg.loga_of_tau.evaluate(jnp.asarray(tau)))
        hier = _ncdm_hierarchy_observables(
            jnp.asarray(y), st["q_ncdm"], st["w_ncdm"], st["M_ncdm"], a, k, idx
        )[:3]
        fluid = (y[idx["ncdm_fluid_delta"]], y[idx["ncdm_fluid_theta"]], y[idx["ncdm_fluid_shear"]])
        for label, f, h in zip(("delta", "theta", "shear"), fluid, hier, strict=True):
            err = abs(float(f) / float(h) - 1.0)
            if err > worst[0]:
                worst = (err, label, k_tau)

    err, label, k_tau = worst
    assert err < PRE_SWITCH_BAR, (
        f"ERROR {name}: fluid {label} off the hierarchy moment by {err:.3e} at k*tau={k_tau:g} "
        f"(k={k:g} Mpc^-1, bar {PRE_SWITCH_BAR:.1e})"
    )


def test_class_mode_pm_matches_class_across_switch_band(nulcdm_cosmology, fast_mode):
    """P_m(z=0) with the "class" fluid mode matches CLASS's fluid-approximation P_m."""
    name, params = nulcdm_cosmology
    if cosmology_reference_dir(name) is None:
        pytest.skip(f"{name} duplicates the fiducial point")
    ref = _reference_pk(name)
    prec = PK_TABLE_SOLVE_PREC
    assert prec.ncdm_fluid_approximation == "class"
    ks = np.array(BAND_K_FAST if fast_mode else BAND_K_FULL)

    pk = np.array([float(clax.compute_pk(params, prec, float(k))) for k in ks])
    pk_class = pk_reference_values(ref, ks, key="pk_m_lin_z0")
    rel = pk / pk_class - 1.0
    bars = np.where(ks < PM_BAO_EDGE_K, PM_BAND_BAR, PM_BAO_EDGE_BAR)
    i = int(np.argmax(np.abs(rel) / bars))
    n_pass = int(np.sum(np.abs(rel) < bars))
    assert n_pass == len(ks), (
        f"ERROR {name}: P_m off CLASS by {rel[i]:+.3e} at k={ks[i]:.2e} Mpc^-1 "
        f"(bar {bars[i]:.1e}; {n_pass}/{len(ks)} k pass)"
    )
