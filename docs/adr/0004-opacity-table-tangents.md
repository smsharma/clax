# ADR 0004: The opacity tables carry their full tangents

Date: 2026-10-20
Status: Accepted
Issue: https://github.com/smsharma/clax/issues/55 (and #45)

## Context

`thermodynamics_solve` returns five opacity tables that `perturbations.py`
reads at every right-hand-side evaluation and in every source function:
kappa_dot (the Thomson scattering rate), exp(-kappa), the visibility g,
dkappa_dot/dln a and g'. Each is a `CubicSpline` in ln a on the knots of the
thermodynamics scan.

From d127731 (May 10, 2026) until this ADR, those five tables were built from
`stop_gradient` copies of their knots and values, with only the n_H_0 ∝ omega_b
factor of kappa_dot restored (kappa_dot_grid times n_H_0 / stop_gradient(n_H_0)).
The commit answered a "~10^8x FD blowup" of the reverse-mode d kappa_dot/d omega_b
through the thermodynamics scan of the time, and its tests checked the tables
at ln a = -8 only, where x_e is constant and the rescale is exact; its own
message records a 10-30% error near recombination.

What the cut did to the derivatives, measured on main at 99b62fe (probe of
Oct 20, 2026, `/lustre/work/n2minh/clax-probes/opacity/`, jobs 27125-27127 and
27159; five cosmologies, both thermodynamics grids, twenty redshifts from
z = 3000 to 0):

- Along omega_cdm and N_ur, which enter x_e only through H(a), the AD tangent of
  every opacity table was exactly zero at every redshift, while the finite
  difference of kappa_dot along omega_cdm reaches d ln kappa_dot / d ln omega_cdm
  of order 1 in the freeze-out tail.
- Along omega_b and Y_He only the n_H_0 path survived: at ln a = -7 (z ~ 1095)
  the reverse-mode derivative of kappa_dot along omega_b was 2.91 against a
  finite difference of 1.42, and that of g had the wrong sign (-0.116 against
  +0.118); at ln a = -6 the errors were ten-fold.
- Downstream (issue #55): dP(k = 0.05 Mpc^-1)/d omega_b from AD was 4.0% off
  converged finite differences and dP(k = 0.1 Mpc^-1)/d omega_cdm +0.66%, both
  in forward and reverse mode. #45 attributed 65% of the omega_cdm bias to this
  cut by a finite difference with the cut tables held fixed. An HMC leapfrog
  follows that force; Fisher, Laplace and MAP estimates inherit the bias.

Two things changed between d127731 and this ADR. ADR 0001 (Aug 29, 2026) made
the production reverse pass a contraction of forward tangents, so reverse-mode
cancellation in the recombination rates no longer reaches the gradient. ADR 0003
(#53, Oct 2026) replaced the clipped explicit recombination steps by CLASS's
Saha -> ODE staging with an L-stable step, after which the forward tangents of
x_e, c_s^2 and T_b at fixed z match Richardson finite differences at 11
cosmologies (`tests/test_recomb_thermo.py::_fd_check`). Neither re-examined the
cut; ADR 0003 lists it under "Unchanged" and #45 under "Not done".

## Decision

Remove the cut. The five opacity tables are `CubicSpline(loga_grid, values)` on
the live knots with the live values, exactly like x_e, T_b and c_s^2;
dkappa_dot/dln a is the derivative of the kappa_dot spline at its own knots and
g' is built from it. The tau_free_streaming block reads the live arrays and
keeps its own `stop_gradient` on the switch time (CLASS's table point has no
derivative either; ADR 0003 / #51).

The primal is unchanged to the last bit: n_H_0 / stop_gradient(n_H_0) is exactly
1.0 in floating point, and the removed `_safe` arrays recomputed exp(-kappa) and
g from the same inputs. Verified by dumping every `ThermoResult` table and scalar
at three cosmologies and two grids on main and on this change (jobs 27164, 27165):
all arrays bitwise identical. Checkpoints, reference bars and every forward
result are therefore untouched; only derivatives change.

## Why the knots matter, and what the "10^8x blow-up" was

A table S(ln a) = sum_i B_i(ln a; x) v_i has knots x_i and values v_i, and both
move with the parameters: the scan is uniform in ln tau, so the knot positions
in ln a shift when tau(a) does (omega_b, omega_cdm, h, N_ur all move them; Y_He
does not). The tangent at fixed ln a is the knot part plus the value part. The
value part alone, the tangent at fixed knot index, differs from the true tangent
by the transport term S'(ln a) d ln a_i/d theta.

The probe's third tree kept the values live and froze only the knots. Along the
knot-moving parameters it was wrong by that transport term, of order 1 in
d ln Q/d ln theta, and exact (1e-11) along Y_He. Along h at z >= 400, where the
true derivative at fixed physical densities is ~1e-7, the same order-1 error
reads as a ratio of 1e7-1e8: that is the blow-up d127731 measured, read against
a vanishing true derivative, and attributed to the Friedmann recursion. The
recursion's tangent is the physical sensitivity of a(tau) and is well behaved;
the knot tangent simply has to accompany the value tangent, which the live-knot
spline does by construction.

## Evidence

Worst error in d ln Q / d ln theta over twenty redshifts, five cosmologies
(fiducial, omega_b +20%, omega_cdm -20%, Y_He = 0.28, N_eff = 2) and both grids
(20000 and 3000 points), AD forward tangent against Richardson finite
differences (steps 1e-3 and 2e-3 relative; `analysis_th.txt`):

| theta | kappa_dot, cut | kappa_dot, full | g, cut | g, full |
|---|---|---|---|---|
| omega_b | 12.6 | 7.5e-3 (a) | 19.7 | 1.7e-2 (b) |
| omega_cdm | 5.1 | 2.5e-7 | 59.5 | 2.2e-5 (b) |
| h | 1.14 | 6.8e-3 (a) | 1.14 | 6.7e-3 (a) |
| Y_He | 2.69 | 6.8e-3 (a) | 19.7 | 6.7e-3 (a) |
| N_ur | 2.7e-2 | 6.8e-3 (a) | 12.3 | 6.7e-3 (a) |

(a) the identical number x_e shows on every tree at z = 600 on the 3000-point
grid at the N_eff = 2 and omega_b +20% points: the hard Taylor switch in the T_b
update at f_e < 1e-3 (ADR 0003, Known limitations), not the opacity tables; on
the 20000-point grid the worst rows are 1e-4 or below. (b) at z = 3000, where
exp(-kappa) goes as exp(-220 d ln omega_b) and a finite difference at relative
step 1e-3 is not converged; the AD tangent differs from the Richardson value by
8e-5 of the log-derivative. Everything else sits at the finite differences' own
uncertainty. No table or tangent is non-finite anywhere.

Reverse mode on a fixed random projection of the same samples: native
`jax.grad` and the production `solve_background_and_thermo` (ADR 0001) agree with
the forward tangent to 6.2e-9 on the full tangents (both were equal to the cut
tangents, and wrong, on main).

The derivative tables dkappa_dot/dln a and g' have a kink at the two ends of the
hydrogen hand-over window (z = 1600 and 1500, ADR 0003): sampled exactly at
z = 1600 the finite difference along omega_cdm on the omega_cdm -20% point is
6% of a log-derivative of 2.4e-3 away from AD, and at every other redshift the
two agree to 1e-10. A finite difference across a kink of a derivative table
measures where the knots fall against the kink; the tests sample 1650 and 1550.

What limits a finite-difference gate along omega_b below z ~ 620 is the primal,
not the derivative. In the spot-check scan at the thermodynamics tests' precision
block (job 27172; omega_b, ln a = -6, -5.5, -5, every opacity table) the
finite-difference error halves each time the step doubles, the signature of a
fixed-size jump in the primal inside the window, and the jump scales with the
grid: 1.7e-2 relative on 3000 points, 1.7e-3 on 10000, 4e-4 on 20000, while
AD is stable across the three grids to 1e-4 and forward equals reverse to
1e-11. The jump is the T_b Taylor switch at f_e < 1e-3 (ADR 0003, Known
limitations) crossing a knot as omega_b moves x_e(z). Where the true derivative
is small this reads as a large relative error: the omega_b derivative of
exp(-kappa) passes through zero near z ~ 400 (x_e n_H goes as H in the
freeze-out tail and the reionisation depth is pinned by tau_reio), and there a
single-step finite difference at 10000 points is 30% off (FD -9.9e-5 against AD
-6.9e-5; -3.98e-4 on 3000 points, -7.7e-5 on 20000, AD -6.4e-5 to -7.0e-5).
The thermodynamics spot checks therefore sample exp(-kappa) along omega_b at
ln a = -5 instead of -6.

P(k) level, omega_cdm: dP(k = 0.1 Mpc^-1)/d omega_cdm by forward mode at the
precision block of `tests/test_pk_forward_mode.py`, against central finite
differences at steps 1e-3 and 5e-4, at the five LCDM grid points (jobs 27176,
27177):

| point | main, AD/FD - 1 | this change, AD/FD - 1 |
|---|---|---|
| fiducial | +0.69%, +0.76% | +0.02%, +0.09% |
| h +10% | +0.67%, +0.72% | +0.00%, +0.05% |
| omega_b +20% | +0.67%, +0.63% | -0.04%, -0.08% |
| omega_cdm -20% | +0.63%, +0.59% | -0.02%, -0.07% |
| n_s +5% | +0.69%, +0.76% | +0.02%, +0.09% |

The finite differences themselves move by 6e-4 to 1e-3 between the two steps
at this precision block, so the remaining 1e-4 to 9e-4 is their scatter. The
primal P(k) values are identical on both trees. This block has `ncdm_q_size = 0`,
so #45's massive-neutrino residual does not enter; with the neutrino
gravitationally negligible #45 had attributed the whole omega_cdm bias to the
cut, which this confirms.

P(k) level, omega_b: dP(k = 0.05 Mpc^-1)/d omega_b at fiducial, the quantity
of `test_public_table_partials_match_fd_full` (production reverse mode through
`compute_pk_table`), on this change (job 27333): AD -3.52446e5 against central
finite differences -3.52533e5 at the test's step (2e-5) and -3.52458e5 at twice
it, i.e. -0.025% and -0.004%; the two finite differences differ by 0.02%. On
main the same AD was 4.0% off (jobs 24324, 24429).

## Tests

- `tests/test_recomb_thermo.py::test_opacity_tangents_match_converged_fd`: the
  five opacity tables through the #48 finite-difference classifier at the 11
  recombination cosmologies (full mode; slow): 21 redshifts along omega_cdm,
  Y_He and N_ur (worst resolved error 1e-4 to 3e-4); z >= 900 along omega_b,
  below which the staircase above sets the finite differences (8e-4 of the
  exp(-kappa) derivative at z = 800 at omega_b +20%, 1.1e-2 at z = 400); z <=
  1200 along h with the small-derivative floor at 1e-5, since at fixed densities
  the tables respond to h at z <= 10 only (AD and FD agree there to 1e-6), the
  z = 30 to 600 derivatives are 1e-7 to 1e-5 of the value under the staircase,
  and above z = 1400 the exp(-kappa) tangent is the trapezoid sum's knot-phase
  sensitivity on a value below 1e-5; where the derivative is below the floor
  the tangent must stay below 1.4e-5 of the value (twice the worst, 6.7e-6).
  Red on main: 55 of 55 cases fail by assertion (job 27183); green on this
  change: 55 of 55 (job 27184), worst resolved error 2.0e-4 (omega_b at
  Y_He = 0.22), 1.8e-4 (h, omega_cdm, N_ur), 9.7e-5 (Y_He).
- `tests/test_pk_forward_mode.py`: the AD-vs-FD bar is 1.8e-3 (was 1%) and the
  test runs at the five LCDM grid points; red on main at +0.59% to +0.76%.
- `tests/test_thermodynamics.py::TestThermoGradients`: the reverse-mode spot
  checks of kappa_dot, exp(-kappa), g and dkappa_dot/dln a now sample ln a = -8,
  -7 and -6 along omega_b and -7, -6 along omega_cdm (they sampled -8 along
  omega_b only). 16 of the 20 cases fail on main (job 27160): along omega_cdm
  AD is exactly 0, along omega_b 64% to 2.6e6% off, g with the wrong sign.
- The P(k)-level gates are the existing
  `test_public_table_partials_match_fd_full` (failing on main, 4% in omega_b)
  and `test_pk_forward_mode.py`.

## Alternatives considered

1. **Keep the cut and restore the x_e path by hand**, e.g. multiplying by
   x_e / stop_gradient(x_e) as well as by n_H_0 / stop_gradient(n_H_0). That
   restores value tangents and not knot tangents: the probe's third tree is
   exactly this situation and is wrong by the transport term along every
   parameter that moves the knots.
2. **A fixed, parameter-independent ln a grid for the scan** (#45's
   suggestion, as CLASS tabulates in z). It would make the per-knot tangent the
   tangent at fixed a and remove the transport term altogether, but it changes
   the primal and every bar set by ADR 0003, and it is not needed: the live-knot
   spline already gives the right tangent. Not pursued.
3. **Forward-only policy for these parameters.** Both modes were wrong; the
   issue is in the primal's tangent graph, not in a mode.

## Consequences

- Every derivative that passes through the thermodynamics now sees the x_e
  response of the Thomson rate: along omega_b, omega_cdm, h, Y_He, N_ur and
  T_cmb, in forward and reverse mode, through `compute_pk`, `compute_pk_table`,
  `compute` and the matter solve. HMC forces along the density parameters
  change at the percent level of the derivative; posteriors do not move, since
  accept/reject uses the primal.
- The "10-30% residual near recombination" and the restriction of the
  thermodynamics gradient tests to ln a = -8 are gone.
- No runtime cost in the primal. The tangent passes carry the cumulative
  optical depth's tangent, which they computed and discarded before.
- Not changed here: the T_b Taylor switch at f_e < 1e-3 (ADR 0003; removed on
  Oct 22, 2026, see CHANGELOG, after which the finite-difference floor along
  omega_b described above is gone and the opacity test gates omega_b at every
  redshift); the
  tau_free_streaming switch time held fixed under AD (#51); whatever part of
  #45's omega_cdm bias sits in the massive-neutrino sector rather than in this
  cut.
