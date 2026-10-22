"""Resolution of the fast-path Bessel lookup table (precision sweep, Oct 2026;
lookup changed to cubic Hermite in #54).

``compute_cls_all_fast`` tabulates j_l(x) and j_l'(x) on a uniform x grid. It
used to interpolate j_l linearly. Linear interpolation of an oscillation of
period 2*pi at spacing h biases the amplitude by about -h^2/12 on average: at 3
points per unit x (h ~ 0.33 for k_max = 1/Mpc) that is -0.9% in Delta_l, which
the Oct 2026 sweep measured as a -1% TT offset at l < 50, +20% in EE at l = 2
and a k_max-dependent C_l; #50 raised the density to 30 points per unit x,
where the bias is below 1e-4. #54 replaced the lookup by a cubic Hermite
interpolant (value and slope at both nodes), whose error scales as h^4: the
default is back at 3 points per unit x and agrees with the 30-point table to
the bar of ``test_default_hermite_table_matches_30_points_per_unit_x``.

``test_linear_interpolation_bias_follows_h2_over_12`` still tests the linear
bias law, with its own helper (``_table_bias``), independent of the fast path;
it documents why a linear lookup would need 30 points per unit x. Cosmology-
independent numerics: exempt from the multi-cosmology rule.
"""
import numpy as np
import pytest

import clax.harmonic as harmonic
from clax.bessel import build_jl_table, sparse_l_grid, spherical_jl


def _table_bias(x_per_unit, l_values, x_max=15000.0):
    """Mean and max relative error of linear interpolation of the j_l table
    against the exact j_l over x in [2 l, 3000], for a production-size table
    (x_max = 15000 as for k_max = 1/Mpc; the 30000-point floor is inactive).
    Returns (mean_rel, max_rel, h)."""
    n_x = max(30000, int(x_max * x_per_unit))
    x_table, jl_table, _ = build_jl_table(int(max(l_values)), n_x=n_x, x_max=x_max)
    x_table = np.asarray(x_table)
    jl_table = np.asarray(jl_table)
    h = float(x_table[1] - x_table[0])
    l_sparse = sparse_l_grid(int(max(l_values)))
    rng = np.random.default_rng(0)
    mean_rel, max_rel = [], []
    for l in l_values:
        row = jl_table[int(np.where(l_sparse == l)[0][0])]
        x = rng.uniform(2.0 * l, 3000.0, 20000)
        exact = np.asarray(spherical_jl(l, x))
        interp = np.interp(x, x_table, row)
        env = np.sqrt(np.mean(exact**2))          # rms envelope of j_l on the window
        keep = np.abs(exact) > 0.5 * env          # avoid zero crossings
        rel = (interp[keep] - exact[keep]) / exact[keep]
        mean_rel.append(rel.mean())
        max_rel.append(np.abs(rel).max())
    return np.array(mean_rel), np.array(max_rel), h


L_VALUES = (2, 10, 50, 200)


# Hermite lookup at the module default density against the same lookup at 30
# points per unit x, on x in [0, 15000] (k_max = 1/Mpc, tau_0 ~ 14000 Mpc), for
# j_l and j_l' at the l below. Bars = 2x the measured worst difference over
# these l, rounded up to 2 significant figures: CPU job 27633 measured worst
# |dj_l| 3.626e-6 and |dj_l'| 3.302e-6, both at l = 2 (l = 30, 300, 1200:
# 3.0e-7, 3.2e-8, 8.1e-9 in j_l). They equal the Hermite lookup's own error at
# the default density against the exact j_l (tests/test_cl_bessel_lookup.py),
# since the 30-point table is 1e4 times closer to exact.
HERMITE_X_MAX = 15000.0
HERMITE_ELLS = (2, 30, 300, 1200)
BAR_JL = 7.3e-6
BAR_JLP = 6.7e-6


@pytest.fixture(scope="module")
def hermite_tables():
    """{x_per_unit: (n_x, jl_table, jlp_table)} for the default density and 30,
    each 'max(30000, ceil(density x x_max))' points on [0, HERMITE_X_MAX] (the
    rule of harmonic._table_grid), rows of sparse_l_grid(max(HERMITE_ELLS))."""
    out = {}
    for dens in sorted({harmonic.JL_TABLE_X_PER_UNIT, 30.0}):
        n_x = max(30000, int(np.ceil(dens * HERMITE_X_MAX)))
        _, jl_t, jlp_t = build_jl_table(max(HERMITE_ELLS), n_x=n_x, x_max=HERMITE_X_MAX)
        out[dens] = (n_x, jl_t, jlp_t)
    return out


def _hermite_lookup(table, ell, x):
    """harmonic._lookup_jl (the fast path's lookup) on one table row."""
    import jax.numpy as jnp
    n_x, jl_t, jlp_t = table
    row = int(np.nonzero(sparse_l_grid(max(HERMITE_ELLS)) == ell)[0][0])
    dx_tab = HERMITE_X_MAX / (n_x - 1)
    i_left, t = harmonic._table_position(jnp.asarray(x), dx_tab, n_x)
    jl, jlp = harmonic._lookup_jl(jl_t[row], jlp_t[row], jnp.float64(ell),
                                  jnp.float64(ell * (ell + 1.0)), i_left, t, dx_tab)
    return np.asarray(jl), np.asarray(jlp)


def test_default_hermite_table_matches_30_points_per_unit_x(hermite_tables):
    """The cubic Hermite lookup at the module default density agrees with the
    same lookup at 30 points per unit x (a table ten times larger) to the bars
    above, for l = 2, 30, 300, 1200 at 20000 seeded-uniform x in [0, 15000]
    plus 2000 x within +-2 of l (where the table switches from backward to
    upward recurrence). This is what lets the default stay at 3 points per
    unit x. Mutation check (CPU job 27636): a linear lookup at 3 points per
    unit x in its place differs from the 30-point table by 2.0e-3 in j_l at
    l = 2, 270 times the bar.
    Cosmology-independent numerics: exempt from the multi-cosmology rule."""
    dens = harmonic.JL_TABLE_X_PER_UNIT
    for ell in HERMITE_ELLS:
        rng = np.random.default_rng(ell)
        near = np.clip(rng.uniform(ell - 2.0, ell + 2.0, 2000), 0.0, None)
        x = np.concatenate([rng.uniform(0.0, HERMITE_X_MAX, 20000), near])
        jl_a, jlp_a = _hermite_lookup(hermite_tables[dens], ell, x)
        jl_b, jlp_b = _hermite_lookup(hermite_tables[30.0], ell, x)
        e_jl = float(np.max(np.abs(jl_a - jl_b)))
        e_jlp = float(np.max(np.abs(jlp_a - jlp_b)))
        print(f"{dens:g} vs 30 pts/unit-x, l={ell}: |dj_l| {e_jl:.3e} (bar {BAR_JL:.1e}), "
              f"|dj_l'| {e_jlp:.3e} (bar {BAR_JLP:.1e})")
        assert e_jl <= BAR_JL, f"ERROR j_l {dens:g} vs 30 pts/unit-x: {e_jl:.2e} > {BAR_JL:.1e} (l={ell})"
        assert e_jlp <= BAR_JLP, f"ERROR j_l' {dens:g} vs 30 pts/unit-x: {e_jlp:.2e} > {BAR_JLP:.1e} (l={ell})"


def test_linear_interpolation_bias_follows_h2_over_12():
    """Mean relative error of the linearly interpolated table is -h^2/12 to
    within 50% at both resolutions (h = 0.33: -0.9%; h = 0.033: -9e-5), and
    the fine table's pointwise error is below 5e-4."""
    mean3, max3, h3 = _table_bias(3.0, L_VALUES)
    mean30, max30, h30 = _table_bias(30.0, L_VALUES)
    for mean, h in ((mean3, h3), (mean30, h30)):
        law = -h**2 / 12
        assert np.all(np.abs(mean - law) < 0.5 * abs(law)), f"h={h:.3f}: bias {mean} vs law {law:.2e}"
    assert np.all(max30 < 5e-4), f"fine-table max error: {max30}"
    assert np.all(max3 > 20 * max30)


def test_fast_path_default_uses_fine_table():
    """compute_cls_all_fast must default to the module constant."""
    import inspect
    sig = inspect.signature(harmonic.compute_cls_all_fast)
    assert sig.parameters["x_per_unit"].default == harmonic.JL_TABLE_X_PER_UNIT


def test_sparse_l_spline_pad_removes_end_error():
    """The dense C_l come from a cubic spline over the sparse l grid; its end
    condition corrupts the last interval unless the sparse grid extends past
    the requested l_max. Synthetic damped-acoustic C_l: unpadded error at
    l_max-60..l_max is ~1-2%, padded ~1e-3 (bar 2e-3: this model's own
    interior spline error, 6 nodes per acoustic period). On CLASS's spectra
    the padded end-condition residual over the last 50 multipoles is 0.03% TT
    (l_max 1000 and 1500) and 0.09% EE (l_max 1500); unpadded 1.94% TT /
    1.80% EE at l_max 1000 and 0.26% / 2.7% at 1500. Pad 100 is adequate and
    larger pads gain nothing."""
    import jax.numpy as jnp
    from clax.interpolation import CubicSpline

    def cl_model(l):
        l = np.asarray(l, dtype=float)
        return np.exp(-(l / 1400.0) ** 2) * (1.0 + 0.3 * np.cos(2 * np.pi * l / 300.0)) / l**2

    l_max = 1000
    dense = np.arange(2, l_max + 1)
    errs = {}
    for pad in (0, harmonic.L_SPLINE_PAD):
        ls = sparse_l_grid(l_max + pad)
        sp = np.asarray(CubicSpline(jnp.array(ls.astype(float)), jnp.array(cl_model(ls))).evaluate(jnp.array(dense.astype(float))))
        err = np.abs(sp / cl_model(dense) - 1.0)
        errs[pad] = err[dense >= l_max - 60].max()
    assert harmonic.L_SPLINE_PAD >= 100
    # this synthetic model has ~0.1% interior spline error (6 nodes per
    # acoustic period); the end condition adds >0.5% unpadded and nothing padded
    assert errs[0] > 5e-3, f"unpadded end error unexpectedly small: {errs[0]}"
    assert errs[harmonic.L_SPLINE_PAD] < 2e-3, f"padded end error: {errs[harmonic.L_SPLINE_PAD]}"
    assert errs[0] > 4 * errs[harmonic.L_SPLINE_PAD]
