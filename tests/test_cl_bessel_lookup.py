"""L0 (#54): the Bessel-table lookup in compute_cls_all_fast against direct
spherical Bessel functions, including x within +-2 of l where the table switches
from backward to upward recurrence, at both table densities in use: 3 points per
unit x (the pre-#50 density) and 30 (JL_TABLE_X_PER_UNIT since #50). The table
is the production one: the rows of _sparse_l_padded(1500) on the fixed range
x_max = 1.05 x 1/Mpc x 20000 Mpc. Cosmology-independent numerics: exempt from the
multi-cosmology rule.

Bars, per density = 2x the Hermite lookup's measured worst error, rounded up to
2 significant figures (l0_measure.py, job 27124; every worst sits at l = 2).
The linear lookup fails them at 30 points per unit x for every tested l (its
worst is ~3e4 times the bar), and at 3 points per unit x for l = 2, 30 and 300;
at 3 points per unit x and l = 1200 its error (4.9e-6 in j_l, 5.9e-5 in the AD
derivative) is below the l = 2-set bars, so that case does not catch it."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import spherical_jn

from clax.harmonic import (L_SPLINE_PAD, _jl_tables, _lookup_jl, _sparse_l_padded,
                           _table_grid, _table_position)

L_MAX = 1500
# points per unit x -> (BAR_JL, BAR_JLP, BAR_DJL): bars on |j_l lookup - j_l|,
# |j_l' lookup - j_l'| and |d/dx j_l lookup - j_l'|.
BARS = {
    3.0: (7.3e-6, 6.7e-6, 6.8e-5),     # job 27124: hermite worst 3.626e-6, 3.302e-6, 3.354e-5
    30.0: (7.3e-10, 6.7e-10, 6.8e-8),  # job 27124: hermite worst 3.635e-10, 3.326e-10, 3.394e-8
}


@pytest.fixture(scope="module", params=sorted(BARS), ids=lambda d: f"{d:g}pts")
def tables(request):
    dens = request.param
    x_max, n_x = _table_grid(1.0, 20000.0, dens)
    _, jl_t, jlp_t = _jl_tables(L_MAX + L_SPLINE_PAD, n_x, x_max)
    return dens, x_max, n_x, jl_t, jlp_t


def _lookup(tables, ell, x, mode):
    _, x_max, n_x, jl_t, jlp_t = tables
    li = int(np.nonzero(_sparse_l_padded(L_MAX) == ell)[0][0])
    dx_tab = x_max / (n_x - 1)
    i_left, t = _table_position(x, dx_tab, n_x)
    return _lookup_jl(jl_t[li], jlp_t[li], jnp.float64(ell), jnp.float64(ell * (ell + 1.0)),
                      i_left, t, dx_tab, mode)


def _samples(ell, x_max):
    rng = np.random.default_rng(ell)
    near = np.clip(rng.uniform(ell - 2.0, ell + 2.0, 2000), 0.0, None)
    return np.concatenate([rng.uniform(0.0, x_max, 20000), near])


@pytest.mark.parametrize("ell", [2, 30, 300, 1200])
def test_hermite_lookup_matches_spherical_bessel(tables, ell):
    dens = tables[0]
    bar_jl, bar_jlp, _ = BARS[dens]
    x = _samples(ell, tables[1])
    jl, jlp = _lookup(tables, ell, jnp.asarray(x), "hermite")
    e_jl = float(np.max(np.abs(np.asarray(jl) - spherical_jn(ell, x))))
    e_jlp = float(np.max(np.abs(np.asarray(jlp) - spherical_jn(ell, x, derivative=True))))
    print(f"{dens:g} pts/unit-x, l={ell}: |dj_l| {e_jl:.2e} (bar {bar_jl:.1e}), "
          f"|dj_l'| {e_jlp:.2e} (bar {bar_jlp:.1e})")
    assert e_jl <= bar_jl, f"ERROR j_l lookup {e_jl:.2e} > {bar_jl:.1e} (l={ell}, {dens:g} pts/unit-x)"
    assert e_jlp <= bar_jlp, f"ERROR j_l' lookup {e_jlp:.2e} > {bar_jlp:.1e} (l={ell}, {dens:g} pts/unit-x)"


@pytest.mark.parametrize("ell", [2, 30, 300, 1200])
def test_ad_derivative_of_lookup_is_jl_prime(tables, ell):
    dens = tables[0]
    bar_djl = BARS[dens][2]
    x = _samples(ell, tables[1])
    d = jax.vmap(jax.grad(lambda xx: _lookup(tables, ell, xx, "hermite")[0]))(jnp.asarray(x))
    e = float(np.max(np.abs(np.asarray(d) - spherical_jn(ell, x, derivative=True))))
    print(f"{dens:g} pts/unit-x, l={ell}: |d/dx lookup - j_l'| {e:.2e} (bar {bar_djl:.1e})")
    assert np.all(np.isfinite(np.asarray(d))), f"ERROR non-finite AD derivative (l={ell}, {dens:g} pts/unit-x)"
    assert e <= bar_djl, f"ERROR AD derivative of lookup {e:.2e} > {bar_djl:.1e} (l={ell}, {dens:g} pts/unit-x)"
