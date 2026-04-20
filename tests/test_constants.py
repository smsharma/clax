"""Tests physical-constant definitions.

Contract:
- Public constant values match the CLASS-aligned reference definitions.

Scope:
- Covers exact-value constants and one approximate Stefan-Boltzmann check.
- Excludes any physics-layer behavior built on top of these constants.
"""

from clax import constants as const


def test_mpc_over_m():
    """``Mpc_over_m`` matches the reference value; expects exact equality."""
    assert const.Mpc_over_m == 3.085677581282e22


def test_speed_of_light():
    """``c_SI`` matches the reference value; expects exact equality."""
    assert const.c_SI == 2.99792458e8


def test_gravitational_constant():
    """``G_SI`` matches the reference value; expects exact equality."""
    assert const.G_SI == 6.67428e-11


def test_boltzmann_constant():
    """``k_B_SI`` matches the reference value; expects exact equality."""
    assert const.k_B_SI == 1.3806504e-23


def test_planck_constant():
    """``h_P_SI`` matches the reference value; expects exact equality."""
    assert const.h_P_SI == 6.62606896e-34


def test_ev():
    """``eV_SI`` matches the reference value; expects exact equality."""
    assert const.eV_SI == 1.602176487e-19


def test_stefan_boltzmann():
    """``sigma_B`` matches the reference value; expects <1e-4 relative error."""
    assert abs(const.sigma_B - 5.670400e-8) / 5.670400e-8 < 1e-4


def test_tcmb_default():
    """``T_cmb_default`` matches the reference value; expects exact equality."""
    assert const.T_cmb_default == 2.7255
