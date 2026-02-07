"""Test that physical constants match CLASS values exactly."""

from jaxclass import constants as const


def test_mpc_over_m():
    assert const.Mpc_over_m == 3.085677581282e22


def test_speed_of_light():
    assert const.c_SI == 2.99792458e8


def test_gravitational_constant():
    assert const.G_SI == 6.67428e-11


def test_boltzmann_constant():
    assert const.k_B_SI == 1.3806504e-23


def test_planck_constant():
    assert const.h_P_SI == 6.62606896e-34


def test_ev():
    assert const.eV_SI == 1.602176487e-19


def test_stefan_boltzmann():
    """sigma_B should be approximately 5.670400e-8."""
    assert abs(const.sigma_B - 5.670400e-8) / 5.670400e-8 < 1e-4


def test_tcmb_default():
    assert const.T_cmb_default == 2.7255
