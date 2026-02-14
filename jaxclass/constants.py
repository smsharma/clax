"""Physical constants matching CLASS v3.3.4 exactly.

All values taken from class_public-3.3.4/include/background.h lines 584-630.
CLASS uses units where c=1, lengths in Mpc, times in Mpc (i.e., Mpc/c).
Densities are in units of [3c^2/(8 pi G)], so H^2 = sum(rho_i).

References:
    CLASS source: include/background.h
"""

import math

# --- Conversion factors ---
# cf. background.h:584
Mpc_over_m = 3.085677581282e22
"""Conversion factor from meters to megaparsecs."""

# cf. background.h:587
Gyr_over_Mpc = 3.06601394e2
"""Conversion factor from megaparsecs to gigayears (c=1, Julian years of 365.25 days)."""

# --- Fundamental constants (SI) ---
# cf. background.h:589-595
c_SI = 2.99792458e8
"""Speed of light in m/s."""

G_SI = 6.67428e-11
"""Newton's gravitational constant in m^3/kg/s^2."""

eV_SI = 1.602176487e-19
"""1 eV expressed in Joules."""

k_B_SI = 1.3806504e-23
"""Boltzmann constant in J/K."""

h_P_SI = 6.62606896e-34
"""Planck constant in J*s."""

# --- Derived constants ---
sigma_B = 2.0 * math.pi**5 * k_B_SI**4 / (15.0 * h_P_SI**3 * c_SI**2)
"""Stefan-Boltzmann constant in W/m^2/K^4 = kg/K^4/s^3.
   sigma_B = 2 pi^5 k_B^4 / (15 h^3 c^2) = 5.670400e-8
"""

# --- CMB temperature ---
T_cmb_default = 2.7255
"""Default CMB temperature today in Kelvin (Fixsen 2009)."""

# --- Helium fraction ---
Y_He_default = 0.2454006
"""Default primordial helium mass fraction Y_He (BBN-consistent for standard LCDM)."""

# --- Neutrino temperature ratio ---
# T_ncdm / T_gamma = (4/11)^(1/3), standard value
T_ncdm_over_T_cmb_default = (4.0 / 11.0) ** (1.0 / 3.0)
"""Default ncdm temperature relative to photon temperature."""

# --- Useful numerical constants ---
# cf. background.h:627-628
zeta3 = 1.2020569031595942853997381615114499907649862923404988817922
"""Riemann zeta(3), used in neutrino number density."""

zeta5 = 1.0369277551433699263313654864570341680570809195019128119741
"""Riemann zeta(5)."""

# --- Derived unit conversions for CLASS internal units ---

# In CLASS, H0 is stored as H0/c in Mpc^-1.
# H0_SI = H0_class * c_SI / Mpc_over_m
# But for km/s/Mpc: H0_kmsMpc = h * 100

# Photon density parameter: Omega_g = (4/c^3) * sigma_B / (3c^2/(8piG)) * T_cmb^4 / H0^2
# In CLASS units where rho_class = 8piG/(3c^2) * rho_physical:
# rho_g_class = (8piG)/(3c^3) * (4 sigma_B / c) * T_cmb^4
# But CLASS stores H0 as H0/c in Mpc^-1, so H0_class = h * 100 * 1e3 / (c_SI * Mpc_over_m^-1)
# We compute the full prefactor:

# 8 * pi * G / (3 * c^2) in units of Mpc * kg^-1 * s^2
_8piG_over_3c2_ = 8.0 * math.pi * G_SI / (3.0 * c_SI**2)

# rho_class = _8piG_over_3c2_ * rho_physical / Mpc_over_m^-2
# Since CLASS measures H in Mpc^-1, and H^2 = rho_class, we have:
# rho_class [Mpc^-2] = (8piG / 3c^2) * rho_physical [kg/m^3] * Mpc_over_m^2 / c_SI^2
# The factor of 1/c_SI^2 comes from converting s^-2 to Mpc^-2

# Photon energy density today in CLASS units per T_cmb^4:
# rho_g = (8piG/3c^3) * (pi^2/15) * (k_B T)^4 / (hbar^3 c^3)
#       = (8piG/3c^3) * (4 sigma_B / c) * T^4
# In CLASS Mpc^-2 units:
# rho_g_class = (8piG / 3c^2) * (4 sigma_B / c) * T^4 * (Mpc_over_m / c_SI)^2
# Simplify: everything times Mpc_over_m^2 / c_SI^2 to go from s^-2 to Mpc^-2

# This factor converts physical energy density to CLASS rho units (Mpc^-2)
# rho_class = rho_physical * _rho_factor_
_rho_factor_ = _8piG_over_3c2_ * Mpc_over_m**2 / c_SI**2
"""Factor to convert physical energy density [kg/m^3 * c^2 = J/m^3]...
Actually, CLASS defines rho such that H^2 [Mpc^-2] = sum rho_i.
So rho_i [Mpc^-2] = (8piG)/(3c^2) * rho_physical [kg/m^3] * (Mpc/c)^2.
This factor = (8piG)/(3c^4) * Mpc_over_m^2."""

# Radiation constant: a_R = 4 sigma_B / c
a_rad = 4.0 * sigma_B / c_SI
"""Radiation constant a_R = 4 sigma_B / c in J/(m^3 K^4)."""

# Electron mass in eV
m_e_eV = 0.51099895e6
"""Electron mass in eV/c^2."""

# Proton mass in eV
m_p_eV = 938.272046e6
"""Proton mass in eV/c^2."""

# Thomson cross section in m^2
sigma_T = 6.6524616e-29
"""Thomson scattering cross section in m^2, cf. CLASS thermodynamics.h:708."""

# Hydrogen ionization energy in eV
E_ion_H = 13.605693122994
"""Hydrogen ionization energy in eV."""

# Lyman alpha energy in eV
E_Lya = 10.2
"""Hydrogen Lyman-alpha transition energy in eV."""

# Two-photon decay rate of hydrogen 2s state in s^-1
Lambda_2s_H = 8.2245809
"""Two-photon decay rate of hydrogen 2s level in s^-1."""
