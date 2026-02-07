"""Parameter containers for jaxCLASS.

CosmoParams: cosmological parameters, JAX-traced (for autodiff).
PrecisionParams: numerical precision settings, static (not traced).
BackgroundResult: output of the background module.

All result types are frozen dataclasses registered as JAX pytrees.

References:
    CLASS source: include/background.h (struct background)
    CLASS source: include/precisions.h
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional

import jax
import jax.numpy as jnp

from jaxclass.constants import T_cmb_default


# ---------------------------------------------------------------------------
# CosmoParams: traced by JAX for autodiff
# ---------------------------------------------------------------------------

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CosmoParams:
    """Cosmological parameters. All float fields are JAX-traceable.

    These are the parameters that gradients are computed with respect to.
    The integer field N_ncdm is static (controls array shapes).

    Units follow CLASS conventions:
        - omega_b, omega_cdm: physical density parameters Omega_x h^2
        - h: dimensionless Hubble parameter H0/(100 km/s/Mpc)
        - T_cmb: CMB temperature in Kelvin
        - m_ncdm: neutrino mass sum in eV (single species degenerate approx)
        - k_pivot: pivot scale in Mpc^-1
    """

    # Hubble
    h: float = 0.6736

    # Densities
    omega_b: float = 0.02237      # Omega_b h^2
    omega_cdm: float = 0.1200     # Omega_cdm h^2

    # CMB
    T_cmb: float = T_cmb_default  # Kelvin

    # Neutrinos
    N_ur: float = 2.0328          # ultra-relativistic species (effective)
    N_ncdm: int = 1               # number of massive species (STATIC, not traced)
    m_ncdm: float = 0.06          # total mass in eV (single species)
    T_ncdm_over_T_cmb: float = 0.71611  # (4/11)^(1/3)
    deg_ncdm: float = 1.0         # degeneracy factor

    # Primordial
    ln10A_s: float = 3.044
    n_s: float = 0.9649
    alpha_s: float = 0.0          # running of spectral index
    r_t: float = 0.0              # tensor-to-scalar ratio
    n_t: float = 0.0              # tensor spectral index
    k_pivot: float = 0.05         # Mpc^-1

    # Reionization
    tau_reio: float = 0.0544

    # Dark energy (CPL: w(a) = w0 + wa*(1-a))
    w0: float = -1.0
    wa: float = 0.0
    cs2_fld: float = 1.0          # DE sound speed squared

    # Curvature (v1: must be 0)
    Omega_k: float = 0.0

    # Helium
    Y_He: float = 0.2454006

    # --- PyTree registration ---
    def tree_flatten(self):
        # All float fields are children (traced); N_ncdm is aux (static)
        children = []
        child_names = []
        for f in fields(self):
            if f.name == "N_ncdm":
                continue
            children.append(getattr(self, f.name))
            child_names.append(f.name)
        aux_data = {"N_ncdm": self.N_ncdm, "child_names": tuple(child_names)}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        child_names = aux_data["child_names"]
        kwargs = dict(zip(child_names, children))
        kwargs["N_ncdm"] = aux_data["N_ncdm"]
        return cls(**kwargs)

    def replace(self, **kwargs) -> CosmoParams:
        """Return a new CosmoParams with specified fields replaced."""
        current = {f.name: getattr(self, f.name) for f in fields(self)}
        current.update(kwargs)
        return CosmoParams(**current)


# ---------------------------------------------------------------------------
# PrecisionParams: NOT traced by JAX (static, controls array shapes)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrecisionParams:
    """Numerical precision parameters. These are NOT JAX-traced.

    They control grid sizes, tolerances, and solver settings. Changing them
    changes the shapes of arrays, so they must be compile-time constants
    for JIT.
    """

    # Background
    bg_n_points: int = 800          # number of log(a) grid points
    bg_a_ini_default: float = 1e-7  # initial scale factor (needs to be early enough for high-k perturbation ICs)
    bg_tol: float = 1e-10           # ODE tolerance

    # NCDM (massive neutrino) quadrature
    ncdm_q_size: int = 15           # number of momentum bins
    ncdm_q_max: float = 15.0        # max comoving momentum q
    ncdm_bg_n_points: int = 512     # points for pre-tabulation grid

    # Thermodynamics
    th_z_max: float = 5e4           # max redshift for recombination
    th_n_points: int = 10000        # number of z grid points
    th_tol: float = 1e-5            # ODE tolerance

    # Perturbations
    pt_k_min: float = 1e-5          # Mpc^-1
    pt_k_max_cl: float = 5.0        # Mpc^-1 (for C_l computation)
    pt_k_max_pk: float = 50.0       # Mpc^-1 (for P(k) output)
    pt_k_per_decade: int = 30
    pt_l_max_g: int = 17            # photon Boltzmann hierarchy
    pt_l_max_pol_g: int = 17        # photon polarization hierarchy
    pt_l_max_ur: int = 17           # massless neutrino hierarchy
    pt_l_max_ncdm: int = 17         # massive neutrino hierarchy
    pt_tau_n_points: int = 5000
    pt_ode_rtol: float = 1e-5
    pt_ode_atol: float = 1e-10

    # Transfer
    tr_l_max_scalars: int = 2500
    tr_l_max_tensors: int = 500
    tr_l_limber_switch: int = 100
    tr_limber_transition_width: float = 20.0

    # Harmonic
    hr_k_per_decade: int = 40

    # Lensing
    le_l_max: int = 2500

    # ODE solver settings
    ode_max_steps: int = 65536
    ode_adjoint: str = "recursive_checkpoint"  # or "direct"

    @staticmethod
    def fast_cl():
        """Fast preset for C_l iteration.

        k_max=0.15 limits to where hierarchy (l_max=25) is well-converged.
        """
        return PrecisionParams(
            pt_k_max_cl=0.15,        # k*tau_visibility < l_max → hierarchy valid
            pt_k_per_decade=15,
            pt_tau_n_points=2000,
            pt_l_max_g=25,           # larger hierarchy for better convergence
            pt_l_max_pol_g=25,
            pt_l_max_ur=25,
            pt_ode_rtol=1e-4,
            pt_ode_atol=1e-8,
            ode_max_steps=65536,
        )

    @staticmethod
    def medium_cl():
        """Medium precision preset for C_l computation.

        l_max=50 reduces hierarchy truncation artifacts at high multipoles.
        k_max=0.3 extends to smaller scales. More expensive than fast_cl.
        """
        return PrecisionParams(
            pt_k_max_cl=0.3,         # extend to smaller scales
            pt_k_per_decade=20,
            pt_tau_n_points=3000,
            pt_l_max_g=50,           # reduce truncation artifacts
            pt_l_max_pol_g=50,
            pt_l_max_ur=50,
            pt_ode_rtol=1e-5,
            pt_ode_atol=1e-10,
            ode_max_steps=65536,
        )

    @staticmethod
    def science_cl():
        """Science-quality preset targeting <1% C_l accuracy at l=2-2500.

        k_max=0.35 covers l~2500 via k*chi_star. l_max=50 for converged source
        functions. 60 k/decade gives ~13 points per BAO oscillation at k_rec.
        ~270 k-modes total. ~4 min per forward pass on CPU.
        """
        return PrecisionParams(
            pt_k_max_cl=0.35,        # covers l~2500 via k*chi_star
            pt_k_per_decade=60,      # ~13 points/BAO oscillation
            pt_tau_n_points=4000,
            pt_l_max_g=50,           # converged hierarchy
            pt_l_max_pol_g=50,
            pt_l_max_ur=50,
            pt_ode_rtol=1e-6,
            pt_ode_atol=1e-11,
            ode_max_steps=131072,
        )
