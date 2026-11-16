"""Parameter containers for clax.

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

from clax.constants import T_cmb_default


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
    ln10A_s: float = 3.0445224377  # matches A_s=2.1e-9 exactly
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
    ncdm_q_size: int = 5            # perturbation hierarchy q-bins (CLASS default: 5)
    ncdm_bg_q_size: int = 30        # background integration q-points (CLASS default: 11)
    ncdm_q_max: float = 15.0        # max comoving momentum q
    ncdm_bg_n_points: int = 512     # points for pre-tabulation grid
    ncdm_fluid_approximation: str = "class"  # mb, hu, class, none (CLASS default: class)
    ncdm_fluid_trigger_tau_over_tau_k: float = 31.0  # CLASS default late-time switch

    # Thermodynamics
    # 5e4 is a practical floor, not a tunable knob: the thermo table's first
    # knot (z=th_z_max) must map to a conformal time earlier than perturbation
    # start tau_ini (~0.1-0.5 Mpc), because CubicSpline.evaluate clips to the
    # table boundary. Below the first knot, kappa_dot would freeze at its
    # boundary value instead of scaling as a^-2. Measured: th_z_max=5e3 puts
    # the first knot at tau=80.7 Mpc (> tau_ini) and gives P(k=0.01)=4.77e15
    # (wrong); th_z_max=5e4 gives P(k=0.01)=7.94e4, matching th_z_max=5e5 to
    # 2e-6 while running ~6x faster than 5e3 (60s vs 366s at k=0.01).
    th_z_max: float = 5e4           # max redshift for recombination
    th_n_points: int = 20000        # number of z grid points
    th_tol: float = 1e-5            # ODE tolerance
    # Reverse-mode AD rule for the fused background+thermodynamics solve
    # (clax.thermodynamics.solve_background_and_thermo), issue #30:
    #   "stable" — jax.custom_vjp whose backward pass computes the params
    #              cotangent via a batched forward-mode basis (jacfwd over
    #              the CosmoParams leaves). Fixes the ~2% h-gradient error
    #              from catastrophic cancellation in the recombination-era
    #              native backward pass. Forward-mode (jax.jvp) through the
    #              fused solve is BLOCKED in this mode (custom_vjp caveat).
    #   "native" — plain background_solve + thermodynamics_solve calls;
    #              required for jax.jvp/jacfwd through the fused solve.
    # Mirrors the ode_adjoint precedent below (static string flag).
    th_grad_mode: str = "stable"    # or "native"

    # Perturbations
    pt_k_min: float = 1e-5          # Mpc^-1
    pt_k_max_cl: float = 5.0        # Mpc^-1 (for C_l computation)
    pt_k_max_pk: float = 50.0       # Mpc^-1 (for P(k) output)
    pt_k_per_decade: int = 30
    pt_k_grid_type: str = "log"     # "log" | "chebyshev" (Lobatto nodes in log10 k; issue #31)
    pt_l_max_g: int = 17            # photon Boltzmann hierarchy
    pt_l_max_pol_g: int = 17        # photon polarization hierarchy
    pt_l_max_ur: int = 17           # massless neutrino hierarchy
    pt_l_max_ncdm: int = 17         # massive neutrino hierarchy
    pt_tau_n_points: int = 5000
    pt_ode_rtol: float = 1e-5
    pt_ode_atol: float = 1e-10
    # Radiation streaming approximation (RSA) triggers, as CLASS precisions.h:
    # rsa_on once tau/tau_k > radiation_streaming_trigger_tau_over_tau_k (45) and
    # tau > tau_free_streaming, the first time after recombination at which the
    # Thomson mean free path exceeds radiation_streaming_trigger_tau_c_over_tau
    # (5) conformal times, tau_c/tau = 1/(kappa' tau) > 5 (thermodynamics.c;
    # ThermoResult.tau_free_streaming). Until Oct 12, 2026 the second condition
    # was kappa'/(aH) < 5, which fires ~70 Mpc too early, with the visibility
    # function still at 69% of its peak (docs/highl_residual_2026-10-12.md,
    # branch exp/highl-investigation).
    pt_rsa_trigger_tau_over_tau_k: float = 45.0
    pt_rsa_trigger_tau_c_over_tau: float = 5.0

    # Transfer
    tr_l_max_scalars: int = 2500
    tr_l_max_tensors: int = 500
    tr_l_limber_switch: int = 100
    tr_limber_transition_width: float = 20.0

    # Harmonic
    hr_k_per_decade: int = 40
    hr_n_k_fine: int = 5000          # fine k-grid for source interpolation
    hr_l_max: int = 2500             # max multipole for C_l computation
    hr_l_limber: int = 0             # use Limber approx for l >= this (0=disabled)
    # Mpc; upper bound on the conformal age tau_0 that fixes the fast path's
    # Bessel-table range, x_max = 1.05 pt_k_max_cl hr_tau0_max, independent of
    # cosmology so the table is built once (#54, ADR 0005). Fiducial tau_0 is
    # ~14100 Mpc; compute_cls_all_fast raises (also under jit) if tau_0 exceeds
    # this bound. Raising it widens the table proportionally (82 MiB at fit_cl).
    hr_tau0_max: float = 20000.0

    # Lensing
    le_l_max: int = 2500

    # ODE solver settings
    ode_max_steps: int = 65536
    ode_adjoint: str = "recursive_checkpoint"  # or "direct"
    # Perturbation ODE solver.
    #   "kvaerno5" — 5th-order ESDIRK (diffrax default, robust, Newton iteration)
    #   "rodas5"   — 5th-order Rosenbrock (no Newton, LU only; ~1.3x faster on
    #                CPU, potentially 3-5x on GPU; batched automatically for
    #                table solves)
    pt_ode_solver: str = "kvaerno5"

    # Memory management
    # >0 exact chunk size, 0 backend-aware auto-batching, <0 force full vmap.
    pt_k_chunk_size: int = 0

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
            # Oct 2026: tolerance rung carried over from the planck_cl sweep
            # (docs/precision_sweep_2026-10-10.md), not measured on this
            # preset's 20 k/decade, k_max 0.3 grid. At l_max 50, 60 k/decade,
            # k_max 1.0 (config E6, Kvaerno5) rtol 1e-4 / atol 1e-6 differs
            # from 1e-6 / 1e-11 by 0.094% at l=2 and 0.003% at l>=30; this
            # preset's old 1e-5 / 1e-10 and the new rung agree to < 0.01% for
            # l>=30 at the 12/10/17 hierarchy (sweep configs F vs G6).
            pt_ode_rtol=1e-4,
            pt_ode_atol=1e-6,
            # 4096 was sufficient with Kvaerno5 at rtol 1e-4 / atol 1e-8
            # (sweep config G4, hierarchy 12/10/17); 8192 carried over from
            # planck_cl, not measured on this grid (Kvaerno5 here by default).
            ode_max_steps=8192,
        )

    @staticmethod
    def science_cl():
        """Science-quality preset targeting <1% C_l accuracy at l=20-150.

        k_max=0.35, l_max=50. 200 k/decade resolves the oscillatory T_l(k)
        transfer function (Bessel period ~2.3e-4 Mpc^-1 vs k-spacing ~1.2e-4).
        ~900 k-modes total. Achieves sub-percent EE at l=12-100 and TT at
        l=20-150 (except l=100 at ~2%). ~10 min on V100 GPU.
        """
        return PrecisionParams(
            pt_k_max_cl=0.35,        # covers l~2500 via k*chi_star
            pt_k_per_decade=200,     # resolves T_l(k) Bessel oscillations
            pt_tau_n_points=4000,
            pt_l_max_g=50,           # converged hierarchy
            pt_l_max_pol_g=50,
            pt_l_max_ur=50,
            # Oct 2026: tolerance rung carried over from the planck_cl sweep
            # (docs/precision_sweep_2026-10-10.md), not measured on this
            # preset's 200 k/decade, k_max 0.35 grid. At l_max 50, 60 k/decade,
            # k_max 1.0 (config E6, Kvaerno5) rtol 1e-4 / atol 1e-6 differs
            # from the old 1e-6 / 1e-11 by 0.094% at l=2 and 0.003% at l>=30
            # (config E, rtol 1e-4 / atol 1e-8, is the one identical to 0.005%).
            pt_ode_rtol=1e-4,
            pt_ode_atol=1e-6,
            # 4096 was sufficient with Kvaerno5 at rtol 1e-4 / atol 1e-8
            # (sweep config G4, hierarchy 12/10/17); 8192 carried over from
            # planck_cl, not measured on this grid (Kvaerno5 here by default).
            ode_max_steps=8192,
        )

    @staticmethod
    def planck_cl():
        """Science preset, measured as shipped before #54, i.e. with the fast
        path's LINEAR Bessel lookup at 30 points per unit x, rebuilt on every
        call (rodas5; job 24668, before #53;
        V100, compute_cls_all_fast at n_k_fine 20000) against CLASS with a
        massless neutrino and RECFAST: TT within 0.086% (l <= 30), 0.233%
        (l <= 500), 0.262% (l <= 1000); EE within 0.60% (l = 2-4), 0.46%
        (l = 5-500), 0.98% (l = 5-1000); TE/sqrt(TT EE) within 0.37% (0.33%
        in job 24668, 0.37% in job 24639). The 0.98% EE number to l = 1000 is
        a single multipole at l ~ 526 and is bounded by the sparse-l-grid
        interpolation of the C_l stage, not by this preset: sparse_l_grid
        steps by 20 to l = 500 and by 50 from 550, and splining CLASS's own
        spectra on that grid gives EE 1.13% at l = 526 and TT 0.16% at l = 527
        from the grid alone (CHANGELOG, Oct 12, 2026, follow-ups). Stages on
        a V100, cached: thermodynamics ~63 s, perturbations ~30 s massless
        (~47 s with the preset's 5 neutrino q-bins, which
        scripts/benchmark_speed.py keeps at m_ncdm = 0), harmonic ~13 s;
        ~124 s end to end. After the #53 recombination rewrite (job 27000,
        Oct 20, 2026): TT 0.086 / 0.238 / 0.271%, EE 0.596 / 0.455 / 0.981%,
        TE/sqrt(TT EE) 0.327%, within 0.01 pp of the above; stages
        thermodynamics 59.6 s, perturbations 27.8 s, harmonic 13.5 s,
        101.5 s end to end. With #54's cubic Hermite lookup at 3 points per
        unit x (table built once per process), against the same tree without
        it (main 5ac3df3, which has the CLASS radiation-streaming trigger;
        tests/test_preset_accuracy.py::TestPlanckClPreset, V100, jobs 30151 /
        30152): EE at l = 2-4 0.596% -> 0.151%, the linear table's l = 2 bias
        removed; every other band within 0.04 percentage points (TT 0.086 ->
        0.079 / 0.238 -> 0.245 / 0.370 -> 0.375%, EE 5-500 0.455 -> 0.450%,
        5-1000 0.981 -> 0.986%, TE/sqrt(TT EE) 0.327 -> 0.287%). The stage
        timings above were not re-measured with it.

        Retuned Oct 2026 (docs/precision_sweep_2026-10-10.md). The previous
        values (Kvaerno5, l_max 50, rtol 1e-6, atol 1e-11, max_steps 131072)
        cost 4606 s per perturbation solve on a V100 and give the same spectra
        to 0.1%: the solve is step-count-bound and the tolerance ladder is
        converged at rtol 1e-4 / atol 1e-6 for every hierarchy size, so the
        retune costs nothing at l <= 2000. l_max 17 for all species keeps the
        photon-truncation effect in the reionization EE bump below 0.05%
        (12/10/17 gives 0.18%). hr_n_k_fine 20000 together with the fast
        path's 30-point-per-unit-x linear Bessel table (since #54 the default
        is the cubic Hermite lookup at 3 points per unit x, which agrees with
        the Hermite lookup at 30 to 1.6e-5 in TT at fit_cl, jobs
        27208-27214) removes the former -1% TT
        offset at l < 50 and the +20% EE error at l=2. Above l ~ 1000 the
        C_l stage is percent-level (TT rms 1.7-2% at l = 1000-2000, EE 3-8%,
        common to the exact-j_l and table paths: perturbation k sampling /
        source interpolation); 120 k/decade halves that at ~1.4x the cost.
        300 k-modes.
        th_n_points: 100000 is kept for the science preset; 20000 (as in
        planck_fast) changes C_l by <= 0.02% to l=500, <= 0.05% to l=1000 and
        ~0.13% above, and cuts thermodynamics from ~59 s to ~13 s on a V100.
        Both numbers predate the Saha -> ODE recombination staging (#48, Oct
        2026): 100000 was chosen for the former explicit Heun stepping and is
        not re-tuned for the staging, where x_e at 20000 points is within
        1.1e-4 of 100000 over z 200-8000 (largest at z = 200), both 2.0e-4 from
        CLASS (the HeIII-era offset near z ~ 5900); the C_l effect of 20000 on
        the staged recombination is unmeasured.
        Sweep history (Kvaerno5 candidate, configs P1/P1m, V100 cached):
        perturbations 105 s (massless), 189 s (one 0.06 eV neutrino, 5
        q-bins), harmonic 13 s; TT within 0.26% and EE within 0.31%
        (l = 30-500) of CLASS massless RECFAST up to l = 1000. rodas5 (default
        since Oct 12, 2026) matches that candidate to 0.08% at l=2 and
        <= 0.03% elsewhere at 29.5 s vs 110.6 s of perturbations;
        thermodynamics (~60 s at th_n_points 100000) then dominates.
        GPU memory: with n_k_fine 20000 and n_tau 5000 every (n_k, n_tau)
        working array of the fast C_l path is 1e8 doubles (800 MB) and the
        stage materialises several of them, so budget of order 10 GB or more;
        validated only on a 32 GB V100.
        """
        return PrecisionParams(
            pt_k_max_cl=1.0,         # covers l~2500 with margin
            # 30 / 60 / 120 per decade give identical C_l only to l <= 300
            # (sweep G6k30 / G6k120); at 60 the C_l stage is at TT rms 1.7-2%
            # over l = 1000-2000; 120 halves the l > 1000 error at ~1.4x the
            # perturbation cost.
            pt_k_per_decade=60,
            pt_tau_n_points=5000,    # 10000 and 20000 identical; 2000 costs 0.6% low-l EE
            th_n_points=100000,      # 20000: <=0.05% to l=1000, 46 s cheaper; not re-tuned for #48
            pt_l_max_g=17,
            pt_l_max_pol_g=17,
            pt_l_max_ur=17,
            pt_ode_rtol=1e-4,
            pt_ode_atol=1e-6,
            # 4096 was sufficient with Kvaerno5 at rtol 1e-4 / atol 1e-8 (sweep
            # config G4, hierarchy 12/10/17); rodas5's step count differs, and
            # the validation of this preset passed at 8192 (job 24668).
            ode_max_steps=8192,
            hr_n_k_fine=20000,       # 5000 is percent-level above l~500
            # Oct 12, 2026: rodas5 with cubic Hermite dense output (rosenbrock.py)
            # matches kvaerno5 to 0.08% at l=2 and <= 0.03% elsewhere at 3.7x
            # less perturbation time (29.5 s vs 110.6 s on a V100).
            pt_ode_solver="rodas5",
        )

    @staticmethod
    def planck_fast():
        """planck_cl's perturbation and C_l-stage settings with a cheaper
        thermodynamics grid and a shorter l range.

        Same perturbation stage as planck_cl (k_max 1.0, 60 k/decade, 5000
        tau points, l_max 17/17/17, rtol 1e-4 / atol 1e-6, max_steps 8192,
        rodas5 with the cubic Hermite dense output) and the same fast-path
        C_l settings (hr_n_k_fine 20000, the default Hermite Bessel lookup at
        3 points per unit x since #54; 30-point linear lookup before),
        with two changes:
        - th_n_points 20000 (planck_cl: 100000): thermodynamics ~13 s instead
          of ~59-63 s on a V100; C_l differ by <= 0.02% to l=500, <= 0.05%
          to l=1000 and ~0.13% above (precision-sweep candidate runs, Oct 12,
          2026, before #53; not re-tuned for the Saha -> ODE recombination
          staging, see planck_cl).
        - hr_l_max 1500 (planck_cl: 2500): the table C_l stage scales as
          n_l x n_k_fine x n_tau, so the shorter l range roughly halves the
          harmonic work. planck_cl's stage is ~13 s at l_max 2500 with
          n_k_fine 20000 on a V100; planck_fast's is not separately
          benchmarked (scripts/benchmark_speed.py has no planck_fast entry).

        Accuracy to l=1000 is planck_cl's up to the th_n_points differences
        above; this preset is not separately validated against CLASS
        (tests/test_preset_accuracy.py covers planck_cl and fit_cl). It is
        the --fast preset of the matched-species perturbation comparisons in
        tests/test_perturbations.py.
        GPU memory: with n_k_fine 20000 and n_tau 5000 every (n_k, n_tau)
        working array of the fast C_l path is 1e8 doubles (800 MB) and the
        stage materialises several of them, so budget of order 10 GB or more;
        validated only on a 32 GB V100.
        """
        return PrecisionParams(
            pt_k_max_cl=1.0,         # same as planck_cl
            pt_k_per_decade=60,      # same as planck_cl
            pt_tau_n_points=5000,    # same as planck_cl
            th_n_points=20000,       # 5x fewer than planck_cl (100000)
            pt_l_max_g=17,           # same as planck_cl (Oct 2026 retune)
            pt_l_max_pol_g=17,
            pt_l_max_ur=17,
            pt_ode_rtol=1e-4,        # same as planck_cl
            pt_ode_atol=1e-6,        # same as planck_cl
            ode_max_steps=8192,      # same as planck_cl
            hr_n_k_fine=20000,       # same as planck_cl
            hr_l_max=1500,           # halves the n_l x n_k_fine x n_tau table work vs 2500
            pt_ode_solver="rodas5",  # same as planck_cl (Oct 12, 2026)
        )

    @staticmethod
    def fit_cl():
        """Fast preset for fitting / HMC, targeting <2% C_l accuracy at l<600
        with the default massive neutrino (5 q-bins, l_max_ncdm 8, no fluid
        approximation; #57).

        Reduces resolution everywhere for speed (planck_cl values in
        parentheses):
        - 20 k/decade (60) -> ~100 k-modes (300)
        - l_max 17/17/17 hierarchy (17; the same since the Oct 2026 retune,
          which moved planck_cl from 50 and fit_cl from 25)
        - 2000 tau points (5000)
        - 3000 thermodynamics points (100000)
        - ncdm_q_size 5 (5), pt_l_max_ncdm 8 (17), ncdm_fluid_approximation
          "none" ("class"): the massive-neutrino hierarchy at CLASS's q-bin
          count, truncated at l = 8 and followed as a hierarchy to today (106
          state variables; 61 without the hierarchy, 154 at l_max 17 with the
          fluid variables). Until Oct 20, 2026 the preset set ncdm_q_size 0,
          which paired the species' massive background density with the
          massless-neutrino perturbations (#57): TT 14.6-15.4% off CLASS at
          l <= 30 at every LCDM grid point (job 27008). Measured against CLASS
          (HyRec) references (diags/diag_fitcl_qgrid.py, jobs 27173-27746,
          Oct 20-22, 2026; docs/fitcl_qgrid_2026-10-20.md), as TT 2-30 /
          TT 2-500 / EE 5-500 / TE/sqrt(TT EE) 2-500:
            LCDM grid (0.06 eV)  0.47-0.54 / 0.56-0.69 / 1.20-1.65 / 0.87-1.13%
            0.15 eV              0.48 / 0.52 / 1.18 / 0.87%
            0.30 eV              0.52 / 0.64 / 1.16 / 0.86%
            0.50 eV              0.53 / 0.63 / 1.19 / 0.85%
            massless (RECFAST)   0.48 / 0.48 / 1.21%.
          Why these three values. The Gauss-Laguerre q-grid is converged at 5
          bins (8 and 12 bins reproduce it to 0.02%). With the "class" fluid
          approximation TT at l <= 30 carries a deficit that grows with the
          mass, 0.60 / 0.71 / 0.98 / 1.29% at 0.06 / 0.15 / 0.30 / 0.50 eV,
          and P_m is 0.94% off CLASS at k = 4e-3 Mpc^-1 at 0.30 eV: the late
          ISW is sourced at k ~ 1e-3-1e-2 Mpc^-1 below z ~ 2, where k tau > 31
          and clax's implementation of that approximation (#46) is active;
          with the hierarchy kept to today both read 0.5% and 0.1%, flat in
          mass. 3 q-bins sit closer to CLASS with the approximation on
          (0.47-0.61%) only because two errors cancel; with it off they read
          0.93% at 0.50 eV. Cost: 18.5-21.6 s end to end on a V100 across two
          nodes (jobs 27746, 27711; PT 8.9-11.0 s), against 19.8-20.3 s for 3
          bins with the approximation and 19.1 s for the old preset (jobs
          27173, 27711). l_max_ncdm 8 vs 17: with the approximation on the
          C_l agree to 0.011% at l <= 500 at every measured point
          (tests/test_preset_accuracy.py test_lmax_ncdm_truncation guards
          0.1%, measured <= 0.010% with the approximation off); in linear P_m
          the truncation raises the small-scale power by +0.06% at k = 0.05
          rising to +0.9% at k = 1 Mpc^-1 (+0.6% with the approximation on),
          the same at every mass and solver tolerance and with the batched
          solver taking the same 324 steps:
          the species' free streaming while still relativistic, not step
          control (diags/diag_pk_batched_steps.py). fit_cl is a C_l preset;
          its linear P_m is 1.2-1.5% off CLASS near k = 0.2 Mpc^-1 for every
          hierarchy (the 20 k per decade), not changed here.
        - ode_max_steps 8192 (8192; was 1024)
        - rtol 1e-4 / atol 1e-6 (the same; Oct 2026): the converged rung of
          the tolerance ladder. The old 1e-3 / 1e-4 was never validated at
          low l and leaves 1.8% TT at l=2 even with the Hermite dense output.
        - fast all-l table Bessel (hr_n_k_fine 20000, hr_l_max 1500; planck_cl
          20000, 2500): precomputed j_l(x) and j_l'(x) tables with the
          T0+T1+T2 transfer contributions, one JIT instead of one per sparse
          l (83 of them at l_max 1500). hr_n_k_fine was 5000 until #54: with
          5000 fine-k points the trapezoid integral over k samples |T_l|^2
          about once per oscillation near l ~ 1200, so dC_l/dh flips sign
          along h; 20000 makes it monotone at +3.7% forward time (measured
          against 5000, eager, V100).

        V100 timings with the shipped hierarchy (diags/diag_fitcl_qgrid.py,
        cached medians): job 27746 (LCDM grid) BG 0.5 s, TH 1.7 s, PT 8.9 s,
        HR 7.3 s, 18.5 s end to end; job 27711 (massive-neutrino grid, another
        node) PT 11.0 s, 21.6 s; first call 85 s (compile + run). The same
        script on job 27173's node: 19.1 s with the old ncdm_q_size 0 (PT
        7.2 s), 19.8 s with 3 bins and the fluid approximation (PT 9.3 s),
        21.1 s with 5 bins and the approximation (PT 11.0 s), 23.5 s at
        l_max_ncdm 17 (PT 13.5 s).

        Earlier records, all with ncdm_q_size 0. V100 timings, cached, rodas5
        at rtol 1e-4 / atol 1e-6, from two
        scripts that time the harmonic stage differently (quote the script
        with the number). The timings and CLASS accuracies below were
        measured at n_k_fine 5000, before it was raised to 20000 (+3.7%
        forward time; TT at l = 100-1200 moves by up to 1.6% toward CLASS,
        EE/TE at l = 500-1200 by 2e-3 / 6e-3 away, job 27208), and before #54's
        fixed-range Bessel table (built once per process and reused) and
        Hermite lookup: diags/diag_preset_solver_choice.py (job 24652,
        config E; C_l to l_max 1500 with n_k_fine 5000): BG 0.7 s, TH 2.4 s,
        PT 9.8 s, HR 3.8 s, 16.8 s end to end (kvaerno5: 61.0 s);
        scripts/benchmark_speed.py (job 24668, hr_l_max 1500, n_k_fine 5000):
        BG 0.6 s, TH 2.1 s, PT 8.4 s, HR 7.6 s, 18.6 s end to end (job 24621:
        HR 7.2 s). Accuracy vs CLASS (massless RECFAST; job 24668, before
        #53): TT 0.46% at l<=30, 0.69% at l<=500; EE 1.25% at l=5-500; after
        #53 (job 27000): 0.46% / 0.46% / 1.19%, 16.9 s end to end. The
        perturbation k-grid limits l > 500 (TT -6.4% at l=1000).
        """
        return PrecisionParams(
            pt_k_max_cl=1.0,         # keep full k range for l coverage
            pt_k_per_decade=20,      # sparse perturbation k-grid (interp compensates)
            pt_tau_n_points=2000,    # sufficient for source resolution
            th_n_points=3000,        # RECFAST with fewer steps (was 5000)
            pt_l_max_g=17,           # CLASS default hierarchy (59 vs 83 state vars at l_max=25)
            pt_l_max_pol_g=17,
            pt_l_max_ur=17,
            ncdm_q_size=5,           # 0 until Oct 20, 2026 (#57: 15% TT at l<=30); converged (8, 12 bins agree)
            pt_l_max_ncdm=8,         # vs 17: C_l identical to 0.011% at l<=500; +0.6% in P_m at k=1 (docstring)
            ncdm_fluid_approximation="none",  # "class" mode (#46) costs 1% TT at l<=30 by 0.30 eV (docstring)
            pt_ode_rtol=1e-4,        # Oct 2026: converged rung of the tolerance ladder (was 1e-3)
            pt_ode_atol=1e-6,        # (was 1e-4)
            # was 1024; 4096 was sufficient with Kvaerno5 at rtol 1e-4 / atol
            # 1e-8 (sweep config G4); rodas5's step count differs, and the
            # validation passed at 8192 (job 24668).
            ode_max_steps=8192,
            # 5000 undersamples |T_l|^2 near l~1200: dC_l/dh flips sign along h (#54); planck_cl already 20000
            hr_n_k_fine=20000,
            hr_l_max=1500,           # reduced max multipole
            # rodas5 with cubic Hermite dense output (Oct 12, 2026). With the
            # earlier linear dense output the late ISW was wrong (TT -46% at
            # l=2 at these tolerances); now rodas5 matches kvaerno5 to 0.08%
            # at l=2 and 0.03% elsewhere at 9.8 s vs 54.7 s of perturbations.
            # rtol 1e-3 / atol 1e-4 would save another 5.6 s but leaves 1.8%
            # at l=2 and 1.3% at l=3.
            pt_ode_solver="rodas5",
        )
