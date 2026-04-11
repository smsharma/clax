"""Probe ncdm shear sensitivity to perturbation quadrature and hierarchy size.

This diagnostic is meant to answer a narrow question:

- does the late-time `theta_ncdm` / `shear_ncdm` mismatch move materially when
  we increase `ncdm_q_size`?
- does it move materially when we increase `pt_l_max_ncdm`?

If the answer is "yes", the main bug is likely quadrature/truncation quality.
If the answer is "no", the main bug is more likely in the `Psi_2` evolution
equation or a deeper convention mismatch.

Usage:
    python diags/diag_ncdm_shear_convergence.py
    python diags/diag_ncdm_shear_convergence.py --fast
    python diags/diag_ncdm_shear_convergence.py --k 0.1 --q-sizes 5,9,15 --lmax-values 17,25,35
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from clax.background import background_solve
from clax.params import PrecisionParams
from clax.thermodynamics import thermodynamics_solve
from tests.pk_test_utils import (
    class_perturbation_components,
    load_perturbation_reference,
    perturbation_match_tau_samples,
    solve_matched_perturbation_components,
)
from clax.params import CosmoParams


def rel_err(value: np.ndarray, reference: np.ndarray, eps: float = 1.0e-30) -> np.ndarray:
    """Return elementwise relative error with safe normalization."""
    return np.abs(value - reference) / (np.abs(reference) + eps)


def parse_int_list(text: str) -> list[int]:
    """Parse a comma-separated integer list."""
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def summarize_field(field: str, tau_samples: np.ndarray, clax_values: dict[str, np.ndarray], class_values: dict[str, np.ndarray]) -> str:
    """Return a one-line summary for one perturbation field."""
    err = rel_err(clax_values[field], class_values[field])
    worst_idx = int(np.argmax(err))
    final_idx = -1
    return (
        f"{field}: max_rel={100*err[worst_idx]:.2f}% at tau={tau_samples[worst_idx]:.1f} "
        f"final_rel={100*err[final_idx]:.2f}% "
        f"(clax_final={clax_values[field][final_idx]:+.6e}, class_final={class_values[field][final_idx]:+.6e})"
    )


def run_case(k: float, prec: PrecisionParams, tau_samples: np.ndarray, class_values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Solve one direct mode and return matched clax perturbation components."""
    params = CosmoParams()
    t0 = time.time()
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    elapsed = time.time() - t0
    clax_values = solve_matched_perturbation_components(
        params,
        prec,
        bg,
        th,
        k,
        tau_samples,
        tau_ini_mode="direct",
    )
    return {"elapsed_s": np.array([elapsed]), **clax_values}


def print_case(label: str, tau_samples: np.ndarray, clax_values: dict[str, np.ndarray], class_values: dict[str, np.ndarray]) -> None:
    """Print the key `ncdm` error summaries for one precision choice."""
    print(label)
    print(f"  setup: bg+th={float(clax_values['elapsed_s'][0]):.1f}s")
    print(f"  {summarize_field('delta_ncdm', tau_samples, clax_values, class_values)}")
    print(f"  {summarize_field('theta_ncdm', tau_samples, clax_values, class_values)}")
    print(f"  {summarize_field('shear_ncdm', tau_samples, clax_values, class_values)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Use the reduced late-time tau sample set")
    parser.add_argument("--k", type=float, default=0.05, help="Reference mode in Mpc^-1")
    parser.add_argument("--q-sizes", default="5,9,15", help="Comma-separated perturbation q sizes")
    parser.add_argument("--lmax-values", default="17,25,35", help="Comma-separated ncdm l_max values")
    args = parser.parse_args()

    k = float(args.k)
    q_sizes = parse_int_list(args.q_sizes)
    lmax_values = parse_int_list(args.lmax_values)

    ref_pert = load_perturbation_reference(k)
    tau_samples = perturbation_match_tau_samples(np.asarray(ref_pert["tau_Mpc"]), args.fast)
    base_prec = PrecisionParams.planck_fast()

    params = CosmoParams()
    base_bg = background_solve(params, base_prec)
    class_values = class_perturbation_components(base_bg, ref_pert, tau_samples)

    print(f"k={k:.3f} Mpc^-1")
    print("tau samples:", ", ".join(f"{tau:.1f}" for tau in tau_samples))
    print(
        f"CLASS final values: "
        f"delta_ncdm={class_values['delta_ncdm'][-1]:+.6e}  "
        f"theta_ncdm={class_values['theta_ncdm'][-1]:+.6e}  "
        f"shear_ncdm={class_values['shear_ncdm'][-1]:+.6e}"
    )

    print("\n=== q-size scan (fixed l_max_ncdm) ===")
    for q_size in q_sizes:
        prec = dataclasses.replace(base_prec, ncdm_q_size=q_size)
        clax_values = run_case(k, prec, tau_samples, class_values)
        print_case(
            f"q_size={q_size}, l_max_ncdm={prec.pt_l_max_ncdm}",
            tau_samples,
            clax_values,
            class_values,
        )

    print("\n=== l_max scan (fixed q_size) ===")
    for l_max in lmax_values:
        prec = dataclasses.replace(base_prec, pt_l_max_ncdm=l_max)
        clax_values = run_case(k, prec, tau_samples, class_values)
        print_case(
            f"q_size={prec.ncdm_q_size}, l_max_ncdm={l_max}",
            tau_samples,
            clax_values,
            class_values,
        )


if __name__ == "__main__":
    main()
