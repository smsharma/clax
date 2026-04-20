#!/usr/bin/env python3
"""Generate selected CLASS linear-matter P(k) references on the fiducial k-grid.

This script rewrites the requested ``reference_data/*/pk.npz`` files using the
same 500-point logarithmic ``k`` grid stored in
``reference_data/lcdm_fiducial/pk.npz``. The output schema matches the fiducial
reference as closely as possible:

- ``k``
- ``pk_m_lin_z0`` and legacy alias ``pk_lin_z0``
- ``pk_cb_lin_z0``
- ``pk_m_z{z}``, ``pk_cb_z{z}``, and legacy alias ``pk_z{z}`` for
  ``z in {0.0, 0.5, 1.0, 2.0}``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


REFERENCE_DIR = Path(__file__).resolve().parents[1] / "reference_data"
FIDUCIAL_PK_PATH = REFERENCE_DIR / "lcdm_fiducial" / "pk.npz"
Z_PK = (0.0, 0.5, 1.0, 2.0)

FIDUCIAL_CLASS_PARAMS = {
    "h": 0.6736,
    "omega_b": 0.02237,
    "omega_cdm": 0.1200,
    "A_s": 2.1e-9,
    "n_s": 0.9649,
    "tau_reio": 0.0544,
    "N_ur": 2.0328,
    "N_ncdm": 1,
    "m_ncdm": 0.06,
    "T_ncdm": 0.71611,
    "output": "mPk dTk",
    "P_k_max_1/Mpc": 50.0,
    "z_max_pk": max(Z_PK),
    "tol_background_integration": 1e-12,
}


def _background_density_at_z(bg, z: float, key: str) -> float:
    """Return a CLASS background density interpolated to redshift ``z``."""
    z_bg = np.asarray(bg["z"])[::-1]
    rho_bg = np.asarray(bg[key])[::-1]
    return float(np.interp(z, z_bg, rho_bg))


def _compute_pk_components(cosmo, k_eval: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray]:
    """Return CLASS total-matter and cb-only linear spectra at redshift ``z``."""
    pk_m = np.asarray([cosmo.pk_lin(float(k), z) for k in k_eval])
    transfer = cosmo.get_transfer(z)
    bg = cosmo.get_background()

    k_transfer = np.asarray(transfer["k (h/Mpc)"]) * cosmo.h()
    rho_b = _background_density_at_z(bg, z, "(.)rho_b")
    rho_cdm = _background_density_at_z(bg, z, "(.)rho_cdm")

    delta_b = np.asarray(transfer["d_b"])
    delta_cdm = np.asarray(transfer["d_cdm"])
    delta_tot = np.asarray(transfer["d_tot"])
    delta_cb = (rho_b * delta_b + rho_cdm * delta_cdm) / (rho_b + rho_cdm)
    cb_to_m_ratio = (delta_cb / delta_tot) ** 2
    pk_cb = pk_m * np.interp(np.log(k_eval), np.log(k_transfer), cb_to_m_ratio)
    return pk_m, pk_cb


def _model_params() -> dict[str, dict[str, float | int | str]]:
    """Return CLASS parameter dictionaries for the requested reference models."""
    models: dict[str, dict[str, float | int | str]] = {}

    def with_overrides(**overrides):
        params = dict(FIDUCIAL_CLASS_PARAMS)
        params.update(overrides)
        return params

    models["massive_nu_015"] = with_overrides(m_ncdm=0.15)
    models["w0wa_m09_01"] = with_overrides(w0_fld=-0.9, wa_fld=0.1, Omega_Lambda=0.0)

    for name in ("omega_cdm_high", "omega_cdm_low", "omega_b_high", "omega_b_high", "tau_high", "tau_low", "ns_high", "ns_low", "h_high", "h_low"):
        params_path = REFERENCE_DIR / name / "params.npz"
        params_npz = np.load(params_path)
        overrides = {key: float(params_npz[key]) for key in params_npz.files}
        models[name] = with_overrides(**overrides)

    return models


def regenerate_selected_pk_references() -> None:
    """Regenerate the selected ``pk.npz`` files from CLASS."""
    from classy import Class

    k_eval = np.asarray(np.load(FIDUCIAL_PK_PATH)["k"])
    models = _model_params()

    print(f"Using fiducial k-grid from {FIDUCIAL_PK_PATH}")
    print(f"k-grid length={len(k_eval)}, k_min={k_eval[0]:.6g}, k_max={k_eval[-1]:.6g}")

    for name, params in models.items():
        cosmo = Class()
        cosmo.set(params)
        cosmo.compute()

        payload: dict[str, np.ndarray] = {"k": k_eval}
        for z in Z_PK:
            pk_m, pk_cb = _compute_pk_components(cosmo, k_eval, z)
            payload[f"pk_m_z{z}"] = pk_m
            payload[f"pk_cb_z{z}"] = pk_cb
            payload[f"pk_z{z}"] = pk_m
            if z == 0.0:
                payload["pk_m_lin_z0"] = pk_m
                payload["pk_cb_lin_z0"] = pk_cb
                payload["pk_lin_z0"] = pk_m

        out_path = REFERENCE_DIR / name / "pk.npz"
        np.savez(out_path, **payload)
        cosmo.struct_cleanup()
        cosmo.empty()

        print(f"  wrote {out_path}")


if __name__ == "__main__":
    regenerate_selected_pk_references()
