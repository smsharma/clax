"""fit_cl massive-neutrino q-grid sweep (smsharma/clax#57, Oct 20, 2026).

fit_cl ships with ``ncdm_q_size=0``, which pairs the 0.06 eV species' massive
background density with the massless-neutrino perturbations (#57): TT is 15%
off CLASS at l <= 30 at every LCDM grid point. This sweep measures fit_cl with
a real hierarchy, ``ncdm_q_size`` q-bins times ``pt_l_max_ncdm + 1``
multipoles, to pick the smallest grid that holds the preset's target bars and
to record its cost.

Configs (``PrecisionParams.fit_cl()`` with the two fields replaced):
    Q0    q 0, l_max_ncdm 17   shipped (the #57 baseline)
    Q3    q 3, l_max_ncdm 17
    Q4    q 4, l_max_ncdm 17
    Q5    q 5, l_max_ncdm 17   CLASS defaults (ncdm_q_size 5, l_max_ncdm 17)
    Q5L8  q 5, l_max_ncdm 8
    Q3L8  q 3, l_max_ncdm 8

Grids (``--grid``):
    lcdm  the five LCDM grid cosmologies (locked to tests/conftest.py
          ``COSMOLOGY_GRID_LCDM``, default 0.06 eV neutrino) against
          ``reference_data/<name>/cls.npz`` (CLASS defaults, HyRec), then the
          massless fiducial (``CosmoParams(m_ncdm=0.0)``) against
          ``reference_data/cls_massless_recfast.npz``.
    nu    the massive-neutrino points of ``COSMOLOGY_GRID_NULCDM`` that differ
          from fiducial: m_ncdm 0.06 (= fiducial), 0.15 and 0.30 (CLASS
          references ``reference_data/massive_nu_015`` and ``massive_nu_030``,
          the latter generated Oct 20, 2026), plus 0.50 eV beyond the test grid
          (reference generated on demand, not checked in). A point without a
          CLASS reference at report time is scored against the ``Q5`` run at
          the same point (marked "vs Q5"); run ``Q5`` first. A reference that
          appears after the run is picked up by ``--report`` ("ref added").

Bands as tests/test_preset_accuracy.py ``TestFitClPreset``: TT 2-30, TT 2-500,
EE 5-500, TEn 2-500 with TEn = (TE - ref) / sqrt(TT_ref EE_ref); TT 2-1000 is
reported for information only (fit_cl's k-grid limits l > 500). With ``--pk``
the linear P_m(k, z = 0) from ``clax.compute_pk_table`` (the ``Rodas5Batched``
path fit_cl uses for P(k)) is scored on the reference k grid restricted to
1e-3 <= k <= 1 Mpc^-1 against ``pk_m_lin_z0`` where a reference exists. The
first cosmology of each config compiles; the later ones are cached, and the
report quotes the median cached per-stage time. JSON is rewritten after every
cosmology, so a killed job keeps what it finished.

Usage (igpu compute node, /lustre/work/n2minh/std/clax/presets/fitcl-qgrid.sbatch):
    python diags/diag_fitcl_qgrid.py --configs Q0,Q3,Q4,Q5,Q5L8,Q3L8 --out r.json
    python diags/diag_fitcl_qgrid.py --configs Q5,Q5L8,Q3,Q3L8 --grid nu --pk --out nu.json
    python diags/diag_fitcl_qgrid.py --report r.json [--md r.md]
"""
import argparse
import dataclasses
import json
import os
import socket
import subprocess
import sys
import time
import traceback

sys.path.insert(0, ".")

import numpy as np

REF_MASSLESS = "reference_data/cls_massless_recfast.npz"
REF_CFG = "Q5"            # the differential reference where CLASS has none
PK_KRANGE = (1e-3, 1.0)   # Mpc^-1; fit_cl's pt_k_max_cl is 1.0
MASSLESS = "massless_fiducial"

# Locked to tests/conftest.py COSMOLOGY_GRID_LCDM / COSMOLOGY_GRID_NULCDM.
GRID_LCDM = {
    "lcdm_fiducial": {},
    "h_high": {"h": 0.6736 * 1.10},
    "omega_b_high": {"omega_b": 0.02237 * 1.20},
    "omega_cdm_low": {"omega_cdm": 0.1200 * 0.80},
    "ns_high": {"n_s": 0.9649 * 1.05},
}
GRID_NU = {
    "lcdm_fiducial": {},
    "massive_nu_015": {"m_ncdm": 0.15},
    "massive_nu_030": {"m_ncdm": 0.30},
    # Beyond the test grid: the q-bin decision at a mass that is non-relativistic
    # before recombination. Reference from `generate_class_reference.py --model
    # massive_nu_050` (not checked in); scored only when that directory exists.
    "massive_nu_050": {"m_ncdm": 0.50},
}
GRIDS = {"lcdm": GRID_LCDM, "nu": GRID_NU}

CONFIGS = {
    "Q0": dict(q=0, l_max=17, note="shipped fit_cl (#57 baseline)"),
    "Q3": dict(q=3, l_max=17, note="3 q-bins"),
    "Q4": dict(q=4, l_max=17, note="4 q-bins"),
    "Q5": dict(q=5, l_max=17, note="5 q-bins (CLASS defaults)"),
    "Q5L8": dict(q=5, l_max=8, note="5 q-bins, l_max_ncdm 8"),
    "Q3L8": dict(q=3, l_max=8, note="3 q-bins, l_max_ncdm 8"),
    # q-convergence of the Gauss-Laguerre momentum grid (finding of Oct 20, 2026:
    # the 5-bin low-l TT deficit grows with the mass); l_max_ncdm 8 throughout.
    "Q8L8": dict(q=8, l_max=8, note="8 q-bins, l_max_ncdm 8"),
    "Q12L8": dict(q=12, l_max=8, note="12 q-bins, l_max_ncdm 8"),
    # The same hierarchies with the ncdm fluid approximation off (full hierarchy
    # at all times; CLASS's references keep CLASS's own fluid approximation):
    # the late ISW at l = 2-30 is sourced at k ~ 1e-3-1e-2 Mpc^-1 below z ~ 2,
    # where k tau > 31 and the "class" fluid mode (#46) is active.
    "Q3L8F0": dict(q=3, l_max=8, fa="none", note="3 q-bins, l_max_ncdm 8, no fluid approximation"),
    "Q5L8F0": dict(q=5, l_max=8, fa="none", note="5 q-bins, l_max_ncdm 8, no fluid approximation"),
    "Q12L8F0": dict(q=12, l_max=8, fa="none", note="12 q-bins, l_max_ncdm 8, no fluid approximation"),
}
BANDS = [("tt", 2, 30), ("tt", 2, 500), ("ee", 5, 500), ("ten", 2, 500), ("tt", 2, 1000)]
PROBE_L = (2, 3, 5, 10, 20, 30, 50, 100, 200, 500)


def _git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _save(results, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f)
    os.replace(tmp, path)


def _load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _build(name):
    from clax.params import PrecisionParams
    cfg = CONFIGS[name]
    extra = {"ncdm_fluid_approximation": cfg["fa"]} if "fa" in cfg else {}
    return cfg, dataclasses.replace(PrecisionParams.fit_cl(), ncdm_q_size=cfg["q"],
                                    pt_l_max_ncdm=cfg["l_max"], **extra)


def _errs(cls, ref, l_read):
    """Per-l errors for l = 2..l_read against a reference indexed by l from 0."""
    n = min(len(cls["tt"]), len(ref["tt"]), l_read + 1)
    ell = np.arange(2, n)
    tt, ee, te = (np.asarray(cls[k])[2:n] for k in ("tt", "ee", "te"))
    rtt, ree, rte = (np.asarray(ref[k])[2:n] for k in ("tt", "ee", "te"))
    return dict(ell=ell, tt=tt / rtt - 1, ee=ee / ree - 1, ten=(te - rte) / np.sqrt(rtt * ree))


def _band(err, spec, lo, hi):
    m = (err["ell"] >= lo) & (err["ell"] <= hi)
    a = np.abs(err[spec][m])
    i = int(np.argmax(a))
    return float(a[i]), int(err["ell"][m][i])


def _bands(cls, ref):
    e = _errs(cls, ref, 1000)
    return {f"{sp} {lo}-{hi}": _band(e, sp, lo, hi) for sp, lo, hi in BANDS}


def _pk_band(pk, ref_pk):
    r = np.abs(np.asarray(pk["pk"]) / np.asarray(ref_pk) - 1.0)
    i = int(np.argmax(r))
    return float(r[i]), float(pk["k"][i])


def _ref_paths(cname):
    d = os.path.join("reference_data", cname)
    cls = REF_MASSLESS if cname == MASSLESS else os.path.join(d, "cls.npz")
    pk = os.path.join(d, "pk.npz")
    return (cls if os.path.exists(cls) else None), (pk if os.path.exists(pk) else None)


def run_config(name, out_path, results, grid, want_pk):
    import jax
    import jax.numpy as jnp
    import clax
    from clax.params import CosmoParams
    from clax.background import background_solve
    from clax.thermodynamics import thermodynamics_solve
    from clax.perturbations import perturbations_solve, _perturbation_solve_setup
    from clax.harmonic import compute_cls_all_fast

    cfg, prec = _build(name)
    rec = results.get(name) or dict(name=name, **cfg, cosmologies={})
    rec.update(settings={k: getattr(prec, k) for k in
                         ("ncdm_q_size", "pt_l_max_ncdm", "ncdm_fluid_approximation",
                          "pt_k_per_decade", "pt_tau_n_points", "th_n_points", "pt_l_max_g",
                          "pt_l_max_ur", "ode_max_steps", "pt_ode_solver", "pt_ode_rtol",
                          "pt_ode_atol", "hr_l_max", "hr_n_k_fine")},
               status="running", host=socket.gethostname(), git=_git_head(),
               device=str(jax.devices()[0].device_kind), jax=jax.__version__,
               started=time.strftime("%Y-%m-%d %H:%M:%S"))
    results[name] = rec
    _save(results, out_path)
    print(f"\n=== {name}: {cfg['note']} | q {cfg['q']} l_max_ncdm {cfg['l_max']} fa {prec.ncdm_fluid_approximation} | grid {grid}"
          f"{' + pk' if want_pk else ''} ===", flush=True)

    def stage(fn, leaf):
        t0 = time.perf_counter()
        r = fn()
        jax.block_until_ready(leaf(r))
        return r, time.perf_counter() - t0

    def one_cls(p, label):
        bg, t_bg = stage(lambda: background_solve(p, prec), lambda r: r.conformal_age)
        th, t_th = stage(lambda: thermodynamics_solve(p, prec, bg), lambda r: r.tau_star)
        if "n_eq" not in rec:
            try:
                rec["n_eq"] = int(_perturbation_solve_setup(p, prec, bg, th)[1])
            except Exception as e:  # informational only
                rec["n_eq"] = f"? ({type(e).__name__})"
        pt, t_pt = stage(lambda: perturbations_solve(p, prec, bg, th), lambda r: r.source_T0)
        cls, t_hr = stage(lambda: compute_cls_all_fast(pt, p, bg, l_max=prec.hr_l_max,
                                                       n_k_fine=prec.hr_n_k_fine),
                          lambda r: r["tt"])
        n_k = int(len(pt.k_grid))
        del pt
        print(f"  {label}: bg {t_bg:.1f}s th {t_th:.1f}s pt {t_pt:.1f}s hr {t_hr:.1f}s "
              f"(n_k={n_k}, n_eq={rec['n_eq']})", flush=True)
        return cls, dict(bg=t_bg, th=t_th, pt=t_pt, hr=t_hr, total=t_bg + t_th + t_pt + t_hr, n_k=n_k)

    def one_pk(p, k, label):
        res, t_pk = stage(lambda: clax.compute_pk_table(p, prec, z=0.0, k_eval=jnp.asarray(k)),
                          lambda r: r.pk_grid)
        print(f"  {label}: pk {t_pk:.1f}s ({len(k)} k in [{k[0]:.3g}, {k[-1]:.3g}])", flush=True)
        return np.asarray(res.pk_grid, dtype=float), t_pk

    points = list(GRIDS[grid].items()) + ([(MASSLESS, {"m_ncdm": 0.0})] if grid == "lcdm" else [])
    pk_k, n_pk_done = None, 0
    if want_pk:
        d = np.load(os.path.join("reference_data", "lcdm_fiducial", "pk.npz"))
        k = np.asarray(d["k"])
        pk_k = k[(k >= PK_KRANGE[0]) & (k <= PK_KRANGE[1])]
    n_cls_done = 0   # the first C_l solve of this process compiles (also on resume)
    for cname, ov in points:
        p = CosmoParams(**ov)
        ref_cls, ref_pk = _ref_paths(cname)
        crec = rec["cosmologies"].get(cname) or {}
        crec.update(ref=ref_cls, ref_pk=ref_pk)
        try:
            if crec.get("status") != "ok":
                crec["cached"] = n_cls_done > 0
                n_cls_done += 1
                cls, crec["timing"] = one_cls(p, f"{cname} ({'cached' if crec['cached'] else 'compile+run'})")
                crec["cls"] = {k: np.asarray(cls[k], dtype=float).tolist() for k in ("tt", "ee", "te")}
                finite = all(np.isfinite(np.asarray(cls[k])).all() for k in ("tt", "ee", "te"))
                del cls
                if ref_cls:
                    crec["bands"] = _bands(crec["cls"], np.load(ref_cls))
                    print("    " + ", ".join(f"{k} {100 * v[0]:.3f}% (l={v[1]})"
                                             for k, v in crec["bands"].items()), flush=True)
                crec["status"] = "ok" if finite else "nonfinite"
            if want_pk and "pk" not in crec:
                pk, t_pk = one_pk(p, pk_k, f"{cname} P(k)")
                crec["pk"] = {"k": pk_k.tolist(), "pk": pk.tolist()}
                crec["timing_pk"] = t_pk
                crec["cached_pk"] = n_pk_done > 0   # first P(k) of this process compiles
                n_pk_done += 1
                if ref_pk:
                    d = np.load(ref_pk)
                    m = (d["k"] >= PK_KRANGE[0]) & (d["k"] <= PK_KRANGE[1])
                    assert np.allclose(d["k"][m], pk_k), f"{ref_pk}: k grid differs from lcdm_fiducial"
                    crec["pk_band"] = _pk_band(crec["pk"], d["pk_m_lin_z0"][m])
                    print(f"    P_m lin z=0: {100 * crec['pk_band'][0]:.3f}% (k={crec['pk_band'][1]:.3g})",
                          flush=True)
        except Exception as e:  # one bad point must not kill the sweep
            crec["status"] = "error"
            crec["error"] = f"{type(e).__name__}: {str(e)[:800]}"
            crec["traceback"] = traceback.format_exc()[-3000:]
            print(f"  ERROR {crec['error']}", flush=True)
        rec["cosmologies"][cname] = crec
        _save(results, out_path)
    rec["status"] = "ok" if all(c.get("status") == "ok" for c in rec["cosmologies"].values()) else "partial"
    rec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _save(results, out_path)
    try:
        jax.clear_caches()
    except Exception:
        pass


def _scored(results, name, cname):
    """(bands, pk_band, tag): against CLASS when a reference exists (stored at
    run time, or found now: tag 'ref added'), else against the REF_CFG run at
    the same point (tag 'vs REF_CFG')."""
    c = results[name].get("cosmologies", {}).get(cname, {})
    if c.get("status") != "ok":
        return None, None, c.get("status", "-")
    bands, pk_band, tag = c.get("bands"), c.get("pk_band"), ""
    if bands is None or (pk_band is None and "pk" in c):
        # A CLASS reference that appeared after the run (e.g. massive_nu_030,
        # generated Oct 20, 2026): score the stored spectra against it.
        ref_cls, ref_pk = _ref_paths(cname)
        if bands is None and ref_cls:
            bands, tag = _bands(c["cls"], np.load(ref_cls)), "ref added"
        if pk_band is None and "pk" in c and ref_pk:
            d = np.load(ref_pk)
            m = (d["k"] >= PK_KRANGE[0]) & (d["k"] <= PK_KRANGE[1])
            pk_band, tag = _pk_band(c["pk"], d["pk_m_lin_z0"][m]), "ref added"
    base = results.get(REF_CFG, {}).get("cosmologies", {}).get(cname, {})
    if bands is None and name != REF_CFG and base.get("status") == "ok":
        bands = _bands(c["cls"], {k: np.array(base["cls"][k]) for k in ("tt", "ee", "te")})
        tag = f"vs {REF_CFG}"
    if pk_band is None and "pk" in c and name != REF_CFG and "pk" in base:
        pk_band = _pk_band(c["pk"], base["pk"]["pk"])
        tag = tag or f"vs {REF_CFG}"
    return bands, pk_band, tag


def report(path, md_path=None):
    results = _load(path)
    lines = []

    def P(s=""):
        lines.append(s)
        print(s)

    names = [n for n in CONFIGS if n in results]
    cnames = []
    for n in names:
        for c in results[n].get("cosmologies", {}):
            if c not in cnames:
                cnames.append(c)
    grid_pts = [c for c in cnames if c != MASSLESS]
    P("# fit_cl massive-neutrino q-grid sweep (#57)")
    P()
    P("Errors in percent: max |C_l/ref - 1| over the band (TEn = (TE - ref)/sqrt(TT_ref EE_ref)); "
      "grid columns are the maximum over the grid points that have a CLASS reference, with the point "
      "and l of the maximum; massless = CosmoParams(m_ncdm=0) vs the massless-RECFAST reference. "
      f"Points without a CLASS reference are scored against the {REF_CFG} run (marked). "
      "Timings: median of the cached cosmologies, seconds.")
    P()
    hdr = ("| cfg | q | l_max | n_eq | pt | hr | total | pk | " +
           " | ".join(f"grid {sp.upper()} {lo}-{hi}" for sp, lo, hi in BANDS) +
           " | " + " | ".join(f"massless {sp.upper()} {lo}-{hi}" for sp, lo, hi in BANDS[:4]) + " |")
    P(hdr)
    P("|" + "---|" * (hdr.count("|") - 1))
    for n in names:
        r = results[n]
        cos = r.get("cosmologies", {})
        cached = [c["timing"] for k, c in cos.items() if c.get("status") == "ok" and c.get("cached")]
        med = {k: float(np.median([t[k] for t in cached])) for k in ("pt", "hr", "total")} if cached else {}
        pks = [c["timing_pk"] for k, c in cos.items() if c.get("cached_pk") and "timing_pk" in c]
        row = [n, str(r["q"]), str(r["l_max"]), str(r.get("n_eq", "-"))]
        row += [f"{med[k]:.1f}" if k in med else "-" for k in ("pt", "hr", "total")]
        row.append(f"{np.median(pks):.1f}" if pks else "-")
        for sp, lo, hi in BANDS:
            key = f"{sp} {lo}-{hi}"
            best = [(c["bands"][key][0], c["bands"][key][1], k) for k, c in cos.items()
                    if k in grid_pts and c.get("status") == "ok" and c.get("bands")]
            if best:
                m, l_at, at = max(best)
                row.append(f"{100 * m:.3f} ({at.replace('lcdm_', '')}, l={l_at})")
            else:
                row.append("-")
        for sp, lo, hi in BANDS[:4]:
            c = cos.get(MASSLESS, {})
            row.append(f"{100 * c['bands'][f'{sp} {lo}-{hi}'][0]:.3f}" if c.get("status") == "ok" else "-")
        P("| " + " | ".join(row) + " |")
    P()
    P("Per cosmology (TT 2-30 / TT 2-500 / EE 5-500 / TEn 2-500 [/ P_m lin z=0, 1e-3-1 Mpc^-1], percent):")
    P()
    P("| cfg | " + " | ".join(cnames) + " |")
    P("|" + "---|" * (len(cnames) + 1))
    for n in names:
        cells = []
        for k in cnames:
            bands, pk_band, tag = _scored(results, n, k)
            if bands is None and pk_band is None:
                cells.append(tag or ("(reference)" if n == REF_CFG else "-"))
                continue
            s = " / ".join(f"{100 * bands[f'{sp} {lo}-{hi}'][0]:.2f}" for sp, lo, hi in BANDS[:4]) if bands else "-"
            if pk_band is not None:
                s += f" / pk {100 * pk_band[0]:.2f} (k={pk_band[1]:.2g})"
            cells.append(s + (f" ({tag})" if tag else ""))
        P(f"| {n} | " + " | ".join(cells) + " |")
    P()
    P(f"Fiducial TT and EE vs CLASS (HyRec, 0.06 eV) at l = {PROBE_L} (percent):")
    P()
    for n in names:
        c = results[n].get("cosmologies", {}).get("lcdm_fiducial", {})
        if c.get("status") != "ok" or not c.get("ref"):
            continue
        e = _errs(c["cls"], np.load(c["ref"]), 1000)
        P(f"    {n} TT: " + " ".join(f"{100 * e['tt'][l - 2]:+.2f}" for l in PROBE_L))
        P(f"    {n} EE: " + " ".join(f"{100 * e['ee'][l - 2]:+.2f}" for l in PROBE_L))
    P()
    if md_path:
        with open(md_path, "w") as f:
            f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="Q0,Q3,Q4,Q5,Q5L8,Q3L8")
    ap.add_argument("--grid", default="lcdm", choices=sorted(GRIDS))
    ap.add_argument("--pk", action="store_true", help="also score linear P_m(k, z=0)")
    ap.add_argument("--out", default="")
    ap.add_argument("--report", default=None, help="results JSON to summarise")
    ap.add_argument("--md", default="", help="write the report markdown here")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.report:
        report(args.report, args.md or None)
        return
    if args.list:
        for n in CONFIGS:
            cfg, prec = _build(n)
            print(n, cfg["note"], "q", prec.ncdm_q_size, "l_max_ncdm", prec.pt_l_max_ncdm,
                  prec.ncdm_fluid_approximation, prec.pt_ode_solver, prec.pt_ode_rtol, prec.pt_ode_atol)
        return
    if not args.out:
        ap.error("--out is required for a run")
    names = [n.strip() for n in args.configs.split(",") if n.strip()]
    results = _load(args.out)
    print(f"configs={names} grid={args.grid} pk={args.pk} out={args.out} git={_git_head()}", flush=True)
    for n in names:
        run_config(n, args.out, results, args.grid, args.pk)
    print("done", flush=True)


if __name__ == "__main__":
    main()
