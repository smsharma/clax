"""C_l-vs-CLASS acceptance baseline for tests/test_cl_vs_class.py (smsharma/clax#54).

Two commands:

  python scripts/cl_vs_class_baseline.py run <base_tree> <out_dir> <label> <cosmology>   (GPU)
  python scripts/cl_vs_class_baseline.py assemble <out_dir> <label> <base_commit>        (CPU)

`run` evaluates ONE cosmology of tests/conftest.py::CL_VALUE_GRID (read from this
repository's conftest, so the grid is the test fixture's own) on the clax tree <base_tree>,
the branch's base without the #54 code, eagerly, for the preset keys
  fit_cl         PrecisionParams.fit_cl(), n_k_fine = 20000: the fix's preset value, so that
                 fix and base are compared at an equal k-grid and only the Bessel lookup and
                 table differ (ADR 0005, Decision 8);
  fit_cl_nk5000  (base trees only) the base preset's own default 5000, recorded in the JSON
                 as documentation of what the k-grid change alone does; not gated;
  planck_cl      PrecisionParams.planck_cl(), n_k_fine = prec.hr_n_k_fine (20000 on both).
For each: unlensed TT/EE/TE from compute_cls_all_fast(..., l_max=prec.hr_l_max, n_k_fine),
C_l^phiphi from compute_cl_pp(..., l_max), lensed TT/EE/TE/BB from lens_cls, and
max |C/C_CLASS - 1| per l band against this repository's
reference_data/<name>/{cls,cl_pp,cls_lensed}.npz (TE and lensed TE: |dTE| / sqrt(TT EE) of
the reference): the arithmetic of tests/test_cl_vs_class.py::_errors, with
th_grad_mode="native", whose primal equals "stable" (thermodynamics.py docstring). On a
tree whose compute_cls_all_fast / compute_cl_pp take prec= (a tree with #54) prec is
passed; the base trees do not take it. Writes <out_dir>/<label>-<name>.json (band maxima,
argmax l, timings, job and host) and <out_dir>/spec-<label>-<name>-<key>.npz (the spectra).

`assemble` gathers those files for every cosmology of CL_VALUE_GRID into
tests/data/cl_vs_class_baseline.json ({<name>: {<key>: {<spectrum>: {<band>: err}}},
"_provenance": {...}}) and tests/data/cl_vs_class_baseline.npz (the base spectra under
<cosmology>__<preset>__<spectrum>, fit_cl and planck_cl only, float64 indexed by
l = 0..l_max), the two files the test reads. The test recomputes the band maxima from the
npz and checks them against the JSON, so both must come from one run.

History. The Oct 25, 2026 baseline (jobs 30137-30150, V100, JAX 0.9.2) came from this
script's out-of-repo predecessor (/lustre/work/n2minh/clax-probes/cl/v1_baseline.py, sha1
8c502efd) on 5ac3df3. The Nov 17, 2026 baseline is on 6ff1303 (#58, #60 and #61 merged),
the base of the rebased #54 branch.
"""
import ast
import hashlib
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(REPO, "reference_data")
DATA = os.path.join(REPO, "tests", "data")
BANDS = {"2-30": (2, 30), "30-100": (30, 100), "100-500": (100, 500),
         "500-1500": (500, 1500), "1500-2500": (1500, 2500)}
GATED_KEYS = ("fit_cl", "planck_cl")
SPECTRA = ("tt", "ee", "te", "pp", "ltt", "lee", "lte", "lbb")


def load_grid():
    """CL_VALUE_GRID evaluated from tests/conftest.py's source (no pytest import)."""
    src = open(os.path.join(REPO, "tests", "conftest.py")).read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "CL_VALUE_GRID" for t in node.targets):
            return eval(compile(ast.Expression(node.value), "conftest.py", "eval"), {})  # noqa: S307
    raise RuntimeError("CL_VALUE_GRID not found in tests/conftest.py")


def errors(spectra, lmax):
    """tests/test_cl_vs_class.py::_errors' per-band arithmetic, plus the argmax l."""
    out, at = {}, {}
    for s, tup in spectra.items():
        mine, ref = tup[0], tup[1]
        norm = tup[2] if len(tup) == 3 else ref
        out[s], at[s] = {}, {}
        for band, (lo, hi) in BANDS.items():
            hi = min(hi, lmax)
            if lo >= hi:
                continue
            sl = slice(lo, hi + 1)
            r = np.abs(mine[sl] - ref[sl]) / np.abs(norm[sl])
            out[s][band] = float(np.max(r))
            at[s][band] = lo + int(np.argmax(r))
    return out, at


def sha1(path):
    return hashlib.sha1(open(path, "rb").read()).hexdigest()


def cmd_run(tree, out_dir, label, name):
    tree = os.path.realpath(tree)
    sys.path.insert(0, tree)
    import inspect

    import jax
    import jax.numpy as jnp

    import clax
    assert os.path.realpath(clax.__file__).startswith(tree + "/"), (clax.__file__, tree)
    from clax import CosmoParams, PrecisionParams, harmonic
    from clax.harmonic import compute_cls_all_fast
    from clax.lensing import compute_cl_pp, lens_cls
    from clax.perturbations import perturbations_solve
    from clax.thermodynamics import solve_background_and_thermo

    has_prec_hr = "prec" in inspect.signature(compute_cls_all_fast).parameters
    has_prec_pp = "prec" in inspect.signature(compute_cl_pp).parameters
    assert has_prec_hr == has_prec_pp, (has_prec_hr, has_prec_pp)
    kind = "fix" if has_prec_hr else "base"
    lookup = ("hermite" if kind == "fix" else "linear") + f"@{harmonic.JL_TABLE_X_PER_UNIT:g}"
    dev = jax.devices()[0]
    job = os.environ.get("SLURM_JOB_ID")
    host = os.uname().nodename

    def say(msg):
        print(f"V1 {kind} {name}: {msg}", flush=True)

    grid = load_grid()
    assert name in grid, (name, list(grid))
    p = CosmoParams(**grid[name])
    say(f"tree {tree} kind {kind} lookup {lookup} job {job} host {host} device "
        f"{dev.device_kind} jax {jax.__version__} overrides {grid[name]}")
    assert dev.platform == "gpu", f"expected a GPU, got {dev}"

    def kw(prec):
        return {"prec": prec} if has_prec_hr else {}

    u = np.load(os.path.join(REF, name, "cls.npz"))
    ln = np.load(os.path.join(REF, name, "cls_lensed.npz"))
    rp = np.load(os.path.join(REF, name, "cl_pp.npz"))
    os.makedirs(out_dir, exist_ok=True)
    res = {}
    meta = dict(tree=tree, tree_kind=kind, lookup=lookup, job=job, host=host,
                device=dev.device_kind, jax=jax.__version__, cosmology=name,
                overrides=grid[name], date=time.strftime("%b %-d, %Y"), keys={})
    plan = {"fit_cl": (PrecisionParams.fit_cl, [("fit_cl", 20000)]
                       + ([("fit_cl_nk5000", 5000)] if kind == "base" else [])),
            "planck_cl": (PrecisionParams.planck_cl, [("planck_cl", None)])}
    for preset, (mk, keys) in plan.items():
        prec = replace(mk(), th_grad_mode="native")
        lmax = prec.hr_l_max
        t0 = time.perf_counter()
        bg, th = solve_background_and_thermo(p, prec)
        pt = perturbations_solve(p, prec, bg, th)
        jax.block_until_ready(pt)
        t1 = time.perf_counter()
        pp = compute_cl_pp(pt, p, bg, th, lmax, **kw(prec))
        jax.block_until_ready(pp)
        t2 = time.perf_counter()
        say(f"{preset}: preset hr_n_k_fine {prec.hr_n_k_fine} hr_l_max {lmax} solver "
            f"{prec.pt_ode_solver} rtol {prec.pt_ode_rtol}; bg+th+pt {t1 - t0:.1f} s, "
            f"cl_pp {t2 - t1:.1f} s")
        for key, nk in keys:
            nk = prec.hr_n_k_fine if nk is None else nk
            if key == "fit_cl" and kind == "fix":
                assert prec.hr_n_k_fine == nk == 20000, prec.hr_n_k_fine
            if key == "fit_cl_nk5000":
                assert prec.hr_n_k_fine == 5000, prec.hr_n_k_fine
            if key == "planck_cl":
                assert nk == 20000, nk
            t0 = time.perf_counter()
            cl = compute_cls_all_fast(pt, p, bg, l_max=lmax, n_k_fine=nk, **kw(prec))
            jax.block_until_ready(cl)
            t1 = time.perf_counter()
            lt, le, lte, lb = lens_cls(cl["tt"], cl["ee"], cl["te"], jnp.zeros_like(cl["tt"]),
                                       pp, l_max=lmax)
            jax.block_until_ready(lb)
            t2 = time.perf_counter()
            spectra = {
                "tt": (np.asarray(cl["tt"]), u["tt"]), "ee": (np.asarray(cl["ee"]), u["ee"]),
                "te": (np.asarray(cl["te"]), u["te"], np.sqrt(u["tt"] * u["ee"])),
                "pp": (np.asarray(pp), rp["pp"]),
                "ltt": (np.asarray(lt), ln["tt"]), "lee": (np.asarray(le), ln["ee"]),
                "lte": (np.asarray(lte), ln["te"], np.sqrt(ln["tt"] * ln["ee"])),
                "lbb": (np.asarray(lb), ln["bb"]),
            }
            fin = {s: bool(np.all(np.isfinite(v[0][2:lmax + 1]))) for s, v in spectra.items()}
            e, at = errors(spectra, lmax)
            res[key] = e
            meta["keys"][key] = dict(n_k_fine=nk, l_max=lmax, preset=preset, argmax_l=at,
                                     finite=fin, harmonic_s=t1 - t0, lens_s=t2 - t1)
            np.savez(os.path.join(out_dir, f"spec-{label}-{name}-{key}.npz"),
                     **{s: v[0] for s, v in spectra.items()})
            say(f"{key}: n_k_fine {nk}; harmonic {t1 - t0:.1f} s, lens {t2 - t1:.1f} s; "
                f"finite {all(fin.values())}")
            for s in spectra:
                say(f"{key} {s:3s} " + "  ".join(f"{b} {v:.3e}@{at[s][b]}" for b, v in e[s].items()))
    out_json = os.path.join(out_dir, f"{label}-{name}.json")
    with open(out_json, "w") as f:
        json.dump({name: res, "_meta": meta}, f, indent=1)
    say(f"wrote {out_json}")
    say("RESULT " + json.dumps({name: res}))
    say("DONE")


def cmd_assemble(out_dir, label, base_commit):
    grid = load_grid()
    res, jobs, hosts, spectra = {}, {}, {}, {}
    devices, jaxv, kinds, trees, dates, nk, lmax = set(), set(), set(), set(), set(), {}, {}
    for name in grid:
        d = json.load(open(os.path.join(out_dir, f"{label}-{name}.json")))
        meta = d["_meta"]
        res[name] = d[name]
        jobs[name], hosts[name] = meta["job"], meta["host"]
        devices.add(meta["device"]); jaxv.add(meta["jax"]); kinds.add(meta["tree_kind"])
        trees.add(meta["tree"]); dates.add(meta["date"])
        for key, m in meta["keys"].items():
            nk.setdefault(key, set()).add(m["n_k_fine"])
            lmax.setdefault(key, set()).add(m["l_max"])
            assert all(m["finite"].values()), (name, key, m["finite"])
        for key in GATED_KEYS:
            z = np.load(os.path.join(out_dir, f"spec-{label}-{name}-{key}.npz"))
            for s in SPECTRA:
                spectra[f"{name}__{key}__{s}"] = z[s]
    assert kinds == {"base"}, kinds
    assert len(trees) == 1 and len(devices) == 1 and len(jaxv) == 1, (trees, devices, jaxv)
    assert all(len(v) == 1 for v in nk.values()) and all(len(v) == 1 for v in lmax.values())
    nk = {k: v.pop() for k, v in nk.items()}
    lmax = {k: v.pop() for k, v in lmax.items()}
    os.makedirs(DATA, exist_ok=True)
    npz_path = os.path.join(DATA, "cl_vs_class_baseline.npz")
    json_path = os.path.join(DATA, "cl_vs_class_baseline.json")
    np.savez(npz_path, **spectra)
    rel = os.path.relpath(os.path.abspath(__file__), REPO)
    prov = {
        "what": ("max |C_l / C_l^CLASS - 1| per l band of the base tree, the branch's base "
                 "without the #54 code (TE and lensed TE: |dTE| / sqrt(TT EE) of the "
                 "reference), against reference_data/<name>/{cls,cl_pp,cls_lensed}.npz. "
                 "tests/test_cl_vs_class.py passes a band if fix error <= base error + 1e-4 "
                 "or the spectrum moved by at most 1e-4 of the base spectrum; bands starting "
                 "at l >= 30 are checked l by l on the base spectra of spectra_npz (every l "
                 "must pass), the 2-30 bands on these band maxima "
                 "(docs/adr/0005-cl-differentiability.md, Decision 8)"),
        "base_commit": base_commit,
        "base_tree": trees.pop(),
        "n_k_fine": nk,
        "n_k_fine_note": ("fit_cl is measured at n_k_fine 20000 = the fix's preset value "
                          "(equal k-grid, so that only the Bessel lookup and table differ); "
                          "fit_cl_nk5000 is the base preset's own default, recorded as "
                          "documentation only and not gated"),
        "lookup": {"base": "linear interpolation of the j_l table at 30 points per unit x, "
                           "table rebuilt on every call",
                   "fix": "cubic Hermite interpolation at 3 points per unit x, table built once"},
        "l_max": lmax,
        "calls": ("eager; prec = replace(preset, th_grad_mode='native'); "
                  "compute_cls_all_fast(pt, p, bg, l_max=hr_l_max, n_k_fine=NK); "
                  "compute_cl_pp(pt, p, bg, th, hr_l_max); lens_cls(tt, ee, te, 0, pp, "
                  "l_max=hr_l_max)"),
        "script": rel,
        "script_sha1": sha1(os.path.abspath(__file__)),
        "jobs": jobs,
        "hosts": hosts,
        "device": devices.pop(),
        "jax": jaxv.pop(),
        "date": sorted(dates)[-1] if len(dates) == 1 else sorted(dates),
        "reference_note": ("the lcdm_fiducial, omega_b_high and h_high cls.npz / cls_lensed.npz "
                           "predate the CLASS build that wrote cl_pp.npz (a fresh run of that "
                           "build differs by up to ~2.5e-3 in lensed TT at fiducial, job "
                           "26859); the gate compares fix and base against the same files, so "
                           "the offset cancels to first order"),
        "spectra_npz": {
            "file": os.path.relpath(npz_path, REPO),
            "sha1": sha1(npz_path),
            "keys": ("<cosmology>__<preset>__<spectrum>, preset in {fit_cl, planck_cl}, "
                     "spectrum in tt/ee/te/pp/ltt/lee/lte/lbb; float64 C_l indexed by "
                     f"l = 0..l_max (fit_cl {lmax['fit_cl']}, planck_cl {lmax['planck_cl']})"),
            "source": f"the base spectra of the same jobs ({rel} run: spec-{label}-*.npz): "
                      "the arrays the errors above were computed from",
        },
    }
    with open(json_path, "w") as f:
        json.dump({**res, "_provenance": prov}, f, indent=1)
    print(f"wrote {json_path} ({len(res)} cosmologies) and {npz_path} ({len(spectra)} arrays)")


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "run":
        cmd_run(*sys.argv[2:6])
    elif len(sys.argv) == 5 and sys.argv[1] == "assemble":
        cmd_assemble(*sys.argv[2:5])
    else:
        sys.exit(__doc__)
