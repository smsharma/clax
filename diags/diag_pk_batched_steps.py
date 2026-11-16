"""Why fit_cl's batched P(k) moves with the ncdm state dimension (#57 follow-up, Oct 22, 2026).

The q-grid sweep (diags/diag_fitcl_qgrid.py, docs/fitcl_qgrid_2026-10-20.md)
found that linear P_m(k, z=0) from ``compute_pk_table`` differs between
massive-neutrino hierarchies (q-bins x l_max_ncdm) by 0.2-1.0% at
k = 0.2-1 Mpc^-1, by amounts identical at 0.06, 0.15 and 0.30 eV. The matter
solve runs ``Rodas5Batched``: every k-mode of a chunk takes the same steps,
chosen by one filtered RMS norm over six variables (eta, delta_cdm, delta_b,
F_g_0, theta_b, F_g_1; ``_scalar_pid_filtered_rms_norm_batched``), none of
them ncdm. This diagnostic separates step-control truncation error from
physics, all at the fiducial cosmology (0.06 eV):

  A  tolerance ladder: P_m for four hierarchies at rtol/atol 1e-4/1e-6 (the
     preset), 1e-5/1e-7, 1e-6/1e-8, through the public ``compute_pk_table``;
     if the spread between hierarchies shrinks with the tolerance it is the
     step control.
  B  per-mode reference: the same four hierarchies with
     ``pt_ode_solver="kvaerno5"`` (the per-mode path, its own step sequence
     per k) at rtol 1e-7 / atol 1e-10 (and 1e-6 / 1e-9 for one, to show the
     reference's own convergence): the absolute error of each batched run
     and the physical difference between hierarchies.
  C  step counts: the batched solve of ``_perturbations_solve_mpk_impl``
     mirrored chunk by chunk outside ``lax.map`` so ``sol.stats`` is
     concrete; delta_m(k, tau_0) from the mirror is checked against A.
  D  lensing: C_l^phiphi (``compute_cl_pp``, nonlinear "none" and
     "halofit") and lensed TT/EE for the shipped 3-bin and the 5-bin /
     l_max 17 hierarchies, since only the halofit/EPT modulator reads the
     batched solve; the linear lensing source comes from the C_l path.

Hierarchies: Q0 (q 0, l_max 17: the pre-#57 preset), Q3L8 (q 3, l_max 8:
shipped), Q5L8 (q 5, l_max 8), Q5 (q 5, l_max 17: CLASS defaults). P_m is
scored on the reference k grid of ``reference_data/lcdm_fiducial/pk.npz``
restricted to 1e-3 <= k <= 1 Mpc^-1, also against CLASS ``pk_m_lin_z0``.

Usage (igpu compute node):
    python diags/diag_pk_batched_steps.py --sections A,B,C,D --out r.json
    python diags/diag_pk_batched_steps.py --report r.json [--md r.md]
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

HIER = {
    "Q0": dict(q=0, l_max=17),
    "Q3L8": dict(q=3, l_max=8),
    "Q5L8": dict(q=5, l_max=8),
    "Q5": dict(q=5, l_max=17),
    # fluid approximation off (the preset after Oct 22, 2026, and its l_max 17 counterpart)
    "Q5L8F0": dict(q=5, l_max=8, fa="none"),
    "Q5F0": dict(q=5, l_max=17, fa="none"),
}
LADDER = [(1e-4, 1e-6), (1e-5, 1e-7), (1e-6, 1e-8)]
REF_TOL = (1e-7, 1e-10)       # first run (job 27685): the hierarchies ran out of the 8192-step budget
REF_TOL_CHECK = (1e-6, 1e-9)
REF_TOL2 = (1e-6, 1e-9)       # second run: with ode_max_steps 32768
REF_MAX_STEPS2 = 32768
K_RANGE = (1e-3, 1.0)
K_PROBE = (0.01, 0.05, 0.1, 0.2, 0.5, 1.0)
L_PP = 1500
L_LENSED = 1500      # lens_cls wants every array at exactly l_max + 1; ratios are read to L_READ
L_READ = 1000


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


def _prec(name, rtol, atol, solver="rodas5", max_steps=None):
    from clax.params import PrecisionParams
    h = HIER[name]
    extra = {"ncdm_fluid_approximation": h["fa"]} if "fa" in h else {}
    if max_steps is not None:
        extra["ode_max_steps"] = max_steps
    return dataclasses.replace(PrecisionParams.fit_cl(), ncdm_q_size=h["q"], pt_l_max_ncdm=h["l_max"],
                               pt_ode_rtol=rtol, pt_ode_atol=atol, pt_ode_solver=solver, **extra)


def _k_ref():
    d = np.load(os.path.join("reference_data", "lcdm_fiducial", "pk.npz"))
    k = np.asarray(d["k"])
    m = (k >= K_RANGE[0]) & (k <= K_RANGE[1])
    return k[m], np.asarray(d["pk_m_lin_z0"])[m]


def _timed(fn, leaf):
    import jax
    t0 = time.perf_counter()
    r = fn()
    jax.block_until_ready(leaf(r))
    return r, time.perf_counter() - t0


def _pk_table(params, prec, k):
    import jax.numpy as jnp
    import clax
    res, t = _timed(lambda: clax.compute_pk_table(params, prec, z=0.0, k_eval=jnp.asarray(k)),
                    lambda r: r.pk_grid)
    return np.asarray(res.pk_grid, dtype=float), t


def run_A(results, out, params, k, pk_class):
    sec = results.setdefault("A", {})
    for name in HIER:
        for rtol, atol in LADDER:
            key = f"{name}@{rtol:.0e}"
            if sec.get(key, {}).get("status") == "ok":
                continue
            rec = dict(name=name, rtol=rtol, atol=atol, status="running")
            try:
                pk, t = _pk_table(params, _prec(name, rtol, atol), k)
                rec.update(pk=pk.tolist(), t=t, vs_class=float(np.max(np.abs(pk / pk_class - 1))), status="ok")
                print(f"  A {key}: {t:.1f}s  vs CLASS max {100 * rec['vs_class']:.3f}%", flush=True)
            except Exception as e:
                rec.update(status="error", error=f"{type(e).__name__}: {str(e)[:600]}",
                           traceback=traceback.format_exc()[-2000:])
                print(f"  A {key}: ERROR {rec['error']}", flush=True)
            sec[key] = rec
            _save(results, out)


def run_B(results, out, params, k, pk_class):
    sec = results.setdefault("B", {})
    todo = [(n,) + REF_TOL2 + (REF_MAX_STEPS2,) for n in ("Q5L8F0", "Q5F0", "Q5L8", "Q5")]
    for name, rtol, atol, max_steps in todo:
        key = f"{name}@kv5@{rtol:.0e}"
        if sec.get(key, {}).get("status") == "ok":
            continue
        rec = dict(name=name, rtol=rtol, atol=atol, solver="kvaerno5", max_steps=max_steps, status="running")
        try:
            pk, t = _pk_table(params, _prec(name, rtol, atol, solver="kvaerno5", max_steps=max_steps), k)
            rec.update(pk=pk.tolist(), t=t, vs_class=float(np.max(np.abs(pk / pk_class - 1))), status="ok")
            print(f"  B {key}: {t:.1f}s  vs CLASS max {100 * rec['vs_class']:.3f}%", flush=True)
        except Exception as e:
            rec.update(status="error", error=f"{type(e).__name__}: {str(e)[:600]}",
                       traceback=traceback.format_exc()[-2000:])
            print(f"  B {key}: ERROR {rec['error']}", flush=True)
        sec[key] = rec
        _save(results, out)


def run_C(results, out, params, k):
    """Mirror of clax.perturbations._solve_mpk_batched_rosenbrock.solve_batch, chunk by
    chunk in Python so diffrax's stats are concrete. Same pieces, same calls."""
    import jax
    import jax.numpy as jnp
    import diffrax
    from clax import perturbations as P
    from clax.rosenbrock import Rodas5Batched
    from clax.background import background_solve
    from clax.thermodynamics import thermodynamics_solve
    sec = results.setdefault("C", {})
    rtol, atol = LADDER[0]
    for name in HIER:
        if sec.get(name, {}).get("status") == "ok":
            continue
        rec = dict(name=name, rtol=rtol, atol=atol, status="running")
        try:
            prec = _prec(name, rtol, atol)
            # the k range compute_pk_table would solve for this k_eval
            import clax
            prec = clax._resolve_pk_precision(prec, k_eval=jnp.asarray(k), kmax=None)
            bg = background_solve(params, prec)
            th = thermodynamics_solve(params, prec, bg)
            pid_config = P._resolve_scalar_pid_config()
            (idx, n_eq, k_grid, tau_grid, tau_ini, n_tau, tau_max,
             ncdmfa_mode_code, ncdmfa_trigger, args_ncdm,
             l_max_g, l_max_pol, l_max_ur, l_max_ncdm) = P._perturbation_solve_setup(
                params, prec, bg, th, n_tau_override=P._mpk_tau_n_points(prec), tau_max_factor=1.0)
            q_ncdm, w_ncdm, M_ncdm, dlnf0_ncdm = args_ncdm
            n_k = len(k_grid)
            batch_size = P._resolve_pt_k_batch_size(
                prec, n_k=n_k, n_tau=n_tau,
                n_outputs=P._pt_saved_output_count(solve_kind="mpk"), solve_kind="mpk")
            shared_args = (bg, th, params, idx, l_max_g, l_max_pol, l_max_ur,
                           ncdmfa_mode_code, ncdmfa_trigger,
                           q_ncdm, w_ncdm, M_ncdm, dlnf0_ncdm,
                           prec.pt_rsa_trigger_tau_over_tau_k)

            def f_single(t, y, k_scalar):
                return P._perturbation_rhs(t, y, (k_scalar,) + shared_args)

            def vf_batched(t, y_batch, args):
                if isinstance(args, tuple) and len(args) == 2:
                    return jax.tree_util.tree_map(jnp.zeros_like, y_batch)
                return jax.vmap(f_single, in_axes=(None, 0, 0))(t, y_batch, args)

            @jax.jit
            def solve_batch(k_batch):
                y0_batch = jax.vmap(
                    lambda ki: P._adiabatic_ic(ki, jnp.array(tau_ini), bg, params, idx, n_eq,
                                               args_ncdm=args_ncdm))(k_batch)
                sol = diffrax.diffeqsolve(
                    diffrax.ODETerm(vf_batched), solver=Rodas5Batched(),
                    t0=tau_ini, t1=tau_max, dt0=tau_ini * 0.1, y0=y0_batch,
                    saveat=diffrax.SaveAt(ts=tau_grid),
                    stepsize_controller=P._make_scalar_pid_controller_batched(
                        prec=prec, k_batch=k_batch, idx=idx, config=pid_config),
                    adjoint=diffrax.DirectAdjoint(), max_steps=prec.ode_max_steps,
                    args=(f_single, k_batch))
                y_last = sol.ys[-1]
                dm = jax.vmap(lambda y, ki: P._extract_delta_m(
                    y, ki, tau_grid[-1], bg, idx, q_ncdm=q_ncdm, w_ncdm=w_ncdm, M_ncdm=M_ncdm,
                    ncdmfa_mode_code=ncdmfa_mode_code, ncdmfa_trigger=ncdmfa_trigger))(y_last, k_batch)
                return dm, sol.stats["num_steps"], sol.stats["num_accepted_steps"], sol.stats["num_rejected_steps"]

            n_tail = n_k % batch_size
            pad = (batch_size - n_tail) if n_tail else 0
            k_padded = jnp.concatenate([k_grid, jnp.full(pad, k_grid[-1])])
            chunks = k_padded.reshape(-1, batch_size)
            t0 = time.perf_counter()
            dms, steps, acc, rej = [], [], [], []
            for c in chunks:
                dm, ns, na, nr = solve_batch(c)
                jax.block_until_ready(dm)
                dms.append(np.asarray(dm)); steps.append(int(ns)); acc.append(int(na)); rej.append(int(nr))
            t = time.perf_counter() - t0
            dm_all = np.concatenate(dms)[:n_k]
            rec.update(n_eq=int(n_eq), n_k=int(n_k), batch_size=int(batch_size), n_chunks=len(chunks),
                       tau_max=float(tau_max), n_tau=int(n_tau), k_grid=np.asarray(k_grid).tolist(),
                       delta_m_tau0=dm_all.tolist(), num_steps=steps, num_accepted=acc, num_rejected=rej,
                       t=t, status="ok")
            print(f"  C {name}: n_eq {n_eq} n_k {n_k} batch {batch_size} chunks {len(chunks)} "
                  f"steps/chunk {steps} rejected {rej} ({t:.1f}s)", flush=True)
        except Exception as e:
            rec.update(status="error", error=f"{type(e).__name__}: {str(e)[:600]}",
                       traceback=traceback.format_exc()[-2500:])
            print(f"  C {name}: ERROR {rec['error']}", flush=True)
        sec[name] = rec
        _save(results, out)


def run_D(results, out, params):
    import jax.numpy as jnp
    from clax.background import background_solve
    from clax.thermodynamics import thermodynamics_solve
    from clax.perturbations import perturbations_solve, perturbations_solve_mpk
    from clax.harmonic import compute_cls_all_fast
    from clax.lensing import compute_cl_pp, lens_cls
    sec = results.setdefault("D", {})
    rtol, atol = LADDER[0]
    for name in ("Q5L8F0", "Q5F0", "Q3L8", "Q5"):
        if sec.get(name, {}).get("status") == "ok":
            continue
        rec = dict(name=name, status="running")
        try:
            prec = _prec(name, rtol, atol)
            bg = background_solve(params, prec)
            th = thermodynamics_solve(params, prec, bg)
            pt, t_pt = _timed(lambda: perturbations_solve(params, prec, bg, th), lambda r: r.source_T0)
            pt_mpk, t_mpk = _timed(lambda: perturbations_solve_mpk(params, prec, bg, th), lambda r: r.delta_m)
            cls, t_hr = _timed(lambda: compute_cls_all_fast(pt, params, bg, l_max=L_PP, n_k_fine=prec.hr_n_k_fine),
                               lambda r: r["tt"])
            pp0, t_pp0 = _timed(lambda: compute_cl_pp(pt, params, bg, th, l_max=L_PP), lambda r: r)
            pph, t_pph = _timed(lambda: compute_cl_pp(pt, params, bg, th, l_max=L_PP, nonlinear="halofit",
                                                      pt_mpk=pt_mpk), lambda r: r)
            tt, ee, te = (np.asarray(cls[x], dtype=float) for x in ("tt", "ee", "te"))
            bb = jnp.zeros_like(cls["tt"])
            lensed = {}
            for tag, pp in (("none", pp0), ("halofit", pph)):
                out4, t_l = _timed(lambda: lens_cls(cls["tt"], cls["ee"], cls["te"], bb, pp, l_max=L_LENSED),
                                   lambda r: r[0])
                lensed[tag] = dict(tt=np.asarray(out4[0], dtype=float).tolist(),
                                   ee=np.asarray(out4[1], dtype=float).tolist(), t=t_l)
            rec.update(t=dict(pt=t_pt, mpk=t_mpk, hr=t_hr, pp_none=t_pp0, pp_halofit=t_pph),
                       unlensed=dict(tt=tt.tolist(), ee=ee.tolist()),
                       pp=dict(none=np.asarray(pp0, dtype=float).tolist(), halofit=np.asarray(pph, dtype=float).tolist()),
                       lensed=lensed, status="ok")
            r_hf = np.asarray(pph)[2:L_PP + 1] / np.asarray(pp0)[2:L_PP + 1] - 1
            print(f"  D {name}: pt {t_pt:.1f}s mpk {t_mpk:.1f}s; halofit/none on C_l^pp: "
                  f"{100 * r_hf[98]:+.2f}% at l=100, {100 * r_hf[498]:+.2f}% at l=500, {100 * r_hf[-1]:+.2f}% at l={L_PP}",
                  flush=True)
        except Exception as e:
            rec.update(status="error", error=f"{type(e).__name__}: {str(e)[:600]}",
                       traceback=traceback.format_exc()[-2500:])
            print(f"  D {name}: ERROR {rec['error']}", flush=True)
        sec[name] = rec
        _save(results, out)


def _at(k, arr, kk):
    return arr[int(np.argmin(np.abs(k - kk)))]


def report(path, md_path=None):
    r = _load(path)
    lines = []

    def P(s=""):
        lines.append(s)
        print(s)

    k, pk_class = _k_ref()
    P("# fit_cl batched P(k) vs the ncdm state dimension (#57 follow-up)")
    P()
    P(f"Fiducial cosmology (0.06 eV). P_m(k, z=0) on {len(k)} reference k in [{k[0]:.3g}, {k[-1]:.3g}] Mpc^-1. "
      "Percent throughout.")
    A, B, C, D = (r.get(s, {}) for s in "ABCD")

    def pk_of(sec, key):
        c = sec.get(key, {})
        return np.array(c["pk"]) if c.get("status") == "ok" else None

    if A:
        P()
        P("## A: tolerance ladder, batched Rodas5 (P_m / P_m[Q5 at the same tolerance] - 1 at k = "
          + ", ".join(f"{x:g}" for x in K_PROBE) + "; last column max |P_m / P_CLASS - 1|)")
        P()
        P("| hierarchy | rtol | " + " | ".join(f"k={x:g}" for x in K_PROBE) + " | vs CLASS | t (s) |")
        P("|" + "---|" * (len(K_PROBE) + 4))
        for rtol, atol in LADDER:
            base = pk_of(A, f"Q5@{rtol:.0e}")
            for name in HIER:
                c = A.get(f"{name}@{rtol:.0e}", {})
                pk = pk_of(A, f"{name}@{rtol:.0e}")
                if pk is None or base is None:
                    P(f"| {name} | {rtol:.0e} | " + " | ".join(["-"] * len(K_PROBE)) + f" | {c.get('status', '-')} | - |")
                    continue
                d = pk / base - 1
                P(f"| {name} | {rtol:.0e} | " + " | ".join(f"{100 * _at(k, d, x):+.3f}" for x in K_PROBE)
                  + f" | {100 * c['vs_class']:.3f} | {c['t']:.1f} |")
    if B:
        P()
        P("## B: per-mode Kvaerno5 reference (each hierarchy's own converged answer)")
        P()
        P("| run | vs CLASS max | P_m / P_m[Q5 kv5 1e-7] - 1 at k = " + ", ".join(f"{x:g}" for x in K_PROBE) + " | t (s) |")
        P("|---|---|---|---|")
        base = pk_of(B, f"Q5@kv5@{REF_TOL[0]:.0e}")
        for key, c in B.items():
            pk = pk_of(B, key)
            if pk is None:
                P(f"| {key} | {c.get('status')} | - | - |")
                continue
            d = pk / base - 1 if base is not None else np.zeros_like(pk)
            P(f"| {key} | {100 * c['vs_class']:.3f} | " + " ".join(f"{100 * _at(k, d, x):+.4f}" for x in K_PROBE)
              + f" | {c['t']:.1f} |")
        if A:
            P()
            P("Batched run minus its own Kvaerno5 reference (P_batched / P_kv5 - 1), i.e. the step-control error of each run:")
            P()
            P("| hierarchy | rtol | " + " | ".join(f"k={x:g}" for x in K_PROBE) + " | max over k |")
            P("|" + "---|" * (len(K_PROBE) + 3))
            for name in HIER:
                ref = pk_of(B, f"{name}@kv5@{REF_TOL[0]:.0e}")
                if ref is None:
                    continue
                for rtol, atol in LADDER:
                    pk = pk_of(A, f"{name}@{rtol:.0e}")
                    if pk is None:
                        continue
                    d = pk / ref - 1
                    P(f"| {name} | {rtol:.0e} | " + " | ".join(f"{100 * _at(k, d, x):+.3f}" for x in K_PROBE)
                      + f" | {100 * np.max(np.abs(d)):.3f} |")
    if C:
        P()
        P("## C: batched solve mirrored chunk by chunk at the preset tolerance (1e-4 / 1e-6)")
        P()
        P("| hierarchy | n_eq | n_k | batch | steps per chunk (accepted+rejected) | rejected | delta_m(tau0)^2 / Q5 - 1 at k = "
          + ", ".join(f"{x:g}" for x in K_PROBE) + " | mirror vs A at k=1 |")
        P("|---|---|---|---|---|---|---|---|")
        c5 = C.get("Q5", {})
        for name, c in C.items():
            if c.get("status") != "ok":
                P(f"| {name} | {c.get('status')} | | | | | | |")
                continue
            kg = np.array(c["k_grid"]); dm = np.array(c["delta_m_tau0"])
            cells = "-"
            if c5.get("status") == "ok":
                d = (dm / np.array(c5["delta_m_tau0"])) ** 2 - 1
                cells = " ".join(f"{100 * _at(kg, d, x):+.3f}" for x in K_PROBE)
            chk = "-"
            a = A.get(f"{name}@{LADDER[0][0]:.0e}", {})
            if a.get("status") == "ok" and c5.get("status") == "ok" and name != "Q5":
                a5 = np.array(A[f"Q5@{LADDER[0][0]:.0e}"]["pk"])
                ra = _at(k, np.array(a["pk"]) / a5 - 1, 1.0)
                rc = _at(kg, (dm / np.array(c5["delta_m_tau0"])) ** 2 - 1, 1.0)
                chk = f"A {100 * ra:+.3f} / C {100 * rc:+.3f}"
            P(f"| {name} | {c['n_eq']} | {c['n_k']} | {c['batch_size']} | {c['num_steps']} | {c['num_rejected']} | {cells} | {chk} |")
    if D:
        P()
        P(f"## D: lensing at the preset tolerance; max |ratio - 1| over l = 2..{L_READ} (C_l^pp to {L_PP}, lensed to {L_LENSED})")
        P()

        def mx(x, y, lo, hi):
            x, y = np.array(x)[lo:hi + 1], np.array(y)[lo:hi + 1]
            d = x / y - 1
            i = int(np.argmax(np.abs(d)))
            return f"{100 * abs(d[i]):.3f}% (l={lo + i})"

        ref = np.load(os.path.join("reference_data", "lcdm_fiducial", "cls.npz"))
        refl = np.load(os.path.join("reference_data", "lcdm_fiducial", "cls_lensed.npz"))
        for name, c in D.items():
            if c.get("status") != "ok":
                P(f"- {name}: {c.get('status')}")
                continue
            P(f"- {name} vs CLASS: C_l^pp (linear) {mx(c['pp']['none'], ref['pp'], 2, L_READ)}; "
              f"lensed TT {mx(c['lensed']['none']['tt'], refl['tt'], 2, L_READ)}, EE {mx(c['lensed']['none']['ee'], refl['ee'], 5, L_READ)} "
              f"(unlensed TT {mx(c['unlensed']['tt'], ref['tt'], 2, L_READ)}); halofit / linear on C_l^pp "
              f"{mx(c['pp']['halofit'], c['pp']['none'], 2, L_READ)}; lensed / unlensed TT {mx(c['lensed']['halofit']['tt'], c['unlensed']['tt'], 2, L_READ)}")
        for a_name, b_name in (("Q5L8F0", "Q5F0"), ("Q3L8", "Q5")):
            a, b = D.get(a_name, {}), D.get(b_name, {})
            if a.get("status") != "ok" or b.get("status") != "ok":
                continue
            P(f"- {a_name} / {b_name}: unlensed TT {mx(a['unlensed']['tt'], b['unlensed']['tt'], 2, L_READ)}, EE {mx(a['unlensed']['ee'], b['unlensed']['ee'], 5, L_READ)}; "
              f"C_l^pp linear {mx(a['pp']['none'], b['pp']['none'], 2, L_READ)}, halofit {mx(a['pp']['halofit'], b['pp']['halofit'], 2, L_READ)}; "
              f"lensed TT (halofit pp) {mx(a['lensed']['halofit']['tt'], b['lensed']['halofit']['tt'], 2, L_READ)}, "
              f"EE {mx(a['lensed']['halofit']['ee'], b['lensed']['halofit']['ee'], 5, L_READ)}")
    P()
    if md_path:
        with open(md_path, "w") as f:
            f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", default="A,B,C,D")
    ap.add_argument("--out", default="")
    ap.add_argument("--report", default=None)
    ap.add_argument("--md", default="")
    args = ap.parse_args()
    if args.report:
        report(args.report, args.md or None)
        return
    if not args.out:
        ap.error("--out is required for a run")
    import jax
    from clax.params import CosmoParams
    results = _load(args.out)
    results["meta"] = dict(git=_git_head(), host=socket.gethostname(), device=str(jax.devices()[0].device_kind),
                           jax=jax.__version__, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    params = CosmoParams()
    k, pk_class = _k_ref()
    secs = [s.strip().upper() for s in args.sections.split(",") if s.strip()]
    print(f"sections={secs} out={args.out} git={results['meta']['git']} device={results['meta']['device']}", flush=True)
    for s in secs:
        print(f"=== section {s} ===", flush=True)
        if s == "A":
            run_A(results, args.out, params, k, pk_class)
        elif s == "B":
            run_B(results, args.out, params, k, pk_class)
        elif s == "C":
            run_C(results, args.out, params, k)
        elif s == "D":
            run_D(results, args.out, params)
        try:
            jax.clear_caches()
        except Exception:
            pass
    print("done", flush=True)


if __name__ == "__main__":
    main()
