"""
Report-14/15 diagnostics + PPC aggregation (claims-ledger rows C14, C15).

Transcribes, into a single citable artifact, the MCMC diagnostics (C14, plan
SS7.4.3) and posterior-predictive p-values (C15, plan SS7.4.4) across all 20
fitted application conditions (4 studies x 5 temperatures).

Sources (committed application outputs):
  reports/applications/<study>/data/diagnostics_T*.txt   (CmdStan `diagnose`)
  reports/applications/<study>/data/fit_summary.json     (ppc_p_values per cond)

C14: every fit should be clean (treedepth / E-BFMI / R-hat / ESS satisfactory),
with at most 1-2 divergent transitions in the two highest GPT-4o temperatures.
C15: every PPC p-value (ll / modal / prob) should sit in [0.3, 0.7].

Run:
  python working_papers/seu_sensitivity_methodology/spikes/report1415_diagnostics_ppc_spike.py
"""
from __future__ import annotations

import glob
import json
import os
import re

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(THIS_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(PAPER_DIR))
APP_DIR = os.path.join(PROJECT_ROOT, "reports", "applications")

STUDIES = {
    "temperature_study": "GPT-4o x Insurance",
    "claude_insurance_study": "Claude x Insurance",
    "gpt4o_ellsberg_study": "GPT-4o x Ellsberg",
    "ellsberg_study": "Claude x Ellsberg",
}

PPC_LO, PPC_HI = 0.3, 0.7


def _flagged_param_families(txt: str) -> list[str]:
    """Return the sorted set of parameter-name prefixes flagged R-hat > 1.01.

    The CmdStan `diagnose` output lists offending parameter names (e.g. eta[12],
    psi[3,1], beta[2,1]) after the 'greater than 1.01' line. We collapse to the
    family prefix (token before any '[') to characterise WHICH parameters mix
    poorly -- structural (alpha/beta/delta) vs high-dimensional per-trial
    nuisance/latent (eta/psi/...).
    """
    m = re.search(r"R-hat greater than 1\.01:\s*(.*?)\nSuch high values",
                  txt, re.DOTALL)
    if not m:
        return []
    names = re.findall(r"([A-Za-z_]+)\s*\[", m.group(1))
    return sorted(set(names))


def aggregate_diagnostics() -> dict:
    per_fit = []
    n_total = 0
    n_clean = 0
    structural = {"alpha", "beta", "delta"}
    for sdir, label in STUDIES.items():
        for f in sorted(glob.glob(os.path.join(APP_DIR, sdir, "data", "diagnostics_T*.txt"))):
            n_total += 1
            txt = open(f).read()
            cond = re.search(r"T(\d+_\d+)", f).group(1).replace("_", ".")
            no_div = "No divergent transitions found" in txt
            m = re.search(r"(\d+) of (\d+)\s*\(([\d.]+)%\).*diverg", txt, re.IGNORECASE)
            n_div = 0 if no_div else (int(m.group(1)) if m else None)
            div_pct = 0.0 if no_div else (float(m.group(3)) if m else None)
            families = _flagged_param_families(txt)
            checks = {
                "treedepth_ok": "Treedepth satisfactory" in txt,
                "ebfmi_ok": "E-BFMI satisfactory" in txt,
                "ess_ok": "effective sample size satisfactory" in txt,
                "rhat_ok": "R-hat values satisfactory" in txt,
                "no_problems": "no problems detected" in txt,
            }
            fully_clean = all(checks.values()) and no_div
            if fully_clean:
                n_clean += 1
            per_fit.append({
                "study": label, "temperature": cond,
                "n_divergences": n_div, "divergence_pct": div_pct,
                "rhat_flagged_families": families,
                "structural_param_flagged": bool(set(families) & structural),
                "fully_clean": fully_clean,
                **checks,
            })
    nonclean = [r for r in per_fit if not r["fully_clean"]]
    all_families = sorted({fam for r in per_fit for fam in r["rhat_flagged_families"]})
    return {
        "n_total_fits": n_total,
        "n_fully_clean": n_clean,
        "all_rhat_ess_ok": all(r["rhat_ok"] and r["ess_ok"] for r in per_fit),
        "alpha_ever_rhat_flagged": any("alpha" in r["rhat_flagged_families"] for r in per_fit),
        "any_structural_param_flagged": any(r["structural_param_flagged"] for r in per_fit),
        "all_flagged_param_families": all_families,
        "ess_ok_all_fits": all(r["ess_ok"] for r in per_fit),
        "max_divergence_pct": max((r["divergence_pct"] or 0) for r in per_fit),
        "n_fits_with_divergences": sum(1 for r in per_fit if (r["n_divergences"] or 0) > 0),
        "n_fits_rhat_flagged": sum(1 for r in per_fit if not r["rhat_ok"]),
        "total_divergences": sum((r["n_divergences"] or 0) for r in per_fit),
        "fits_with_divergences": [
            {"study": r["study"], "temperature": r["temperature"],
             "n_divergences": r["n_divergences"], "divergence_pct": r["divergence_pct"]}
            for r in per_fit if (r["n_divergences"] or 0) > 0
        ],
        "nonclean_fits": nonclean,
        "per_fit": per_fit,
    }


def aggregate_ppc() -> dict:
    pvals = []
    outside = []
    for sdir, label in STUDIES.items():
        fs = json.load(open(os.path.join(APP_DIR, sdir, "data", "fit_summary.json")))
        for t, info in fs.items():
            for stat, p in info["ppc_p_values"].items():
                pvals.append(p)
                if not (PPC_LO <= p <= PPC_HI):
                    outside.append({"study": label, "temperature": t,
                                    "stat": stat, "p": round(p, 4)})
    return {
        "n_pvalues": len(pvals),
        "n_fits": len(pvals) // 3,
        "min": min(pvals), "max": max(pvals),
        "mean": sum(pvals) / len(pvals),
        "n_in_band": sum(1 for p in pvals if PPC_LO <= p <= PPC_HI),
        "band": [PPC_LO, PPC_HI],
        "outside_band": outside,
    }


def main() -> None:
    diag = aggregate_diagnostics()
    ppc = aggregate_ppc()

    print("=== C14: MCMC diagnostics across all 20 fits ===")
    print(f"  total fits: {diag['n_total_fits']}   fully clean (no problems "
          f"detected): {diag['n_fully_clean']}")
    print(f"  alpha (structural) ever R-hat-flagged: {diag['alpha_ever_rhat_flagged']}")
    print(f"  any structural param (alpha/beta/delta) flagged: {diag['any_structural_param_flagged']}")
    print(f"  ESS satisfactory in every fit: {diag['ess_ok_all_fits']}")
    print(f"  R-hat>1.01 fits: {diag['n_fits_rhat_flagged']}  | "
          f"flagged families: {diag['all_flagged_param_families']}")
    print(f"  divergences: total {diag['total_divergences']} across "
          f"{diag['n_fits_with_divergences']}/{diag['n_total_fits']} fits; "
          f"max per-fit {diag['max_divergence_pct']:.2f}%")
    if diag["fits_with_divergences"]:
        print("  per-fit divergences:")
        for r in diag["fits_with_divergences"]:
            print(f"     {r['study']:22s} T={r['temperature']:4s}  "
                  f"{r['n_divergences']}/4000 ({r['divergence_pct']:.2f}%)")
    if diag["nonclean_fits"]:
        print("  R-hat>1.01 fits (nuisance-parameter detail):")
        for r in diag["nonclean_fits"]:
            if not r["rhat_ok"]:
                print(f"     {r['study']:22s} T={r['temperature']:4s}  "
                      f"families={r['rhat_flagged_families']}  "
                      f"structural_flagged={r['structural_param_flagged']}")

    print("\n=== C15: PPC p-values across all 20 fits (3 stats each) ===")
    print(f"  n p-values: {ppc['n_pvalues']}  (= {ppc['n_fits']} fits x 3 stats)")
    print(f"  range: [{ppc['min']:.3f}, {ppc['max']:.3f}]   mean: {ppc['mean']:.3f}")
    print(f"  in [{PPC_LO}, {PPC_HI}]: {ppc['n_in_band']}/{ppc['n_pvalues']}")
    if ppc["outside_band"]:
        print("  outside band:")
        for o in ppc["outside_band"]:
            print(f"     {o['study']} T={o['temperature']} {o['stat']}={o['p']}")
    else:
        print("  outside band: none")

    out = {
        "spec": {
            "rows": ["C14", "C15"],
            "sections": ["7.4.3", "7.4.4"],
            "studies": STUDIES,
            "app_dir": APP_DIR,
        },
        "C14_diagnostics": diag,
        "C15_ppc": ppc,
    }
    out_path = os.path.join(THIS_DIR, "report1415_diagnostics_ppc_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
