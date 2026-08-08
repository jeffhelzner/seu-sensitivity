"""
Phase D power analysis for ``h_m01_size`` (study plan §8.5).

Regime (a) -- near-deterministic / pseudo-replication -- is implemented first
because it SOLVES FOR ``num_problems``, and every other regime and the whole API
budget price off that number.

What this adds over :mod:`analysis.hierarchical_parameter_recovery`
-------------------------------------------------------------------
Recovery reports bias / rmse / coverage / interval width for a parameter.  Power
needs a DECISION layer on top: for each simulated study, did the design let us
sign the target contrast, and would we have called it? So each iteration records

* ``correct_sign``   -- posterior mean has the sign of the true slope
* ``excludes_zero``  -- the central interval excludes 0
* ``rope_decision``  -- interval lies outside the §8.3 ROPE (|Δ log α| > log 1.25)
* ``type_s``         -- the interval excludes zero *with the wrong sign*
                        (a confidently wrong answer, far worse than a miss)

and across iterations reports the rate of each.  Coverage is retained because
under pseudo-replication it is the diagnostic that matters most: the inference
model assumes independent observations, so when the generating regime correlates
presentations the intervals should get too narrow and coverage should fall below
nominal.  Power computed without watching coverage would look *better* as the
data got worse.

Cost discipline (§ Phase C lesson)
----------------------------------
``n_iterations`` is a parameter and every run writes ``timing.json`` with
measured per-iteration wall clock, so a grid is sized from a measurement rather
than from an estimate.  Phase C's a-priori estimate was wrong by 1.8x because it
benchmarked different sampler settings than the config actually used; the timing
artefact therefore records the sampler settings alongside the seconds.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:  # pragma: no cover - progress bar is cosmetic
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


#: §8.3 default region of practical equivalence on the log-alpha scale.
DEFAULT_ROPE_LOG = float(np.log(1.25))


def build_pseudorep_design(
    base_design: Any,
    *,
    num_presentations: int = 2,
) -> Dict[str, Any]:
    """
    Turn a one-row-per-menu design into one row per PRESENTATION.

    ``HierarchicalStudyDesign`` emits independent observations; the real design
    presents each menu ``num_presentations`` times with the same items in a
    different order.  Each base row is therefore replicated, and ``menu_id``
    records the grouping.

    ``s`` is recomputed from the replicated menu sizes rather than tiled, so it
    stays centered on the realised mean -- the sim rejects a mis-centered ``s``,
    and tiling a vector centered on the pre-replication mean would trip it only
    when the replication factor changed, which is a nasty thing to debug.
    """
    data = base_design.get_data_dict()
    I = np.asarray(data["I"], dtype=int)
    cell = np.asarray(data["cell"], dtype=int)
    n_menus = I.shape[0]

    order = np.repeat(np.arange(n_menus), num_presentations)
    I_rep = I[order]
    cell_rep = cell[order]
    menu_id = (order + 1).astype(int)

    sizes = I_rep.sum(axis=1).astype(float)
    s = sizes - sizes.mean()

    out = dict(data)
    out["I"] = I_rep.tolist()
    out["cell"] = cell_rep.tolist()
    out["M_total"] = int(I_rep.shape[0])
    out["M_per_cell"] = [
        int(np.sum(cell_rep == j + 1)) for j in range(int(data["J"]))
    ]
    out["s"] = s.tolist()
    out["menu_size"] = sizes.astype(int).tolist()
    out["mean_menu_size"] = float(sizes.mean())
    out["n_menus"] = int(n_menus)
    out["menu_id"] = menu_id.tolist()
    out["num_presentations"] = int(num_presentations)
    return out


class HierarchicalPowerAnalysis:
    """Simulate under a stress regime, fit the independence model, score it."""

    def __init__(
        self,
        sim_model: Any,
        inference_model: Any,
        study_design: Any,
        output_dir: str,
        *,
        n_iterations: int = 20,
        n_mcmc_samples: int = 2000,
        n_mcmc_warmup: Optional[int] = None,
        n_mcmc_chains: int = 4,
        adapt_delta: float = 0.95,
        num_presentations: int = 2,
        rho_copy: float = 0.0,
        sim_overrides: Optional[Dict[str, Any]] = None,
        sim_only_keys: Sequence[str] = (),
        rope_log: float = DEFAULT_ROPE_LOG,
        interval_prob: float = 0.90,
        seed: int = 12345,
    ):
        self.sim_model = sim_model
        self.inference_model = inference_model
        self.study_design = study_design
        self.output_dir = output_dir
        self.n_iterations = n_iterations
        self.n_mcmc_samples = n_mcmc_samples
        self.n_mcmc_warmup = n_mcmc_warmup or n_mcmc_samples // 2
        self.n_mcmc_chains = n_mcmc_chains
        self.adapt_delta = adapt_delta
        self.num_presentations = num_presentations
        self.rho_copy = rho_copy
        self.sim_overrides = dict(sim_overrides or {})
        self.sim_only_keys = tuple(sim_only_keys)
        self.rope_log = rope_log
        self.interval_prob = interval_prob
        self.seed = seed

        os.makedirs(self.output_dir, exist_ok=True)

    # -- Main loop --

    def run(self) -> Dict[str, Any]:
        sim_data = build_pseudorep_design(
            self.study_design, num_presentations=self.num_presentations
        )
        sim_data["rho_copy"] = float(self.rho_copy)
        sim_data.update(self.sim_overrides)

        lower_q = (1.0 - self.interval_prob) / 2.0
        upper_q = 1.0 - lower_q

        records: List[Dict[str, Any]] = []
        durations: List[float] = []

        for iteration in tqdm(range(self.n_iterations), desc="power"):
            started = time.time()

            sim_fit = self.sim_model.sample(
                data=sim_data,
                seed=self.seed + iteration,
                iter_sampling=1,
                iter_warmup=0,
                chains=1,
                fixed_param=True,
                adapt_engaged=False,
            )
            draw = sim_fit.draws_pd().iloc[0]

            true_gamma_size = float(draw["gamma_size"])
            agreement_rate = float(draw.get("agreement_rate", float("nan")))
            y = [int(draw[f"y[{m + 1}]"]) for m in range(sim_data["M_total"])]

            inference_data = {
                k: v
                for k, v in sim_data.items()
                if k
                not in (
                    "gamma0_mean",
                    "gamma0_sd",
                    "gamma_sd",
                    "sigma_cell_sd",
                    "beta_sd",
                    "rho_copy",
                    "n_menus",
                    "menu_id",
                    "num_presentations",
                )
                + self.sim_only_keys
            }
            inference_data["y"] = y

            fit = self.inference_model.sample(
                data=inference_data,
                seed=self.seed + 1000 + iteration,
                iter_sampling=self.n_mcmc_samples,
                iter_warmup=self.n_mcmc_warmup,
                chains=self.n_mcmc_chains,
                adapt_delta=self.adapt_delta,
                show_progress=False,
            )

            posterior = fit.draws_pd()["gamma_size"].to_numpy()
            record = self._score(
                posterior, true_gamma_size, lower_q, upper_q
            )
            record.update(
                {
                    "iteration": iteration + 1,
                    "true_gamma_size": true_gamma_size,
                    "agreement_rate": agreement_rate,
                    "seconds": time.time() - started,
                }
            )
            try:
                diag = fit.diagnose()
                record["divergences"] = "no problems" not in (diag or "").lower()
            except Exception:  # pragma: no cover - diagnose is best effort
                record["divergences"] = None

            records.append(record)
            durations.append(record["seconds"])

        summary = self._summarize(records, sim_data)
        self._write(records, summary, sim_data, durations)
        return summary

    # -- Scoring --

    def _score(
        self,
        posterior: np.ndarray,
        truth: float,
        lower_q: float,
        upper_q: float,
    ) -> Dict[str, Any]:
        mean = float(posterior.mean())
        lower = float(np.quantile(posterior, lower_q))
        upper = float(np.quantile(posterior, upper_q))

        excludes_zero = bool(lower > 0 or upper < 0)
        correct_sign = bool(np.sign(mean) == np.sign(truth)) if truth != 0 else None
        # A type-S error is a CONFIDENT wrong sign, which is qualitatively worse
        # than failing to detect: it would be reported as a finding.
        type_s = bool(
            excludes_zero and truth != 0 and np.sign(mean) != np.sign(truth)
        )
        outside_rope = bool(lower > self.rope_log or upper < -self.rope_log)

        return {
            "posterior_mean": mean,
            "ci_lower": lower,
            "ci_upper": upper,
            "ci_width": upper - lower,
            "covered": bool(lower <= truth <= upper),
            "excludes_zero": excludes_zero,
            "correct_sign": correct_sign,
            "type_s": type_s,
            "outside_rope": outside_rope,
        }

    def _summarize(
        self, records: List[Dict[str, Any]], sim_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        def rate(key: str) -> Optional[float]:
            vals = [r[key] for r in records if r[key] is not None]
            return float(np.mean(vals)) if vals else None

        widths = np.array([r["ci_width"] for r in records], dtype=float)
        errors = np.array(
            [r["posterior_mean"] - r["true_gamma_size"] for r in records],
            dtype=float,
        )
        seconds = np.array([r["seconds"] for r in records], dtype=float)

        return {
            "n_iterations": len(records),
            "regime": "a_pseudo_replication",
            "rho_copy": self.rho_copy,
            "num_presentations": self.num_presentations,
            "n_menus": sim_data["n_menus"],
            "M_total": sim_data["M_total"],
            "J": sim_data["J"],
            "menus_per_cell": sim_data["n_menus"] / sim_data["J"],
            "power_excludes_zero": rate("excludes_zero"),
            "power_outside_rope": rate("outside_rope"),
            "correct_sign_rate": rate("correct_sign"),
            "type_s_rate": rate("type_s"),
            "coverage": rate("covered"),
            "nominal_coverage": self.interval_prob,
            "mean_ci_width": float(widths.mean()),
            "bias": float(errors.mean()),
            "rmse": float(np.sqrt((errors ** 2).mean())),
            "mean_agreement_rate": float(
                np.nanmean([r["agreement_rate"] for r in records])
            ),
            "seconds_per_iteration": float(seconds.mean()),
            "total_seconds": float(seconds.sum()),
            "sampler": {
                "iter_warmup": self.n_mcmc_warmup,
                "iter_sampling": self.n_mcmc_samples,
                "chains": self.n_mcmc_chains,
                "adapt_delta": self.adapt_delta,
            },
            "provisional": True,
            "frozen_at": None,
        }

    def _write(
        self,
        records: List[Dict[str, Any]],
        summary: Dict[str, Any],
        sim_data: Dict[str, Any],
        durations: List[float],
    ) -> None:
        with open(os.path.join(self.output_dir, "iterations.json"), "w") as fh:
            json.dump(records, fh, indent=2)
        with open(os.path.join(self.output_dir, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        # Timing is its own artefact so a grid can be costed without reloading
        # the (much larger) iteration records.
        with open(os.path.join(self.output_dir, "timing.json"), "w") as fh:
            json.dump(
                {
                    "seconds_per_iteration": summary["seconds_per_iteration"],
                    "seconds": durations,
                    "sampler": summary["sampler"],
                    "M_total": sim_data["M_total"],
                    "n_menus": sim_data["n_menus"],
                    "J": sim_data["J"],
                    "measured_not_estimated": True,
                },
                fh,
                indent=2,
            )
