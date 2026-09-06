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
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from utils.cmdstan_artifacts import gzip_csv_files

try:  # pragma: no cover - progress bar is cosmetic
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


#: §8.3 default region of practical equivalence on the log-alpha scale.
DEFAULT_ROPE_LOG = float(np.log(1.25))

DIAGNOSTIC_EXCLUDE_PREFIXES = (
    "log_lik", "y_pred", "eta", "alpha_obs", "log_alpha_obs",
    "T_obs", "T_rep", "ppc_",
)


def fit_diagnostics(fit: Any, *, seconds: float, max_treedepth: int) -> Dict[str, Any]:
    """Return all-parameter mixing diagnostics, excluding per-observation arrays."""
    summary = fit.summary()
    parameters = [
        name for name in summary.index
        if not name.startswith(DIAGNOSTIC_EXCLUDE_PREFIXES)
    ]
    bulk_columns = [
        column for column in summary.columns
        if "ESS_bulk" in column or column == "N_Eff"
    ]
    rhat_columns = [column for column in summary.columns if "hat" in column.lower()]
    bulk = (
        {name: float(summary.loc[name, bulk_columns[0]]) for name in parameters}
        if bulk_columns else {}
    )
    rhat = (
        {name: float(summary.loc[name, rhat_columns[0]]) for name in parameters}
        if rhat_columns else {}
    )
    bulk = {name: value for name, value in bulk.items() if np.isfinite(value)}
    rhat = {name: value for name, value in rhat.items() if np.isfinite(value)}

    method_variables = fit.method_variables()
    treedepth = np.asarray(method_variables["treedepth__"])
    divergent = np.asarray(method_variables["divergent__"])
    minimum_bulk = min(bulk.values()) if bulk else None
    return {
        "seconds": seconds,
        "mean_treedepth": float(treedepth.mean()),
        "max_treedepth_reached": int(treedepth.max()),
        "treedepth_saturated_share": float((treedepth >= max_treedepth).mean()),
        "divergences": int(divergent.sum()),
        "ess_bulk": bulk,
        "rhat": rhat,
        "min_ess_bulk": minimum_bulk,
        "max_rhat": max(rhat.values()) if rhat else None,
        "ess_bulk_per_1000_seconds": (
            minimum_bulk / (seconds / 1000.0)
            if minimum_bulk is not None and seconds > 0 else None
        ),
        "worst_ess": sorted(bulk.items(), key=lambda item: item[1])[:12],
        "worst_rhat": sorted(rhat.items(), key=lambda item: -item[1])[:12],
    }


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
    # Regime (b) supplies a real vector here; zeros reproduce the plain model.
    out["cell_offset"] = [0.0] * int(data["J"])
    return out


def interaction_component(offsets: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    The part of a per-cell shift that the design matrix CANNOT absorb.

    ``h_m01_size`` writes ``log_alpha_cell[j] = gamma0 + X[j] * gamma +
    sigma_cell * z[j]``.  Anything lying in the column space of ``[1 X]`` is
    therefore soaked up by the main effects and is not an interaction at all;
    only the residual is.  With 6 models x 3 prompts the residual space is
    exactly the 10-dimensional model x prompt interaction space -- the two are
    the SAME subspace, verified by rank -- so this projection is what turns a
    stated cell shift into the quantity ``sigma_cell * z`` has to represent.

    It matters for scoring as well as for construction: a spike on ONE cell is
    NOT orthogonal to the main effects, so part of it is legitimately absorbed
    by ``gamma`` and the posterior residual must be compared against the
    projected truth, not against the raw spike.
    """
    A = np.column_stack([np.ones(len(offsets)), np.asarray(X, dtype=float)])
    coef, *_ = np.linalg.lstsq(A, np.asarray(offsets, dtype=float), rcond=None)
    return np.asarray(offsets, dtype=float) - A @ coef


def sparse_interaction_offsets(
    n_cells: int, spike_cell: int, magnitude: float
) -> np.ndarray:
    """
    The regime (b) SPARSE truth: one cell shifted, every other cell exactly 0.

    This is the substantive form of RQ3's question.  "Do prompts move some
    models more than others" is a claim about CONCENTRATION, and its sharpest
    version is that a single (model, prompt) combination reacts while the rest
    are null.  A Gaussian ``sigma_cell`` cannot represent that: it assumes the
    interaction dimensions are exchangeable, so it must either shrink the one
    real deviation or inflate all of them.  Whether it nonetheless LOCALIZES
    the spike is the whole question, and it is answerable with the existing
    model -- only a failure here would justify building a horseshoe variant.

    The vector is centered so the spike does not masquerade as a shift in the
    grand level, which ``gamma0`` would absorb anyway.
    """
    offsets = np.zeros(int(n_cells), dtype=float)
    offsets[int(spike_cell)] = float(magnitude)
    return offsets - offsets.mean()


def matched_dense_offsets(
    sparse_offsets: np.ndarray, X: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """
    The control arm: a DENSE Gaussian interaction of the SAME magnitude.

    Matching is on the INTERACTION COMPONENT, not on the raw vector, and it is
    matched exactly rather than in expectation.  Both arms therefore put the
    identical amount of energy into the subspace ``sigma_cell * z`` models, and
    differ only in whether that energy sits in one cell or is spread over all
    of them.  Without this control, "the spiked cell was flagged" could not be
    separated from "some cell is always flagged when sigma_cell is large" --
    the same confound the MAR arm removes in regime (c).

    The returned vector lies entirely in the interaction subspace, so the dense
    arm has no main-effect leakage; the sparse arm does, and reports it.
    """
    target = interaction_component(sparse_offsets, X)
    draw = rng.normal(size=len(sparse_offsets))
    resid = interaction_component(draw, X)
    norm = np.linalg.norm(resid)
    if norm == 0:  # pragma: no cover - measure-zero
        return np.zeros_like(target)
    return resid * (np.linalg.norm(target) / norm)


def localization_scores(
    residual_draws: np.ndarray,
    true_residual: np.ndarray,
    spike_cell: int,
    lower_q: float,
    upper_q: float,
) -> Dict[str, Any]:
    """
    Can a Gaussian ``sigma_cell`` point at the cell that actually moved?

    ``residual_draws`` is (draws, J) of ``sigma_cell * z_alpha[j]`` -- the
    posterior of the cell-level deviation that is not explained by the main
    effects.  ``spike_cell`` is the cell the answer is scored against: the
    spiked one in the sparse arm, and the LARGEST TRUE deviation in the dense
    control, which is the honest analogue of the same question.  Two things are
    reported, and they answer different halves of it:

    * ``spike_rank`` / ``spike_is_max`` -- IDENTIFICATION.  Is the cell that
      moved most the largest posterior deviation?  Chance is 1/J.
    * ``spike_flagged`` and ``other_flagged_rate`` -- CALIBRATION.  Does that
      cell's interval exclude zero more often than an unspiked one?  A
      procedure that flags the right cell but also flags half the others has
      not localized anything.

    ``recovered_fraction`` measures the SMEARING the exchangeable prior causes:
    a Gaussian random effect pulls a lone large deviation toward the others, so
    a value well below 1 is the attenuation a sparsity prior would avoid.
    """
    means = residual_draws.mean(axis=0)
    lower = np.quantile(residual_draws, lower_q, axis=0)
    upper = np.quantile(residual_draws, upper_q, axis=0)
    flagged = (lower > 0) | (upper < 0)

    spike = int(spike_cell)
    truth = np.asarray(true_residual, dtype=float)
    order = np.argsort(-np.abs(means))
    others = np.delete(np.abs(means), spike)
    return {
        "posterior_mean_residual": [float(v) for v in means],
        "true_residual": [float(v) for v in truth],
        "flagged": [bool(v) for v in flagged],
        "n_flagged": int(flagged.sum()),
        "residual_sd_across_cells": float(means.std(ddof=1)),
        "spike_cell": spike,
        "spike_rank": int(np.where(order == spike)[0][0]) + 1,
        "spike_is_max": bool(order[0] == spike),
        "spike_flagged": bool(flagged[spike]),
        "other_flagged_rate": float(
            (flagged.sum() - int(flagged[spike])) / (len(means) - 1)
        ),
        "spike_z": float(
            (abs(means[spike]) - others.mean()) / others.std(ddof=1)
        ),
        "recovered_fraction": (
            float(means[spike] / truth[spike]) if truth[spike] != 0 else None
        ),
    }


def agreement_by_size(
    y: Sequence[int], menu_id: Sequence[int], menu_size: Sequence[int]
) -> Dict[str, Any]:
    """
    Share of menus whose presentations all chose the same item, BY MENU SIZE.

    This is the input to §8.5(e)(iii): the position-stability subset is defined
    WITHIN size stratum and then downsampled to the smallest stable count, so
    the retention of the whole subset is set by the WORST size -- which is the
    largest, since agreement falls as choice mass spreads over more near-tied
    alternatives (1 - sum p_i^2 rises with n).

    An absolute zero-flip rule would therefore be dominated by size-2 menus and
    near-empty at size 8, destroying the very predictor variance that identifies
    gamma_size. Reporting retention per size is what makes that visible.
    """
    by_menu: Dict[int, List[int]] = {}
    size_of: Dict[int, int] = {}
    for choice, menu, size in zip(y, menu_id, menu_size):
        by_menu.setdefault(int(menu), []).append(int(choice))
        size_of[int(menu)] = int(size)

    stable_by_size: Dict[int, List[bool]] = {}
    for menu, choices in by_menu.items():
        stable_by_size.setdefault(size_of[menu], []).append(
            len(set(choices)) == 1
        )

    per_size = {
        str(size): {
            "n_menus": len(flags),
            "stable_share": float(np.mean(flags)),
        }
        for size, flags in sorted(stable_by_size.items())
    }
    shares = [v["stable_share"] for v in per_size.values()]
    counts = [
        v["n_menus"] * v["stable_share"] for v in per_size.values()
    ]
    # A size-stratified subset can keep at most the smallest stable count from
    # every size, so the balanced subset size is that minimum times the number
    # of sizes.
    balanced_total = min(counts) * len(counts) if counts else 0.0
    total_menus = sum(v["n_menus"] for v in per_size.values())
    return {
        "per_size": per_size,
        "worst_size_stable_share": float(min(shares)) if shares else None,
        "balanced_subset_menus": float(balanced_total),
        "balanced_subset_retention": (
            float(balanced_total / total_menus) if total_menus else None
        ),
    }


def selective_refusal_mask(
    eta_gap: np.ndarray,
    refusal_rate: float,
    concentration: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Which observations survive selective refusal (§6.4, regime (c)).

    Refusals are NOT missing at random.  A model that declines to choose does so
    disproportionately on menus where the alternatives are close -- which are
    exactly the menus that carry the most information about alpha.  Dropping
    them leaves an easier-looking choice set, so the fitted alpha should be
    biased UPWARD; sizing that bias is the point of the regime.

    ``concentration`` interpolates the targeting: 0 is missing-at-random (the
    null case, which should produce no bias, only lost precision), and larger
    values concentrate refusals on the smallest gaps.  The rate is held exactly
    so that MAR and targeted variants drop the same COUNT and differ only in
    WHICH observations go -- otherwise a bias comparison would confound
    targeting with sample size.
    """
    n = eta_gap.size
    n_drop = int(round(refusal_rate * n))
    if n_drop <= 0:
        return np.ones(n, dtype=bool)

    if concentration <= 0:
        drop_idx = rng.choice(n, size=n_drop, replace=False)
    else:
        # Smaller gap => larger weight. Ranks are used rather than raw gaps so
        # the targeting strength does not depend on the arbitrary scale of eta.
        ranks = eta_gap.argsort().argsort().astype(float)
        weights = np.exp(-concentration * ranks / max(n - 1, 1))
        weights = weights / weights.sum()
        drop_idx = rng.choice(n, size=n_drop, replace=False, p=weights)

    mask = np.ones(n, dtype=bool)
    mask[drop_idx] = False
    return mask


def subset_stan_data(data: Dict[str, Any], mask: np.ndarray) -> Dict[str, Any]:
    """
    Restrict a Stan data dict to the observations selected by *mask*.

    ``s`` is RE-CENTERED on the surviving observations, because the sim and the
    inference model both require a centered covariate and dropping a
    gap-correlated subset shifts its mean.  Leaving it centered on the full
    sample would be a silent specification error.
    """
    out = dict(data)
    I = np.asarray(data["I"], dtype=int)[mask]
    cell = np.asarray(data["cell"], dtype=int)[mask]
    sizes = I.sum(axis=1).astype(float)

    out["I"] = I.tolist()
    out["cell"] = cell.tolist()
    out["M_total"] = int(I.shape[0])
    out["M_per_cell"] = [
        int(np.sum(cell == j + 1)) for j in range(int(data["J"]))
    ]
    out["s"] = (sizes - sizes.mean()).tolist()
    out["menu_size"] = sizes.astype(int).tolist()
    out["mean_menu_size"] = float(sizes.mean())
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
        max_treedepth: int = 12,
        num_presentations: int = 2,
        rho_copy: float = 0.0,
        refusal_rates: Sequence[float] = (),
        refusal_concentration: float = 4.0,
        sparse_interaction: Optional[Dict[str, Any]] = None,
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
        self.max_treedepth = max_treedepth
        self.num_presentations = num_presentations
        self.rho_copy = rho_copy
        self.refusal_rates = tuple(refusal_rates)
        self.refusal_concentration = refusal_concentration
        self.sparse_interaction = dict(sparse_interaction or {})
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

        # -- Regime (b): install the SPARSE truth (§8.5(b)) --------------------
        # The exchangeable Gaussian term is switched OFF so the entire
        # cell-level structure is the supplied spike; otherwise a random
        # sigma_cell would add a second, dense interaction on top of the sparse
        # one and the two arms would no longer differ only in shape.
        spike_offsets = None
        design_X = None
        if self.sparse_interaction:
            J = int(sim_data["J"])
            design_X = np.asarray(sim_data["X"], dtype=float)
            spike_cell = self.sparse_interaction.get("spike_cell")
            spike_cell = J - 1 if spike_cell is None else int(spike_cell)
            spike_offsets = sparse_interaction_offsets(
                J, spike_cell, float(self.sparse_interaction.get("magnitude", 1.0))
            )
            sim_data["sigma_cell_sd"] = 0.0
            self.sparse_interaction["spike_cell"] = spike_cell

        records: List[Dict[str, Any]] = []
        durations: List[float] = []
        rng = np.random.default_rng(self.seed)

        for iteration in tqdm(range(self.n_iterations), desc="power"):
            started = time.time()

            if spike_offsets is not None:
                sim_data["cell_offset"] = spike_offsets.tolist()

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

            # The full gamma vector, not just gamma_size.  RQ4's ordering
            # statistic is a function of the gamma_model contrasts, so the
            # SAMPLING spread of gamma-hat across simulated studies is the one
            # input regime (f) cannot get without real fits.  Capturing it here
            # means the expensive J=18 run serves both purposes at once.
            n_predictors = int(sim_data["P"])
            true_gamma = [
                float(draw[f"gamma[{p + 1}]"]) for p in range(n_predictors)
            ]

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
                    "cell_offset",
                )
                + self.sim_only_keys
            }
            inference_data["y"] = y

            chain_dir = Path(self.output_dir) / f"iteration_{iteration + 1:03d}" / "chains" / "main"
            chain_dir.mkdir(parents=True, exist_ok=True)
            fit_started = time.time()
            fit = self.inference_model.sample(
                data=inference_data,
                seed=self.seed + 1000 + iteration,
                iter_sampling=self.n_mcmc_samples,
                iter_warmup=self.n_mcmc_warmup,
                chains=self.n_mcmc_chains,
                adapt_delta=self.adapt_delta,
                max_treedepth=self.max_treedepth,
                show_progress=False,
                output_dir=str(chain_dir),
            )
            fit_seconds = time.time() - fit_started
            chain_files = gzip_csv_files(fit.runset.csv_files)
            diagnostics = fit_diagnostics(
                fit, seconds=fit_seconds, max_treedepth=self.max_treedepth
            )

            posterior = fit.draws_pd()["gamma_size"].to_numpy()
            record = self._score(
                posterior, true_gamma_size, lower_q, upper_q
            )
            draws = fit.draws_pd()
            gamma_mean = [
                float(draws[f"gamma[{p + 1}]"].mean()) for p in range(n_predictors)
            ]
            record.update(
                {
                    "iteration": iteration + 1,
                    "true_gamma_size": true_gamma_size,
                    "true_gamma": true_gamma,
                    "posterior_mean_gamma": gamma_mean,
                    "agreement_rate": agreement_rate,
                    "retention": agreement_by_size(
                        y, sim_data["menu_id"], sim_data["menu_size"]
                    ),
                    "chain_files": [str(path) for path in chain_files],
                    "diagnostics": diagnostics,
                    "seconds": time.time() - started,
                }
            )

            # -- Regime (c): selective-refusal missingness (§6.4) -------------
            # Refit the SAME simulated data with a gap-targeted subset removed.
            # Pairing complete and filtered on one dataset removes the
            # between-simulation variance, which otherwise swamps a bias this
            # size at n=6.
            if self.refusal_rates:
                eta_gap = np.array(
                    [float(draw[f"eta_gap[{m + 1}]"])
                     for m in range(sim_data["M_total"])]
                )
                true_gamma0 = float(draw["gamma0"])
                complete_gamma0 = float(draws["gamma0"].mean())
                record["true_gamma0"] = true_gamma0
                record["gamma0_complete"] = complete_gamma0
                record["refusal"] = {}
                for rate in self.refusal_rates:
                    for label, concentration in (
                        ("mar", 0.0),
                        ("targeted", self.refusal_concentration),
                    ):
                        mask = selective_refusal_mask(
                            eta_gap, rate, concentration, rng
                        )
                        sub = subset_stan_data(inference_data, mask)
                        sub["y"] = [
                            v for v, keep in zip(inference_data["y"], mask) if keep
                        ]
                        sub_fit = self.inference_model.sample(
                            data=sub,
                            seed=self.seed + 2000 + iteration,
                            iter_sampling=self.n_mcmc_samples,
                            iter_warmup=self.n_mcmc_warmup,
                            chains=self.n_mcmc_chains,
                            adapt_delta=self.adapt_delta,
                            max_treedepth=self.max_treedepth,
                            show_progress=False,
                        )
                        sub_draws = sub_fit.draws_pd()
                        record["refusal"][f"{label}_{rate}"] = {
                            "rate": rate,
                            "concentration": concentration,
                            "n_kept": int(mask.sum()),
                            "mean_eta_gap_kept": float(eta_gap[mask].mean()),
                            "mean_eta_gap_dropped": (
                                float(eta_gap[~mask].mean())
                                if (~mask).any() else None
                            ),
                            "gamma0_mean": float(sub_draws["gamma0"].mean()),
                            "gamma0_shift_vs_complete": float(
                                sub_draws["gamma0"].mean() - complete_gamma0
                            ),
                            "gamma_size_mean": float(
                                sub_draws["gamma_size"].mean()
                            ),
                        }
                record["seconds"] = time.time() - started

            # -- Regime (b): sparse vs matched dense interaction (§8.5(b)) ----
            # The base fit above IS the sparse arm.  The control re-simulates
            # with a dense interaction of the same interaction-subspace
            # magnitude, under the SAME sim seed, so the two arms share gamma,
            # beta, delta and the copy pattern and differ only in the SHAPE of
            # the cell-level structure.  Fitting is not paired on one dataset
            # here (unlike regime (c)) because the arms ARE different truths;
            # sharing the RNG stream is the strongest pairing available.
            if spike_offsets is not None:
                record["sparse"] = {}
                J = len(spike_offsets)
                true_sparse_resid = interaction_component(
                    spike_offsets, design_X
                )
                residual_draws = np.column_stack(
                    [
                        draws["sigma_cell"].to_numpy()
                        * draws[f"z_alpha[{j + 1}]"].to_numpy()
                        for j in range(J)
                    ]
                )
                record["sparse"]["sparse"] = localization_scores(
                    residual_draws,
                    true_sparse_resid,
                    int(self.sparse_interaction["spike_cell"]),
                    lower_q,
                    upper_q,
                )
                record["sparse"]["sparse"]["main_effect_leak"] = float(
                    np.linalg.norm(spike_offsets - true_sparse_resid)
                    / np.linalg.norm(spike_offsets)
                )

                dense_offsets = matched_dense_offsets(
                    spike_offsets, design_X, rng
                )
                dense_sim_data = dict(sim_data)
                dense_sim_data["cell_offset"] = dense_offsets.tolist()
                dense_sim_fit = self.sim_model.sample(
                    data=dense_sim_data,
                    seed=self.seed + iteration,
                    iter_sampling=1,
                    iter_warmup=0,
                    chains=1,
                    fixed_param=True,
                    adapt_engaged=False,
                )
                dense_draw = dense_sim_fit.draws_pd().iloc[0]
                dense_data = dict(inference_data)
                dense_data["y"] = [
                    int(dense_draw[f"y[{m + 1}]"])
                    for m in range(sim_data["M_total"])
                ]
                dense_fit = self.inference_model.sample(
                    data=dense_data,
                    seed=self.seed + 3000 + iteration,
                    iter_sampling=self.n_mcmc_samples,
                    iter_warmup=self.n_mcmc_warmup,
                    chains=self.n_mcmc_chains,
                    adapt_delta=self.adapt_delta,
                    max_treedepth=self.max_treedepth,
                    show_progress=False,
                )
                dense_draws = dense_fit.draws_pd()
                dense_resid_draws = np.column_stack(
                    [
                        dense_draws["sigma_cell"].to_numpy()
                        * dense_draws[f"z_alpha[{j + 1}]"].to_numpy()
                        for j in range(J)
                    ]
                )
                true_dense_resid = interaction_component(
                    dense_offsets, design_X
                )
                record["sparse"]["dense"] = localization_scores(
                    dense_resid_draws,
                    true_dense_resid,
                    int(np.argmax(np.abs(true_dense_resid))),
                    lower_q,
                    upper_q,
                )
                record["seconds"] = time.time() - started

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

        # Sampling spread of gamma-hat around truth, per predictor.  This is the
        # TRUE sampling SE (it includes any inflation from pseudo-replication),
        # not the posterior SD -- which is exactly the distinction that matters,
        # since the whole point of regime (a) is that the posterior SD is too
        # small when presentations are correlated.
        gamma_se: Optional[List[float]] = None
        if len(records) >= 2 and records[0].get("true_gamma") is not None:
            errors_gamma = np.array(
                [
                    np.asarray(r["posterior_mean_gamma"], dtype=float)
                    - np.asarray(r["true_gamma"], dtype=float)
                    for r in records
                ]
            )
            gamma_se = [float(v) for v in errors_gamma.std(axis=0, ddof=1)]

        return {
            "n_iterations": len(records),
            "regime": (
                "b_sparse_interaction"
                if self.sparse_interaction
                else "c_selective_refusal"
                if self.refusal_rates
                else "a_pseudo_replication"
            ),
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
            "gamma_sampling_se": gamma_se,
            "P": sim_data.get("P"),
            "menu_sizes": sorted(set(int(v) for v in sim_data["menu_size"])),
            "retention": self._mean_retention(records),
            "refusal": self._mean_refusal(records),
            "sparse_interaction": self._mean_sparse(records),
            "seconds_per_iteration": float(seconds.mean()),
            "total_seconds": float(seconds.sum()),
            "sampler": {
                "iter_warmup": self.n_mcmc_warmup,
                "iter_sampling": self.n_mcmc_samples,
                "chains": self.n_mcmc_chains,
                "adapt_delta": self.adapt_delta,
                # Recorded because it is a COST driver and a validity
                # condition, and because leaving it unrecorded is how this
                # harness silently ran at the CmdStan default of 10 while
                # hierarchical_parameter_recovery ran at 12 -- so Phase D
                # timings were not comparable to Phase C's.
                "max_treedepth": self.max_treedepth,
            },
            "provisional": True,
            "frozen_at": None,
        }

    @staticmethod
    def _mean_retention(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Average the per-size stability shares across iterations."""
        entries = [r["retention"] for r in records if r.get("retention")]
        if not entries:
            return None
        sizes = sorted(entries[0]["per_size"], key=int)
        return {
            "per_size_stable_share": {
                size: float(
                    np.mean([e["per_size"][size]["stable_share"] for e in entries])
                )
                for size in sizes
            },
            "worst_size_stable_share": float(
                np.mean([e["worst_size_stable_share"] for e in entries])
            ),
            "balanced_subset_retention": float(
                np.mean([e["balanced_subset_retention"] for e in entries])
            ),
        }

    @staticmethod
    def _mean_refusal(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Average the paired refusal shifts across iterations.

        The headline is ``gamma0_shift_vs_complete``: the change in the shared
        log-alpha level caused by removing a gap-targeted subset of the SAME
        data. §6.4 predicts it is positive -- dropping close-call menus leaves an
        easier-looking choice set, so the model reads the decider as more
        deterministic than it is. The MAR arm at the same rate is the control:
        it drops the same COUNT, so any difference between the two is the cost
        of TARGETING rather than of lost sample size.
        """
        entries = [r["refusal"] for r in records if r.get("refusal")]
        if not entries:
            return None
        keys = sorted(entries[0])
        return {
            key: {
                "rate": entries[0][key]["rate"],
                "concentration": entries[0][key]["concentration"],
                "mean_gamma0_shift": float(
                    np.mean([e[key]["gamma0_shift_vs_complete"] for e in entries])
                ),
                "mean_gamma_size": float(
                    np.mean([e[key]["gamma_size_mean"] for e in entries])
                ),
                "mean_eta_gap_kept": float(
                    np.mean([e[key]["mean_eta_gap_kept"] for e in entries])
                ),
                "mean_eta_gap_dropped": float(
                    np.mean([
                        e[key]["mean_eta_gap_dropped"] for e in entries
                        if e[key]["mean_eta_gap_dropped"] is not None
                    ])
                ),
                "n_kept": int(np.mean([e[key]["n_kept"] for e in entries])),
            }
            for key in keys
        }

    @staticmethod
    def _mean_sparse(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Average the regime (b) localization scores across iterations.

        Read the two arms side by side. ``spike_is_max_rate`` in the SPARSE arm
        against 1/J is whether a Gaussian ``sigma_cell`` identifies the cell
        that moved; ``spike_flagged_rate`` against ``other_flagged_rate`` is
        whether it does so with a usable error rate. The DENSE arm is scored on
        its own largest true deviation, so it says what the same procedure
        achieves when the interaction is exchangeable -- the shape the model
        actually assumes. If sparse is no worse than dense, the existing model
        answers RQ3 and no horseshoe variant is needed.

        ``mean_recovered_fraction`` is the attenuation: how much of the true
        deviation survives the exchangeable prior's shrinkage.
        """
        entries = [r["sparse"] for r in records if r.get("sparse")]
        if not entries:
            return None
        n_cells = len(entries[0]["sparse"]["posterior_mean_residual"])
        return {
            "n_cells": n_cells,
            "chance_is_max_rate": 1.0 / n_cells,
            "arms": {
                arm: {
                    "spike_is_max_rate": float(
                        np.mean([e[arm]["spike_is_max"] for e in entries])
                    ),
                    "mean_spike_rank": float(
                        np.mean([e[arm]["spike_rank"] for e in entries])
                    ),
                    "spike_flagged_rate": float(
                        np.mean([e[arm]["spike_flagged"] for e in entries])
                    ),
                    "other_flagged_rate": float(
                        np.mean([e[arm]["other_flagged_rate"] for e in entries])
                    ),
                    "mean_spike_z": float(
                        np.mean([e[arm]["spike_z"] for e in entries])
                    ),
                    "mean_recovered_fraction": float(
                        np.mean([e[arm]["recovered_fraction"] for e in entries])
                    ),
                    "mean_residual_sd_across_cells": float(
                        np.mean(
                            [e[arm]["residual_sd_across_cells"] for e in entries]
                        )
                    ),
                }
                for arm in ("sparse", "dense")
            },
            "main_effect_leak": float(
                np.mean([e["sparse"]["main_effect_leak"] for e in entries])
            ),
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
