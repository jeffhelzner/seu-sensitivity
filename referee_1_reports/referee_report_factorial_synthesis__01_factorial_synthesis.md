# Referee Report: 2×2 Factorial Synthesis: LLM × Task

**Report reviewed:** `factorial_synthesis/01_factorial_synthesis.qmd`
**Review date:** 2026-04-02
**Reviewer role:** Referee for *Judgment and Decision Making*

---

## Summary

This report presents a post-hoc synthesis of four previously reported studies, organized as a 2×2 factorial design crossing LLM (GPT-4o vs. Claude 3.5 Sonnet) with task domain (insurance triage, K=3, vs. Ellsberg gambles, K=4) to determine which factor drives the previously observed non-replication of a temperature–sensitivity relationship. The design is clean and well-motivated: two studies had been conducted (GPT-4o × Insurance, Claude × Ellsberg); these confounded LLM and task; the two missing cells were subsequently run. The report re-analyses the posterior draws from all four cells and concludes that **LLM is the dominant factor** — GPT-4o shows a consistent negative temperature–α slope (P(slope < 0) ≈ 0.99 and 0.98) while Claude does not (P ≈ 0.56 and 0.77) — with task playing a secondary role and no detectable interaction. The report is visually effective, well-structured, and carefully hedged on most claims. However, the evidential basis for several headline conclusions is weaker than the text suggests, the statistical approach to the factorial analysis is informal, and important interpretive opportunities are missed.

**Recommendation: Minor Revision**

---

## Strengths

1. **Elegant factorial logic that directly addresses the original confound.** The non-replication in the Ellsberg study simultaneously changed the LLM and the task. The decision to run the two missing cells to decompose this confound reflects sound experimental reasoning, and the 2×2 structure is communicated with textbook clarity. The factorial framing transforms a failed replication into a substantive finding — the LLM matters, the task does not — which is a more informative conclusion than "the effect did not replicate."

2. **Exceptional data visualization.** The report's 11 figures form a coherent visual narrative. The 2×2 forest plot grid (@fig-forest-2x2) immediately communicates the core pattern — declining posteriors in the top row, overlapping posteriors in the bottom row. The interaction plots (@fig-interaction, @fig-interaction-slopes) are textbook presentations of the factorial logic. The slope density overlay (@fig-slope-comparison) effectively compresses the four cells into a single panel. The figures are well-labelled, with informative captions that correctly describe what each panel shows. This is a model for visual communication of Bayesian results.

3. **Quantitative interaction analysis.** The difference-in-differences computation on the slope draws is the right approach for testing whether the 2×2 pattern is additive. The report correctly identifies this as a formal test of the interaction term: if the LLM effect were task-specific (or vice versa), this quantity would deviate from zero. The execution is transparent — the code is shown, the draws are propagated properly, and the uncertainty is reported.

4. **Honest reporting of mixed evidence.** The Claude × Ellsberg cell (P(slope < 0) ≈ 0.77) is described as a "weak decline" rather than spun as either a positive finding or a null, and the strict monotonicity probabilities (all below 0.13, including for GPT-4o) are reported without selective emphasis. The discussion accurately characterizes the temperature–sensitivity finding as "LLM-specific" rather than universal.

5. **Reproducibility design.** The report loads pre-computed posterior draws from the four individual cell reports, making the synthesis fully reproducible from frozen artefacts. The data directory table in the Reproducibility section provides a clear audit trail. The analysis code is embedded and executable.

---

## Major Issues

### 1. The "dominant LLM effect" claim is overstated relative to the evidence

**Description:** The report's central conclusion is that "the LLM is the dominant factor." However, the quantitative evidence for the LLM main effect is more equivocal than the text implies. The critical statistic — P(GPT-4o slope < Claude slope) — is 0.817 for the insurance task and 0.797 for the Ellsberg task. These are directionally informative but far from the decisiveness conveyed by the language of "dominance." In the JDM community, a posterior probability of ~0.80 would typically be characterised as "suggestive" or "moderately supported," not as a settled conclusion. By contrast, the within-LLM evidence (P(slope < 0) ≈ 0.99 for GPT-4o, ≈ 0.56 for Claude) is more decisive — but this is evidence about each LLM's slope *individually*, not about the *between-LLM contrast*.

The mismatch arises because the slope draws for the four cells are independent (computed from separate model fits with no shared parameters). This means the between-cell comparisons carry substantial uncertainty from combining two already-uncertain estimates. The report does not distinguish between the strong within-cell evidence (GPT-4o's slope is negative) and the weaker between-cell evidence (GPT-4o's slope is *more negative than Claude's*).

**Recommendation:** (a) Report the P(GPT slope < Claude slope) values prominently in the Discussion, not only in the LLM effect table. (b) Soften the "dominant factor" language to something like "the LLM factor accounts for most of the qualitative variation in temperature–sensitivity patterns." (c) Add a sentence acknowledging that the between-LLM comparison, while directionally clear, carries more uncertainty than the within-LLM evidence. (d) Consider whether a joint hierarchical model (estimating LLM and task effects within a single model) would yield tighter between-cell contrasts; if not feasible, note this as a limitation.

**Severity:** Major — the headline conclusion requires calibration to the actual evidential strength.

### 2. The interaction analysis is under-powered; "minimal interaction" is an absence-of-evidence claim

**Description:** The interaction term (difference-in-differences of slopes) has a median of −1.8 and a 90% CI of [−87.8, 79.0], with P(interaction > 0) ≈ 0.49 and P(interaction < 0) ≈ 0.51. The report interprets this as evidence that "the LLM and task effects are approximately additive" and that the CI "spanning zero is consistent with a purely additive model."

This interpretation conflates absence of evidence with evidence of absence. A 90% CI that spans nearly 170 units of slope (from −88 to +79) is not informative — it is compatible with massive interactions in either direction. The correct conclusion is that the data cannot distinguish between additive and non-additive structures, not that the structure is additive. This is precisely the inferential error that equivalence testing and region-of-practical-equivalence (ROPE) analyses were developed to address (Kruschke, 2013; Lakens, 2017). Without defining what magnitude of interaction would be "meaningful" and showing that the posterior concentrates within that ROPE, the claim of minimal interaction is unsupported.

The extreme width of the CI reflects the propagation of uncertainty through the difference-in-differences: each slope draw is computed by regression on five points with substantial posterior uncertainty, and two such differences are then subtracted. This is a fundamental statistical limitation of the approach, not a data quality issue.

**Recommendation:** (a) Replace "minimal interaction" with language reflecting the actual epistemic state: "the data are uninformative about the presence or magnitude of an interaction." (b) If maintaining the additivity claim, define a ROPE for the interaction (e.g., |interaction| < 20, or some fraction of the within-cell slope range) and report the posterior probability of falling within it. (c) Acknowledge that the 2×2 design with independent model fits is structurally under-powered to detect interactions; a joint hierarchical model would be needed for sharper interaction estimates. (d) Adjust conclusion #3 ("Minimal interaction") accordingly.

**Severity:** Major — the additivity claim is a key conclusion (it determines the interpretation of the original non-replication) and is not supported by the evidence as currently presented.

### 3. Temperature range confound compromises the factorial comparison

**Description:** The GPT-4o cells use temperatures in {0.0, 0.3, 0.7, 1.0, 1.5} while the Claude cells use {0.0, 0.2, 0.5, 0.8, 1.0}. The report acknowledges this in a callout box and states that "comparisons across LLMs therefore focus on the qualitative pattern (monotonic decline vs. flat / non-monotonic) rather than quantitative slope magnitudes." However, the analysis repeatedly computes and compares quantitative slopes across LLMs (e.g., @tbl-llm-effect reports "GPT-4o slope (med) = −30.8" vs. "Claude slope (med) = −3.6" and computes P(GPT slope < Claude slope) = 0.817). The slope Δα/ΔT is defined by a regression over the full temperature grid, so the GPT-4o slope is estimated over a wider range (ΔT = 1.5) with a different grid spacing than the Claude slope (ΔT = 1.0).

This is more than a minor caveat — it confounds the LLM comparison with the range comparison. A flatter true relationship would produce a less negative slope over a narrower range even if the underlying sensitivity function were identical. The callout box notes this, but the subsequent analysis ignores the constraint. The problem is compounded in the interaction analysis, where the difference-in-differences of slopes inherits the range asymmetry.

**Recommendation:** (a) Compute a "matched-range" slope for GPT-4o using only the draws at T ∈ {0.0, 0.3, 0.7, 1.0} (dropping T = 1.5) and report the LLM comparison on this restricted grid. This does not fully resolve the confound (the grid points still differ), but it narrows it. (b) Alternatively, compute the proportional change (α_max / α_min) or the rank correlation between temperature and α, which are less sensitive to the absolute temperature scale. (c) Discuss explicitly whether the qualitative conclusion (GPT-4o declining, Claude flat) would change if the GPT-4o grid were restricted to [0, 1.0]. The figures suggest it would not, but this should be verified quantitatively, not assumed.

**Severity:** Major — the confound directly affects the primary quantitative comparison and the interaction estimate. The qualitative conclusion is likely robust, but the current analysis does not demonstrate this.

### 4. No formal hypotheses are stated before results

**Description:** The report moves directly from the design summary to the results matrix without articulating formal hypotheses. The introduction implicitly sets up three hypotheses (LLM effect, task effect, interaction), but these are not stated as directional predictions with clear operationalizations. For a JDM audience, the distinction between confirmatory and exploratory analysis is important (Simmons et al., 2011). Which predictions were pre-registered or at least articulated before the data were analysed? Were the individual cell studies run *because* a factorial design was planned, or was the factorial framing imposed post-hoc?

The introductory callout box previews the key finding ("The LLM is the dominant factor") before any results are presented, which further blurs the confirmatory–exploratory boundary. This is unusual in JDM reporting and removes any tension about the results.

**Recommendation:** (a) Add a brief "Hypotheses" subsection between the Design Summary and Results Matrix that states the predictions being tested (e.g., "H1: P(slope < 0) will be higher for GPT-4o than Claude within both tasks; H2: The task effect will be smaller than the LLM effect; H3: The interaction will be negligible"). (b) Clearly indicate whether these hypotheses were formulated before the missing cells were run, or whether the synthesis is exploratory. (c) Move the key-finding callout box to the Discussion or, at minimum, label it as a preview rather than a premise. (d) If the factorial structure was not fully planned a priori (i.e., the missing cells were run reactively after the non-replication), acknowledge this as an important methodological caveat — the design is sound regardless, but transparency about the design chronology matters.

**Severity:** Major — the absence of explicit hypotheses and confirmatory/exploratory labelling does not meet JDM transparency norms.

---

## Minor Issues

### 1. Strict monotonicity probabilities are surprisingly low for GPT-4o and deserve comment

**Description:** The P(strict monotonicity ↓) values are 0.1247 for GPT-4o × Insurance and 0.0902 for GPT-4o × Ellsberg. These are remarkably low for a relationship described as "clear monotonic negative." The summary table (@tbl-monotonicity) reports the "Pattern" as "Declining" for GPT-4o cells, but the strict monotonicity probabilities — which measure whether α is strictly decreasing at *every* consecutive temperature step — tell a different story. A probability below 0.13 means that in nearly 90% of posterior draws, at least one adjacent pair reverses.

**Recommendation:** Add a paragraph in the Results or Discussion noting that while the global slope is clearly negative, the trajectory is not strictly monotonic — there are local reversals (most likely at adjacent temperatures like T=0.3/0.7), indicating that the relationship is a noisy decline rather than a smooth monotonic function. This is relevant for the theoretical interpretation: a strict monotonic decline would support a simple "temperature adds noise" mechanism, while a noisy decline is consistent with more complex effects.

### 2. The report cites no references

**Description:** The References section is empty (`::: {#refs} :::`). The report makes no citations to the JDM literature, the SEU/expected utility literature, the LLM decision-making literature, or the Bayesian methods literature. For a JDM journal submission, this is a significant omission. The theoretical framing (SEU, Ellsberg paradox, softmax choice models) has a rich intellectual history that should be acknowledged. The experimental design choices (factorial structure, temperature manipulation) should be connected to the literature on moderator analysis and experimental methodology.

**Recommendation:** Add citations to at least: (a) Ellsberg (1961) for the Ellsberg task; (b) Luce (1959) or McFadden (1974) for the softmax/logit choice model; (c) Camerer (1995) or Hey & Orme (1994) for experimental methods in decision under uncertainty; (d) relevant LLM decision-making papers (e.g., Binz & Schulz, 2023; Hagendorff et al., 2023); (e) Kruschke (2013) for the Bayesian approach.

### 3. No posterior predictive checks are reported in the synthesis

**Description:** The individual cell reports include posterior predictive checks that assess model fit. The synthesis report does not mention these at all. While it would be redundant to repeat the full checks, a brief summary (e.g., "Posterior predictive p-values were in the acceptable range [0.3, 0.7] for all four cells; see individual reports") would assure the reader that the model adequately describes the data in all conditions being compared.

**Recommendation:** Add a one-paragraph summary of posterior predictive adequacy across all four cells, with cross-references to the individual reports.

### 4. The grand summary figure (@fig-grand-summary) plots all cells on the same x-axis despite different temperature grids

**Description:** The figure overlays GPT-4o trajectories (plotted at T = 0.0, 0.3, 0.7, 1.0, 1.5) and Claude trajectories (plotted at T = 0.0, 0.2, 0.5, 0.8, 1.0) on the same x-axis. This creates a visual impression of direct comparability that the text correctly warns against. A reader glancing at the figure without reading the caption might conclude that GPT-4o and Claude were tested at the same temperatures.

**Recommendation:** Either (a) add light vertical marker bands at the two different temperature grids to make the non-alignment visible, or (b) use a faceted version (left panel: GPT-4o temperatures, right panel: Claude temperatures) with matched y-axes. Alternatively, use a normalised x-axis (e.g., temperature quantiles or proportion of maximum temperature) to enable more legitimate visual comparison.

### 5. The interaction plot (@fig-interaction) uses P(slope < 0) as the dependent variable, which is non-linear and potentially misleading

**Description:** The interaction analysis is conducted on the slope draws (a linear scale), but the interaction *plot* uses P(slope < 0) as the y-axis. P(slope < 0) is a tail probability — a non-linear transformation of the slope distribution's location and spread. Two conditions with identical slope distributions but different variances would plot at different P values. The roughly parallel lines in @fig-interaction may not imply additive effects on the slope scale. The slope-based interaction plot (@fig-interaction-slopes) is more appropriate and should be the primary figure.

**Recommendation:** Either remove @fig-interaction or relegate it to a secondary role. Make @fig-interaction-slopes the primary interaction figure, as it operates on the same scale as the quantitative interaction analysis.

### 6. The slope computation method is informal

**Description:** The slope is computed by fitting a simple least-squares regression to the α draws at each temperature level, independently for each posterior draw. This is a reasonable descriptive summary, but it treats all temperature levels as equally spaced and equally weighted, and it does not account for the different spacing of the GPT-4o and Claude grids. More importantly, it computes the slope *outside* the Bayesian model — the Stan models estimate α separately at each temperature, and the slope is imposed post-hoc. This means the slope uncertainty reflects only α posterior uncertainty, not model uncertainty about the functional relationship between temperature and α.

**Recommendation:** Acknowledge in the Methods or Analysis section that the slope is a derived summary computed from independent per-temperature posteriors, not a model parameter. Note that this approach cannot distinguish between linear and non-linear temperature–α relationships (both would produce a negative slope).

---

## Suggestions for Improvement

1. **Consider a joint hierarchical model.** The current synthesis analyses four independent model fits by combining their posterior draws. A hierarchical model that estimates LLM and task effects as group-level parameters — treating the four cells as exchangeable (or partially pooled) members of a factorial structure — would yield tighter between-cell contrasts and formal effect-size estimates. This would substantially strengthen the main-effect and interaction claims. Such a model could be fit directly to the four sets of α posterior draws (a meta-analytic approach) without reprocessing the raw choice data.

2. **Compute effect sizes.** The report discusses "dominant" and "secondary" effects but provides no formal effect-size measures. For a 2×2 factorial, natural effect-size quantities include the variance explained by each factor and the interaction, or the standardized mean difference between factor levels. Even an informal variance decomposition (what fraction of the between-cell slope variability is attributable to LLM vs. task?) would strengthen the "dominance" claim.

3. **Connect to the LLM interpretability literature.** The finding that temperature affects GPT-4o's decision sensitivity but not Claude's is potentially informative about how these models implement temperature scaling internally. The Discussion section mentions "mechanistic investigation" as a future direction but does not connect to existing work on LLM temperature, sampling strategies, or the constitutional AI / RLHF processes that differentiate these models. Even brief references to this literature would position the finding within a broader research program.

4. **Report the absolute α levels.** The synthesis focuses entirely on the *slope* of the temperature–α relationship and does not discuss the *level*. GPT-4o at T=0.0 has α ≈ 128 (Insurance) or ≈ 110 (Ellsberg), while Claude at T=0.0 has α ≈ 28 (Insurance) or ≈ 55 (Ellsberg). These baseline differences are substantial (4–5× for insurance, 2× for Ellsberg) and raise a separate question: is GPT-4o generally more SEU-sensitive than Claude, regardless of temperature? This level effect is orthogonal to the slope effect but equally interesting for a JDM audience.

5. **Discuss the oscillatory Claude pattern.** Both Claude cells show a non-monotonic, oscillatory pattern (the individual cell reports describe this in detail). The synthesis report mentions this briefly but does not engage with it. Is the oscillation a systematic feature of Claude's temperature implementation, or is it noise? Do the oscillation patterns align across the two Claude cells (e.g., both dip at the second temperature level)? If so, this would be a substantive finding in its own right.

6. **Add a Methods section.** The report jumps from design to results without a clear Methods section describing how the secondary analyses (slope computation, main effects, interaction) were conducted. While the code is embedded and transparent, a brief prose section explaining the analytical approach would make the report accessible to readers who do not read Python code.

---

## Summary Table

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Research Question / Hypotheses | Good | Clear question; hypotheses not formally stated before results |
| Experimental Design | Excellent | Textbook 2×2 factorial that cleanly addresses the original confound |
| Operationalization / Measurement | Good | α well-defined via component studies; synthesis inherits their operationalization |
| Evidential Strength | Adequate | Strong within-cell evidence; between-cell comparisons (~P ≈ 0.80) weaker than claimed |
| Robustness / Generalizability | Adequate | Temperature range confound unresolved; only 2 LLMs and 2 tasks |
| Causal Reasoning / Interpretation | Good | Factorial logic is sound; interaction claim overstated given wide CI |
| Relation to Literature | Insufficient | No references cited; no connection to JDM, EU theory, or LLM literature |
| Computational Methodology | Good | Slope computation is informal but adequate; no independent model fitting |
| Exposition / Transparency | Excellent | Outstanding figures; transparent code; clear structure |

---

## Confidential Comments to the Editor

This synthesis report is the natural capstone of the four individual cell studies and makes a genuine contribution by resolving the original confound. The factorial design is elegant, the visualization is among the best I have seen in a computational JDM submission, and the overall conclusion (GPT-4o shows the effect, Claude does not, task matters less) is likely correct.

My main concern is that the quantitative evidence for the between-LLM comparison is weaker than the framing suggests. The P(GPT slope < Claude slope) values of ~0.80 and the extremely wide interaction CI do not support the confident language of "dominant factor" and "minimal interaction." A reader who examines the numbers carefully will notice this gap. The qualitative conclusion is well-supported — the GPT-4o posteriors decline clearly while the Claude posteriors do not — but the formal factorial analysis does not yet match the informal visual impression.

The absence of any references is unusual and should be addressed before publication. The report reads more like a project-internal document than a journal submission; situating it within the JDM and LLM literature would substantially improve its suitability for the venue.

The temperature range confound (OpenAI [0, 1.5] vs. Anthropic [0, 1.0]) is a structural limitation that cannot be fully resolved but should be addressed more seriously than the current callout box. A matched-range reanalysis would take minimal additional work and would substantially strengthen the cross-LLM comparison.

These issues are all addressable within a minor revision. The core contribution — a clean factorial decomposition of a non-replication — is sound and valuable.
