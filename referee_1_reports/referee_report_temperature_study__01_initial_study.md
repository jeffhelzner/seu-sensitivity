# Referee Report: Temperature and SEU Sensitivity: Initial Results

**Report reviewed:** `temperature_study/01_initial_study.qmd`
**Review date:** 2026-04-02
**Reviewer role:** Referee for *Judgment and Decision Making*

---

## Summary

This report presents the first empirical application of the SEU sensitivity framework, investigating whether increasing the sampling temperature of GPT-4o reduces the model's estimated sensitivity (α) to subjective expected utility maximization in an insurance claims triage task. Using a well-calibrated Bayesian softmax choice model (m_01) fit independently at five temperature levels, the authors find strong evidence that the global slope Δα/ΔT is negative (P > 0.99), with the sharpest separation between greedy decoding (T = 0.0) and high-temperature conditions (T ≥ 1.0). The finding that T = 0.3 and T = 0.7 are statistically indistinguishable is reported honestly and discussed thoughtfully. The report is clearly written, methodologically rigorous, and transparent in its limitations. However, several issues—primarily concerning the independence assumption across conditions, the limited generalizability claim available from a single-LLM/single-task design, and the absence of prior sensitivity analysis—should be addressed before the work meets full publication standards.

**Recommendation: Minor Revision**

---

## Strengths

1. **Exemplary prior calibration workflow.** The grid search over 12 candidate lognormal priors, evaluated by prior predictive simulation against the actual study design, is a model of principled prior selection. The rationale for departing from the foundational prior is clearly explained, the calibration target (SEU-maximizer selection rate) is substantively meaningful, and the numerical considerations (softmax overflow) are transparent. This is the kind of prior predictive workflow that JDM researchers adopting Bayesian methods should emulate.

2. **Honest reporting of null and ambiguous results.** The near-indistinguishability of α at T = 0.3 and T = 0.7 could easily have been buried under the strong global slope result. Instead, the pairwise comparison table, the strict monotonicity probability (~0.12), and the coarser monotonicity test are all reported prominently. The three competing interpretations (threshold effect, insufficient power, genuine non-monotonicity) are appropriately hedged and none is favored without evidence.

3. **Thorough validation pipeline.** Running both parameter recovery (20 iterations) and SBC (200 simulations) at the actual study scale (M = 300, D = 32) is expensive and commendable. The validation is not perfunctory—it confirms that α is identifiable under the study-specific design and that posterior uncertainty is correctly calibrated. The candid discussion of the β–δ identification issue and its irrelevance to the primary α analysis demonstrates mature modeling judgment.

4. **Position counterbalancing and data quality.** The 3× presentation design with position shuffling is a clear improvement over a pilot that had position-bias confounds. Achieving 0% NA rates across all conditions is noteworthy for LLM-based experiments and is properly contrasted with the pilot's ad hoc imputation.

5. **Reproducibility infrastructure.** The frozen data snapshot with version-controlled artifacts, the refitting instructions, and the complete analysis pipeline from raw data to reported conclusions set a high standard for computational reproducibility. The separation between frozen data and live pipeline re-runs is well-engineered.

---

## Major Issues

### 1. Independence assumption across temperature conditions

**Description:** Each temperature condition is fit as a fully independent estimation problem—separate posterior for α, β, δ—using the same underlying set of R = 30 insurance claims. The report correctly notes this (§Discussion, "Model Adequacy") but does not engage with the inferential consequences. Because the same claims appear at every temperature, the α estimates are not independent across conditions: any idiosyncratic features of this particular claim set (e.g., claims that happen to be especially easy or hard to rank) affect all five estimates simultaneously. The pairwise comparisons and monotonicity tests treat draws from separate posteriors as if they were from a joint distribution, but the correlation structure induced by shared stimuli is not modeled.

**Recommendation:** Acknowledge this limitation explicitly in the pairwise comparison section (§5.2). The statement "the indicator function is evaluated draw-by-draw from the *independent* posteriors" should be accompanied by a caveat that these are independent *conditional on the shared design*, and that a hierarchical model jointly estimating α(T) would properly account for between-condition correlations. This does not require re-analysis, but the reader should understand that the pairwise probabilities may be somewhat overconfident.

**Severity:** Major — the structure of inference across conditions is a core design concern and the current presentation could mislead readers about the evidential strength of pairwise comparisons.

### 2. No prior sensitivity analysis

**Description:** The report documents a principled prior selection (Lognormal(3.0, 0.75)) but does not test whether the qualitative conclusions—especially the global slope and pairwise ordering—are robust to perturbation of this choice. This is listed as a planned "Next Step" (§7, item 4) but is not executed. Given that the prior is moderately informative (90% CI: [5.5, 67]) and placed based on prior predictive simulation against the *same data design*, the concern is whether the posterior is partially echoing the prior rather than being driven by the data.

**Recommendation:** Refit at least the extreme conditions (T = 0.0 and T = 1.5) under two alternative priors—e.g., Lognormal(2.5, 1.0) (wider, lower median) and Lognormal(3.5, 0.5) (narrower, higher median)—and report whether the qualitative ordering and slope sign are stable. This need not be exhaustive; the goal is to demonstrate that the primary conclusion is not an artifact of the specific prior chosen. Given that M = 300 observations per condition is substantial, the prior should be swamped by the data—but this should be demonstrated, not assumed.

**Severity:** Major — prior sensitivity is a standard expectation in Bayesian analyses reported to the JDM community (cf. Kruschke, 2013), and the absence of any robustness check for the primary parameter is a notable gap.

### 3. Limited generalizability acknowledged but underweighted

**Description:** The findings are based on a single LLM (GPT-4o), a single task domain (insurance claims triage), and a single embedding model (text-embedding-3-small). The "Next Steps" section mentions replication across LLMs as future work, but the discussion section does not sufficiently caution the reader against generalizing the temperature–sensitivity relationship to other settings. Cross-study context from later reports in this series (e.g., the Claude × Insurance study and the Ellsberg study) shows that the temperature–α effect does *not* replicate with Claude 3.5 Sonnet—the slope is essentially zero—suggesting the finding may be GPT-4o–specific.

**Recommendation:** While this report was presumably written before the cross-LLM results were available, the discussion should include an explicit boundary statement: "These results establish the temperature–sensitivity relationship for GPT-4o on the insurance triage task. Whether this relationship generalizes to other LLMs, task domains, or embedding methods is an open empirical question." If the cross-LLM studies were completed before revision, referencing their results in a brief "Update" note would substantially strengthen the report.

**Severity:** Major — a JDM reviewer would flag the single-LLM, single-task design as a significant limitation, especially for claims framed as being about "LLM decision-making" generally (as in the abstract/introduction).

---

## Minor Issues

### 1. Causal language in the introduction

**Description:** The introduction states that "increasing the external sampling temperature should *decrease* the estimated α" and later that "temperature *controls* the performance noise." These are causal claims, but the study design—five independent cross-sectional fits to data collected at different temperatures—is better described as observational within a manipulated factor design. The temperature manipulation is controlled, but the mechanism by which temperature enters the choice process is not modeled.

**Recommendation:** Replace "should decrease" with "is expected to be associated with lower" in the introduction. Replace "controls" with "modulates" or "is associated with" in the discussion. The directional prediction is fine; the issue is the implicit causal mechanism claim.

### 2. SBC with 200 simulations: power disclaimer

**Description:** The SBC section appropriately notes that 200 simulations yield modest chi-square power (~10 expected counts per bin). However, the conclusion states that α shows "approximately uniform ranks" without quantifying how much miscalibration could go undetected. A reader unfamiliar with SBC might take the non-rejection as strong evidence of calibration.

**Recommendation:** Add a sentence noting the approximate detectable effect size—e.g., "With 200 simulations and 20 bins, the chi-square test has approximately 50% power to detect a 40% deviation from uniformity in any single bin." This helps the reader calibrate the strength of the SBC evidence.

### 3. Feature construction confound

**Description:** The PCA projection is fitted on pooled embeddings across all temperature conditions, which is appropriate for ensuring a shared coordinate system. However, because the LLM's assessments at different temperatures may produce systematically different embedding distributions, the PCA axes are partially influenced by the temperature manipulation. The report does not discuss whether this could create an aliasing between temperature effects on *assessment quality* and temperature effects on *choice sensitivity*.

**Recommendation:** Add a brief note in the Feature Construction section acknowledging this potential confound. If the explained variance ratios or loadings differ systematically by temperature in the pooled PCA, this would be informative. Even a sentence noting "Because assessments vary with temperature, the shared PCA basis may partly reflect temperature-induced variation in the embedding space, not only variation across claims" would suffice.

### 4. Posterior predictive checks: limited test statistics

**Description:** The three PPC test statistics (log-likelihood, modal choice, mean probability) are all global summaries. These would not detect localized misfit—e.g., systematic over- or under-prediction of choices for specific claims, or differential model adequacy across problems with different numbers of alternatives.

**Recommendation:** Consider adding a problem-level or alternative-level PPC (e.g., calibration plot of predicted vs. observed choice frequencies per alternative, or a PPC conditional on the number of alternatives N_m). This is a suggestion rather than a requirement, as the global PPCs are adequate for the present claims.

### 5. Notation: "between-condition factor"

**Description:** §2.1 describes temperature as a "between-condition factor." The standard JDM terminology would be "between-subjects factor" (each temperature is a separate "subject" in the sense of separate data collection) or simply "experimental factor." Since the "subjects" here are LLM runs rather than human participants, a brief clarification of the analogy would help JDM readers.

**Recommendation:** Rephrase to: "Five temperature levels define the experimental factor. Each level constitutes an independent data collection (analogous to a between-subjects design, where each 'subject' is a separate LLM run at a fixed temperature)."

### 6. Missing specification of choice prompt

**Description:** The report describes the task (insurance claims triage) and the assessment process (LLM evaluates each claim individually), but does not include the actual prompt text shown to the LLM when making choices. For a JDM audience, the exact wording of the decision prompt is important for evaluating whether the task genuinely engages the intended decision-theoretic constructs.

**Recommendation:** Include the choice prompt text (or at minimum a representative example) in an appendix or callout box, or provide a cross-reference to where it can be found in the codebase.

---

## Suggestions for Improvement

1. **Hierarchical temperature model.** The report correctly identifies this as a natural extension (§7, item 2). Even a simple model—e.g., log α(T) = a + b·T with priors on a and b—would directly estimate the slope with proper uncertainty, account for between-condition correlation, and provide a cleaner test of monotonicity. This would address Major Issue 1 and substantially strengthen the contribution.

2. **Connect to inverse temperature in statistical mechanics.** The foundational report (Report 1) mentions the Boltzmann/inverse-temperature interpretation. The empirical report could draw this connection more explicitly: the study is, in effect, examining whether the *external* softmax temperature (token sampling) interacts predictably with the *estimated internal* inverse temperature (α). This framing would resonate with readers familiar with computational rationality and resource-rational analysis (Griffiths et al., 2015; Lieder & Griffiths, 2020).

3. **Effect size interpretation.** The slope Δα/ΔT is reported but its practical significance is not contextualized. How much does the choice probability of the EU-maximizing alternative change across the temperature range? Converting the α difference into predicted choice probability differences (using the monotonicity theorem from Report 1) for a representative set of decision problems would make the effect size tangible for a JDM audience.

4. **Visualize the coarse monotonicity result.** The text reports P(coarse monotonicity) ≈ 0.55 (or similar), but this is not visualized. A small schematic showing the three-group ordering (T=0.0 > mid-range > high-T) alongside the five-group strict ordering would clarify the structure of the evidence.

5. **Position-bias analysis.** The report mentions that position counterbalancing was implemented but does not analyze whether position effects were in fact present (or absent) in the collected data. A brief check—e.g., whether choice frequency varies by presentation position—would validate the design decision and is straightforward to compute from the existing data.

6. **Broader JDM literature engagement.** The discussion connects the findings to the foundational framework's theorems but does not engage with the broader literature on decision noise, stochastic choice, and the interpretation of softmax temperature in human decision-making (e.g., Stahl & Wilson, 1994; McKelvey & Palfrey, 1995, on quantal response equilibrium; Rieskamp, 2008, on evidence accumulation). Even brief connections would situate the contribution for a JDM audience.

---

## Summary Table

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Research Question / Hypotheses | Good | Clear directional hypothesis, well-motivated; could be more explicit about confirmatory vs. exploratory |
| Experimental Design | Good | Sound manipulation, position counterbalancing; single-LLM/single-task limits generalizability |
| Operationalization / Measurement | Good | α is well-defined; feature construction is principled but PCA confound is unaddressed |
| Evidential Strength | Good | Strong global slope evidence; honest about weak pairwise separation; independence assumption unaddressed |
| Robustness / Generalizability | Adequate | No prior sensitivity analysis; no cross-LLM or cross-task evidence (in this report); acknowledged as limitation |
| Causal Reasoning / Interpretation | Good | Mostly appropriate hedging; some causal language in intro/discussion could be softened |
| Relation to Literature | Adequate | Connects to own foundational framework; limited engagement with broader JDM and stochastic choice literature |
| Computational Methodology | Excellent | Parameter recovery, SBC, PPCs all at study scale; priors calibrated via prior predictive simulation |
| Exposition / Transparency | Excellent | Clear writing, well-organized, frozen data snapshot, complete reproducibility pipeline |

---

## Confidential Comments to the Editor

This is a solid, carefully executed study that establishes a meaningful empirical finding within a well-validated Bayesian framework. The primary contribution—demonstrating that LLM sampling temperature has a predictable effect on estimated EU sensitivity for GPT-4o—is novel and relevant to the growing literature on LLM decision-making. The major issues identified (independence assumption, prior sensitivity, generalizability framing) are all addressable without fundamental redesign; the recommended revisions would strengthen the evidential claims without changing the core narrative. The computational methodology is among the most thorough I have seen in a JDM-targeted manuscript, and the reproducibility infrastructure is exemplary.

One concern for the editor: subsequent reports in this series show that the temperature–α effect does not replicate with Claude 3.5 Sonnet on either the same task or a different task, suggesting the finding may be GPT-4o–specific. This does not diminish the initial report's contribution (the finding is still real and interesting), but raises questions about how the report should be framed if published independently of the series. I would recommend that the authors address this in the discussion, either by referencing the cross-LLM results or by explicitly scoping the claims to GPT-4o.

The manuscript is close to publication quality for a JDM venue. Minor revision is warranted; a major revision would be appropriate only if the authors are unable to provide prior sensitivity analysis (which I expect would confirm the robustness of the main finding given M = 300).
