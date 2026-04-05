# Referee Report: Temperature and SEU Sensitivity — Risky Alternatives Extension

**Report reviewed:** `01_risky_alternatives_study.qmd`  
**Path:** `reports/applications/temperature_study_with_risky_alts/01_risky_alternatives_study.qmd`  
**Date:** April 2, 2026

---

## Summary

This report extends the initial temperature–sensitivity study (Report 1) by introducing risky alternatives — choice problems with explicitly stated probability simplexes — alongside the original uncertain alternatives, and fitting three augmented models (m_11, m_21, m_31) that differ in how sensitivity is shared or separated across decision contexts. The study produces two substantive findings: (i) the negative temperature–sensitivity relationship from Report 1 replicates robustly under all three model specifications, and (ii) the LLM (GPT-4o) exhibits systematically *lower* EU sensitivity in the risky context than in the uncertain context, a finding that is both surprising and theoretically provocative. The experimental design is clean, the model comparison logic is well-executed, and the posterior predictive check analysis provides genuine discriminative power across models. The report is well-written and the findings are appropriately hedged. There are, however, several issues — primarily concerning the interpretation of the α/ω gap, the absence of formal model comparison, and the independence assumption — that require attention before the work meets full JDM journal standards.

**Recommendation: Minor Revision**

---

## Strengths

1. **Well-motivated model family with clear nesting structure.** The progression from m_11 (shared α) through m_21 (separate α, ω) to m_31 (proportional ω = κ·α) constitutes an exemplary model comparison strategy. Each model encodes a substantively different structural hypothesis about the relationship between uncertain and risky sensitivity, and the nesting relationships are cleanly exposited. The decision to present all three rather than selecting one ex ante is appropriate for an application report in a new research program.

2. **Posterior predictive checks provide genuine model discrimination.** The PPC analysis is the analytical highlight of the report. The demonstration that m_11's shared α produces systematically inflated risky-choice PPC statistics (modal, prob), while m_21's separate ω resolves this miscalibration, is compelling model-adequacy evidence. This goes beyond routine diagnostics to reveal substantive model misfit with interpretive consequences — exactly what PPCs should do.

3. **Clean replication of the temperature–sensitivity effect.** The core finding from Report 1 — that α declines with sampling temperature, with posterior probability exceeding 0.99 for a negative global slope — is reproduced under three different model specifications with both partially overlapping and novel data. The pairwise comparison structure (strong separation at extremes, near-indistinguishability of T = 0.3 and T = 0.7) replicates exactly, lending substantial credibility to the phenomenon.

4. **Thoughtful cross-study comparison.** The comparison of α posteriors across m_01, m_11, m_21, and m_31 is well-designed and reveals interpretable patterns: m_11's tighter but downward-biased α (pulled by the risky context), and the recovery of m_01-like values by m_21 and m_31 when the risky context is given its own parameter. This provides implicit validation that the m_01 estimates in Report 1 were not confounded by the absence of risky data.

5. **Risky alternative design is well-constructed.** The 30-alternative pool with corner, balanced, and intermediate probability profiles provides good coverage of the simplex. Maintaining parallel design features (100 base × 3 presentations, 2–4 alternatives per problem, position counterbalancing) across the two contexts enables clean comparison.

---

## Major Issues

### 1. The α > ω Finding Lacks a Formal Aggregate Test

**Description:** The report's most novel empirical finding — that GPT-4o is *less* sensitive to EU in the risky context (stated probabilities) than in the uncertain context (inferred probabilities) — is presented via per-temperature point estimates and credible intervals from m_21 and m_31, supplemented by qualitative commentary. However, there is no formal aggregate test of this claim. Given that the finding is counterintuitive (one might expect greater sensitivity when probabilities are known), the evidentiary burden is correspondingly higher.

**Recommendation:** Compute and report the posterior probability P(α > ω) at each temperature under m_21, and the posterior probability P(κ < 1) at each temperature under m_31. Additionally, compute an aggregate measure: the posterior distribution of the mean difference across temperatures (draw-wise, average α_t − ω_t over t = 0.0, ..., 1.5, then report median and 90% CI of this average). This would give the reader a single, interpretable summary of the strength of evidence for context-dependent sensitivity.

**Severity:** Major — the central novel finding of the report lacks the formal quantification needed to evaluate its strength.

### 2. No Formal Model Comparison

**Description:** The report fits three models and evaluates their adequacy via PPCs, concluding that m_21 is best-calibrated. However, there is no formal model comparison using information criteria (e.g., WAIC, LOO-CV) or Bayes factors. The PPC analysis is valuable for detecting specific forms of misfit, but it does not quantify the predictive performance tradeoff between the models' different parameterizations. Given that the three models differ in parameter count (m_11: 3 free; m_21: 4; m_31: 4) and that m_31 nests m_11 (when κ = 1), information-theoretic comparison would sharpen the model selection argument.

**Recommendation:** Report LOO-CV (via Pareto-smoothed importance sampling, which CmdStanPy supports straightforwardly) or WAIC for each model at each temperature. If this is computationally infeasible, at minimum report the combined log-likelihood evaluated at posterior means as a rough comparison. The PPC analysis should be framed as a complement to, not a substitute for, information-theoretic model comparison.

**Severity:** Major — the model comparison story is incomplete without some quantitative comparison of predictive performance.

### 3. Independence Assumption in Cross-Study Comparisons

**Description:** As flagged in the referee report for Report 1, the per-temperature fits are independent — no joint model pools information across temperatures — and the cross-study comparison with m_01 treats independently estimated posteriors as if they were draws from a joint distribution. In this report, the issue is compounded: the uncertain-choice data in the augmented models are *the same data* as in m_01, so the m_01 and m_11/m_21/m_31 posteriors for α are not independent even conditional on temperature. The cross-study comparison (§6) implicitly treats them as independent.

**Recommendation:** Acknowledge explicitly that the cross-study comparison is informal because the models share the same uncertain-choice data. The m_21/m_31 α estimates are expected to be similar to m_01 because they are fitting the same uncertain observations — this is not independent confirmation so much as a consistency check. Frame it as such. If the authors wish to claim that the augmented models "recover m_01-like α values," they should note that this is a necessary consequence of the shared likelihood, not an independent finding.

**Severity:** Major — could mislead readers about the degree of independent replication.

### 4. Interpretations of ω < α Are Speculative and Underqualified

**Description:** The Discussion (§7) offers three interpretations of the finding that risky sensitivity is lower than uncertain sensitivity: (a) a format effect, (b) calibration asymmetry from the β·w → softmax mapping, and (c) utility estimation precision (fine EU differences among similar simplexes). These are interesting but entirely post hoc, and they range from the plausible (c) to the speculative (a). The format effect interpretation, in particular, is presented without any supporting evidence and would be difficult to distinguish from the calibration asymmetry account without additional experiments.

**Recommendation:** (i) Label these interpretations explicitly as post hoc. (ii) Note that interpretation (b) — the adaptive β layer effectively sharpening probabilities — is partially testable: one could examine whether the estimated subjective probabilities ψ_r under the fitted model are more "peaked" (lower entropy) than the stated risky simplexes, which would be consistent with this account. (iii) For interpretation (c), report the distribution of EU differences among alternatives in risky vs. uncertain problems to assess whether risky problems genuinely present finer distinctions. (iv) Acknowledge that any of these explanations, or some combination, could be operative; the current data do not discriminate among them.

**Severity:** Major — the interpretations carry much of the discussion's weight but are not grounded in evidence.

---

## Minor Issues

### 1. Prior Sensitivity Analysis Still Missing

**Description:** The same α prior — Lognormal(3.0, 0.75) — is carried forward from Report 1 without robustness checking, and the ω prior in m_21 adopts the same hyperparameters by symmetry. The κ prior in m_31 — Lognormal(0, 0.5) — is moderately informative (90% CI [0.44, 2.28]) but not sensitivity-checked. The referee report for Report 1 flagged the absence of prior sensitivity analysis as a major issue.

**Recommendation:** Refit at least one temperature condition (e.g., T = 0.0 or T = 1.0) under the m_21 model with alternative priors — e.g., α ~ Lognormal(2.5, 1.0), ω ~ Lognormal(2.5, 1.0) — and verify that the qualitative finding (ω < α) and the PPC pattern are preserved. Report prior-to-posterior contraction ratios for the key parameters.

### 2. Description of Risky Decision Problems Is Incomplete

**Description:** The report specifies the risky alternative pool (S = 30 simplexes) and the design (100 base × 3 presentations), but does not describe how the decision problems are *constructed*: how were the 2–4 alternatives per problem selected from the pool of 30? Random sampling? Stratified by entropy or EU dominance? The answers affect whether there are easy/hard subsets and may interact with the sensitivity estimates.

**Recommendation:** Describe the problem construction algorithm. If random, state the seed or note reproducibility. If stratified, state the strata and rationale.

### 3. N ≈ 300 Notation Is Imprecise

**Description:** The report consistently writes "N ≈ 300" for risky problems, with the approximation sign suggesting slight variation across temperatures. The data quality section notes only 1 NA out of 1,500, so the actual counts are presumably 300, 300, 300, 300, 299. This should be stated exactly.

**Recommendation:** Report exact N per temperature, or confirm that N = 300 at four temperatures and N = 299 at T = 1.5 (after NA removal). The "≈" creates unnecessary ambiguity.

### 4. PPC Test Statistics Could Be Described More Precisely

**Description:** Three PPC statistics are named (ll, modal, prob) but their formal definitions are not stated. "Log-likelihood" is straightforward, but "modal choice frequency" and "mean choice probability" could mean several different things depending on whether they are computed per-problem, per-alternative, or globally.

**Recommendation:** Provide one-sentence formal definitions. E.g., "Modal: the fraction of problems where the most probable alternative under the model is the one actually chosen. Prob: the average predicted probability assigned to the observed choice."

### 5. Forest Plot Legend Could Be Clearer

**Description:** In the cross-study comparison (§6, fig-cross-study), four models are plotted with different markers and offsets. The legend shows only model names (m_01, m_11, m_21, m_31) without noting that m_01 is from a *different study* with a *different model structure* (uncertain-only). A reader unfamiliar with the series might not understand the comparison.

**Recommendation:** Add parenthetical labels: "m_01 (Report 1, uncertain only)" / "m_11 (shared α, this report)" / "m_21 (separate α, ω)" / "m_31 (proportional ω = κ·α)".

### 6. Causal Language

**Description:** Several passages use causal framing (e.g., "the risky context *pulling* α downward" in §6; "temperature *controls* sensitivity" in the introduction). Given that each temperature is fit independently and the design is between-condition (not within-subject with counterbalancing), causal claims are unwarranted.

**Recommendation:** Replace causal language with associational framing: "is associated with," "co-varies with," etc.

---

## Suggestions for Improvement

### 1. Entropy Analysis of Risky vs. Uncertain Alternative Sets

The finding that ω < α is potentially interpretable in terms of the information structure of the two decision contexts. Computing and comparing the entropy of EU-difference distributions for risky vs. uncertain problems would help ground the discussion: if risky problems present smaller EU differences (because stated probabilities cluster near the centroid), then lower sensitivity is a rational response. This could transform a post hoc interpretation into a data-supported explanation.

### 2. Joint Hierarchical Model Across Temperatures

As suggested in the referee report for Report 1, a hierarchical model with temperature as a continuous predictor — e.g., log α(T) = a + b·T, log ω(T) = c + d·T — would obviate the independence issue, directly estimate slope parameters, and test whether the temperature effect on ω parallels that on α. The m_31 structure (ω = κ·α) would be particularly amenable to a hierarchical extension where κ is allowed to vary with temperature.

### 3. Connect to the Risk–Ambiguity Distinction in JDM

The uncertain/risky distinction maps directly onto the ambiguity/risk distinction that is central to JDM since Ellsberg (1961). The finding that sensitivity differs between risk and ambiguity (with ambiguity yielding *higher* sensitivity) connects to — but partially inverts — the common human finding of ambiguity aversion. The report should discuss whether the α > ω pattern is consistent with an "ambiguity preference" interpretation or whether the softmax framework precludes direct comparison. This connection would significantly strengthen the report's integration with the JDM literature.

### 4. Within-Model Pairwise Contrast for ω

The pairwise comparison analysis is performed only for α. A parallel table for ω (under m_21) would clarify whether the temperature–sensitivity gradient is equally strong in the risky context. If P(ω_{T=0} > ω_{T=1.5}) is substantially lower than the analogous probability for α, this would suggest that risky sensitivity is both lower and less temperature-responsive — a more nuanced finding than "both decline."

### 5. Report β Estimates or Diagnostics

The β matrix (mapping features to subjective probabilities) is estimated jointly with α in all three models, but β estimates are not reported or discussed. Since the interpretation of α depends on having a well-estimated β (poorly estimated β could absorb sensitivity variation), at minimum reporting that β estimates are stable across models and temperatures would strengthen the claim that variation in α/ω is genuine.

---

## Summary Table

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Research Question / Hypotheses | Excellent | Three clear, well-motivated research questions with explicit pre-specification of confirmatory (replication of temperature effect) vs. exploratory (context-dependent sensitivity) goals. |
| Experimental Design | Good | Clean parallel design with counterbalancing; minor gap in describing risky problem construction algorithm. |
| Operationalization / Measurement | Good | The uncertain/risky distinction is well-operationalized; the mapping to risk/ambiguity JDM constructs could be made more explicit. |
| Evidential Strength | Good | Temperature replication is strong (P > 0.99 for negative slope across models); the ω < α finding is directionally consistent but lacks formal aggregate quantification. |
| Robustness / Generalizability | Adequate | Multiple model specifications provide internal robustness; no prior sensitivity analysis, single LLM/task, no hierarchical pooling. Generalizability explicitly limited by design but should reference cross-LLM findings from other reports. |
| Causal Reasoning / Interpretation | Adequate | Causal language occasionally employed where only association is warranted; post hoc interpretations of ω < α not clearly labeled. |
| Relation to Literature | Adequate | Connection to Report 1 is thorough; connection to the broader JDM literature on risk vs. ambiguity is underdeveloped. |
| Computational Methodology | Excellent | Clean MCMC diagnostics across all 15 fits; PPC analysis discriminates models effectively; prior predictive calibration adapted for augmented design. |
| Exposition / Transparency | Excellent | Well-organized, clear writing; complete frozen data snapshot; transparent analysis pipeline with inline code. |

---

## Confidential Comments to the Editor

This report represents a genuine empirical contribution to the emerging literature on LLM decision-making. The finding that GPT-4o's EU sensitivity differs between risk and ambiguity contexts — and in the opposite direction from what a naive rational-agent account would predict — is novel and thought-provoking. The model comparison exercise is well-executed and the PPC-based evidence for context-dependent sensitivity is convincing.

The major revisions are all tractable: computing formal aggregate tests (P(α > ω), P(κ < 1)), adding model comparison via information criteria, reframing the cross-study comparison, and labeling interpretations as post hoc. None requires new data collection or fundamental redesign.

The absence of prior sensitivity analysis, flagged in both prior referee reports in this series, is becoming a recurring concern. The authors should be encouraged to address it in this report, as doing so would also resolve the outstanding issue from Report 1 and strengthen the overall manuscript sequence.

For JDM venue suitability: the risk/ambiguity angle is the most promising hook. If the authors develop the connection to Ellsberg-style ambiguity phenomena and frame the ω < α finding in terms of how LLMs process risk vs. ambiguity, this report would speak directly to central concerns of the JDM community.
