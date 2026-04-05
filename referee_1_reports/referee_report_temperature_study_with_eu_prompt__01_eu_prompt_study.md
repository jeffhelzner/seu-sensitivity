# Referee Report: Temperature and SEU Sensitivity: EU-Prompt Variation

**Report reviewed:** `temperature_study_with_eu_prompt/01_eu_prompt_study.qmd`
**Review date:** 2026-04-02
**Reviewer role:** Referee for *Judgment and Decision Making*

---

## Summary

This report investigates whether explicitly instructing GPT-4o to maximize expected utility increases estimated sensitivity (α) to SEU maximization, using the same problems, assessments, embeddings, and model (m_01) as the base temperature study but with a modified choice prompt. The key finding is a null-to-negative effect: the EU-prompt does not raise α, and point estimates are *lower* at four of five temperature levels, though no individual per-temperature difference is statistically decisive. The temperature–sensitivity gradient is preserved under the EU-prompt. The report is clearly written, the experimental design is clean (only the choice prompt varies), and the interpretation — centered on the gap between articulable commitment and behavioral competence — is thoughtful and well-connected to both the SEU sensitivity framework and the cognitive science literature. However, the report suffers from several issues: the evidential basis for its strongest interpretive claims is thin (the per-temperature differences are individually non-decisive and no aggregate test is reported), the cross-study comparison relies on independently estimated posteriors without acknowledging the inferential limitations this creates, and the speculative discussion of *why* the EU-prompt reduces α, while interesting, is given more space than the evidence warrants. With targeted revisions, this would make a solid contribution to the literature on LLM decision-making and prompt sensitivity.

**Recommendation: Minor Revision**

---

## Strengths

1. **Clean experimental isolation.** The design's greatest strength is what it *doesn't* change. By reusing problems, assessments, and embeddings from the base study and modifying only the choice prompt, the report achieves a rare degree of experimental control. The change is confined to the decision rule stage of the SEU decomposition, with belief formation held constant. This design discipline is exemplary and makes the comparison interpretable in a way that a full-pipeline re-run would not.

2. **Honest reporting of a null/negative result.** The authors find the opposite of the naive prediction — the EU-prompt decreases rather than increases α — and report this straightforwardly without spinning it as a positive finding. The per-temperature comparison table, the overlaid posterior densities, and the explicit statement that "90% credible intervals on α_base − α_EU contain zero at every temperature" are models of honest inferential reporting. A JDM referee would appreciate this transparency.

3. **Substantive and well-grounded discussion.** The three interpretations offered for why the EU-prompt might reduce sensitivity — prompt interference with implicit competence, imperfect explicit calculation, and frame narrowing — are all cognitively plausible, clearly distinguished, and appropriately hedged. The connection to the "competence vs. articulable reasoning" theme and to the commitment–performance distinction from the foundational framework is genuinely insightful and adds theoretical value beyond the empirical result.

4. **Efficient use of the factorial structure.** The report demonstrates good scientific economy: by holding everything constant except the choice prompt, it required only 1,500 new API calls rather than the full ~3,150 of the base study. This is both cost-effective and methodologically sound — the shared assessment/embedding pipeline eliminates a major source of between-study variation.

5. **Thorough cross-study comparison infrastructure.** The slope comparison, monotonicity analysis, and pre-computed cross-study analysis JSON represent a systematic approach to between-condition inference. The visualization choices — overlaid densities, difference violins, summary heatmap — are well-suited to the comparison task and effectively communicate the pattern-level nature of the finding.

---

## Major Issues

### 1. No aggregate test of the prompt effect

**Issue title:** Pattern-level claim without pattern-level evidence

**Description:** The report's central finding is that the EU-prompt lowers α at four of five temperatures — described as "a pattern-level finding, not a temperature-specific one." However, no aggregate statistical quantity directly tests this pattern. The evidence consists of five independent per-temperature comparisons, each individually non-decisive (the highest P(base > EU) is 0.85 at T = 1.5). The sign consistency across conditions is noted qualitatively but not tested quantitatively. A JDM referee would expect at least one formal test of the aggregate hypothesis — e.g., a pooled or averaged Δα across temperatures, or a posterior probability that the *average* difference is positive, or a Bayesian sign test on the five pairwise comparisons.

**Recommendation:** Compute and report P(mean Δα > 0) by averaging the draw-wise α_base − α_EU across the five temperatures. This is straightforward with the existing 4,000 posterior draws and would provide a single quantity that directly addresses the pattern-level claim. Also consider reporting the posterior distribution of the number of temperatures at which base > EU (out of 5), compared to the null expectation of 2.5 under no effect. Either test would give the reader a principled basis for evaluating the pattern-level claim.

**Severity:** Major — the central claim lacks direct statistical support. The individual comparisons are honestly reported as non-decisive, but the pattern-level inference is left to the reader's subjective impression rather than being formalized.

### 2. Independence assumption in cross-study comparisons

**Issue title:** Independent posteriors treated as joint draws

**Description:** The P(α_base > α_EU) quantities are computed by comparing draw *i* from the base study's posterior with draw *i* from the EU-prompt study's posterior. This is valid only if the two sets of draws are drawn from a proper joint posterior, which they are not — they are from independent model fits. The report acknowledges this in the Limitations section (point 1), noting that "a joint model that shares structure across prompt conditions would provide sharper inference." However, the acknowledgment is too buried and too brief for the weight the cross-study comparison bears in the analysis.

**Recommendation:** Move the caveat about independent posteriors to the Cross-Study Comparison section itself (§4), immediately before or after the per-temperature comparison table. Explain that the draw-by-draw comparison is valid under the assumption that the MCMC chains have mixed well (which diagnostics confirm) but does not account for the correlation structure induced by shared stimuli. Note that this may make the per-temperature P(base > EU) values somewhat unreliable as measures of directional confidence. A parenthetical would suffice: "These probabilities are computed from independent posteriors matched by draw index; a joint model over prompt conditions would properly account for shared design structure."

**Severity:** Major — the cross-study comparison is the core analysis, and the inferential basis needs to be transparent where the comparison is presented, not only in a late limitations section.

### 3. No prior sensitivity analysis or model robustness checks

**Issue title:** Absence of robustness checks

**Description:** The report uses the same m_01 model and Lognormal(3.0, 0.75) prior as the base study, which is appropriate for comparability. However, no sensitivity analysis is conducted to test whether the qualitative finding (EU-prompt lowers α) is robust to alternative prior specifications or model variants. Given that the per-temperature differences are not individually decisive, it is especially important to demonstrate that the directional pattern is not an artifact of the specific prior. The base study's referee report flagged this same issue; it is equally relevant here.

**Recommendation:** Refit at least the extreme conditions (T = 0.0 and T = 1.5, where the differences are largest) under one alternative prior — e.g., Lognormal(2.5, 1.0) — and verify that the sign of α_base − α_EU is preserved. Alternatively, report the prior-to-posterior contraction (how much narrower is the posterior than the prior?) to demonstrate that the data dominate the prior at every temperature. Either approach would address the concern without major additional computation.

**Severity:** Major — this is a standard expectation for Bayesian analyses in JDM venues, and the absence is more concerning here than in the base study because the effect of interest is smaller and less decisive.

---

## Minor Issues

### 1. Speculative discussion is disproportionate to the evidence

**Description:** The Discussion section devotes approximately 40% of its length to three speculative interpretations of why the EU-prompt reduces α, plus a subsection on the T = 1.0 exception. These are interesting and well-written, but the empirical finding they explain — a directional pattern with no individually decisive comparisons and no aggregate test — is not strong enough to support this degree of interpretive elaboration. The discussion risks conveying more certainty about the effect's existence than the evidence warrants.

**Recommendation:** Retain the three interpretations but shorten each by ~30%, and add an explicit caveat at the start of the subsection: "The following interpretations are speculative, conditional on the directional pattern being genuine — a proposition that the current data support at a suggestive but not decisive level." The T = 1.0 exception subsection should be shortened to 2–3 sentences or moved to a footnote, as it reads as over-interpreting the one reversal in a noisy five-point comparison.

### 2. Missing exact prompt text in the report body

**Description:** The report describes the EU-prompt modification in the Introduction and notes the added paragraph, but the actual prompt text is quoted only in summary. A JDM referee would want to see the *complete* choice prompt (both base and EU-prompt versions) to evaluate whether the instruction is well-operationalized and whether it might have unintended effects (e.g., priming the model toward particular response patterns, adding cognitive load that affects token generation).

**Recommendation:** Include the complete EU choice prompt in a callout box or appendix, and ideally also the base study's choice prompt for direct comparison. The prompt text exists in `data/prompts.yaml` and should be brought into the report body.

### 3. PPC results deserve more discussion

**Description:** The posterior predictive checks are reported briefly ("All PPC p-values fall within [0.42, 0.67]") with a one-sentence conclusion. Given that the EU-prompt modifies the choice process, it is worth noting explicitly that the m_01 model — which has no prompt-condition parameter — fits the EU-prompt data as well as the base data. This has an important implication: the model's softmax structure is flexible enough to accommodate whatever the EU-prompt does to choices, which means the prompt effect is absorbed entirely by changes in α (and β, δ) rather than requiring a structural change to the model.

**Recommendation:** Add 2–3 sentences interpreting the PPC results in terms of model adequacy under the new prompt condition. This strengthens the argument that the α comparison is meaningful.

### 4. The summary heatmap is visually misleading

**Description:** The heatmap in the Discussion uses a red-green color scale with limits [0.2, 0.8], which makes the P(base > EU) = 0.75 at T = 0.0 appear visually dramatic. On a [0, 1] scale, 0.75 would look much more moderate. Additionally, the use of red/green may be inaccessible to colorblind readers.

**Recommendation:** Consider using a diverging blue-red or viridis colormap, and setting the scale to [0, 1] to avoid visual exaggeration. Alternatively, note the truncated scale explicitly in the figure caption.

### 5. Cross-reference to the risky alternatives study

**Description:** The Discussion mentions that the T = 0.3 ≈ T = 0.7 plateau "persists across prompt conditions" (§Monotonicity), implying this is a robust structural feature. However, the report does not reference the risky alternatives study (Temperature Study 3), which could provide additional evidence on whether this plateau is task-specific or general.

**Recommendation:** Add a brief cross-reference to the risky alternatives study when discussing the persistence of the mid-range plateau, noting whether it also appears in that variant.

### 6. Figure 5 (comparison summary) right panel: axis label

**Description:** The right panel's y-axis label is "α_base − α_EU" in plain text, whereas the rest of the report uses proper mathematical notation ($\alpha_{\text{base}} - \alpha_{\text{EU}}$). This is a minor formatting inconsistency.

**Recommendation:** Use LaTeX-rendered axis labels in the matplotlib code to maintain consistency.

---

## Suggestions for Improvement

1. **Joint hierarchical model over prompt conditions.** This is correctly identified as a "Next Step" and would substantially strengthen the contribution. A model with prompt condition as a covariate (e.g., log α = a + b·T + c·EU + d·T×EU) would directly estimate the EU-prompt main effect and its interaction with temperature, properly accounting for the shared design. This would resolve Major Issues 1 and 2 simultaneously.

2. **Effect size in choice probability space.** The α differences (e.g., Δα ≈ −17 at T = 0.0) are difficult for a JDM reader to interpret without context. Converting the α difference into predicted changes in choice probability for a representative problem (e.g., "a Δα of −17 corresponds to reducing the EU-maximizing choice probability from X% to Y% for a problem with utility differences of Z") would make the practical significance tangible. The softmax function and the foundational report's monotonicity theorem provide the tools for this conversion.

3. **Connect to the "telling more than we know" literature.** The finding that explicit EU instructions degrade performance resonates with Nisbett & Wilson's (1977) classic work on the limits of introspective access, and with the broader literature on verbal overshadowing (Schooler & Engstler-Schooler, 1990). While these are human-cognition findings, the analogy is apt: asking the model to articulate a decision process may interfere with a well-functioning implicit process. Drawing this connection would strengthen the report's engagement with the JDM literature.

4. **Bayesian meta-analytic combination.** If the sign pattern (base > EU at 4/5 temperatures) recurs in other application studies, a Bayesian meta-analytic model across the entire factorial design would provide authoritative evidence for the prompt effect. The factorial synthesis report (Report 7) may address this, but noting the possibility here would be valuable.

5. **Chain-of-thought follow-up.** The Discussion mentions testing "chain-of-thought variants" as a next step. This is especially well-motivated given the interpretive framework: if the EU-prompt's negative effect stems from forcing imprecise explicit calculation, then chain-of-thought scaffolding (which provides workspace for more careful computation) might mitigate the interference. This prediction is testable and would help discriminate among the three proposed interpretations.

6. **Report the base study's prompt for contrast.** Including the base study's choice prompt alongside the EU-prompt would allow the reader to evaluate the *magnitude* of the prompt manipulation, not just its presence. A side-by-side comparison would make the "minimal intervention" claim verifiable.

---

## Summary Table

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Research Question / Hypotheses | Good | Clear question, well-motivated; the prediction of *increased* α is natural but the null/negative finding is reported honestly |
| Experimental Design | Excellent | Exemplary control — reuses everything except the choice prompt; clean isolation of the decision-rule stage |
| Operationalization / Measurement | Good | α is well-defined; prompt modification is well-targeted; exact prompt text should be in the report body |
| Evidential Strength | Adequate | Per-temperature differences are individually non-decisive; no aggregate test of the pattern-level claim; evidence is suggestive but falls short of conclusive |
| Robustness / Generalizability | Adequate | No prior sensitivity analysis; single LLM, single task, single prompt formulation; limitations acknowledged but robustness not demonstrated |
| Causal Reasoning / Interpretation | Good | Appropriately hedged; speculative discussion is thoughtful but disproportionate to evidence base |
| Relation to Literature | Adequate | Good connection to own framework; the "competence vs. articulable reasoning" theme could be connected to Nisbett & Wilson, verbal overshadowing literature |
| Computational Methodology | Good | Clean MCMC diagnostics, adequate PPCs; inherits validated methodology from base study; no independent validation needed |
| Exposition / Transparency | Good | Clear writing, well-organized visualizations; frozen data snapshot; prompt text and independence caveat should be more prominent |

---

## Confidential Comments to the Editor

This report presents a genuinely interesting finding — that telling an LLM to maximize expected utility actually *reduces* its measured EU-sensitivity — but the evidence is weaker than the interpretive discussion suggests. The per-temperature comparisons are individually non-decisive, and the absence of an aggregate test means the central claim rests on the reader's subjective evaluation of a 4-out-of-5 directional pattern. This is suggestive but not compelling by JDM standards, where a well-calibrated reader might note that under the null hypothesis (no prompt effect), observing 4/5 same-sign differences has probability ~0.19 by a binomial sign test — not conventionally significant.

The clean experimental design is the report's strongest asset. By holding everything constant except the choice prompt, the authors have created a comparison that is as interpretable as a between-subjects experiment can be in this setting. The discussion of why the EU-prompt might reduce sensitivity is among the most thoughtful in the series and raises interesting questions about the relationship between explicit instruction and implicit competence in LLMs.

The recommended revisions are straightforward: (1) compute an aggregate test of the prompt effect, (2) move the independence caveat to where the comparison is presented, (3) add a prior sensitivity check, and (4) include the full prompt texts. None of these requires new data collection or fundamental redesign. If the aggregate test confirms the directional pattern (which I suspect it will, given that the draws are consistent at 4/5 temperatures with modest effect sizes), the report would be a solid contribution. If the aggregate test is ambiguous, the report remains valuable as an honest null finding with a well-designed study.

The report fits naturally into the series as the second cell in what becomes a 2×2 factorial (LLM × task), and its findings are consistent with the emerging theme that LLM decision-making is more sensitive to architectural parameters (temperature) than to prompt-level interventions. This is a useful finding for the JDM community's understanding of LLM rationality, even if the evidence for the specific EU-prompt effect is not individually definitive.
