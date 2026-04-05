# Referee Report: Temperature and SEU Sensitivity: EU-Prompt Variation

**Report Under Review:** `temperature_study_with_eu_prompt/01_eu_prompt_study.qmd`

**Reviewer:** JDM Referee (methodologically sophisticated)

**Date:** April 2, 2026

---

## Summary

This report investigates whether explicitly instructing GPT-4o to maximize expected utility increases its estimated sensitivity (α) to SEU maximization, compared to a base study with an implicit decision prompt. The design is elegant: by reusing assessments and embeddings from the base temperature study, the authors isolate the prompt intervention to the decision rule while holding belief formation constant. The central finding—that the EU-prompt *decreases* rather than increases α at most temperatures—is counterintuitive and thoughtfully discussed. However, the statistical evidence for this pattern is weaker than the narrative suggests: per-temperature 90% credible intervals all contain zero, and no formal test of the pattern-level claim is provided. The report meets many JDM standards but would benefit from more conservative framing of the central finding and clearer pre-specification of hypotheses.

**Recommendation: Minor Revision**

The work is methodologically sound and contributes a genuinely surprising finding. The issues identified are addressable without major restructuring, primarily requiring more conservative interpretation of the evidence and clearer hypothesis specification.

---

## Strengths

1. **Clean experimental design with minimal intervention.** The study isolates a single manipulation (adding one paragraph to the choice prompt) while holding all other components constant—same problems, same assessments, same embeddings, same model. This is a textbook example of experimental control, allowing clean attribution of any differences to the prompt modification.

2. **Elegant decomposition of belief formation and decision rule.** By reusing assessments from the base study, the design separates the two stages of SEU-based decision making. This is conceptually valuable: we learn that the EU-prompt affects the *application* of beliefs to choices, not the beliefs themselves.

3. **Transparent reporting of counterintuitive results.** The authors anticipated that explicit EU instructions would increase sensitivity and honestly report that the opposite occurred. This intellectual honesty is commendable and stands in contrast to publication-bias-driven reporting norms.

4. **Thoughtful discussion of multiple interpretations.** The discussion offers three substantive explanations for why explicit EU instructions might reduce sensitivity (prompt interference, imperfect explicit calculation, frame narrowing), all grounded in cognitive science concepts. This moves beyond "we found X" to "here's how to think about X."

5. **Strong computational diagnostics and model fit.** MCMC diagnostics are clean across all conditions (_R̂_, ESS, E-BFMI all satisfactory; zero divergences). Posterior predictive checks indicate adequate model fit at every temperature level. Readers can trust that the statistical conclusions are not artifacts of computational problems.

---

## Major Issues

### Issue 1: Overclaiming the Pattern-Level Finding

**Description:** The report's central claim is that "the EU-prompt *lowers* estimated α at three of five temperatures" and that this constitutes "a pattern-level finding." However, the evidence for this pattern is weaker than the narrative implies:

- All five per-temperature 90% credible intervals on αbase − αEU contain zero
- No individual P(base > EU) exceeds 0.85 (the maximum is 0.81 at T=0.0)
- No formal statistical test of the pattern claim (e.g., probability that ≥4 of 5 differences are positive) is reported

The report acknowledges that "the individual per-temperature differences are not decisive," but this caveat is buried beneath stronger language in the abstract and summary sections (e.g., "The EU-prompt does not increase α").

**Recommendation:** Either (a) provide a formal test of the pattern-level hypothesis (e.g., compute the joint posterior probability that ≥4 of 5 differences have the same sign, using the matched draws), or (b) reframe the finding as "suggestive but not statistically decisive." The abstract and summary should reflect the uncertainty, not just the point estimates.

**Severity:** Major—affects the headline conclusion.

### Issue 2: Absence of Pre-Specified Hypotheses

**Description:** The report does not articulate a priori hypotheses before presenting results. The opening question ("does explicitly telling the LLM to maximize expected utility increase α?") is framed as if it were a confirmatory test, but no prediction is stated. The "naive expectation" that EU-prompt should increase α is mentioned only in the Discussion, making it unclear whether this was a pre-specified hypothesis or a post-hoc framing device.

JDM standards (per Simmons et al., 2011) require distinguishing confirmatory from exploratory analyses. If this study was designed to test a specific prediction (EU-prompt → higher α), that prediction should be stated in the Introduction. If the study was exploratory, it should be labeled as such.

**Recommendation:** Add a clear hypothesis statement in the Introduction, explicitly noting whether the prediction was pre-registered or specified before data analysis. If the study was exploratory, acknowledge this and frame the findings accordingly.

**Severity:** Major—affects interpretation of evidential strength.

### Issue 3: Limited Discussion of Alternative Explanations for the Core Finding

**Description:** While the Discussion offers three thoughtful interpretations of why EU-prompt might reduce sensitivity (prompt interference, imperfect calculation, frame narrowing), it does not seriously consider the possibility that the observed pattern is **noise**. Given that no per-temperature difference achieves conventional credibility thresholds, and the T=1.0 condition shows a reversal, the most parsimonious explanation may be that there is no robust EU-prompt effect at all—the differences are sampling variability.

**Recommendation:** Add a fourth interpretation: "No robust effect." Discuss what evidence would be needed to distinguish this from the "interference" explanations—e.g., replication, larger sample sizes, or the joint hierarchical model proposed in Next Steps.

**Severity:** Major—affects the balance of the interpretation.

---

## Minor Issues

### Issue 4: The T=1.0 Exception Is Underexplored

**Description:** At T=1.0, the EU-prompt yields *higher* α than the base study (median 43.3 vs 39.1, P(base > EU) = 0.37). This reversal is noted but the explanation is highly speculative ("one speculative explanation: at moderate-to-high temperature, the base model's implicit heuristics may be sufficiently disrupted..."). If the pattern-level claim is to be sustained, the exception should either be explained more rigorously or acknowledged as weakening the pattern.

**Recommendation:** Either provide additional analysis of the T=1.0 condition (e.g., examining choice distributions, PPC fit quality) or acknowledge that this exception weakens confidence in the pattern-level claim.

### Issue 5: "Choice Probability" Language Could Be Misleading

**Description:** The report refers to "P(base > EU)" throughout without always clarifying that this is a posterior probability computed from matched MCMC draws, not a frequentist p-value or a probability statement about a future replication. For JDM readers accustomed to frequentist reporting, this could cause confusion.

**Recommendation:** Add a brief methodological note (perhaps in a callout box) explaining how the posterior probabilities are computed and how to interpret them. Something like: "P(base > EU) denotes the proportion of posterior draws where αbase exceeds αEU—a Bayesian analogue to asking 'how probable is it that the true base α exceeds the true EU α?'"

### Issue 6: Missing Connection to Human JDM Literature on Instruction Effects

**Description:** The finding that explicit EU instructions reduce performance has intriguing parallels in human cognition research—e.g., "verbal overshadowing" (Schooler & Engstler-Schooler, 1990) and the expertise reversal effect (Kalyuga et al., 2003). The Discussion mentions the "well-documented finding in cognitive science that conscious rule-following can degrade the performance of automatic, well-practiced skills" but provides no citations.

**Recommendation:** Add citations to the relevant cognitive science literature on instruction effects and automaticity. This would strengthen the theoretical grounding and situate the findings within the broader JDM/cognitive science literature.

### Issue 7: Figure Captions Could Be More Informative

**Description:** Some figure captions (e.g., Figure 1 "Forest plot...") describe what is shown but not what the reader should conclude. The captions could be more informative by noting the key visual takeaway.

**Recommendation:** Enhance figure captions to include interpretive guidance—e.g., "The forest plot shows declining α with temperature; note the substantial overlap between T=0.3 and T=0.7."

---

## Suggestions for Improvement

1. **Joint hierarchical model.** The report correctly identifies this as a limitation and a next step. If feasible within the revision timeline, fitting a model that includes prompt condition as a covariate would substantially strengthen the inference. At minimum, this should be flagged as essential for any future claims about the EU-prompt effect.

2. **Power/precision analysis.** How large would the base-vs-EU difference need to be for the current design to detect it with high probability? A retrospective power analysis would help readers understand whether the non-decisive results reflect a small true effect or insufficient precision.

3. **Replication with alternative EU-prompt formulations.** Could the negative effect be specific to the particular wording used? A conceptual replication with chain-of-thought prompting or different EU framings would address this.

4. **Connection to "prompting for rationality" literature.** There is a growing literature on whether prompting LLMs to be "rational" or "careful" improves their decision quality (e.g., recent work on chain-of-thought prompting and reasoning). Situating this finding within that literature would increase the report's appeal to AI/ML audiences while maintaining JDM relevance.

5. **Effect size interpretation.** The report focuses on whether effects are "decisive" but could also discuss whether the observed differences (e.g., Δα ≈ −17 at T=0.0) are *practically meaningful*. What does a 17-point decrease in α mean for choice behavior?

---

## Summary Table

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Research Question / Hypotheses | Adequate | Clear question but no pre-specified hypothesis |
| Experimental Design | Excellent | Clean isolation of prompt manipulation; strong controls |
| Operationalization / Measurement | Excellent | α is well-defined; feature construction is transparent |
| Evidential Strength | Adequate | Per-temperature CIs all contain zero; pattern claim needs formal test |
| Robustness / Generalizability | Adequate | Limitations acknowledged; single task/model constraint |
| Causal Reasoning / Interpretation | Good | Multiple interpretations offered; could add "null effect" possibility |
| Relation to Literature | Adequate | JDM connections present but could be strengthened |
| Computational Methodology | Excellent | Clean diagnostics; good PPCs; reproducible pipeline |
| Exposition / Transparency | Good | Clear writing; could improve figure captions and terminology |

---

## Confidential Comments to the Editor

This is a well-executed study with a genuinely interesting finding. The counterintuitive result—that telling an LLM to maximize expected utility appears to *reduce* its EU-aligned behavior—is exactly the kind of discovery that advances understanding, regardless of whether it ultimately replicates.

My primary concern is that the current framing overclaims the strength of the evidence. The per-temperature comparisons are individually non-decisive, and no formal test of the pattern-level hypothesis is provided. If the authors can either (a) provide such a test or (b) reframe the findings more conservatively, I believe the report would meet JDM publication standards.

The study's value lies partly in its conceptual contribution (distinguishing belief formation from decision rule application) and partly in raising important questions about the relationship between LLM "competence" and "articulable reasoning." These contributions stand regardless of whether the EU-prompt effect proves robust—but the report should not claim more certainty than the data support.

I recommend acceptance conditional on the major revisions outlined above. If the authors prefer to frame the study as exploratory hypothesis-generation rather than hypothesis-testing, the evidentiary bar is lower, but this should be explicit.
