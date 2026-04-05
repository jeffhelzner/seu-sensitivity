# Referee Report: Temperature and SEU Sensitivity: Risky Alternatives Extension

**Report Under Review:** `01_risky_alternatives_study.qmd`  
**Path:** `reports/applications/temperature_study_with_risky_alts/01_risky_alternatives_study.qmd`  
**Venue:** *Judgment and Decision Making* or comparable JDM journal

---

## Summary

This report extends the initial temperature study by introducing risky alternatives—where outcome probabilities are stated explicitly—alongside the original uncertain alternatives. The study addresses three empirical questions: whether the temperature–sensitivity relationship replicates with the augmented data, whether sensitivity differs between risky and uncertain contexts, and whether any difference is proportionally linked across temperatures. The authors fit three model variants (m_11, m_21, m_31) to data from five temperature conditions (T = 0.0 to 1.5), each with approximately 300 uncertain and 300 risky decisions. The findings are substantive: the temperature–α relationship replicates robustly across all model specifications; the LLM exhibits systematically lower sensitivity in the risky context (ω < α); and the m_21 model (separate sensitivities) provides the best posterior predictive calibration. The work is methodologically rigorous, with excellent MCMC diagnostics and transparent reporting. However, the theoretical interpretation of the risky/uncertain sensitivity gap is post-hoc, and the design cannot definitively adjudicate between the candidate explanations offered.

**Recommendation: Minor Revision**

The empirical contribution is solid and the findings are clearly presented. The revisions required are primarily interpretive and connective—strengthening theoretical grounding and engaging more deeply with the JDM literature on risk versus ambiguity.

---

## Strengths

1. **Clean Replication of Core Finding.** The negative relationship between temperature and α—the central finding of Report 1—replicates exactly across m_11, m_21, and m_31. The pairwise comparison structure (strong separation between T = 0.0 and T ≥ 1.0, near-indistinguishability of T = 0.3 and T = 0.7) is preserved. This robustness across model specifications and doubled data provides strong evidential support for the temperature–sensitivity relationship.

2. **Informative Model Comparison via PPCs.** The report employs posterior predictive checks with multiple test statistics (log-likelihood, modal choice frequency, mean choice probability) separately for uncertain and risky contexts. The finding that m_11's shared α produces systematic miscalibration in the risky context—resolved by m_21's separate ω—provides genuine model-selection information without relying solely on information criteria. This approach is particularly appropriate for JDM audiences who may be skeptical of purely statistical model comparison.

3. **Novel Empirical Finding: Context-Dependent Sensitivity.** The consistent finding that ω < α (risky sensitivity lower than uncertain sensitivity) across all temperatures is a genuinely new empirical result. The m_31 proportionality parameter κ clustering below 1.0 (medians 0.71–0.94) provides a compact summary of this effect. This adds substantive value beyond the temperature study by revealing that the LLM's decision-making differs between stated and inferred probabilities.

4. **Rigorous Computational Execution.** All 15 fits (3 models × 5 temperatures) achieve clean MCMC diagnostics: no divergences, satisfactory R̂ and ESS values, and appropriate posterior predictive calibration. The joint estimation from uncertain and risky data substantially tightens the α posteriors relative to the m_01 analysis—the m_11 standard deviations are approximately half those of m_01. This precision gain is appropriately attributed to the doubled data.

5. **Transparent Cross-Study Comparison.** The comparison figure showing α posteriors across m_01, m_11, m_21, and m_31 is exemplary. It demonstrates both qualitative replication (the ordering of temperatures is preserved) and quantitative differences (m_21's uncertain-α estimates closely match m_01, confirming the original estimates were not biased). This cross-study triangulation strengthens confidence in the findings across the series.

---

## Major Issues

### Issue 1: Post-Hoc Interpretation of the Risky/Uncertain Sensitivity Gap

**Description:** The finding that ω < α is robustly documented but its interpretation is largely speculative. The Discussion offers three candidate explanations: (a) format effects in numerical vs. linguistic probability processing, (b) calibration asymmetry from the learned β mapping, and (c) utility estimation precision effects from fine EU differences. None of these can be distinguished by the current design—all are post-hoc rationalizations of an observed pattern. The report does not commit to a preferred interpretation or articulate what evidence would discriminate among them.

This matters for a JDM audience because the rich literature on risk versus ambiguity (Ellsberg, 1961; Camerer & Weber, 1992; Trautmann & van de Kuilen, 2015) has developed precise theoretical machinery for understanding how known versus unknown probabilities affect decision-making. The current report does not engage with this literature, leaving the ω < α finding theoretically orphaned.

**Recommendation:** 
(a) Add an explicit discussion connecting to the risk/ambiguity distinction in JDM. The uncertain context is closer to "ambiguity" (probabilities must be inferred) while the risky context is canonical "risk" (stated probabilities). Classical results suggest ambiguity aversion, which would predict *more* conservative (potentially more EU-aligned?) choices under uncertainty. The finding that the LLM is *more* sensitive under uncertainty than under risk is interesting precisely because it may contradict intuitions from the human literature.
(b) Articulate testable predictions that would discriminate among the candidate explanations. For example, the "format effect" hypothesis might be tested by presenting risky alternatives in natural language; the "calibration asymmetry" hypothesis might be tested by using fixed (non-learned) probability mappings.
(c) Acknowledge that the current finding is descriptive—a robust empirical pattern—and that mechanistic explanation requires follow-up study.

**Severity:** Major—the theoretical interpretation of the headline finding needs grounding.

---

### Issue 2: No Formal Model Comparison

**Description:** The report presents three models (m_11, m_21, m_31) and concludes from PPC calibration that m_21 is "best-calibrated." While PPC-based model assessment is valuable, it is not a formal model comparison. The report does not report WAIC, LOO-CV, Bayes factors, or any quantitative measure of relative fit that accounts for model complexity. The m_21 model has an additional free parameter (ω); its better PPC calibration may simply reflect overfitting rather than capturing genuine structure.

This is consequential for interpretation: if m_21 is preferred only because it has more flexibility, the "different sensitivities" interpretation is less compelling than if m_21 is preferred despite its complexity.

**Recommendation:** Add formal model comparison using leave-one-out cross-validation (LOO-CV) or the widely applicable information criterion (WAIC). Report the elpd differences with standard errors across models. If m_21 substantially outperforms m_11 by these criteria—not just by PPC p-values closer to 0.5—the interpretation is strengthened. If the difference is within noise, the report should acknowledge that the evidence for separate sensitivities is suggestive but not definitive.

**Severity:** Major—model comparison claims require formal support.

---

### Issue 3: Causal Attribution of the α/ω Difference

**Description:** The report interprets ω < α as reflecting a genuine difference in how the LLM processes risky versus uncertain alternatives. However, the uncertain and risky decision problems differ in ways beyond just the probability format:

- **Stimulus complexity:** Uncertain alternatives are derived from natural-language claim descriptions processed through embedding and PCA; risky alternatives are presented as explicit probability simplexes.
- **Dimensionality:** Uncertain alternatives have D = 32 features mapped through a K × D matrix β; risky alternatives have K probabilities.
- **Estimation burden:** The uncertain model estimates β, ψ, and α jointly; the risky model takes probabilities as given.

These differences confound format (stated vs. inferred probabilities) with representation, dimensionality, and estimation pathway. The claim that "the LLM processes explicit probabilities differently from inferred ones" is plausible but cannot be definitively attributed without controlling for these factors.

**Recommendation:** Acknowledge these confounds explicitly and discuss their implications. Consider whether a matched design—where uncertain alternatives use the same 30 stimulus profiles as risky alternatives, just with probabilities inferred versus stated—would provide cleaner attribution. This could be a suggestion for future work. Hedge the current interpretation appropriately: the finding is that *as operationalized in this design*, risky choices show lower sensitivity, but this may reflect task structure rather than probability-format effects per se.

**Severity:** Major—causal claims need explicit acknowledgment of alternative explanations.

---

## Minor Issues

### Issue 4: Risky Alternatives Pool Design Underexplained

**Description:** The 30 risky alternatives are described briefly—spanning "corner alternatives," "balanced alternatives," and "intermediate cases"—but the actual design of this pool is not fully detailed. How were the 30 probability simplexes selected? Were they chosen to maximize discriminability, to span the simplex uniformly, to match the expected utility distribution of uncertain alternatives, or by some other criterion? The design of the alternative pool affects identification and the interpretation of ω.

**Recommendation:** Add a table or figure showing the distribution of the risky probability simplexes (e.g., a ternary plot or histogram of entropy values). Specify the design rationale. If the alternatives were chosen to span a range of expected utilities comparable to the uncertain context, note this.

---

### Issue 5: Wide Credible Intervals on κ

**Description:** The m_31 proportionality parameter κ has 90% credible intervals that include 1.0 at some temperatures (e.g., T = 1.0, T = 1.5 based on the visual inspection of the forest plot). While the medians cluster below 1.0, the intervals are wide enough that κ = 1 (equivalence to m_11) is not definitively excluded. The report notes this in passing ("has wide credible intervals; a study with larger N...would improve precision") but does not quantify the probability that κ < 1 across conditions.

**Recommendation:** Report P(κ < 1) explicitly for each temperature condition. This would clarify whether the "κ below 1" finding is merely a point-estimate tendency or is credibly different from 1.0. If P(κ < 1) is, say, 0.85 across all conditions, that is quite different from 0.99.

---

### Issue 6: Limited Engagement with Human Decision-Making Literature

**Description:** The report frames the risky/uncertain distinction in terms of LLM processing (format effects, calibration asymmetry) but does not engage with the extensive JDM literature on this topic. Classic work on ambiguity aversion (Ellsberg, 1961), probability weighting (Kahneman & Tversky, 1979; Prelec, 1998), and source preference (Tversky & Wakker, 1995) provides rich theoretical resources. The "uncertainty" context in this study is analogous to ambiguity (unknown probabilities), and the "risk" context is canonical risk (known probabilities). How does the LLM's behavior compare to typical human patterns?

**Recommendation:** Add 2–3 paragraphs connecting to the JDM literature on risk versus ambiguity. The Ellsberg study in this series (Report 4) presumably engages more directly with ambiguity; cross-reference and note similarities/differences. Discuss whether the ω < α finding is consistent or inconsistent with human ambiguity aversion (humans typically avoid ambiguous options; does lower ω mean less avoidance or something else?).

---

### Issue 7: No Sample Size or Power Justification

**Description:** The study uses M = 300 uncertain and N ≈ 300 risky decisions per temperature condition. No justification is provided for these numbers. Are they sufficient to estimate κ with useful precision? The wide intervals on κ suggest the study may be underpowered for fine discrimination between m_11 and m_31.

**Recommendation:** Add a brief note on sample size rationale. If simulation-based power analysis informed the design, reference it. If not, acknowledge this as a limitation and quantify what precision gains would require (e.g., "doubling N would reduce κ interval width by approximately X%").

---

### Issue 8: Minor Presentation Issues

**Description:** 
- The PPC comparison figure (@fig-ppc-comparison) scatter points are somewhat difficult to distinguish with similar colors and overlapping positions.
- The report uses "uncertain" for what the JDM literature usually calls "ambiguous" situations—this may cause confusion for readers familiar with the risk/ambiguity distinction.

**Recommendation:** Consider alternative visual encodings for the PPC plot (e.g., connected lines by model, faceting by context). Add a note clarifying the terminology choice—"uncertain" is used here to refer to situations where probabilities are inferred, which the decision-theory literature often terms "ambiguous."

---

## Suggestions for Improvement

1. **Add Summary Statistics on EU Alignment by Context.** The report focuses on the sensitivity parameter but does not report raw choice behavior. What fraction of choices are EU-maximizing in each context at each temperature? This descriptive statistic would complement the model-based α/ω estimates and be more accessible to readers unfamiliar with softmax models.

2. **Consider a Hierarchical Model Across Temperatures.** Both this report and Report 1 fit temperatures independently. A model expressing α(T) = exp(γ₀ + γ₁T) would directly estimate the functional relationship and potentially narrow uncertainty. At least discuss why this was not pursued.

3. **Report Utility Parameter (δ) Summaries.** The utility increments δ are shared across contexts in all three models. How stable are these estimates across temperatures? If utility is consistently estimated, this strengthens the interpretation that α/ω differences are about sensitivity, not utility.

4. **Discuss Practical Implications.** The finding that temperature affects LLM rationality has implications for AI deployment. A paragraph noting that greedy decoding produces more EU-aligned choices—potentially desirable or undesirable depending on application—would broaden the paper's relevance.

5. **Pre-Registration Statement.** Was the decision to compare m_11, m_21, and m_31 pre-specified, or were these models considered after observing initial results? Clarifying the confirmatory versus exploratory status of the model comparison would enhance transparency.

---

## Summary Table

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Research Question / Hypotheses | Good | Clear extension of Report 1; three well-defined questions |
| Experimental Design | Good | Clean 5×2 design; confounds between risk/uncertainty operationalization noted |
| Operationalization / Measurement | Good | α and ω clearly defined; risky alternatives pool could be better described |
| Evidential Strength | Good | Strong replication; ω < α finding robust but needs formal model comparison |
| Robustness / Generalizability | Adequate | Single task domain, single LLM; no prior sensitivity analysis |
| Causal Reasoning / Interpretation | Adequate | Post-hoc interpretations; confounds acknowledged but not resolved |
| Relation to Literature | Adequate | Limited engagement with risk/ambiguity literature |
| Computational Methodology | Excellent | Clean diagnostics; appropriate PPCs across models |
| Exposition / Transparency | Excellent | Clear writing; effective visualizations; honest limitations |

---

## Confidential Comments to the Editor

This is a methodologically sound piece that advances the series with genuinely new findings. The ω < α result—that LLMs are less sensitive to EU differences when probabilities are stated explicitly—is counterintuitive and interesting. However, the theoretical interpretation remains underdeveloped, and this limits the paper's contribution to JDM theory.

The major revisions requested are addressable without fundamental restructuring: adding formal model comparison (LOO-CV), engaging with the risk/ambiguity literature, and acknowledging causal limitations more explicitly. The work represents a genuine empirical contribution—documenting context-dependent sensitivity in LLM decision-making—even if the explanation for this phenomenon remains open.

One concern for the series as a whole: the accumulation of findings across multiple reports (temperature effects, risky/uncertain differences, and presumably LLM and task effects in later reports) would benefit from an integrating synthesis. The upcoming factorial synthesis report (Report 7) may address this, but the current report reads somewhat incrementally—"we extended the design, and here are more results." A sharper statement of what the risky-alternatives extension teaches us about the underlying phenomenon would strengthen the contribution.

I see no ethical concerns with this work.

---

*Report prepared following the JDM referee prompt template.*
