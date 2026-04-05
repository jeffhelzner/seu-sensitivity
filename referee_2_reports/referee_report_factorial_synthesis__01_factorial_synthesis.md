# Referee Report: 2×2 Factorial Synthesis: LLM × Task

**Report Under Review:** `01_factorial_synthesis.qmd`  
**Path:** `reports/applications/factorial_synthesis/01_factorial_synthesis.qmd`  
**Venue:** *Judgment and Decision Making* or comparable JDM journal

---

## Summary

This synthesis report addresses a crucial interpretive ambiguity from the earlier studies in this series: whether the failure to replicate the temperature–sensitivity relationship observed in the GPT-4o × Insurance study was due to the change in LLM (from GPT-4o to Claude), the change in task (from insurance to Ellsberg), or their interaction. The authors complete a 2×2 factorial design by running the two missing cells and conduct a systematic cross-study analysis. The central finding—that the LLM is the dominant factor, with GPT-4o showing clear temperature–α relationships regardless of task while Claude does not—is well-supported by the presented evidence. The analysis of main effects and interaction is sensible, and the conclusion that the effects are approximately additive is reasonable given the data. However, the report has important methodological limitations: it synthesizes pre-computed data without fitting a unified hierarchical model, the temperature scales differ between LLMs making quantitative comparisons difficult, and the formal assessment of interaction relies on descriptive rather than inferential methods.

**Recommendation: Minor Revision**

The empirical contribution is valuable and the conclusions are defensible, but the statistical analysis could be strengthened and the interpretive framing needs refinement.

---

## Strengths

1. **Elegant Resolution of a Confounded Design.** The original Temperature → Ellsberg study changed both LLM and task simultaneously, precluding attribution. This report completes the factorial by running the missing cells (GPT-4o × Ellsberg, Claude × Insurance), enabling clean decomposition into main effects. This is textbook experimental design logic applied thoughtfully to an ongoing research program.

2. **Consistent Data Collection and Modeling Across Cells.** All four cells use approximately M ≈ 300 observations, D = 32 embedding dimensions, R = 30 distinct alternatives, and the same two-stage feature construction pipeline. The use of m_01 (K=3) for insurance cells and m_02 (K=4) for Ellsberg cells, each with task-appropriate prior calibration, reflects careful attention to design consistency where possible.

3. **Clear and Effective Visualizations.** The forest plots (@fig-forest-2x2), density overlays (@fig-density-2x2), and interaction plots (@fig-interaction, @fig-interaction-slopes) efficiently communicate the key finding: GPT-4o posteriors shift leftward at higher temperatures while Claude posteriors do not. The pairwise comparison heatmaps (@fig-heatmaps-2x2) provide cell-by-cell detail. The summary figure (@fig-grand-summary) is publication-ready.

4. **Transparent Handling of Cross-LLM Temperature Incomparability.** The report explicitly acknowledges that temperature values are not comparable across providers (OpenAI's [0.0, 2.0] vs. Anthropic's [0.0, 1.0]) and appropriately frames the comparison in terms of qualitative patterns ("monotonic decline vs. flat/non-monotonic") rather than quantitative slope magnitudes. This intellectual honesty is commendable.

5. **Appropriate Use of Difference-in-Differences for Interaction.** The quantitative interaction analysis (@interaction-quantitative) correctly computes the difference-in-differences of slopes and reports that the 90% CI spans zero. This is the correct way to assess interaction in a 2×2 design and supports the conclusion of approximate additivity. The inference that "the non-replication was driven primarily by the LLM change" follows logically from the pattern of results.

---

## Major Issues

### Issue 1: No Unified Hierarchical Model for the Factorial Design

**Description:** The report synthesizes pre-computed slope draws from four independently fit models. This "post-hoc synthesis" approach treats each cell as a separate experiment and computes factorial quantities by combining posterior draws across cells. A more statistically coherent approach would fit a single hierarchical model with LLM and task factors (and their interaction) built into the model structure. The current approach (a) cannot share information across cells, (b) treats the cells as though they have no common structure, and (c) may underestimate uncertainty about the interaction if there are shared sources of variation.

The foundational methodology validated α recovery within cells, but cross-cell comparisons introduce a new level of analysis that has not been validated. Are slope differences across cells estimated with appropriate coverage? The current approach assumes yes, but this is not demonstrated.

**Recommendation:** At minimum, discuss why a unified model was not used—if it was computational complexity, model specification difficulty, or the desire to maintain consistency with the individual cell reports, say so explicitly. Ideally, conduct a sensitivity analysis by fitting a simple hierarchical model (e.g., α ~ LLM × Task × Temperature with partial pooling) to at least one comparison pair to assess whether conclusions change. If this is beyond scope, acknowledge the limitation clearly and frame the analysis as exploratory rather than definitive.

**Severity:** Major—the statistical model does not match the factorial structure of the question being asked.

---

### Issue 2: Interaction Assessment is Descriptive, Not Inferential

**Description:** The report concludes there is "minimal interaction" because the 90% CI for the difference-in-differences spans zero and the interaction plot lines are "roughly parallel." This is a reasonable descriptive judgment, but it does not constitute a formal test. The 90% CI spanning zero is consistent with either (a) no true interaction, or (b) an underpowered study that cannot detect a moderate interaction. With only four cells and substantial uncertainty in each slope estimate, the study may have limited power to detect anything other than a large interaction.

Additionally, the probability statements (e.g., P(interaction > 0) = 0.XXX) are treated as though they directly answer the question "is there an interaction?" but the relevant question for establishing additivity is whether the interaction is *practically negligible*, not whether it is *statistically zero*.

**Recommendation:** (1) Report a rough power estimate for detecting an interaction of a specified effect size, or acknowledge that the study may be underpowered for interaction detection. (2) Consider a more stringent criterion for claiming additivity—e.g., that the 90% CI for interaction is entirely within a region of practical equivalence (ROPE). (3) Soften the language from "minimal interaction" to "no strong evidence of interaction, though the study may have limited power to detect moderate interactions."

**Severity:** Major—overclaiming support for additivity without addressing power concerns.

---

### Issue 3: Priors Differ Across Models (m_01 vs. m_02)

**Description:** The insurance cells use m_01 with prior α ~ Lognormal(3.0, 0.75), while the Ellsberg cells use m_02 with prior α ~ Lognormal(3.5, 0.75). This difference was calibrated separately for each task based on prior predictive analysis for K=3 vs. K=4 settings. While this is methodologically defensible for within-task comparisons, it complicates cross-task comparisons in the factorial: the prior-induced "pull" on posterior α estimates differs between insurance and Ellsberg cells.

If the priors are meaningfully different (median 20 vs. 30), and if the data are only moderately informative, then some of the observed cross-task differences in α levels could reflect prior differences rather than data differences. The slope comparisons may be more robust (since the prior is the same within each task), but this warrants discussion.

**Recommendation:** Add a paragraph discussing whether prior differences could affect cross-cell comparisons. Consider (a) reporting the prior-induced α expectations for each cell, (b) noting that the slope analysis (within-task changes across temperature) is less affected than level comparisons, or (c) conducting a sensitivity analysis re-fitting one cell with the other prior to assess robustness.

**Severity:** Major—unaddressed methodological asymmetry that could affect factorial conclusions.

---

## Minor Issues

### Issue 4: No Model Adequacy Assessment at the Synthesis Level

**Description:** The individual cell reports presumably include MCMC diagnostics and posterior predictive checks for each fit. However, the synthesis report does not address whether model adequacy varies systematically across the factorial. If, for example, the model fits the Claude × Insurance cell poorly but the GPT-4o × Ellsberg cell well, this could confound the LLM effect with model misspecification. A table summarizing R-hat, ESS, and key PPC metrics for all four cells would reassure readers.

**Recommendation:** Include a summary table of diagnostics across all four cells, or at minimum note that all cells passed standard diagnostic checks and direct readers to the individual reports.

---

### Issue 5: Thresholds for "Declining" vs. "Flat" Are Arbitrary

**Description:** Table @tbl-monotonicity uses verbal labels ("Declining," "Weak decline," "Flat/non-monotonic") based on P(slope < 0) thresholds that appear to be > 0.9 for "Declining" and > 0.7 for "Weak decline." These thresholds are not justified. Why is 0.9 the cutoff for "Declining"? A P(slope < 0) of 0.85 might be quite convincing in some contexts. The thresholds affect the narrative framing (e.g., Claude × Ellsberg at P ≈ 0.77 is labeled "Weak" rather than "Some evidence").

**Recommendation:** Either justify the thresholds based on decision-theoretic grounds (e.g., "we adopt 0.9 as a conventional threshold analogous to one-sided p < 0.10") or present the probabilities without categorical labels and let readers draw their own conclusions.

---

### Issue 6: Limited Connection to JDM Literature on Cross-Subject Variation

**Description:** The finding that different LLMs show different temperature–sensitivity relationships has an analog in the human JDM literature: individual differences in decision noise, rationality, and consistency. Work by Bruhin et al. (2010) on heterogeneity across decision-makers, Hey and Orme (1994) on error models, and more recent work on computational rationality (Lieder & Griffiths, 2020) could contextualize the LLM findings. Why might different "decision-makers" (here, LLMs) exhibit different temperature–sensitivity profiles? Is this analogous to human individual differences in decision quality?

**Recommendation:** Add 1–2 paragraphs in the Discussion connecting the LLM-specific findings to the broader JDM literature on individual differences in choice quality and noise. This would situate the contribution for a JDM audience.

---

### Issue 7: Missing Discussion of Potential Confounds with Model Training

**Description:** GPT-4o and Claude 3.5 Sonnet differ not only in their temperature implementations but also in their training data, RLHF procedures, and potentially in task-specific fine-tuning. The observed LLM difference could reflect (a) different temperature implementations, (b) different baseline decision-making tendencies, or (c) different training histories that interact with temperature. The report attributes the effect to "temperature implementation" but this is one of several possible explanations.

**Recommendation:** Add a sentence or two acknowledging that the LLM effect could reflect multiple factors beyond temperature implementation per se. The mechanistic investigation mentioned in Future Directions is appropriate, but the current interpretation should be more cautious.

---

### Issue 8: Reproducibility Section Could Be More Detailed

**Description:** The Reproducibility section lists the data directories but does not provide version information for the frozen data, commit hashes, or instructions for re-running the synthesis analysis. For readers who want to reproduce the figures, the path from "access these directories" to "generate this report" is unclear.

**Recommendation:** Add a brief note on how to regenerate the synthesis (e.g., "render this Quarto document with dependencies X") and consider versioning the frozen data with a date stamp or hash.

---

## Suggestions for Improvement

1. **Add a Visual Summary of the Factorial Structure.** A 2×2 table with thumbnails of the α trajectories in each cell, positioned at the start of the Results section, would immediately orient readers to the key pattern before the detailed figures.

2. **Report Bayes Factors for Key Comparisons.** Computing Bayes factors for the main effect contrasts (e.g., GPT slope < Claude slope) would provide a more direct quantification of the evidence for the LLM effect than posterior probabilities alone.

3. **Consider a Meta-Analytic Framework.** The slope estimates from each cell could be pooled in a simple random-effects meta-analysis, providing a formal estimate of the grand mean slope and heterogeneity across cells. This would complement the factorial ANOVA-style analysis.

4. **Discuss Implications for Practitioners.** The finding that GPT-4o's decision-making quality degrades with temperature while Claude's does not has practical implications for LLM deployment in decision-support systems. A sentence or two on this would broaden the paper's appeal.

5. **Pre-Registration Statement.** Were the factorial design and analysis plan pre-registered after the initial confounded studies? If so, noting this would strengthen the confirmatory framing; if not, the analysis should be described as exploratory.

---

## Summary Table

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Research Question / Hypotheses | Excellent | Clearly motivated by prior confound; well-structured factorial |
| Experimental Design | Good | Clean 2×2 design; inconsistent temperature scales across LLMs |
| Operationalization / Measurement | Good | Consistent within cells; priors differ across tasks |
| Evidential Strength | Good | Strong LLM effect; interaction assessment underpowered |
| Robustness / Generalizability | Adequate | Two LLMs, two tasks; no unified model; limited robustness checks |
| Causal Reasoning / Interpretation | Good | Appropriately attributes effect to LLM; some over-interpretation |
| Relation to Literature | Adequate | Limited connection to JDM work on individual differences |
| Computational Methodology | Adequate | Synthesis of pre-computed data; no validation of cross-cell analysis |
| Exposition / Transparency | Excellent | Clear writing; outstanding figures; reproducible structure |

---

## Confidential Comments to the Editor

This synthesis report makes a genuine empirical contribution by resolving the LLM-vs-task confound that plagued the earlier studies. The conclusion that the LLM is the dominant factor is well-supported and has practical implications for LLM deployment. However, the statistical methodology—synthesizing pre-computed posteriors rather than fitting a unified model—is a limitation that the authors should acknowledge more clearly. The claim of "minimal interaction" based on a 90% CI spanning zero is also overstated given likely power limitations.

For a JDM venue, the report would benefit from stronger connections to the human decision-making literature—particularly work on individual differences in choice quality and the sources of decision noise. The finding that different "agents" (LLMs) exhibit qualitatively different temperature–behavior relationships is interesting but unexplained at a mechanistic level.

The work is suitable for publication with the revisions outlined above. The major methodological concerns (Issues 1–3) require substantive attention, while the minor issues and suggestions are largely clarificatory.

I see no ethical concerns with this work.

---

*Report prepared following the JDM referee prompt template.*
