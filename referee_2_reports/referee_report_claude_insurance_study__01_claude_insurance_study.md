# Referee Report: Temperature and SEU Sensitivity: Claude × Insurance Study

**Report Under Review:** `01_claude_insurance_study.qmd`  
**Path:** `reports/applications/claude_insurance_study/01_claude_insurance_study.qmd`  
**Venue:** *Judgment and Decision Making* or comparable JDM journal

---

## Summary

This report presents a well-designed empirical study that isolates the effect of LLM identity on the temperature–sensitivity relationship by pairing Claude 3.5 Sonnet with the same insurance triage task used in the initial temperature study (which employed GPT-4o). The central finding—that the monotonic negative temperature–α relationship observed with GPT-4o is **not replicated** with Claude—is clearly supported by the evidence. The posterior slope is approximately −3 with $P(\text{slope} < 0) \approx 0.56$ (essentially indistinguishable from chance), and the α estimates show a non-monotonic, oscillatory pattern. This represents strong evidence that the temperature–sensitivity effect is LLM-specific rather than a universal property of temperature scaling in language models. The methodological rigor is commendable: parameter recovery validates model identifiability, posterior predictive checks confirm adequate fit, and the findings are honestly reported without overstatement.

**Recommendation: Minor Revision**

The work makes a clear empirical contribution to the factorial design and the core findings are convincingly established. The revisions required are primarily clarificatory, involving theoretical interpretation and connections to the broader literature.

---

## Strengths

1. **Clean Isolation of the LLM Effect.** The experimental design holds the task constant (insurance triage, K=3, R=30, ~300 observations per temperature) while varying only the LLM (Claude vs. GPT-4o). This allows for unconfounded attribution of the non-replication to the LLM factor. The explicit positioning within a 2×2 factorial framework (Table 1) clarifies exactly what this cell contributes to the overall research program and facilitates future cross-cell comparisons.

2. **Consistent Modeling Approach.** The use of the identical m_01 model with identical priors as the initial temperature study ensures that any observed difference is attributable to the LLM, not modeling choices. The design comparison table (Table 4) transparently enumerates the shared and varying parameters across studies, demonstrating methodological discipline.

3. **Rigorous Parameter Recovery.** The 20-iteration parameter recovery study validates that α is identifiable under this design, with approximately 95% coverage, low bias, and reasonable RMSE. The decision to leverage the initial study's SBC (given structural model identity) is defensible and appropriately noted.

4. **Honest Reporting of Non-Replication.** The report does not spin the near-chance $P(\text{slope} < 0) \approx 0.56$ as evidence against an effect or search for post-hoc explanations that preserve the original hypothesis. Instead, the oscillatory pattern and flat overall trend are presented straightforwardly. The contrast with the initial study (slope ≈ −25, $P < 0.01$ vs. slope ≈ −3, $P \approx 0.56$) is visually and numerically compelling.

5. **Excellent Exposition and Visualization.** The forest plot, density overlays, cross-study comparison figure, and pairwise heatmap effectively communicate the findings. The summary callout box at the start orients readers immediately. The reproducibility section with clear data file documentation supports transparency.

---

## Major Issues

### Issue 1: Limited Theoretical Discussion of Why Claude Differs from GPT-4o

**Description:** The report establishes *that* Claude shows a flat, non-monotonic temperature–α relationship while GPT-4o shows a monotonic decline, but offers limited theoretical framework for *why* this might be expected. The Discussion section describes the findings but does not engage with potential mechanisms. Are there architectural, training, or RLHF differences between Claude 3.5 Sonnet and GPT-4o that might explain the divergence? Does Claude's temperature parameter interact differently with its softmax layer? Is Claude's base behavior already near-ceiling or near-floor on this task, limiting temperature's dynamic range?

**Recommendation:** Add 2–3 paragraphs addressing potential mechanisms for the LLM-specific nature of the temperature effect. Possibilities include: (a) different base levels of decision noise in each model; (b) differences in how temperature affects assessment text vs. choice behavior; (c) floor/ceiling effects if Claude is already highly consistent or highly random at baseline. Acknowledge these as speculative but note which could be tested in future work. This will help readers understand what the LLM-specificity finding implies about the nature of the temperature–sensitivity relationship.

**Severity:** Major—the interpretive value of the finding is limited without theoretical context.

---

### Issue 2: Non-Comparable Temperature Ranges Complicate Cross-Study Inference

**Description:** The initial temperature study used $T \in \{0.0, 0.3, 0.7, 1.0, 1.5\}$ over OpenAI's $[0.0, 2.0]$ range, while this study uses $T \in \{0.0, 0.2, 0.5, 0.8, 1.0\}$ over Anthropic's $[0.0, 1.0]$ range. The absolute temperature values differ at 4 of the 5 levels. While the report correctly notes this (and it is unavoidable given API constraints), the implications are understated. The narrower range in this study (absolute span = 1.0 vs. 1.5) may reduce statistical power to detect a temperature effect. More importantly, if the temperature–α relationship is nonlinear in the initial study, extrapolating expectations from GPT-4o's behavior at T=1.5 to Claude's behavior at T=1.0 may be inappropriate.

**Recommendation:** (1) Discuss whether the temperature parameter is likely comparable across providers in terms of the *effective entropy* it induces—i.e., does T=0.5 for Anthropic have the same effect on next-token entropy as T=0.5 for OpenAI? (2) Consider a supplementary analysis restricting the initial study comparison to $T \in \{0.0, 0.3, 0.7, 1.0\}$ to provide a more comparable baseline. (3) Acknowledge in the limitations that cross-provider temperature comparisons require caution.

**Severity:** Major—the comparison is the central contribution, so the comparability of conditions is essential.

---

### Issue 3: Formal Statistical Test of LLM Effect Deferred

**Description:** The report presents side-by-side comparisons (Figure 5, Table 8) demonstrating the GPT-4o vs. Claude contrast but does not compute a formal statistical quantity for the LLM effect. The Discussion mentions that "the full 2×2 factorial analysis (Phase 8 report) will formalise this comparison," but for a standalone publication, readers will expect at least a posterior probability that the Claude slope is smaller than the GPT-4o slope, or credible interval for their difference.

**Recommendation:** Compute and report $P(\text{slope}_{\text{GPT-4o}} < \text{slope}_{\text{Claude}})$ using joint sampling from the two studies' posterior slope distributions. If computational infrastructure makes this difficult (given that the fits were conducted separately), use a Monte Carlo approximation: draw from each study's slope posterior independently and compute the overlap. This quantity directly addresses the claim that the effect is LLM-specific.

**Severity:** Major—the core claim requires a formal contrast, not just visual comparison.

---

## Minor Issues

### Issue 4: SBC Justification by Proxy May Concern Some Readers

**Description:** The report notes that "SBC was not performed for this cell" because "the m_01 model passed SBC in the initial temperature study report." While this reasoning is sound—the model is structurally identical—some readers may object that SBC validates the model under a specific data-generating process, and the Claude assessment text may have different statistical properties than GPT-4o text (e.g., different embedding distributions, different covariance structure in β).

**Recommendation:** Add a sentence explicitly noting that SBC validates the *model*, not the *data source*, and that the parameter recovery study provides study-specific validation. If SBC is computationally tractable, consider running it as a supplementary analysis for completeness.

---

### Issue 5: Oscillatory Pattern Not Characterized Beyond Description

**Description:** The report describes the α estimates as showing a "dip at T=0.2, a rise at T=0.5 and T=0.8, then another dip at T=1.0," and notes this "echoes the oscillating pattern observed in the Ellsberg study with Claude." However, this pattern is not formally characterized. Is the oscillation statistically meaningful, or is it consistent with sampling noise around a flat line? The pairwise comparisons in Table 7 show several comparisons near 0.50, suggesting limited power to detect adjacent differences.

**Recommendation:** Consider computing the posterior probability that the pattern is *merely flat* (i.e., consistent with a horizontal line at the overall mean α) versus genuinely non-monotonic. A simple approach: compute the posterior SD of α at each temperature and check whether the observed pattern falls within the range expected from sampling variability. Alternatively, acknowledge that the oscillation may simply be noise consistent with no temperature effect.

---

### Issue 6: Limited Connection to JDM Literature on Model Heterogeneity

**Description:** The finding that temperature–sensitivity effects are LLM-specific resonates with a large JDM literature on individual differences in decision-making (e.g., some humans show strong EU-alignment while others are noise-dominated). However, the report does not connect to this literature. Are there known factors that predict individual-level α in human choice studies? If so, do LLMs vary in analogous ways?

**Recommendation:** Add a paragraph connecting the LLM-specificity finding to the JDM literature on individual differences in decision noise or rationality. Relevant references might include work on heterogeneity in stochastic choice parameters (Wilcox, 2011), individual differences in utility curvature and noise (Hey & Orme, 1994), or computational vs. strategic sources of choice inconsistency.

---

### Issue 7: Reproducibility Section Lacks Git Hash or Versioning

**Description:** The reproducibility section lists data files but does not indicate a git commit hash, version tag, or date for the frozen snapshot. For a reader attempting to reproduce the analysis in the future, knowing which version of the codebase was used is important.

**Recommendation:** Add a git commit hash or version identifier for the data snapshot and analysis code.

---

### Issue 8: No Discussion of Potential Position Bias Differences Across LLMs

**Description:** The position counterbalancing design (100 base problems × 3 presentations with shuffling) addresses systematic position bias, but the report does not check whether Claude shows the same position bias patterns as GPT-4o. If Claude has stronger or weaker position effects, this could confound the comparison.

**Recommendation:** Add a brief check of position bias (e.g., choice rate by position) and note whether it is comparable across the two studies. If the data are available, include in a supplementary analysis or note in limitations if not examined.

---

## Suggestions for Improvement

1. **Add a Pooled Graph Showing All Four Factorial Cells.** A single figure with α vs. temperature for all four cells of the 2×2 design (including GPT-4o × Ellsberg from the companion study) would visually summarize the factorial structure and highlight the interaction pattern. This could be a teaser for the factorial synthesis report.

2. **Report Absolute α Levels, Not Just Slopes.** The focus on slopes is appropriate for testing the temperature effect, but the absolute α values (e.g., α ≈ 26–33 for Claude vs. α ≈ 23–45 for GPT-4o) may have interpretive significance. Is Claude generally more or less consistent than GPT-4o? A brief comment on mean α differences would round out the comparison.

3. **Consider Discussing Implications for Applied LLM Deployment.** The finding that temperature–sensitivity effects are LLM-specific has practical implications: users cannot assume that temperature tuning will have the same behavioral effects across providers. A sentence in the Discussion noting this applied relevance would broaden the paper's appeal.

4. **Expand the Limitations Section.** Currently implicit, the limitations (non-comparable temperature ranges, deferred SBC, deferred interaction test) should be collected in a dedicated paragraph for transparency.

5. **Pre-Registration Transparency.** If the analysis followed a pre-specified plan, note this. If the comparison approach was developed after seeing the initial study results, acknowledge this as post-hoc (though well-motivated).

---

## Summary Table

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Research Question / Hypotheses | Good | Clear question within factorial design; directional prediction inherited from initial study |
| Experimental Design | Excellent | Clean manipulation; same task holds LLM comparison constant |
| Operationalization / Measurement | Good | Same task, same model as initial study; α comparability assumed |
| Evidential Strength | Good | Strong evidence for non-replication; formal interaction test deferred |
| Robustness / Generalizability | Adequate | Single task, SBC by proxy; broader generalization via factorial design |
| Causal Reasoning / Interpretation | Adequate | Mechanism for LLM-specificity underspecified |
| Relation to Literature | Adequate | Thin connection to JDM individual differences literature |
| Computational Methodology | Excellent | Validated parameter recovery; clean diagnostics; adequate PPCs |
| Exposition / Transparency | Excellent | Clear writing; excellent figures; honest reporting of null trend |

---

## Confidential Comments to the Editor

This report represents a methodologically sound contribution to the factorial design investigating temperature–sensitivity relationships in LLM decision-making. The central finding—that the effect is LLM-specific, with Claude showing no temperature–α relationship on the same task where GPT-4o showed a clear monotonic decline—is convincingly established and makes a substantive empirical contribution.

The main limitation for a JDM audience is the lack of theoretical engagement with *why* different LLMs might respond differently to temperature manipulation. The finding is descriptively interesting but becomes more valuable with mechanistic interpretation. I encourage the authors to at least speculate on potential sources of the LLM difference, even if definitive tests are beyond the scope of this study.

The issue of non-comparable temperature ranges across providers is inherent to working with different LLM APIs, but it does complicate the comparison that is the core contribution of this cell. The authors handle this appropriately by noting it, but a more detailed discussion of whether "temperature" means the same thing across providers would strengthen the inferential foundation.

I recommend acceptance with minor revisions, primarily focused on (1) adding theoretical context for the LLM-specificity finding, (2) addressing the temperature comparability issue more explicitly, and (3) computing a formal statistical quantity for the GPT-4o vs. Claude contrast.

The work raises no ethical concerns and contributes usefully to the growing literature on LLM decision-making and rationality.

---

*Report prepared following the JDM referee prompt template.*
