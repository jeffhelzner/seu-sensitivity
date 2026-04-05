# Referee Report: Temperature and SEU Sensitivity: GPT-4o × Ellsberg Study

**Report reviewed:** `gpt4o_ellsberg_study/01_gpt4o_ellsberg_study.qmd`
**Review date:** 2026-04-02
**Reviewer role:** Referee for *Judgment and Decision Making*

---

## Summary

This report fills a critical cell in the project's 2×2 factorial design (LLM × Task), pairing GPT-4o with the Ellsberg gamble task (K=4) to disambiguate whether the temperature–sensitivity effect observed in the initial study is driven by the LLM, the task, or both. The study finds a clear negative temperature–α relationship (slope median ≈ −38, P(slope < 0) ≈ 0.98), with α declining from ≈110 at T = 0.0 to ≈52 at T = 1.5. This replicates the qualitative pattern from the initial GPT-4o × Insurance study and, together with the Claude × Insurance null result, provides strong evidence that the LLM — not the task domain — is the dominant factor. The report is competently structured, transparent in its methods, and appropriately situated within the factorial program. However, its contribution is primarily *diagnostic* (resolving a confound) rather than substantively novel, and several interpretive and methodological issues — notably the comparability of α across model variants with different K and priors, the absence of prior sensitivity analysis, and the limited engagement with the ambiguity aversion literature — should be addressed before the report meets full JDM publication standards.

**Recommendation: Minor Revision**

---

## Strengths

1. **Clean factorial logic and clear scientific role.** The report's primary contribution is resolving the confound from the Ellsberg study, where both LLM and task were changed simultaneously. By pairing GPT-4o with Ellsberg gambles while holding all other design parameters constant (same temperature levels, same M ≈ 300, same model variant m_02), the study isolates the task dimension of the factorial. The 2×2 design matrix is presented early and clearly, and the reader always knows which comparison is being made and why. This is textbook experimental logic, well-executed.

2. **Direct replication of temperature–sensitivity effect across domains.** The finding that GPT-4o shows P(slope < 0) ≈ 0.98 on Ellsberg gambles — comparable to P > 0.99 on insurance triage — is substantively important. It demonstrates that the effect is not an artifact of the specific insurance triage task or its K=3 structure. The consistency across two qualitatively different decision domains (applied risk assessment vs. abstract gambles) meaningfully strengthens the external validity of the GPT-4o finding.

3. **Model validation appropriate for the cell.** The parameter recovery analysis (20 iterations, M ≈ 300, K=4, D=32) confirms that α is identifiable under the study's specific design. The decision to omit SBC — justified by the structural identity of m_02 across data sources — is reasonable and honestly reported. The posterior predictive checks at all five temperature levels provide adequate model adequacy assurance.

4. **Cross-study comparison is well-structured.** The dual-panel comparison figure (GPT-4o across tasks; Ellsberg across LLMs) is effective at communicating the factorial decomposition. The quantitative comparison table with slopes, P(slope < 0), and P(strict monotonicity) across all available cells provides a compact summary of the program's key findings and allows the reader to evaluate the evidence directly.

5. **Reproducibility.** The frozen data snapshot, Stan-ready JSON files, and refitting instructions maintain the high reproducibility standard established in earlier reports.

---

## Major Issues

### 1. Cross-model comparability of α values

**Description:** The report notes that GPT-4o's α declines from ≈110 to ≈52 on Ellsberg (K=4), compared with ≈41 to ≈9 on Insurance (K=3). The raw α magnitudes differ by roughly 3–5× across the two tasks. While the *direction* of the temperature–α slope is consistent, the *levels* are not directly comparable because the two studies use different model variants (m_01 vs. m_02) with different priors — Lognormal(3.0, 0.75) vs. Lognormal(3.5, 0.75) — and different numbers of consequences (K=3 vs. K=4). The report acknowledges this in passing ("the magnitude may differ due to different K values and prior scales") but does not unpack the implications. A JDM reader might reasonably ask: does GPT-4o at T=1.5 on Ellsberg (α ≈ 52) exhibit *more* or *less* adherence to EU maximization than GPT-4o at T=0.0 on Insurance (α ≈ 41)? The answer depends on how α interacts with K and the utility scale, and the report does not address this.

**Recommendation:** Add a paragraph in the Discussion (or a callout box in §5) explaining why absolute α values are not comparable across cells with different K or different priors. Ideally, convert α values into a common metric — e.g., the implied probability of choosing the EU-maximizing alternative in a canonical problem — to give the reader a task-independent sense of decision quality. Even without this conversion, the current comparison should be accompanied by a more explicit caveat than "the magnitude may differ."

**Severity:** Major — without this clarification, readers may draw incorrect quantitative inferences from the cross-study comparison figures and tables.

### 2. No prior sensitivity analysis

**Description:** The m_02 prior on α is Lognormal(3.5, 0.75), with median ≈33 and 90% CI ≈ [10, 124]. The posterior α values at the lowest temperatures (≈110) are near the upper end of this prior, while values at the highest temperature (≈52) are closer to the median. This raises the question of whether the prior is exerting meaningful influence on the posterior, particularly at the extremes. The report inherits the m_02 prior from the Ellsberg study without testing whether the qualitative conclusions — the sign and strength of the slope — are robust to reasonable perturbations of the prior hyperparameters.

**Recommendation:** Refit at least the boundary conditions (T = 0.0 and T = 1.5) under one or two alternative priors (e.g., a wider Lognormal(3.5, 1.0) and a shifted Lognormal(4.0, 0.75)) and verify that the slope sign and P(slope < 0) remain stable. Given M ≈ 300, the data should overwhelm moderate prior perturbations, but this should be demonstrated rather than assumed — particularly since the estimated α at T = 0.0 (≈110) sits in the tail of the current prior.

**Severity:** Major — this is a standard expectation for Bayesian analyses in JDM venues (cf. Kruschke, 2013) and is especially important when posterior estimates fall near the prior boundaries.

### 3. Limited engagement with ambiguity aversion literature

**Description:** The study uses Ellsberg-style gambles, which are the canonical stimuli for studying ambiguity aversion — one of the most extensively studied phenomena in behavioral decision theory (Ellsberg, 1961; Camerer & Weber, 1992; Trautmann & van de Kuilen, 2015). Yet the report treats the Ellsberg task entirely as a vehicle for the temperature–sensitivity analysis, with no discussion of whether GPT-4o exhibits ambiguity aversion, how temperature might modulate it, or what the estimated α values imply about the LLM's treatment of unknown probabilities. For a JDM audience, Ellsberg gambles are not merely a "different task domain" — they are a theoretically rich context that invites specific predictions about the relationship between uncertainty attitudes and choice sensitivity.

**Recommendation:** Add a subsection (or expand the Discussion) addressing at least two questions: (a) Does GPT-4o exhibit behaviour consistent with ambiguity aversion in its gamble assessments at low temperatures? This could be examined by checking whether the model's estimated subjective probabilities (ψ) reflect a known-urn preference, even without new model fits. (b) Is there a theoretical reason to expect temperature to interact differently with ambiguity-sensitive vs. risk-only decisions? Even a brief discussion connecting the results to the ambiguity aversion literature would significantly strengthen the report's appeal to a JDM audience.

**Severity:** Major — a report using Ellsberg gambles that does not engage with the ambiguity aversion literature will appear under-theorized to JDM reviewers.

---

## Minor Issues

### 1. Slope comparison across cells with different temperature ranges

**Description:** The cross-study comparison table presents slopes for all four factorial cells, but the Claude cells use a narrower temperature range ([0.0, 1.0]) than the GPT-4o cells ([0.0, 1.5]). Since the slope Δα/ΔT is a linear regression coefficient over different covariate ranges, direct comparison of slope magnitudes is confounded by range differences. The GPT-4o slope may appear steeper in part because it is estimated over a wider range that includes the potentially influential T = 1.5 point.

**Recommendation:** Acknowledge this range difference explicitly when comparing slopes. Consider computing a restricted slope for the GPT-4o cells using only T ∈ {0.0, 0.3, 0.7, 1.0} to produce a more directly comparable quantity for the Claude cells.

### 2. Largest drop between T = 1.0 and T = 1.5 not interrogated

**Description:** The Discussion notes that "the largest drop is between T = 1.0 and T = 1.5" but does not explore this finding. This is potentially important: if the temperature–sensitivity relationship is nonlinear (concave, with the largest marginal effect at high temperatures), this has implications for both theory and practical use of LLMs. Moreover, T = 1.5 is beyond the Anthropic API range, so this observation applies only to GPT-4o.

**Recommendation:** Add 1–2 sentences discussing whether the large T = 1.0 → T = 1.5 drop suggests a nonlinear relationship or a possible threshold effect, and note that this high-temperature regime is unavailable for Claude, limiting comparability.

### 3. Strict monotonicity probability not reported numerically in the text

**Description:** The code block computes and prints P(strictly decreasing), but this value is not directly stated in the Discussion narrative. The reader must locate the code output rather than finding it in the interpretive text.

**Recommendation:** Report the strict monotonicity probability explicitly in the Discussion summary (e.g., "P(α strictly decreasing across all five temperatures) = X.XX").

### 4. PCA variance explanation not interpreted

**Description:** The PCA summary is printed (total variance explained, cumulative variance by component) but not discussed. For a JDM audience unfamiliar with PCA, even a brief note — e.g., "32 components capture X% of the variance in GPT-4o's Ellsberg gamble assessments, indicating that the embedding space is [high/moderate]-dimensional" — would be helpful.

**Recommendation:** Add one sentence interpreting the PCA variance statistics.

### 5. Missing description of Ellsberg gamble stimuli

**Description:** The report describes the task as "urns with unknown colour compositions" and notes K = 4 consequences, but does not specify the gamble structures, the urn compositions, or the monetary outcomes. A JDM reader evaluating whether the stimuli genuinely engage Ellsbergian ambiguity would need more detail. The Ellsberg study (Report 4) presumably describes these in detail, but this report should be self-contained enough for a reader to evaluate the stimuli.

**Recommendation:** Add a brief description of the gamble set (e.g., number of unambiguous vs. ambiguous gambles, consequence values, representative example) or provide an explicit cross-reference to the Ellsberg study's stimulus description with a note that the gamble set is identical.

### 6. Data quality: NA summary without context

**Description:** The NA summary is printed but not discussed in the text. The reader sees raw numbers without knowing whether the NA rate is typical, concerning, or noteworthy relative to other cells.

**Recommendation:** Add a sentence comparing the NA rate to other factorial cells and noting whether it is within acceptable bounds.

---

## Suggestions for Improvement

1. **Implied choice probability conversion.** To make α values interpretable for a JDM audience, compute the implied probability of choosing the EU-maximizing alternative for a canonical Ellsberg problem (e.g., a 3-alternative problem with known consequence values) at each temperature level. This translates the abstract α parameter into decision-theoretic terms that JDM researchers can evaluate against their intuitions.

2. **Explore temperature effects on subjective probability estimates.** The model's β parameters map embeddings to subjective probabilities (ψ). Examining whether the estimated ψ distributions shift with temperature — e.g., whether high-temperature conditions produce more diffuse subjective probability estimates — would provide mechanistic insight beyond the single α summary.

3. **Ambiguity-specific analysis.** If the Ellsberg gamble set includes gambles at different ambiguity levels (unambiguous, moderately ambiguous, highly ambiguous), examine whether the temperature effect on α is modulated by gamble ambiguity. This would connect the study directly to the ambiguity aversion literature and could reveal whether temperature disrupts ambiguity-sensitive processing specifically.

4. **Visualization of the factorial interaction.** While the cross-study comparison figure is effective, a single interaction plot showing the slope (or Δα) as a function of LLM with task as a within-panel factor (or vice versa) would make the factorial structure more visually accessible. This could complement or replace the dual-panel layout.

5. **Broader literature on LLM stochastic choice.** The growing literature on LLM decision-making and rationality (e.g., Binz & Schulz, 2023; Hagendorff et al., 2023) is not referenced. Brief connections to this work — particularly studies examining how LLMs handle decision-theoretic tasks and how API parameters affect output quality — would situate the contribution for readers outside the SEU sensitivity project.

6. **Effect of embedding model on results.** Both GPT-4o cells use OpenAI's text-embedding-3-small, while both Claude cells use a potentially different embedding pipeline. If the embedding model differs across LLM conditions, this could confound the LLM main effect. A brief note clarifying whether the embedding model is held constant across LLM conditions (or explaining why it cannot be) would preempt this concern.

---

## Summary Table

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Research Question / Hypotheses | Good | Clear factorial logic; well-motivated by the need to resolve earlier confounds |
| Experimental Design | Good | Sound manipulation; well-matched to other factorial cells; limited to 2 LLMs × 2 tasks |
| Operationalization / Measurement | Good | α is well-defined; cross-cell comparability of α levels is underaddressed |
| Evidential Strength | Good | Strong slope evidence (P ≈ 0.98); adequate PPCs; no prior sensitivity check |
| Robustness / Generalizability | Adequate | No prior sensitivity analysis; no restricted-range slope comparison; temperature ranges differ across cells |
| Causal Reasoning / Interpretation | Good | Appropriate hedging; factorial logic supports causal decomposition; mechanism not modeled |
| Relation to Literature | Adequate | Ellsberg gambles used without engaging ambiguity aversion literature; no LLM rationality references |
| Computational Methodology | Good | Parameter recovery adequate; SBC justifiably omitted; PPCs pass; MCMC diagnostics clean |
| Exposition / Transparency | Good | Clear structure; good cross-references; some numerical results buried in code output rather than text |

---

## Confidential Comments to the Editor

This report's contribution is primarily methodological-diagnostic: it fills a necessary cell in the 2×2 factorial design and confirms that GPT-4o's temperature–sensitivity effect generalises across task domains. This is genuinely useful work, but the contribution is narrower than the initial temperature study (which established the phenomenon) or the factorial synthesis (which draws the programmatic conclusions). As a standalone JDM submission, it would need to be substantially enriched with ambiguity-focused analyses to justify the Ellsberg gamble framing; as a component of the series, it is well-positioned and clearly necessary.

The major issues identified are all addressable without re-analysis or fundamental restructuring. The prior sensitivity analysis is the most important supplement; the cross-model comparability discussion and ambiguity literature engagement are primarily expository improvements. The report is methodologically sound and the core finding is credible — GPT-4o's temperature–sensitivity effect replicates on Ellsberg gambles, and the LLM is the dominant factorial driver.

One editorial consideration: the report estimates α ≈ 110 at T = 0.0, which is at the tail of the Lognormal(3.5, 0.75) prior (prior 95th percentile ≈ 124). While this does not necessarily indicate a problem — the data may simply favour high sensitivity at greedy decoding — it does underline the importance of the prior sensitivity analysis. If the high-α estimates are partially prior-driven (pulled toward the prior's support region), the slope magnitude could be inflated. I expect this is not the case given M ≈ 300, but it should be verified.

The report is close to JDM publication quality. Minor revision is the appropriate recommendation, contingent on the authors addressing the prior sensitivity and cross-model comparability issues.
