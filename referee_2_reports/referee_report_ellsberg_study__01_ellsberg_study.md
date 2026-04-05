# Referee Report: Temperature and SEU Sensitivity: Ellsberg Study

**Report Under Review:** `01_ellsberg_study.qmd`  
**Path:** `reports/applications/ellsberg_study/01_ellsberg_study.qmd`  
**Venue:** *Judgment and Decision Making* or comparable JDM journal

---

## Summary

This report presents an important replication attempt of the initial temperature study, testing whether the monotonic negative relationship between LLM sampling temperature and SEU sensitivity generalizes beyond GPT-4o on insurance tasks to Claude 3.5 Sonnet on Ellsberg-style urn gambles. The central finding—that the monotonic relationship was **not replicated**—is honestly and thoroughly reported. The posterior slope is weakly negative (median ≈ −19) but with substantial uncertainty encompassing zero (P(slope < 0) ≈ 0.77), and the per-temperature α estimates display a curious non-monotonic pattern. The model fits adequately at every temperature level, ruling out misspecification as an explanation. The authors appropriately acknowledge that the simultaneous change of both task domain and LLM precludes attribution of the non-replication to either factor alone, and propose the natural next steps (a 2×2 factorial design). This is a valuable contribution precisely because it tempers the initial study's optimistic findings and maps the boundary conditions of the temperature–sensitivity relationship.

**Recommendation: Minor Revision**

The work meets JDM standards in its honest reporting of a non-replication and its methodological rigor. The revisions required are primarily about strengthening connections to the Ellsberg literature, clarifying the ambiguity manipulation, and addressing some methodological points.

---

## Strengths

1. **Exemplary Scientific Honesty.** The non-replication is reported transparently and without spin. The Summary of Findings callout box front-loads the negative result, and the Discussion section offers balanced speculation about contributing factors without favoring any particular explanation. This aligns with the JDM community's post-replication-crisis commitment to honest uncertainty communication.

2. **Rigorous Parallel Design.** The study closely mirrors the initial temperature study's structure (M ≈ 300 observations per condition, R = 30 alternatives, D = 32 embedding dimensions, same position-counterbalancing protocol), enabling meaningful cross-study comparison. The differences (K = 4 vs. 3, Claude vs. GPT-4o, temperature range [0, 1] vs. [0, 2]) are systematic and clearly documented in the design comparison table.

3. **Appropriate Prior Calibration for the New Task.** The authors repeated the prior predictive grid search for K = 4 consequences, arriving at Lognormal(3.5, 0.75). The calibration logic—matching the prior-implied SEU-max rate (≈0.76) to the initial study's (≈0.78)—ensures that the prior encodes comparable informativeness. This attention to keeping the inference machinery fair across studies is commendable.

4. **Model Adequacy Verification.** Posterior predictive checks at all five temperature levels show p-values in the acceptable range [0.3, 0.6], demonstrating that the m_02 model describes the choice data adequately. The non-monotonic pattern in α is definitively attributed to the data rather than model misfit. This is crucial for trusting the substantive conclusions.

5. **Clear Implications and Next Steps.** The Discussion explicitly identifies the factorial gap (both LLM and task changed simultaneously) and proposes the two additional conditions needed to disentangle the confound. This constructive framing transforms a negative result into a scientifically useful stepping stone.

---

## Major Issues

### Issue 1: Minimal Engagement with the Ellsberg Paradox Literature

**Description:** The report uses "Ellsberg-style urn gambles" as a task domain but engages only superficially with Ellsberg (1961) and the vast literature on ambiguity aversion. The three-tier ambiguity structure (Tier 1: no ambiguity; Tier 2: moderate; Tier 3: high) is a substantial design feature, yet the analysis does not differentiate behavior by tier or connect findings to predictions from ambiguity-aversion models (e.g., maxmin expected utility, α-MEU, smooth ambiguity models). For a JDM audience familiar with this literature, the lack of engagement is conspicuous. Does Claude exhibit the Ellsberg paradox? Does temperature affect ambiguity attitudes? These questions are latent in the design but unexplored.

**Recommendation:** Add a subsection analyzing α (or observed choices) by ambiguity tier. If the tiered structure was not designed to test ambiguity attitudes, acknowledge this limitation and note that the design could support such analysis with different model specification. At minimum, cite key work on ambiguity aversion (Ellsberg, 1961; Gilboa & Schmeidler, 1989; Machina & Siniscalchi, 2014) and explain why the current model—which assumes SEU—is appropriate despite the presence of ambiguous alternatives.

**Severity:** Major—the literature connection is expected for JDM venues and is central to the task domain.

---

### Issue 2: Confounded Design Limits Interpretability

**Description:** The authors acknowledge that changing both the task domain and the LLM simultaneously precludes clean attribution of the non-replication. This is appropriate, but the framing understates the interpretive difficulty. With two confounded factors, the space of possible explanations is not just "model-specific" vs. "task-specific"—it includes *interaction* effects (Claude might behave like GPT-4o on insurance tasks but differently on Ellsberg gambles). The current design provides no information about interactions.

**Recommendation:** Revise the Discussion to explicitly address the interaction possibility. The statement "it is not possible to attribute the non-replication to either factor alone" is true but incomplete; it should also note that the non-replication could arise from a *combination* of factors that is not decomposable. This strengthens the case for the 2×2 factorial and manages reader expectations about what the current study can and cannot show.

**Severity:** Major—inferential clarity is essential for JDM audiences.

---

### Issue 3: SBC Not Performed for m_02

**Description:** The report relies on a "structural equivalence" argument to justify skipping simulation-based calibration (SBC) for m_02, noting that m_0 and m_01 passed SBC and m_02 differs only in the prior hyperparameters on α. While this argument has merit, it leaves a validation gap. Prior hyperparameters can affect posterior calibration through interactions with the likelihood (e.g., if the prior pulls the posterior away from well-identified regions). The parameter recovery exercise (20 iterations) provides some reassurance but is not a substitute for the systematic coverage checks that SBC provides.

**Recommendation:** Either (a) run SBC for m_02 at reduced scale (e.g., 50–100 simulations with fewer chains) and include the results in an appendix, or (b) strengthen the structural equivalence argument by noting specific conditions under which prior hyperparameters do *not* affect calibration (e.g., posterior dominated by likelihood, prior support contained within high-likelihood region). The current handwave is too brief for readers familiar with SBC methodology.

**Severity:** Major—validation completeness is a methodological standard.

---

## Minor Issues

### Issue 4: Narrower Temperature Range Reduces Power

**Description:** The Anthropic API limits temperature to [0, 1], compared to OpenAI's [0, 2]. The report notes this but understates its implications. The initial study's strongest effect separation was between T ≤ 0.7 and T ≥ 1.0. The Ellsberg study's entire range falls within the initial study's "low-to-moderate" region. Even if Claude's behavior mirrored GPT-4o's, the narrower range would yield weaker effect detection.

**Recommendation:** Add a quantitative comparison of the expected effect size given the range restriction. One approach: using the initial study's slope estimate (−25 per unit T), what would the expected α difference be over [0, 1] vs. [0, 1.5]? This would help readers calibrate whether the weak slope finding reflects a genuine difference or power loss.

---

### Issue 5: Alternative Pool Description Lacks Specificity

**Description:** The alternative pool summary (Table 2) describes three tiers by structural features but provides no examples of specific gambles. What are the actual ball counts and payout mappings? For an Ellsberg-style task, the specific framing matters—readers familiar with the literature will expect to see urn descriptions. Without this, evaluating whether the alternatives genuinely engage ambiguity-related decision processes is difficult.

**Recommendation:** Add an appendix or expanded table with 2–3 example gambles from each tier, including urn size, ball counts (or constraints), payout rule, and expected value (where calculable). If alternatives are templated, describe the template; if hand-crafted, explain the rationale.

---

### Issue 6: Non-Monotonic Pattern Unexplored

**Description:** The α estimates show an intriguing alternating pattern: T = 0.0 and T = 0.5 yield higher α than T = 0.2 and T = 0.8, with T = 1.0 intermediate. The pairwise reversal at T = 0.2 → 0.5 is particularly striking. The Discussion lists possible explanations for the overall non-replication but does not address this specific pattern. Is it consistent with any of the proposed mechanisms? Could it be sampling noise?

**Recommendation:** Add speculation about the non-monotonic structure. One possibility: the intermediate temperatures (0.2, 0.8) may engage different modes of Claude's reasoning, while extremes (0.0, 1.0) and the midpoint (0.5) are more stable. Alternatively, with only ~300 observations per condition and wide posteriors, the alternating pattern may simply reflect posterior uncertainty. Computing the probability that the pattern is exactly as observed under a monotonic null model would help quantify this.

---

### Issue 7: Missing Connection to Human Temperature/Noise Studies

**Description:** The report situates the LLM findings relative to the initial study but not relative to human decision-making research. Are there human studies of decision noise under varying conditions that could inform interpretation? The softmax/inverse-temperature framing has parallels in reinforcement learning, experimental economics (trembling-hand equilibria), and bounded rationality research.

**Recommendation:** Add 1–2 paragraphs connecting to relevant human literature. Possible touchpoints: Hey & Orme (1994) on stochastic choice in humans; Caplin & Dean (2015) on rational inattention; work on cognitive load and decision noise. This would strengthen the JDM audience engagement.

---

### Issue 8: No Cross-Validation or Model Comparison

**Description:** The report fits m_02 at each temperature level and reports posterior predictive p-values, which confirm the model does not grossly misfit. However, there is no model comparison (e.g., against a random-choice null, a simpler model without utility structure, or the m_0 foundational model with uncalibrated prior). Showing that the calibrated model outperforms alternatives would strengthen the evidence that the SEU-sensitivity framework is appropriate.

**Recommendation:** Consider adding LOO-CV or Bayes factor comparison against a baseline model (e.g., uniform choice). If computational cost is prohibitive, acknowledge this as a future direction.

---

### Issue 9: Notation Link to Foundations

**Description:** The report assumes familiarity with the foundational reports' notation (α, β, δ, ψ, η, χ). For readers encountering this report independently, a brief notation summary would improve accessibility. The alternative is to ensure that cross-references to the foundational reports are sufficient (the current link to Report 1 is helpful but not hyperlinked in a way that clarifies notation on first encounter).

**Recommendation:** Add a brief notation table in the Model section or expand the cross-reference to specify which foundational report contains the notation definitions.

---

## Suggestions for Improvement

1. **Tier-Specific Analysis.** Given the three-tier ambiguity structure, it would be scientifically interesting to examine whether α varies by tier within each temperature condition. This could be done by subsetting the data and fitting separate models, or by extending the model to include tier-specific α parameters. Even a descriptive analysis (choice consistency by tier) would add value.

2. **Direct Posterior Comparison with Initial Study.** The cross-study figure plots medians and CIs side by side, but a more direct comparison would overlay posterior densities (or show the posterior difference in slope). This would clarify whether the two slopes are statistically distinguishable or merely point-different.

3. **Qualitative Assessment Analysis.** The report notes that Claude produces natural-language assessments before choosing. For readers interested in understanding *why* the pattern differs, examining exemplar assessments at low vs. high temperature could provide mechanistic insight. Are high-temperature assessments longer, less coherent, or differently framed?

4. **Effect Size Interpretation.** What does the credible interval for the slope [−65, +24] imply practically? If the true slope is −65, what change in SEU-max rate would that imply over the temperature range? Connecting the statistical summaries to behavioral predictions would enhance interpretability.

5. **Pre-Registration Statement.** State whether the study design and analysis plan were pre-registered or determined before data collection. The foundational reports suggest a well-planned sequence, but explicit acknowledgment of the confirmatory vs. exploratory status would strengthen transparency.

---

## Summary Table

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Research Question / Hypotheses | Good | Clear replication hypothesis; extension to new domain/model well-motivated |
| Experimental Design | Good | Clean parallel design; confounding acknowledged but limits inference |
| Operationalization / Measurement | Adequate | Ellsberg gambles appropriate but insufficiently described; ambiguity tiers unexplored |
| Evidential Strength | Good | Honest non-replication; evidence proportional to claims; posterior uncertainty well-characterized |
| Robustness / Generalizability | Adequate | Model fits well but SBC skipped; limited robustness checks |
| Causal Reasoning / Interpretation | Good | Appropriately hedged; multiple explanations offered without overcommitment |
| Relation to Literature | Insufficient | Minimal engagement with Ellsberg/ambiguity literature; no human JDM connections |
| Computational Methodology | Good | Clean diagnostics; adequate PPCs; parameter recovery satisfactory |
| Exposition / Transparency | Excellent | Clear writing; publication-quality figures; frozen data snapshot supports reproducibility |

---

## Confidential Comments to the Editor

This report represents solid scientific practice: the authors designed a study to test generalizability of an earlier finding, obtained a null/ambiguous result, and reported it honestly with appropriate caveats. The non-replication is arguably more informative than a simple replication would have been, as it reveals boundary conditions for the temperature–sensitivity relationship. For a JDM venue, this kind of honest empirical work is valuable.

The primary weakness is the disconnect between the Ellsberg task framing and the analysis strategy. By using Ellsberg gambles without engaging with the ambiguity literature or analyzing behavior by ambiguity tier, the authors miss an opportunity to connect with a large JDM readership interested in ambiguity aversion. This is particularly salient because SEU models explicitly assume no ambiguity aversion—the subjective probabilities in the model are point estimates, not intervals or sets. Using SEU to model choices over ambiguous alternatives is not necessarily wrong (the model may still describe choice patterns), but it warrants discussion.

The confounded design is a significant limitation but one the authors acknowledge and propose to address. I would encourage acceptance conditional on strengthening the literature connections (Major Issue #1) and clarifying the inferential limits (Major Issue #2). The SBC issue (Major Issue #3) is important methodologically but unlikely to change the substantive conclusions.

Overall, this is a well-executed study that advances understanding of when and whether LLM decision-making is affected by sampling temperature. The finding that the effect does not generalize straightforwardly is an important contribution.

---

*Report prepared following the JDM referee prompt template.*
