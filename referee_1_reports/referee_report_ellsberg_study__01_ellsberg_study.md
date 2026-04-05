# Referee Report: Temperature and SEU Sensitivity — Ellsberg Study

**Report under review:** `01_ellsberg_study.qmd`
**Report path:** `reports/applications/ellsberg_study/01_ellsberg_study.qmd`
**Reviewer role:** Referee for *Judgment and Decision Making* or comparable venue

---

## Summary

This report investigates whether the monotonic negative relationship between LLM sampling temperature and sensitivity (α) to subjective expected utility maximisation—established in the initial temperature study using GPT-4o on insurance claims triage—generalises to a different task domain (Ellsberg-style urn gambles, K=4) and a different foundational model (Claude 3.5 Sonnet). The answer is clearly negative: α exhibits a non-monotonic pattern across temperature levels, the posterior slope straddles zero (P(slope < 0) ≈ 0.77), and pairwise comparisons reveal several reversals. The model fits well at every temperature, so the non-replication reflects genuine behavioural differences rather than modelling artefacts. The report is competently executed, honestly reported, and appropriately scoped—but it simultaneously changes two factors (LLM and task), which limits its ability to diagnose the source of the non-replication. In the context of the broader project series, the subsequent Claude × Insurance and GPT-4o × Ellsberg studies resolve this confound, but this report as a standalone contribution would benefit from sharper framing of its diagnostic limitations.

**Recommendation: Minor Revision**

---

## Strengths

1. **Honest reporting of a null result.** The report forthrightly leads with the non-replication rather than burying it or spinning weak evidence as confirmatory. The callout box in the Introduction immediately flags the key finding, and the discussion section avoids overclaiming. This is exactly the kind of transparency the JDM community values (Simmons, Nelson, & Simonsohn, 2011), and it is relatively rare in practice. The willingness to publish a non-replication within one's own research programme is commendable.

2. **Rigorous cross-study comparison methodology.** The report presents a careful side-by-side comparison with the initial temperature study: matched sample sizes (~300 observations per condition), analogous prior calibration procedures (grid search yielding comparable prior-implied SEU-max rates of 0.76 vs. 0.78), and identical analytical pipeline. The design comparison table (@tbl-design-comparison) is exemplary—it makes the two-factor confound immediately transparent to the reader.

3. **Thorough model adequacy assessment.** Posterior predictive checks at every temperature level (all p-values in [0.3, 0.6]) convincingly establish that the non-monotonicity is a feature of the data, not a modelling artefact. This is a critical diagnostic that many studies omit. Combined with clean MCMC diagnostics across all conditions and adequate parameter recovery under m_02, the reader can trust that the statistical apparatus is functioning correctly.

4. **Well-designed Ellsberg gamble stimuli.** The alternative pool is thoughtfully constructed with three ambiguity tiers (no ambiguity, moderate, high), providing a systematic gradient from risk to Knightian uncertainty. The tiered structure connects naturally to Ellsberg's (1961) original paradigm and creates stimulus variation that is scientifically meaningful, not merely cosmetic. The use of K=4 monetary consequences ($0–$3) provides a richer consequence space than the initial study's K=3.

5. **Clear articulation of next steps.** The discussion explicitly identifies the factorial decomposition needed (Claude × Insurance, GPT-4o × Ellsberg) to disentangle model-specific versus task-specific explanations. This is scientifically mature framing that sets up the subsequent reports in the series and shows the authors understand the limitations of their current design.

---

## Major Issues

### 1. Simultaneous Manipulation of Two Factors Limits Interpretability

**Description:** The study changes both the LLM (GPT-4o → Claude 3.5 Sonnet) and the task domain (insurance triage → Ellsberg gambles) relative to the initial study. The report acknowledges this confound in the discussion, but the framing throughout the report sometimes implicitly treats the study as a generalisation test, when it is more accurately a joint-manipulation study with an unresolvable attribution problem. For instance, the Introduction frames the study as addressing whether the initial finding "generalises" — but a single cell of a 2×2 design cannot answer this question.

**Recommendation:** Reframe the study explicitly as one cell of a planned 2×2 factorial design from the outset (Introduction and Design sections), rather than presenting the factorial framing as a post-hoc insight in the Discussion. Even if the factorial design emerged iteratively, the report would be strengthened by stating early on: "This study changed both factors simultaneously. A non-replication therefore cannot be attributed to either factor alone, motivating the factorial completion described in Reports 5–7." This sets reader expectations appropriately and avoids the perception that the confound is an oversight rather than a recognised limitation.

**Severity:** Major — while the issue is acknowledged, the framing affects how the reader interprets the entire contribution. A JDM reviewer would flag this as a design limitation that the paper must address prominently.

### 2. Absence of SBC for m_02 Weakens Validation

**Description:** The report notes that simulation-based calibration (SBC) was not performed for m_02, relying instead on a "structural equivalence argument" (m_02 is structurally identical to m_0 and m_01, which passed SBC; only the α prior hyperparameters differ). While this argument has some force, it is not fully convincing for a JDM audience. The prior change from Lognormal(3.0, 0.75) to Lognormal(3.5, 0.75) shifts the prior median from ~20 to ~33 and the 90% CI upper bound from ~67 to ~124. The wider prior and higher K=4 consequence space could, in principle, interact with the model's posterior geometry in ways that parameter recovery alone (20 iterations) does not fully probe. In particular, parameter recovery tests whether the model can recover known parameters from simulated data, but SBC tests whether the posterior is calibrated across the full prior—a stronger property.

**Recommendation:** Either (a) run SBC for m_02 even at reduced scale (e.g., 50–100 iterations rather than 200), or (b) strengthen the structural equivalence argument by providing quantitative evidence—e.g., comparing the posterior concentration properties (effective sample sizes, tail behaviour) of m_02 fits to those of m_0/m_01 fits that did pass SBC. A brief calibration check (e.g., rank histograms for α from the 20 parameter recovery iterations) would also partially address this gap without full SBC.

**Severity:** Major — while unlikely to change the qualitative conclusions, the absence of SBC for a model variant being applied to real data is a gap in the validation chain that reviewers comfortable with Bayesian methods will notice.

### 3. No Direct Analysis of Ambiguity Tier Effects

**Description:** The alternative pool is carefully structured into three ambiguity tiers, and the Discussion speculates that "if Claude processes ambiguous alternatives differently at different temperatures... this could create non-monotonic patterns in overall α." However, the report performs no analysis of α or choice behaviour stratified by ambiguity tier. This is a missed opportunity: the tiered structure was built into the design precisely to enable such analysis. The absence of any tier-level analysis is particularly notable given that: (a) the Ellsberg paradigm's central contribution to JDM is the distinction between risk and ambiguity, and (b) a JDM reviewer would expect any study invoking Ellsberg's name to engage substantively with ambiguity effects.

**Recommendation:** Add a section (or at minimum an appendix) examining whether choice patterns or model fit differ by ambiguity tier. Possible analyses include: (i) computing the proportion of SEU-maximising choices by tier × temperature, (ii) fitting separate models or computing posterior predictive checks stratified by tier, or (iii) at minimum, descriptive statistics (e.g., choice entropy, NA rates) broken down by tier. If the current model framework does not naturally support tier-level decomposition, acknowledge this as a limitation and discuss what a tier-sensitive analysis would require.

**Severity:** Major — the Ellsberg framing creates an expectation of ambiguity analysis that the report does not deliver. A JDM reviewer would view this as a significant gap between the study's theoretical positioning and its empirical content.

---

## Minor Issues

### 1. Temperature Range Asymmetry Not Adequately Discussed as a Confound

**Description:** The Anthropic API limits temperature to [0, 1], while the initial study used [0, 1.5] with OpenAI. The report notes this in a callout box and mentions "reduced statistical power," but does not quantify the impact. The initial study's strongest separation occurred between T ≤ 0.7 and T ≥ 1.0, with the T=1.5 condition playing a critical role. By construction, this study cannot access the high-temperature regime that was most informative in the initial study. The slope comparison (−19 vs. −25) is therefore not directly comparable because the two regressions span different temperature domains.

**Recommendation:** Add a quantitative comparison that accounts for the range difference. For example, compute the slope for the initial study restricted to T ∈ {0.0, 0.3, 0.7, 1.0} (excluding T=1.5) and compare it to the Ellsberg slope. If the initial study's restricted-range slope is substantially weaker, this would contextualise the Ellsberg non-replication differently. Alternatively, use standardised temperature (z-scored within each study's range) for the slope comparison.

### 2. Prior Calibration Comparison Could Be More Precise

**Description:** The report states that the m_02 prior "yields a prior-implied SEU-max rate of approximately 0.76 for K=4, comparable to the m_01 prior's 0.78 rate for K=3." It would strengthen the comparison to clarify what "comparable" means operationally. Is the goal to match the prior-implied SEU-max rate, the prior-implied choice entropy, or the prior probability mass in a specific α range? The choice of calibration target affects the informativeness of the prior in different ways.

**Recommendation:** Briefly state the calibration criterion explicitly (e.g., "We targeted a prior-implied SEU-max rate within 5 percentage points of the m_01 prior's rate") and note whether alternative calibration criteria were considered.

### 3. Cross-Study Comparison Figure Could Show Overlaid Temperature Scales

**Description:** The cross-study comparison figure (@fig-cross-study) uses separate panels with different x-axis scales (initial study: 0–1.5; Ellsberg: 0–1.0). This makes visual comparison harder than necessary. The reader must mentally align the overlapping temperature range.

**Recommendation:** Add either (a) a shared x-axis panel where both studies' α estimates are overlaid on a common [0, 1.5] scale (with the Ellsberg points stopping at 1.0), or (b) vertical reference lines at the common temperature values to aid visual comparison.

### 4. Pairwise Comparison Evidence Thresholds Not Justified

**Description:** The pairwise comparison table uses evidence thresholds (P > 0.95 for "strong," P > 0.8 for "moderate," etc.) that are not justified or referenced. While these are reasonable conventions, they are arbitrary and may not match the thresholds a JDM reader expects (e.g., some Bayesian JDM papers use Bayes factors or ROPE-based criteria rather than posterior probability thresholds).

**Recommendation:** Either cite a source for the chosen thresholds, briefly justify them (e.g., "following Kruschke's (2013) guidelines for posterior probability interpretation"), or present the probabilities without categorical labels and let the reader interpret them.

### 5. No Discussion of Position-Bias Effectiveness

**Description:** The design uses position counterbalancing (P=3 shuffled presentations per problem), which is good practice. However, the report does not report any diagnostic of whether position bias exists in the data or whether counterbalancing was effective. For a JDM audience familiar with order/position effects in choice experiments (Hertwig & Ortmann, 2001), a brief check would be reassuring.

**Recommendation:** Add a brief note or footnote reporting the proportion of choices selecting the first-listed vs. last-listed alternative, aggregated across conditions. If there is no systematic position bias, state so; if there is, note its magnitude and confirm that counterbalancing mitigates it.

### 6. Missing Details on Embedding Stability Across Temperatures

**Description:** The feature construction process embeds Claude's assessments at each temperature, then pools all embeddings across temperatures for PCA. Since Claude's assessment text presumably varies with temperature (longer, more diverse phrasing at higher T), the embedding distribution may shift with temperature. This means the PCA basis reflects a mixture of temperature-dependent variation and gamble-specific variation. It is unclear whether this could induce spurious non-monotonicity in α.

**Recommendation:** Add a brief diagnostic: e.g., compute pairwise cosine similarity of embeddings for the same gamble across temperature conditions, or report whether the PCA loadings are stable across temperature-stratified subsets. If embedding variation across temperatures is small relative to across-gamble variation, state so.

---

## Suggestions for Improvement

1. **Ambiguity aversion analysis.** Given the Ellsberg framing, consider computing a direct measure of ambiguity aversion: e.g., the proportion of choices favouring Tier 1 (unambiguous) alternatives when paired against Tier 2 or Tier 3 alternatives. This could be reported descriptively and compared across temperatures. Even without formal modelling, this would connect the study more substantively to the Ellsberg tradition and JDM readership expectations.

2. **Effect size contextualisation.** The α estimates in this study (ranging from roughly 20–50) are numerically large but their practical meaning is not always intuitive. Consider adding a "decision quality" translation: e.g., for each temperature, report the implied probability that the LLM selects the SEU-maximising alternative in a representative 3-alternative problem. This makes the α differences tangible for a JDM audience.

3. **Explicit connection to human Ellsberg findings.** The report invokes Ellsberg's (1961) paradigm but does not discuss how human decision-makers typically behave in Ellsberg tasks. A brief paragraph noting the classic finding (humans prefer known-probability gambles, violating Savage's Sure-Thing Principle) and whether Claude's choices exhibit a similar pattern would strengthen the JDM relevance. Do Claude's choices show any analogue of ambiguity aversion? Does this vary with temperature?

4. **Forward reference to factorial resolution.** While the report correctly identifies the need for a 2×2 factorial, it does not reference the fact that these studies have been completed (Reports 5–7 in the series). Adding a brief forward reference in the Discussion (e.g., "The factorial completion is reported in [Reports 5–7], which reveal...") would help readers navigating the series understand that the interpretive limitations of this report are addressed elsewhere.

5. **Robustness to PCA dimensionality.** The report uses D=32 dimensions without reporting sensitivity to this choice. A brief check with D=16 or D=64 (even on a single temperature condition) would strengthen the claim that the results are not artifacts of the dimensionality reduction.

6. **Visualisation of non-monotonicity pattern.** Consider adding a "spaghetti plot" showing individual posterior draws of the α trajectory (α as a function of temperature) for a random subset of draws (e.g., 50). This would make the non-monotonicity visually compelling and show the uncertainty in the trajectory shape, complementing the existing forest plot and density plots.

---

## Summary Table

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Research Question / Hypotheses | Good | Clear hypothesis stated; framing as generalisation test is somewhat misleading given two simultaneous manipulations |
| Experimental Design | Good | Solid within-condition design (300 obs, counterbalancing); but two-factor confound limits cross-study inference |
| Operationalization / Measurement | Good | Well-constructed Ellsberg gambles with ambiguity tiers; α clearly defined; but ambiguity tiers not analysed |
| Evidential Strength | Excellent | Posterior summaries well-reported; null result presented honestly with appropriate uncertainty characterisation |
| Robustness / Generalizability | Adequate | Model adequacy checked via PPC; but no SBC for m_02, no robustness to PCA dimensions, no ambiguity-tier decomposition |
| Causal Reasoning / Interpretation | Good | Appropriately hedged; alternative explanations enumerated; but could more forcefully state the attribution problem |
| Relation to Literature | Adequate | Ellsberg (1961) invoked but not substantively engaged; no discussion of human ambiguity aversion findings; limited JDM citations |
| Computational Methodology | Good | Clean diagnostics, adequate parameter recovery; SBC absence is a gap |
| Exposition / Transparency | Excellent | Well-organised, clear writing, reproducible data snapshot, effective figures |

---

## Confidential Comments to the Editor

This report is one component of a well-designed research programme that ultimately resolves the interpretive ambiguity present in this individual study through a full factorial design (Reports 5–7). As a standalone contribution, the report is a competent null result that meets the basic standards of JDM journals but is somewhat limited in its interpretability due to the two-factor confound. Its primary value is as a motivating study within the series rather than as an independent contribution.

The most substantive empirical gap is the absence of ambiguity-tier analysis. For a study that frames itself around Ellsberg gambles, a JDM reviewer would expect engagement with ambiguity effects. This is an addressable gap that would meaningfully strengthen the paper.

The methodological standards (Bayesian modelling, posterior predictive checks, honest null reporting) exceed those of a typical JDM submission. The transparency and reproducibility practices are exemplary. With the recommended revisions—principally reframing the design limitations, adding ambiguity-tier analysis, and strengthening the connection to the JDM literature on ambiguity—this report would comfortably meet publication standards as part of the series.
