# Referee Report: Temperature and SEU Sensitivity: Initial Results

**Report Under Review:** `01_initial_study.qmd`  
**Path:** `reports/applications/temperature_study/01_initial_study.qmd`  
**Venue:** *Judgment and Decision Making* or comparable JDM journal

---

## Summary

This report presents a well-executed empirical investigation of how LLM sampling temperature affects estimated sensitivity to subjective expected utility (SEU) maximization. The study employs a Bayesian softmax choice model (m_01) with a thoughtfully calibrated prior, fitting independent models to five temperature conditions (T = 0.0, 0.3, 0.7, 1.0, 1.5) with 300 observations each. The central finding—that higher temperature is associated with lower estimated sensitivity (α), with posterior probability exceeding 0.99 for a negative slope—is clearly supported by the evidence presented. The authors are commendably honest about the limits of their findings, including the inability to distinguish α at T = 0.3 vs. T = 0.7. The main concerns involve theoretical motivation, external validity, and some gaps in connecting the findings to the broader JDM literature.

**Recommendation: Minor Revision**

The work is methodologically rigorous and transparently reported. The revisions required are primarily clarificatory and interpretive rather than structural.

---

## Strengths

1. **Rigorous Prior Calibration.** The prior predictive grid search over 12 candidate priors, evaluated against the actual study design, represents best practice in Bayesian modeling. The selection of Lognormal(3.0, 0.75) is well-justified by its implied SEU-maximizer selection rate (~78%) and avoidance of numerical overflow issues. This stands in favorable contrast to much applied Bayesian work that uses convenience priors without domain-specific calibration.

2. **Thorough Validation Under the Intended Design.** The parameter recovery (20 iterations) and simulation-based calibration (200 simulations) are conducted at the actual study scale (M = 300, D = 32), not a toy design. The α parameter shows excellent recovery metrics (approximately 90% coverage, low bias) and uniform SBC rank distributions, providing assurance that the posterior estimates can be trusted. The honest acknowledgment of the β–δ identification issue, inherited from the foundational methodology, is appropriate.

3. **Position Counterbalancing Design.** The use of 100 base problems × 3 presentations with position shuffling addresses a known confound in LLM choice experiments. The report clearly distinguishes this improved design from an earlier pilot that suffered from position bias artifacts. The zero NA rate across all conditions indicates robust data collection.

4. **Honest Reporting of Non-Separation.** The inability to distinguish α at T = 0.3 and T = 0.7 is candidly presented rather than buried. The pairwise comparison table, the strict monotonicity analysis (P ≈ 0.12), and the coarser monotonicity test (P ≈ 0.80) provide a complete picture of where the evidence is strong and where it is weak. This transparency aligns with the JDM community's post-"False-Positive Psychology" emphasis on honest uncertainty communication.

5. **Excellent Exposition and Visualization.** The figures are publication-quality: the forest plot, posterior density overlays, pairwise comparison heatmap, and slope posterior all effectively communicate the findings. The writing is clear and appropriately technical. The separation of frozen data from analysis code supports reproducibility.

---

## Major Issues

### Issue 1: Insufficient Theoretical Motivation for the Directional Hypothesis

**Description:** The report adopts the hypothesis that increasing temperature should *decrease* α on intuitive grounds ("the additional stochasticity in token generation should manifest as noisier, less EU-aligned choices"). While this intuition is reasonable, the theoretical justification is underdeveloped. Why should softmax temperature in token sampling translate linearly (or even monotonically) to softmax sensitivity in a choice model? The LLM's temperature parameter affects the probability distribution over *tokens*, not directly over *choices*. The mapping from token-level entropy to choice-level behavior is mediated by complex linguistic processing.

**Recommendation:** Add a section explicitly addressing the theoretical pathway from token-sampling temperature to choice-level α. Consider three possibilities: (a) temperature adds noise that propagates through to choice, reducing effective α; (b) temperature affects the LLM's reasoning process in ways that reduce expected-utility alignment; (c) temperature affects both the assessment phase (embeddings) and the choice phase. The current design conflates these mechanisms. At minimum, acknowledge that the directional prediction, while confirmed, rests on an assumption about how temperature effects propagate rather than a formal derivation.

**Severity:** Major—the interpretive framework for the central finding needs strengthening.

---

### Issue 2: Independent Fits Preclude Modeling the Temperature–α Relationship

**Description:** Each temperature condition is fit independently, yielding five separate posterior distributions. The slope analysis is then computed as a derived quantity across these independent posteriors. This approach treats the five conditions as separate experiments rather than levels of a factorial manipulation. A more natural statistical model would express α as a function of temperature—e.g., α(T) = exp(γ₀ + γ₁ · T) or a hierarchical model with partial pooling across conditions. The current approach likely underestimates uncertainty about the trend (because it ignores potential regularities across conditions) and cannot directly estimate or test functional relationships.

**Recommendation:** At minimum, discuss why independent fitting was chosen over a hierarchical or regression approach. If computational tractability or model stability was the concern, say so. Consider whether a follow-up analysis with a temperature-hierarchical model would change the substantive conclusions. The authors mention this as a future extension—elevating this to a robustness check (at least with a simple linear-in-log-temperature model) would strengthen the paper.

**Severity:** Major—the statistical modeling does not match the structure of the experimental design.

---

### Issue 3: Limited Discussion of Construct Validity for α

**Description:** For a JDM audience, the claim that α measures "sensitivity to subjective expected utility maximization" requires careful unpacking. The parameter α controls the sharpness of the softmax choice distribution—higher α means choices are more concentrated on high-EU alternatives. But this interpretation assumes the model's *other* parameters (β, δ) correctly capture the subjective utilities. The report acknowledges β–δ identification issues. If the utility parameters are poorly identified, what does α actually measure? The answer may be "the tendency to choose consistently in the direction implied by the fitted model"—which is not quite the same as "sensitivity to EU maximization."

**Recommendation:** Add a brief discussion, perhaps in the Discussion section, addressing what α is *operationally* measuring versus what it is *theoretically* intended to measure. JDM readers familiar with the distinction between weak and strong revealed preference will appreciate clarification on this point. The claim is that observing higher α at lower temperatures supports the inference that choices are more EU-aligned; make explicit the assumptions needed for this inference.

**Severity:** Major—construct validity is a core concern for JDM reviewers.

---

## Minor Issues

### Issue 4: No Formal Power or Sample Size Justification

**Description:** The study uses M = 300 observations per condition. The report notes this arises from 100 base problems × 3 presentations, but provides no power analysis or design rationale for why 300 is sufficient. The standard deviations of the α posteriors (ranging from ~9 to ~19 across conditions) and the observation that T = 0.3 and T = 0.7 cannot be distinguished suggest the study may be underpowered for fine-grained comparisons.

**Recommendation:** Add a brief justification for the sample size, ideally referencing simulation-based design analysis. If the goal was to detect large differences (T = 0.0 vs. T = 1.5) but not adjacent comparisons, say so explicitly. This should also inform the framing of the T = 0.3 vs. T = 0.7 result—is it a true null effect or insufficient power?

---

### Issue 5: Task Domain Specificity Not Adequately Discussed

**Description:** The insurance claims triage task is a specific operationalization. How well does "choosing which insurance claim to investigate" map onto the theoretical construct of decision-making under uncertainty? Insurance triage may engage domain-specific heuristics, knowledge, or training artifacts in GPT-4o. The generality of the findings to other decision tasks (gambles, consumer choice, medical decisions) is unclear.

**Recommendation:** Add a paragraph in the Discussion acknowledging the task-specific nature of the findings. Note that subsequent reports in the series (Ellsberg, Claude × Insurance, etc.) provide partial replication across tasks and models. The "Next Steps" section mentions replication with different LLMs, but task generalization deserves equal emphasis.

---

### Issue 6: Limited Connection to the JDM Literature

**Description:** The report makes only glancing reference to the human decision-making literature. How do these findings relate to work on human decision noise, consistency, error theories, or rational inattention? Does the temperature–α relationship have any analog in human studies (e.g., time pressure, cognitive load)? The theoretical framework borrows from Luce's choice rule and McFadden's random utility theory but does not engage with contemporary JDM research on stochastic choice.

**Recommendation:** Add 2–3 paragraphs connecting the findings to relevant JDM work. Possible connections include: (a) Trembling-hand models and error theories of choice (Hey & Orme, 1994); (b) Rational inattention and costly information processing (Sims, 2003; Caplin & Dean, 2015); (c) Human experimental work on temperature/entropy manipulations (if any exists). This will help situate the contribution for the target readership.

---

### Issue 7: Notation Inconsistencies

**Description:** The report uses M for decision problems, but later refers to N[m] for alternatives per problem, R for distinct alternatives, and the Stan code uses additional index variables. For readers not deeply familiar with the foundational reports, the notation can be confusing. For instance, the phrase "M = 300 observations per temperature condition" could be misread as 300 alternatives rather than 300 choice problems.

**Recommendation:** Include a notation summary table at the start of the Experimental Design or Model section, or ensure consistent verbal glosses throughout (e.g., "M = 300 choice problems" rather than "observations").

---

### Issue 8: Missing Sensitivity Analysis on Prior Choice

**Description:** The grid search identified Lognormal(3.0, 0.75) as the selected prior. The report argues this prior is appropriate but does not show that conclusions are robust to nearby prior specifications (e.g., Lognormal(2.5, 0.75) or Lognormal(3.0, 1.0)). Given the informative nature of the chosen prior, sensitivity analysis is warranted.

**Recommendation:** Either (a) conduct a brief sensitivity analysis refitting one temperature condition under 2–3 alternative priors, or (b) defer to a robustness report in the series and explicitly note this limitation.

---

## Suggestions for Improvement

1. **Add a Schematic or DAG.** A diagram showing the generative process (temperature → token sampling → assessments → embeddings → PCA → choice model → α) would clarify the causal pathway and help readers identify where temperature effects could enter.

2. **Report ESS for the Primary Parameter.** The MCMC diagnostics table reports overall pass/fail but not the actual ESS values for α. Including min/median ESS for α across conditions would increase transparency.

3. **Consider a Bayesian Model Comparison.** Fitting an alternative model (e.g., random choice model with α = 0 constrained) at each temperature and comparing via LOO-CV or Bayes factors would strengthen the claim that the EU-sensitivity model is appropriate for these data.

4. **Discuss Implications for LLM Alignment.** The finding that temperature affects decision-theoretic rationality has potential implications for AI alignment research. A brief comment connecting to the growing literature on LLM rationality (e.g., Binz & Schulz, 2023; Hagendorff et al., 2023) would broaden the paper's appeal.

5. **Pre-Registration Statement.** If the hypotheses and analysis plan were pre-registered (even informally in the prompt/planning documents), stating this would strengthen transparency claims. If not, acknowledge that the analysis is exploratory/confirmatory with respect to a post-hoc motivated prediction.

---

## Summary Table

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Research Question / Hypotheses | Good | Clear question; directional prediction justified intuitively but not formally |
| Experimental Design | Good | Clean 5-level design with position counterbalancing; independent fits are suboptimal |
| Operationalization / Measurement | Adequate | α interpretation requires more discussion of construct validity |
| Evidential Strength | Excellent | Strong evidence for main claim; honest about non-separation |
| Robustness / Generalizability | Adequate | Single task, single LLM, no prior sensitivity analysis |
| Causal Reasoning / Interpretation | Good | Appropriately hedged causal claims; mechanism underspecified |
| Relation to Literature | Adequate | Connections to JDM literature are sparse |
| Computational Methodology | Excellent | Rigorous validation; clean diagnostics; appropriate PPCs |
| Exposition / Transparency | Excellent | Clear writing; excellent figures; reproducible data snapshot |

---

## Confidential Comments to the Editor

This is a solid piece of work that introduces a novel Bayesian framework for studying LLM decision-making. The methodology is sound and the empirical findings are interesting. The main limitation for a JDM audience is the relatively thin engagement with human decision-making theory—the report reads more as a methods-and-results paper than a contribution to decision science. However, as one report in a larger series, this is perhaps acceptable; the foundational reports apparently develop the theoretical connections in more depth.

I recommend the authors attend particularly to Major Issue #3 (construct validity of α), as JDM reviewers will expect clarity on what the key parameter is measuring. The other major issues are important but more easily addressed.

The question of novelty is worth noting: while LLM choice behavior has been studied before, applying a formally specified Bayesian choice model with careful prior calibration and validation is genuinely new. If the authors can better connect this methodological contribution to substantive questions in JDM (human-AI analogies, benchmarking rationality, etc.), the paper would have broader impact.

I see no ethical concerns with this work.

---

*Report prepared following the JDM referee prompt template.*
