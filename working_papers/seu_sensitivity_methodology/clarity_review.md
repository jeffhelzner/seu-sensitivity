# Clarity / pacing review of the draft

Working document for the clarity pass on `paper/_output/paper.pdf`. This is the
**review deliverable**: a prioritized inventory of passages that go too quickly
for the typical reader, each with a concrete recommended fix. **No source edits
have been made.** Approve a subset (or all) and I will apply them in one pass,
then re-render and verify.

## Reader model (locked)

Optimize for a reader who is **quantitatively literate but a specialist in none**
of the paper's three constituencies (Bayesian-workflow/stat.ME, ML/AI,
decision-theory/formal-epistemology). Operating rule: **gloss anything that is a
term of art *within* one field for the others; do not re-explain the shared
undergraduate basics of either statistics or decision theory.** Concretely, we
gloss both (i) specialized workflow terms even statisticians don't all share
(SBC, marginal-vs-joint calibration, bootstrap-over-iterations vs. posterior
interval, $\hat R$, prior-predictive rate) and (ii) all decision-theory terms of
art (Dutch book, independence axiom, ambiguity aversion, MEU, matched design).

## Constraints honored by every proposed fix

- Moderate length appetite: 1–3 sentence targeted additions; prefer
  reorder/split over new prose where possible.
- Edit `.qmd` sources only (`paper.tex`, `_output/` are Quarto-generated).
- Em-dash as `---`; callouts `appearance="simple"`; "gauge" reserved for the
  $\beta$ row-shift only; no figurative-sweep words; honest-reporting (nulls =
  "does not detect" / inconclusive, MDE-calibrated).
- **No numeric changes; `claims_ledger.md` untouched.** (See the behavioral-gloss
  decision note under §7 P3 items — those must stay qualitative unless you
  authorize computing and verifying new rates.)

## Severity legend

- **P1** — a genuine comprehension barrier on a load-bearing passage. Fix first.
- **P2** — a real slowdown for the typical reader; strong ROI.
- **P3** — a helpful gloss/polish; apply if you want the fuller treatment.

**Recommended first edit pass: all P1 + all P2 (13 items).** Hold P3 (11 items)
for a later polish, or fold in per your answer to the severity-floor question.

---

# P1 — fix first

## CR-01 · §4.4 · "SBC" undefined + no intuition for why uniform ranks = calibrated
- **Location:** `_04_m0_implementation.qmd` L104–106: "**Method.** SBC draws a
  parameter from the prior, simulates data, fits the model, and computes the rank
  of the true value within the posterior draws; under correct calibration these
  ranks are uniform [@talts2018]."
- **Category:** (c) undefined term of art · (d) missing intuition.
- **Problem:** "SBC" is used in the heading and here but never expanded; and the
  key fact — *why* uniform ranks certify calibration — is asserted, not
  motivated. This is the diagnostic the whole validation argument rests on.
- **Recommended fix (add one lead-in sentence before "**Method.**"):**
  > Simulation-based calibration (SBC) checks whether the model-plus-sampler
  > returns posteriors that are calibrated on average: if we repeatedly draw a
  > true parameter from the prior, simulate data from it, and refit, a
  > well-calibrated 90% credible interval should contain the truth 90% of the
  > time. The rank test below is the standard way to check this across all
  > quantiles at once.
  Then, after "these ranks are uniform," append the intuition: "--- a true value
  drawn from the prior is equally likely to land at any rank within a correctly
  calibrated posterior, so systematic departures from uniformity flag a mismatch
  between model, data, and sampler."

## CR-02 · §4.4 · marginal-SBC demarcation callout bundles three distinct ideas
- **Location:** `_04_m0_implementation.qmd` L120–121: "**The marginal-SBC
  demarcation.** Two distinct points must be kept apart. *First*, marginal rank
  uniformity is necessary but not sufficient for *joint* [calibration] ..."
- **Category:** (e) dense formalism needing plain-language structure.
- **Problem:** The passage carries three independent claims — (i) marginal ≠
  joint calibration, (ii) weak joint informativeness is not miscalibration,
  (iii) what marginal SBC *does* validate — in running prose. The reader must
  hold all three at once.
- **Recommended fix:** Keep the callout, but split the body into three
  short labeled sentences (bold run-in, not a bullet list, to preserve the
  breakable-callout layout that avoided the earlier overflow):
  > **Marginal vs. joint.** Uniform ranks one parameter at a time show each
  > *marginal* posterior is calibrated; they say nothing about the joint
  > posterior's correlation structure. **Weak informativeness is not
  > miscalibration.** A barely-contracted, correlated $(\beta,\delta)$ posterior
  > is a data limitation, not a sampler error; prior and likelihood can both be
  > correct while the joint posterior stays wide. **What marginal SBC certifies.**
  > It validates the implementation --- likelihood, transforms, sampler --- not
  > the design's power to resolve $(\beta,\delta)$ jointly.

## CR-03 · §7.5.2 · minimum-detectable-effect callout is numbers without a narrative
- **Location:** `_07_application.qmd` L213+: "**Minimum-detectable-effect
  (calibrating the Claude null).** The design could resolve a temperature-on-
  $\alpha$ slope at $P(\text{slope} < 0) \ge 0.95$ only if its magnitude were
  ... $\approx 10$–$12\times$ the observed slope ..."
- **Category:** (d) missing intuition · (e) dense formalism.
- **Problem:** A power calculation delivered as three converging magnitudes with
  no opening framing and no closing takeaway. The reader must reconstruct what
  MDE is and what the ten-fold gap *means* for the Claude null.
- **Recommended fix:** Add an opener and a one-sentence takeaway (honest-reporting
  compliant):
  > Opener: "To read the Claude null correctly we ask the design's *resolution*:
  > what is the smallest temperature-on-$\alpha$ slope this study could reliably
  > detect?"
  > Takeaway (end of callout): "The observed slope sits roughly an order of
  > magnitude below that floor, so the null is best read as *the design cannot
  > resolve an effect this small* --- not as evidence of no effect; even a
  > GPT-4o-sized slope would fall below detection here."

## CR-04 · §8.4 → §7 · model-conditionality caveat is buried past the results
- **Location:** `_08_discussion.qmd` L145–147: "**Interpretive caveat
  (model-conditional reading of $\alpha$).** A low recovered $\alpha$ is *not*,
  in a model-free sense, 'low SEU sensitivity.' The measured $\alpha$ is a
  parameter of a specific likelihood ..."
- **Category:** (f) misplaced caveat · (d) missing frame before interpreting.
- **Problem:** This is the single most important interpretive guardrail, but it
  lands in §8 — after the reader has already formed impressions from the §7.5
  numbers. It should frame the results, not follow them.
- **Recommended fix:** Leave the full statement in §8.4, but surface a compact
  forward-referenced version at the **end of §7.1** (right after the three-layer
  reading guide) as a one–two-sentence callout:
  > A recovered $\alpha$ is a parameter of the §3 softmax-over-SEU likelihood,
  > not a context-free rationality score: the within-design comparisons of §7.5
  > (same model, same prior, same features) are licensed, but an $\alpha$ value
  > read in isolation is not a model-free measure of an agent's rationality. We
  > return to this in §[8.4].
  (Wording mirrors the §8.4 text so the two read as one thread.)

---

# P2 — strong ROI, recommended for the first pass

## CR-05 · §1.1 · Allais / independence axiom / Dutch book cited unglossed
- **Location:** `_01_motivation.qmd` §1.1 ¶3: "the Allais paradox [@allais1953]
  exhibits a pattern of preferences that violates the independence axiom ..."
  and "... a coherence standard on degrees of belief in the Dutch-book tradition
  [@ramsey1926; @definetti1937]".
- **Category:** (c) undefined terms of art.
- **Problem:** Central to decision theory, opaque to statistics/ML readers.
- **Recommended fix:** One clause each, inline: for the independence axiom ---
  "(roughly, that a common component shared by two options should not affect the
  preference between them)"; for Dutch book --- "(a set of degrees of belief is
  faulted if it would sanction a combination of bets guaranteeing a sure loss)".

## CR-06 · §2.6 / §3.1 · $\delta \in \Delta^{K-2}$ vs. "$K-1$ increments" vs. `simplex[K-1]`
- **Location:** `_03_m0_identifiability.qmd` §3.1 (and the §2.6 notation entry):
  "increments $\delta \in \Delta^{K-2}$ ... Stan `simplex[K - 1] delta`".
- **Category:** (a) apparent notation/dimension mismatch.
- **Problem:** $K-2$ superscript vs. "$K-1$ increments" vs. `simplex[K-1]` reads
  as an inconsistency on first pass; the reader can't tell it's the standard
  simplex-dimension convention.
- **Recommended fix:** One clarifying parenthetical at first use: "(the $K-1$
  nonnegative increments sum to one, so they lie on the $(K-2)$-dimensional
  simplex $\Delta^{K-2}$; Stan's `simplex[K-1]` counts the $K-1$ entries, whose
  single sum-to-one constraint removes one degree of freedom)."

## CR-07 · §7.3 · $K=4$ recalibration rationale is buried after the hyperparameters
- **Location:** `_07_application.qmd` §7.3 "Prior calibration anchor," final ¶:
  "The recalibration is needed not because the random-choice baseline changes ...
  but because the larger consequence space ($K=4$) changes the spread of expected
  utilities ..."
- **Category:** (b) inferential leap · (d) buried motivation.
- **Problem:** The "why a different $\alpha$ prior for Ellsberg" answer arrives
  after a dense block of hyperparameter values, by which point the question has
  faded. "Softmax curvature" is also unglossed.
- **Recommended fix:** Move a one-sentence version to the **opening** of the
  calibration-anchor paragraph, phrased as the question first: "Why does the
  $K=4$ Ellsberg task need a different $\alpha$ prior than $K=3$ insurance? Not
  because random choice differs (menu size is the same), but because four
  consequences spread the expected-utility values across alternatives more
  widely, steepening the softmax, so a larger $\alpha$ reaches the same
  prior-implied SEU-max rate." Keep the detailed statement where it is.

## CR-08 · §7.3 · "prior-predictive SEU-maximizer selection rate" and "softmax overflow"
- **Location:** `_07_application.qmd` §7.3: "The calibration target is the prior
  predictive SEU-maximizer selection rate ..." and "... tails that cause softmax
  overflow under the study design."
- **Category:** (c) undefined jargon · (d) unstated consequence.
- **Problem:** "Prior predictive ... rate" and "softmax overflow" are used as if
  self-explanatory; the reader doesn't know how the rate is obtained or why
  overflow blocks inference.
- **Recommended fix:** Gloss the rate at first use ("--- the fraction of
  prior-simulated agents that pick the EU-best available option, computed by
  drawing problems from the design and parameters from the prior"); and gloss
  overflow ("--- extremely large $\alpha$ makes $\exp(\alpha V)$ numerically
  unstable, which corrupts the likelihood").

## CR-09 · §6.2 · reorder the bootstrap explanation; gloss "matched design"
- **Location:** `_06_m1_implementation.qmd` §6.2 callout(s): the bootstrap
  procedure and the separate "why a bootstrap rather than a posterior interval"
  note; and "The matched design fixes one study design across conditions ...".
- **Category:** (e) dense formalism · (c) term of art · (f) ordering.
- **Problem:** The bootstrap is defined procedurally before the reader is told
  what question it answers (across-iteration stability, not posterior
  uncertainty). "Matched design" is used without its payoff.
- **Recommended fix:** (i) Put the "why bootstrap, not a posterior interval"
  sentence *before* the procedure. (ii) One-line gloss at first "matched design":
  "--- the same simulated choices are sliced the same way for every condition, so
  precision differences reflect the estimand, not simulation noise."

## CR-10 · §1.4 · normative-vs-descriptive substitution compressed into one sentence
- **Location:** `_01_motivation.qmd` §1.4 ¶2: "Substituting such a theory is
  therefore more than a change of functional form: the quantity being measured
  changes from conformity to a norm to fit to a behavioral model ...".
- **Category:** (b) inferential leap · (e) dense.
- **Problem:** Three ideas (normative vs. descriptive; functional form vs.
  measured quantity; smoothness for HMC) are layered in quick succession.
- **Recommended fix:** Add one scaffolding sentence up front distinguishing
  *normative* (how one ought to choose) from *descriptive* (how people do), so
  the "quantity being measured changes" clause has a foothold. (The maxmin/HMC
  smoothness point already reads clearly once the norm/description split is
  explicit.)

## CR-11 · §3.4 · "almost unchanged" $(\beta,\delta)$ trade-off stated informally
- **Location:** `_03_m0_identifiability.qmd` §3.4: "a family of $(\beta,\delta)$
  pairs that trades a change in beliefs against a compensating change in utilities
  leaves the implied expected utilities --- and hence the choice probabilities ---
  almost unchanged".
- **Category:** (b) inferential leap · (e) dense.
- **Problem:** "Almost unchanged" is the crux of weak informativeness but is left
  intuitive; a non-Bayesian reader can't tell whether the equivalence is exact.
- **Recommended fix:** Add one sentence grounding it in the product structure:
  "Because each $\eta_r = \bm\psi_r^\top\bm\upsilon$ is a *product* of a
  belief factor and a utility factor, many $(\beta,\delta)$ combinations yield
  nearly the same $\eta$; the data pin down $\eta$ (hence $\alpha$) but resolve
  the two factors composing it only weakly."

## CR-12 · §7.2 · embeddings / PCA / $D=32$ pipeline unglossed
- **Location:** `_07_application.qmd` §7.2 "Feature pipeline": "assessment text
  embedded via `text-embedding-3-small`; pooled-PCA projection to $D=32$."
- **Category:** (c) undefined ML jargon.
- **Problem:** Black boxes for non-ML readers.
- **Recommended fix:** One parenthetical: "(embedding turns each assessment's
  text into a numeric vector; PCA linearly reduces those vectors to their $D=32$
  leading directions of variation --- enough to retain task structure while
  keeping the fit tractable)."

## CR-13 · §7.4 · $\hat R$ / "marginal exceedances," and a second bare "SBC" use
- **Location:** `_07_application.qmd` §7.4 "MCMC diagnostics": "$\hat R > 1.01$
  appears in 4/20 fits as marginal exceedances ... confined to the
  weakly-informed nuisance parameters"; and §7.4 "SBC validates the *(prior,
  likelihood, sampler)* triple ...".
- **Category:** (c) undefined jargon · (d) why it's reassuring.
- **Problem:** $\hat R$ and "marginal exceedance" are unexplained; and the
  reassurance (that exceedances are confined to nuisance parameters, while
  $\alpha$ is clean) is left implicit. The §7.4 "SBC" reuses the term with no
  local reminder.
- **Recommended fix:** Gloss $\hat R$ once ("--- the between-/within-chain
  convergence ratio; values at or below $1.01$ indicate adequate mixing"); state
  the reassurance explicitly ("the exceedances fall on the nuisance parameters
  $(\beta,\delta,\bm\upsilon)$, never on $\alpha$, matching the structural weak
  informativeness of §3.4 rather than signalling a sampler failure"); and add a
  five-word back-reference at the §7.4 "SBC" use ("(SBC, defined in §4.4)").

---

# P3 — fuller treatment (apply if you want the deeper pass)

## CR-14 · §2.2 · three softmax precedents listed without a unifying payoff
- `_02_abstract_model.qmd` §2.2: Luce / McFadden / Boltzmann cited as three
  precedents. Add one sentence stating what they share and why it matters here
  (all give a likelihood exponential in value differences, embedding the two
  interpretable limits in the single $\alpha$; probit does not). *(e)*

## CR-15 · §2.3 · non-monotone non-maximal-probability parenthetical unmotivated
- `_02_abstract_model.qmd` §2.3 Property 1 parenthetical about non-maximal
  probabilities rising before falling; "choice-weighted mean" undefined. Lift it
  out of the parenthetical into one plain sentence and define the mean. *(b)*

## CR-16 · §3.3 · boundary-remark symbols $\eta_y$, $\bar\eta$, "non-regularity"
- `_03_m0_identifiability.qmd` §3.3: define $\eta_y$ (chosen alternative's EU)
  and $\bar\eta$ (menu average) at first use; one clause on why the boundary is
  "non-regular" (information vanishes as $\alpha\to0$). *(a/c)*

## CR-17 · §4.1 · softmax gauge / "softly pins the gauge" unmotivated
- `_04_m0_implementation.qmd` §4.1: add a payoff sentence before the callout
  (choice probabilities depend only on $\beta$ row-contrasts, so absolute levels
  are unidentified) and frame the Gaussian prior as selecting the minimum-norm
  representative rather than fixing a reference row. *(a/d)*

## CR-18 · §5.1 · "belief-formation map"; simplex-tangent-space / affine-spanning
- `_05_m1_identifiability.qmd` §5.1: replace "belief-formation map" with
  "depends only on $\delta$, not on $\beta$"; precede the spanning condition with
  a one-sentence intuition (lotteries must point in enough different directions
  to solve for the $K$ utilities). *(c/e)*

## CR-19 · §5.3 · single-$\alpha$ assumption stated but not motivated
- `_05_m1_identifiability.qmd` §5.3 warning: add one sentence motivating a common
  $\alpha$ across risky and uncertain blocks (same deliberative care), and note
  it is a modeling assumption, testable via block-specific $\alpha$. *(d)*

## CR-20 · §5.5 · identifiability intuition leaps through rank/inversion
- `_05_m1_identifiability.qmd` §5.5: add a concrete $K=3$ sentence (three
  independent lottery directions suffice to solve for three utilities; 15
  lotteries far exceed this) before the linear-functional argument. *(d/e)*

## CR-21 · §6.4.4 · "identifiability ≠ precise estimability" abstract, untethered
- `_06_m1_implementation.qmd` §6.4.4: tie the principle immediately to the
  numbers (the <1% matched $\delta$ CI-width reduction and the null $\alpha$
  gain) so the abstract lesson lands on the concrete result. *(b/d)*

## CR-22 · §7.4 · SBC-reuse remark dense
- `_07_application.qmd` §7.4 "Remark (SBC reuse)": step out the three conditions
  (same model, same prior, same feature pipeline ⇒ reuse) and name which cell is
  spared and why Ellsberg still gets fresh SBC. *(b/e)*

## CR-23 · §7.5.3–4 · raw $\alpha$ medians without a behavioral gloss
- `_07_application.qmd` §7.5.3/§7.5.4: after the median/CI lists, add a
  **qualitative** one-liner on what the decline means behaviorally (as $\alpha$
  falls the model chooses the EU-best option less often, toward random), matching
  the temperature lever. **DECISION NOTE:** attaching specific SEU-max
  *percentages* to given $\alpha$ values would introduce new computed numbers ---
  out of scope under the "no numeric changes" rule unless you authorize computing
  and verifying them against the design. Default: keep the gloss directional. *(d/e)*

## CR-24 · §7.6.2(c) · "SEU is of the wrong form" to test ambiguity — unpacked
- `_07_application.qmd` §7.6.2(c): unpack why SEU+softmax cannot separate
  ambiguity-driven from EU-driven deviation (both look like low $\alpha$), and
  gloss MEU as a model with an explicit ambiguity-attitude parameter. *(b/d)*

---

# Cross-cutting aids (author opted in)

## AID-1 · Reader's-guide paragraph at the end of §1.8
Draft to add after the dependency-chain display in `_01_motivation.qmd` §1.8:
> **How to read this paper.** Readers approaching from statistics or the Bayesian
> workflow may treat §§2–3 and 5 as the modeling setup and concentrate on the
> validation and application (§§4, 6, 7); readers approaching from decision
> theory or formal epistemology will find the conceptual claims in §§1–2 and 8
> and can take the identifiability propositions (§§3, 5) on their statements,
> with proofs in Appendix B. Terms of art from each field are glossed at first
> use for the others.

## AID-2 · Forward-surfaced §8.4 caveat into §7.1
See **CR-04** for the compact §7.1 callout text; the full statement stays at
§8.4 with wording aligned so the two read as one thread.

---

# Out of scope for this pass
New analyses; numeric/claims changes; section reordering; a full glossary box;
figure regeneration. Behavioral result glosses stay qualitative unless
percentages are explicitly authorized (see CR-23).
