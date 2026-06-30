# Changelog

All notable changes to the methodological paper and its supporting materials.

The paper is a **living working paper** with the following release policy:

- **v1 (frozen):** deposited on arXiv (and mirrored to PhilArchive) to provide a
  stable, citable identifier. Frozen so the planned alignment-study companion
  paper can forward-reference a fixed version.
- **v2 (and later):** revisions after the alignment-study companion lands;
  forward references tightened, errata folded in.

Each archive deposit pins the supporting repository to a specific commit/tag.

## [Unreleased]

### Changed
- **Numeric audit (2026-06-30, prompted by the β–δ sign fix): full ledger
  cross-check; one reporting error corrected in §7.5.2.** Cross-checked all 17
  ledger claims (C1–C16, C8b) both ledger↔committed-source and body↔ledger;
  15/17 matched. Fixes: (1) **C10 / §7.5.2** Claude×insurance per-condition
  α values were the posterior **means** but labelled "medians" — replaced
  {74,55,77,74,57} with the true posterior **medians** {71,53,73,71,55} from
  `claude_insurance_study/data/primary_analysis.json` (the −3.6 slope is correct,
  the report-level estimator matching `report11_cross_llm_results.json`).
  (2) **§7.5.1** dropped the inconsistent "draw-wise" label on the GPT-4o
  report-level slope (−31 is the OLS-through-medians estimator; the draw-wise
  population-OLS value is −24.6, the superseded `primary_analysis.json` slope
  field) — number unchanged. (3) **C15** ledger range max corrected 0.656→0.657
  (source 0.6565 rounds up; paper body "[0.32, 0.66]" unaffected). All other
  claims verified correct.
- **Author comment on draft PDF (2026-06-30): β–δ error-correlation claim
  de-signed in §3.4, §4.3, §8.3, App B.2.** Removed "visible negative Pearson
  correlation" / "pronounced negative β–δ error correlation" wording. The
  committed pooled value (`reports/_freeze/foundations/04_parameter_recovery/`
  `fig-beta-delta-correlation`) is r = +0.387 (positive), and the concentration
  sweep (Report 13) shows the across-iteration sign is true-α-regime dependent
  (+0.37 at α₀=1 → −0.32 at α₀=5) and non-gauge-invariant (β[1,1] carries the
  row-shift gauge). Reframed as *correlated* β–δ posterior-mean errors across
  recovery replicates — an illustrative, non-gauge-invariant signature whose
  *presence* (not direction) matters. `claims_ledger` C4/C7 updated to drop the
  "negative" descriptor. No numeric/figure changes (the source report already
  frames the diagnostic as suggestive/non-gauge-invariant).
- **Author comment on draft PDF (`comments.md`, 2026-06-29): §8.1 Levi
  commitment/performance elaboration.** Framing/clarity edit, no numeric changes;
  `claims_ledger` untouched. Added two paragraphs to §8.1
  (`_08_discussion.qmd`, `#sec-disc-meaning`) after the opening restatement.
  Paragraph 1 ties the restated meaning of "sensitivity to SEU maximization" to
  Isaac Levi's distinction between *commitment* to a standard and *performance*
  relative to it [@levi1980, Chapter 1] — reading $\alpha$ as the agent's tendency
  to *perform* in accordance with an SEU commitment (arithmetic-error analogy;
  flagged as the natural epistemological home for the graded notion, since formal
  epistemologists are among the intended readers). Paragraph 2 adds two honesty
  clarifications echoed from the foundational reports: (i) Levi was *not* a defender
  of SEU as the standard — he generalized it to indeterminate/imprecise
  probabilities and utilities [@levi1986], so his substantive decision theory falls
  outside the real-valued template (cross-ref §[1.4](#sec-procedural)); what is
  borrowed is his *meta-level* commitment/performance distinction, not his account
  of the standard; (ii) the distinction is of interest even granting SEU — the
  Kahneman–Tversky violations [@kahneman1979; @tversky1992] pose performance-failure
  vs. absence-of-commitment, which the framework does not settle but, *under the
  assumption* of an SEU commitment, renders tractable as the strength of that
  commitment. Cite keys `levi1980`/`levi1986`/`kahneman1979`/`tversky1992` already
  in the bibliography (no new entries). Also credited the commitment/performance
  distinction to Levi at its *first* substantive use --- §2.5
  (`_02_abstract_model.qmd`, `#sec-conceptual-payoff`), where the "disposition of
  an SEU-committed agent to act in accordance with its commitments" sentence
  previously appeared without attribution --- with a forward pointer to §8.1; and
  reworded the §8.1 opening from "connects to" to "returns to ... introduced in
  §2.5" so the two passages read as one thread rather than two independent
  introductions. Per a follow-up author note, *removed* the §8.1 qualifying
  sentence "We make this connection explicit because formal epistemologists are
  among the intended readers ... natural epistemological home ..." (it read as
  downplaying the connection). Renders clean (quarto 3-pass → pdflatex;
  paper now 51 pp.; verified §8.1 on PDF pp.29–30 via gs rasterize — citations and
  the §1.4 cross-ref resolve, no overfull). GOTCHA: a prior interrupted render left
  a truncated `paper.aux` (tail `st{51}`) that made quarto's pass-2 fail with
  "Missing \begin{document}"; fix is `rm -f paper.aux paper.out paper.toc` then
  re-render — unrelated to the edit.
- **Author comments on draft PDF (`comments.md`, 2026-06-29): §2.2 log-sum-exp
  wording, removal of the invisible "pilot" narrative across Part 6 (+ §1/§8),
  bootstrap aside, and Wilcoxon explanation.** Framing/clarity edits, no numeric
  changes; `claims_ledger` untouched. (1) §2.2 (`_02_abstract_model.qmd`): dropped
  the imprecise clause "since the log-sum-exp normalizer of @eq-softmax is convex"
  (Equation 1 is the choice probability, not on the log scale); concavity is now
  carried solely by the standard-MNL attribution [@mcfadden1974; @train2009]. The
  index-$\alpha V$ scope note and the interpretable-vs-well-behaved split are kept.
  (2) Removed the now-invisible 30-iteration "pilot" / confirmatory-re-run narrative
  (it lived only in internal technical reports, so readers cannot see the headline
  it overturned). Recast *generically* — the methodological caution about
  under-powered recovery studies is retained but no longer framed as a confession
  about unseen internal work. Edits: §6.2 power callout, §6.4.1 opener +
  "reverses the pilot's attribution" → plain forward-ref, §6.4.2 "smaller than the
  pilot's ≈2%" dropped, §6.4.4(ii)/(iii) generalized, @tbl-matched caption
  (`_06_m1_implementation.qmd`); §1.7(c) contribution preview (`_01_motivation.qmd`)
  and §8.2 (`_08_discussion.qmd`); Appendix D.4 parenthetical (`_appendix_d.qmd`,
  spike filename + seed kept as reproducibility provenance). Internal spike
  filenames/seeds removed from the body (§6.2 callout, @tbl-matched caption); kept
  only in Appendix D/E provenance. §7.2 "earlier pilot" → "preliminary elicitation
  runs". (3) §6.2 "Why a bootstrap rather than a posterior interval" body paragraph
  → shortened `.callout-note` aside. (4) §6.4.1 "slightly *adverse*" → "slightly
  *worse* (it favors `m_0`)"; added a one-time gloss on the Wilcoxon signed-rank
  test (paired, distribution-free test of whether per-iteration paired differences
  are centered at zero; reported alongside the bootstrap CI, no normality assumed),
  covering the later §6.4.2/§8.3 uses. Renders clean (quarto → pdflatex 3-pass,
  0 undefined refs/cites; verified §2.2 p.9, §6.2 callout-note + §6.4.1 pp.19–20).
- **Author comments on draft PDF (`comments.md`): §2.2 concavity/probit and §6.2
  matched-comparison clarity.** Two clarity edits, no numeric changes. (1) §2.2
  "Why softmax" paragraph (`_02_abstract_model.qmd`): the log-likelihood-concavity
  claim is now attributed to the references as a standard multinomial-logit result
  [@mcfadden1974; @train2009] with an inline reason (the log-sum-exp normalizer of
  @eq-softmax is convex) and a scope note (concavity is in the index $\alpha V$ /
  the $\alpha$-conditional surface with $V$ fixed, *not* joint concavity in $\alpha$
  and the value parameters, since $\alpha V$ is bilinear). The previously conflated
  clause is split into two distinct virtues: log-odds linearity + single-scalar
  limit structure make $\alpha$ *interpretable*, while concavity makes it
  *well-behaved to estimate*. The probit sentence is reframed as a standard
  discrete-choice contrast (Gaussian latent errors → no closed-form IIA/log-odds,
  no single sensitivity scalar with the two limits) citing @train2009, not Luce.
  (2) §6.2 "Statistical-power note" (`_06_m1_implementation.qmd`): the dense
  callout is unpacked. The callout now carries only the resolved-power message;
  three new body paragraphs spell out what each reported magnitude is (the §4.3
  recovery loop on the matched design, the paired-condition structure, the
  paired-iteration median + bootstrap 90% CI over the $n=100$ iterations) and *why*
  a bootstrap rather than a posterior interval (it summarizes a property of the
  estimator/design under repeated simulation; the Bayesian fit lives inside each
  iteration; distribution-free because the per-iteration paired differences are
  non-normal). Sign-convention sentence retained as its own paragraph. Renders
  clean (quarto → pdflatex 3-pass, 0 undefined refs/cites; verified §2.2 on PDF p.9,
  §6.2 on pp.19--20). `claims_ledger` untouched.
- **Non-pairwise generality / menu-size extension (author idea).** Added a
  conceptual note that the instrument places no restriction on the number of
  alternatives, unlike the classical pairwise prospect-theory paradigm and its
  large-scale replication. (1) §1.1 motivation --- new sentence after the
  prospect-theory passage observing that those demonstrations and the replication
  [@ruggeri2020] were pairwise "appropriately for their purpose," whereas this
  formulation is menu-size-general, opening the possibility of probing how
  sensitivity varies with menu size (forward-ptr to §8.5). (2) §8.5 Limitations
  and extensions --- new "Menu size and sensitivity" named-but-not-pursued item:
  the model is already general (available set $\mathcal{A}_m$, per-problem
  alternative count in the Stan impl), so $\log\alpha = \gamma_0 + \gamma_1 N_m$
  could estimate the sensitivity/menu-size relationship; a model-agnostic
  cognitive-load hypothesis (larger menu → lower $\alpha$) given as one
  illustration with sign left open; caveat that current designs randomize/pool
  $N_m$, so a clean estimate needs a purpose-built design. New `@article{ruggeri2020}`
  (Nature Human Behaviour, 2020) added to `references_extra.bib`. No new analysis
  and no numeric changes; `claims_ledger` untouched. Renders clean (quarto →
  pdflatex 3-pass, 0 undefined refs/cites; cite resolves to "(Ruggeri et al. 2020)").

### Changed (earlier)
- **Author comments on the draft PDF (`comments.md`).** Four edits: (1) §1.1 ---
  added Ramsey (1926) and de Finetti (1937) citations supporting the
  "coherence standard on degrees of belief in the Dutch-book tradition" passage
  (two `@incollection` entries added to `references_extra.bib`); (2) §1.2 title
  "Why labeled accuracy under-determines decision quality" →
  "Challenges for label-based assessment of decision quality" (anchor kept);
  (3) §1.6 Scope --- "Its primary object is" → "Its focus is", and "…material
  sets the terms of that analysis rather than constituting an independent
  contribution." → "…material frames that analysis."; (4) §5.3 callout title
  parenthetical "(essential; stated as a conditional)" removed (the body already
  states both points). No numeric changes; `claims_ledger` untouched. Renders
  clean (quarto → pdflatex 3-pass, 0 undefined refs/cites).

### Added
- **Publish the rendered PDF for a pre-arXiv feedback round.** A Quarto
  `post-render` hook (`paper/publish-pdf.py`) copies `_output/paper.pdf` to a
  clean, committed path, `seu-sensitivity-methodology.pdf`, one directory up, so
  the PDF can be linked publicly (e.g. from the author homepage) without exposing
  the `_output/` build directory and without a hand-maintained duplicate that
  could go stale — it is regenerated on every render. The root `.gitignore` gains
  a single negation un-ignoring that one path (the `*.pdf` rule and the
  `_output/` build artifact are otherwise untouched). The title block gains a
  `subtitle` banner ("Working draft (pre-v1) --- comments welcome") and an
  auto-stamped `date: today`, so feedback maps to a known render. GitHub
  tag/Zenodo DOI deferred to the arXiv freeze (v1).

### Changed
- **§1.1–§1.3 disambiguate two senses of a "labeled set" (author comment).**
  The draft conflated (A) a key of *correct choices* (which alternative an
  informed judge would select) with (B) a key of *realized outcomes*; (A) is the
  primary external contrast for this paper, while (B) is the locus of the
  good-outcome-vs-good-decision warning. Fixes: §1.1 ¶1 now frames the external
  approach as a correct-choice key (examples recast from outcome form — "did the
  gamble pay off" — to choice form — "would an informed judge have taken the same
  gamble"); §1.1 ¶2 drops the misplaced "because the decision is made before the
  relevant uncertainty resolves" (that is outcome/sense-B reasoning) and instead
  grounds the come-apart in the right place (the consistency standard constrains
  how choices hang together, not which alternative a key marks correct). §1.2
  reorganized so its two difficulties map cleanly to the two senses: (1) a
  correct-choice key is impractical to obtain — insurance would require polling
  subject-matter experts on what they would choose, costly and sensitive to which
  experts; (2) an *outcome* key measures the wrong thing (good outcome ≠ good
  decision; outcomes depend on factors outside the agent's control). Removed the
  muddled "Consistency-based evaluation by this route would still require a
  labeled set of decisions." §1.3 sharpened to state the central point plainly:
  internal-consistency evaluation requires **no** labeled set — neither a
  correct-choice key nor an outcome record. Renders clean (quarto→pdflatex 3-pass,
  no errors/undefined). No numeric changes; claims_ledger untouched.
- **§1.1 overclaim softened + intellectual-history corrected (author comment).**
  (1) Dropped the sentence claiming the internal-consistency assessment is "the
  one we care about" and "the harder of the two to score" (both overclaims); the
  analyst/institution/LLM examples are kept but recast as settings where the
  assessment is *especially relevant* because a correct-choice key is unavailable
  or costly. (2) The "long precedent" paragraph now distinguishes two strands of
  internal-consistency critique instead of lumping both under heuristics-and-
  biases: Allais [@allais1953] = a violation of *expected utility* as a standard
  for decision under *risk* (independence axiom), which Kahneman–Tversky took up
  in prospect theory; the "Linda problem" [@tversky1983] = the *heuristics-and-
  biases* program proper, a violation of the *probability axioms* as a coherence
  standard on degrees of belief (Dutch-book tradition), motivating the
  representativeness heuristic. Closing line notes SEU couples both kinds of
  coherence (on beliefs and on preferences). Renders clean; no numeric changes.
- **§1.7 "What is new" layout + framing fixes, and §7.4.2 lemma reframed (author
  comments).** Four items. (1) *Four-vs-five mismatch:* the lead-in promised a
  "four-part" package but the box listed five entries (a)–(e); the lead-in now
  states explicitly that (a) is the *conceptual* contribution and items (b)–(e)
  are the four-part *methodological* package. (2)+(3) *Layout pathologies:* the
  contribution list was a Quarto simple-callout, whose icon+content render as
  side-by-side `minipage`s — the content `minipage` is **unbreakable**, so the
  ~one-page list could not start mid-page (left ~half of the preceding page blank),
  overflowed the box bottom, and ran (e) past the frame mid-sentence. Removed the
  `.callout-note` wrapper so the (a)–(e) list flows as ordinary prose with bold
  run-in headers; page-7 whitespace, box-top gap, and (e) overflow all resolved.
  No text content changed inside (a)–(e). (4) *§7.4.2 "Lemma (SBC inheritance)":*
  renamed to **"Remark (SBC reuse)"** and prefaced as "a workflow convention rather
  than a proved proposition" — it is not a formal theorem (and its own
  *Qualification* calls the inheritance "an approximation … not an identity"), so
  the "lemma without proof" framing was misleading; added a clause stating the work
  it does (spares a second, expensive calibration for the matched Claude×insurance
  cell). NB: this reverses the earlier plan-doc instruction to "state the
  inheritance argument as an explicit lemma." Renders clean (quarto→pdflatex, 3
  passes, no errors/undefined). No numeric changes; `claims_ledger.md` untouched.
- **Abstract + §1.1–§1.3 reframed per author comments (`my_comments.md`).** Two
  clusters. (1) Abstract: the `m_1` α sentence no longer alludes to an unseen
  internal pilot or a "pre-registered re-run" — it now states positively that at
  matched choice count the β-free risky block yields no α-precision gain, so
  finite-sample α precision is driven by data *quantity*, not *type*. (2) §1.1
  retitled "Two questions about a decision" → **"Two approaches to assessing a
  decision maker"** (anchor `#sec-two-questions` kept) and recast around the right
  axis: *external / label-based* criterion (agreement with a curated key) vs.
  *internal* standard of consistency (coherence among the agent's own choices),
  with "internal" qualifying the standard, not the agent's cognition (behavioral
  stance preserved). Added a historical-precedent paragraph: heuristics-and-biases
  (Allais [@allais1953] → independence-axiom violation motivating prospect theory
  [@kahneman1979; @tversky1992]; Linda/conjunction fallacy [@tversky1983]) assess
  against internal-consistency standards, vs. the nudge tradition [@thaler2008nudge]
  scoring against an externally curated (dietician-labeled) good set. §1.2/§1.3
  wording migrated from "normative" to "consistency-based / internal-consistency"
  for the *contrast*; §1.3 heading → "The need for consistency-based evaluation"
  (anchor `#sec-normative` kept). "normative" retained where it legitimately
  describes SEU-as-benchmark and normative-vs-descriptive (CPT) in §1.4/§2/§7.6.4.
  Pre-registration wording also corrected in §6.4/§6.4.4 to "confirmatory,
  higher-powered re-run". Three bib entries added to `references_extra.bib`
  (`allais1953`, `tversky1983`, `thaler2008nudge`). Renders clean
  (quarto→pdflatex, 3 passes, 0 undefined refs/cites; new cites resolve). No
  numeric changes; `claims_ledger.md` untouched.

### Changed (earlier)
- **Appendix E reproducibility wording softened to future tense.** E.0 now reads
  "at the point of archival deposit, a DOI *will be* minted from that tagged commit
  ... and substituted for the commit pin" (was stated as established policy); E.1's
  closing parenthetical now reads the order-of-magnitude figures "are reported as
  such" and "a timed clean-room reproduction ... *will be* the authoritative source
  and *will be* recorded alongside the archival deposit" (was present tense "is ...
  and is recorded"). No promised artifact (Zenodo DOI, timed budget) is implied to
  exist before deposit. Renders clean. These two remain the only open
  reproducibility items, both requiring author action at deposit time.
- **Referee 36-comment revision complete — all 36 comments now Resolved.**
  Verified the remaining 21 pending comments (C4, C5, C8, C9, C11, C14, C15, C16,
  C17, C19, C20, C21, C22, C23, C25, C27, C31, C32, C34, C35, C36) against the
  current `.qmd` sources. A prior edit pass had already applied the substantive
  reframes and fixes; 20 of 21 required only a Status-field update, and only C21
  needed a small clarification:
  - C21 (§7.3 prior calibration): made explicit that the random-choice baseline is
    set by the *menu size* ($1/N_m$, with $N_m\in\{2,3,4\}$ identical across tasks),
    not the consequence count $K$; recalibration is attributed to $K=4$ widening
    the EU spread / softmax curvature, not to a $1/3\to1/4$ baseline shift.
  - Verified in place (no further edits): C4 (§8.4 α conditional on η; near-linear-
    regime trade-off), C5 (§7.5 cross-task read descriptively, no cross-task
    probability promised), C8 (§7.2 urns enter only via LLM-assessment embedding),
    C9 (D.2 4 chains × 1000 = 4000 draws), C11 (D.4 design external/fresh per iter),
    C14 (§2.3 menu-relative sets $\mathcal{A}_m$), C15 (§7.4 SBC-inheritance
    Qualification: provider-specific $w$ ⇒ approximation, not identity), C16 (§7.3
    Ellsberg Lognormal(3.5,0.75) reported), C17 (§7.6.5 cross-domain replication,
    not stage isolation), C19 (§7.7 reversed-precision argument removed →
    partial-pooling/generalization), C20 (§6.3 per-block rate = prior statement;
    posterior-predictive flagged as the real test), C22/C36 (§7.4 υ reclassified as
    global length-$K$ utility on the weakly-informed side, not a per-trial latent),
    C23 (§6.4 sign-convention note; A→B +27.1% [+15.0, +38.5]), C25 (§8.3 marginal
    insufficiency vs weak informativeness kept apart), C27 (§4.4 "(i)" softened —
    marginal ranks do not certify the joint posterior), C31 (§7.6.2 non-reproduction
    inconclusive at the resolution floor), C32 (E.2 report-level −3.6 vs population-
    OLS −2.9 spelled out, C16 named), C34 (§7.5.2 MDE box ~10–12× either 2.9 or 3.6),
    C35 (§7.1 "withholds support" = inconclusive at resolution, not an established
    zero).
  - Flipped all 21 detailed Status fields plus the Overall Feedback status to
    [Resolved] in
    `feedback-measuring-sensitivity-to-subjective-expected-utili-2026-06-25.md`.
  - Render clean (quarto→pdflatex 3 passes, Output created, 0 undefined refs/cites).
    No numeric changes; `claims_ledger` untouched.
- **Referee 36-comment revision (formal/notation set) verified complete.**
  Confirmed against the current `.qmd` sources that the remaining formal comments
  are resolved (no further edits required; all were already applied):
  - C10 (δ simplex dimension): §2.6, §3.1, and the notation glossary use
    δ∈Δ^{K-2} (K−1 increments summing to one; matches Stan `simplex[K - 1]`).
  - C12 (β–δ correlation statistic): §3.4 states precisely what is computed — an
    across-recovery-replicate Pearson correlation of the posterior-mean errors of
    representative components β_{1,1} and δ_1 (Report 4), flagged as an
    illustrative non-gauge-invariant signature, not a within-posterior statistic.
  - C33 (α=0 differentiability): §3.3 boundary remark gives ℓ smooth at α=0 with
    ∂_α ℓ|_0 = η_y − η̄, reframes α=0 as non-regularity, log-odds injective on
    [0,∞), genericity not vacuous. Appendix A carries no separate (incorrect)
    differentiability remark.
  - Audit of early comments (all Resolved): C1/C3/C7 (Appendix B.3/B.4 + §5.1/
    §5.5/§5.6), C2 (§7.6.5↔§7.7 assessment-stage consistency), C6 (Theorem A.1 /
    §2.3 Property 1 log-odds + maximizing-set monotonicity), C18 (A.4 scale pair),
    C24 (§1.7(e)), C26 (§2.2 softmax/Luce link), C28 (§1.6), C29 (§4 gauge prior:
    likelihood flat, Gaussian prior not), C30 (§1.7(a)).
  - Status fields updated in
    `feedback-measuring-sensitivity-to-subjective-expected-utili-2026-06-25.md`.
    Render clean (quarto→pdflatex 3 passes, 0 undefined refs/cites). No numeric
    changes; claims_ledger untouched.
  - C13 (insurance consequences vs actions): VERIFIED resolved — §7.2 defines the
    K=3 consequences as investigator-agreement outcomes (neither/one/both),
    matching `applications/claude_insurance_study/results/problems.json`, and
    names the alternatives as the pool items (menu size N_m∈{2,3,4}). The earlier
    action-like "forward/hold/decline" wording is gone.
- **Author review pass on the draft PDF (prose, terminology, and a table-margin
  fix).** Addressed the comments in `my_comments_on_draft.md`:
  - Abstract: clarified the α sentence ("identifiability in principle does not by
    itself reveal what governs its precision at finite n --- which we find to be
    the *quantity* of choice data, not the *type* of choice").
  - §1.2 "natural way" → "standard way".
  - §1.4: dropped the "likelihood structure is best understood" claim (SEU is not
    a statistical theory); kept the widest-benchmark reason and a plain pointer to
    §§3/5.
  - §1.7: deleted the "one-line gloss in non-technical terms" sentence; item (e)
    heading "An honestly reported full-pipeline application" → "An illustrative
    full-pipeline application"; removed "while honestly declining to claim one in
    the other two" from its plain-terms gloss.
  - §2.6: clarified the unclear "only β indeterminacy used in the paper" → "we
    rely on no other invariance of the belief map".
  - Terminology consistency with `@tbl-notation`: "gauge" is reserved for the β
    additive row-shift; the utility-scale fixing is the υ-endpoint convention /
    affine *indeterminacy*. Fixed §3.2, reworked the unclear §3.4 sentence, and
    mirrored it in Appendix B.2.
  - Removed figurative/metaphorical language paper-wide: "load-bearing"
    (→ essential/key, §5, §8, App D, App E), "formal backbone" (→ formal core),
    "lives in" (→ is in), "mantra" (§6.4.4 + §8.2), "the discipline the paper
    preaches", "cautionary tale" heading (→ "a caution"), "in the loop". §6.4.4(iii)
    rewritten to drop the two flagged sentences while keeping the n=30→100 content.
  - §7.6.5 expanded into two paragraphs: what the insurance α estimand actually is
    (a property of the composed assessment-LLM + feature-pipeline + choice-LLM
    system) and how the Ellsberg replication isolates the choice stage.
  - Appendix E.2 per-figure manifest no longer overflows the right margin: added
    `\usepackage[htt]{hyphenat}`, wrapped the table in `\footnotesize` with
    reduced `\tabcolsep`, set explicit `tbl-colwidths`, and trimmed redundant
    report paths (already named in the surrounding prose). Renders clean
    (quarto → pdflatex, 0 undefined refs, no overfull boxes).
  - No numeric/claims changes; `claims_ledger.md` untouched.

- **Removed the weak-identifiability (Fisher/ridge) apparatus; re-grounded the
  (β, δ) claim on the Bayesian workflow.** The only numerically-supported
  "proposition" (Prop 3.2 / Appendix B.2 — the η-Jacobian Fisher block, condition
  number κ, δ Schur-complement ratio, near-flat-direction analysis) and its
  IV/GMM "weak identification" terminological demarcation are gone. The
  substantive story is unchanged but now rests on existing recovery artifacts:
  uncertain choices leave (β, δ) *weakly informed* — wide marginal CIs and a
  negative β–δ error correlation (Report 4; claims-ledger C4) — while α recovers
  cleanly. Edits: §3 (title, §3.4 rewrite, §3.5 softened to an empirical
  observation, §3.6), Appendix B.2 (now a brief workflow note, slot kept so
  B.3/B.4 do not renumber), §1.7 contributions (b)/(d), §6.4.4(i), §8.3 + §8.6,
  §2.5, and the abstract. "Weak identifiability" / "ridge" wording replaced by
  Bayesian-native phrasing paper-wide. Appendix D.0 and E.2 + `figure_manifest.md`
  drop the Fig B.2 / B.2-Fisher rows; `claims_ledger.md` C4 replaced, C14 reworded,
  the "B.2 spike outcome" gate section removed (gating rule + Report-14 gate
  renumbered). Bootstrap CIs and the §6 matched-recovery story (C1–C3, Report 14)
  are untouched — they belong to the "m_1 adds little" result, which stays. The
  b2 spike + results JSON + `B2_RESULTS.md` + `b2_fisher_spike.png` are archived
  under `local/archive/b2_fisher_weak_id_removed_20260623/`.
- **Figure manifest: remaining `_tbd_` rows resolved (E.2 + `figure_manifest.md`).**
  The two standalone-manifest rows that were still `not-started` are now closed:
  - **§7.5.5 2×2 forest plot** of the per-cell global-slope posteriors is
    `computed`. New artifact `spikes/report_2x2_forest_spike.py` →
    `figures/report_2x2_forest.png`, rendering the canonical *report-level* cell
    summaries (claims_ledger C9/C10/C12/C13), so the figure matches @tbl-2x2 by
    construction. The spike also records a population-OLS cross-check
    (`b_i = Cov(T,α_i)/Var(T)` over the committed per-condition draws) in its
    results JSON --- diagnostic only, not plotted: it agrees on P(slope<0) and on
    the two Ellsberg cells, and reflects the insurance-cell slope-convention
    difference already documented in C16. The body retains @tbl-2x2 (the paper's
    no-embedded-figures convention); the forest plot is a tracked reproducible
    artifact. Appendix E.2 gains a matching `Fig (§7.5.5)` row.
  - **§1.8 dependency display** is reclassified from a `not-started` mermaid
    "diagram" to an **authored** LaTeX `\underbrace` dependency chain in the §1.8
    body (no mermaid, no generated figure) --- matching what the section actually
    contains.
- **Figure manifest: pending foundational rows filled (E.2 + `figure_manifest.md`).**
  The three `pending`/`not-started` foundational-report figures --- §4.3 (`m_0`
  α true-vs-estimated recovery scatter), §4.4 (`m_0` SBC), §6.5 (`m_1` SBC) ---
  are now `computed`. Their provenance is the in-script canonical design of
  `reports/foundations/04_parameter_recovery.qmd` and `06_sbc_validation.qmd`
  (generic `parameter_recovery.py` / `sbc.py` drivers), **not** the smoke-scale
  `configs/*recovery*.json` / `configs/*sbc*.json` files the placeholders
  referenced; the Config column now records the inline canonical design
  (recovery `m_0`: M25 K3 D5 R15, 50 iters; SBC `m_0`/`m_1`: N_sbc=999 thin4 1
  chain). Seeds: design np 42; recovery 12345+i sim / 54321+i fit (D.2); SBC
  123+i per draw. E.2 intro prose updated (no rows remain pending).
- **Appendix D.3 SBC seed convention corrected.** D.3 previously claimed SBC
  per-draw seeds follow the D.2 "12345+i / 54321+i simulate/fit" pair; the actual
  `analysis/sbc.py` implementation uses a **single** seed `123 + i` per replicate
  (the `_sbc` program draws the true parameters and simulates its dataset in the
  same pass it fits), with the study design fixed once under NumPy seed 42.
  Corrected to match the code so the manifest's D.3 reference is accurate.


### Added
- **Appendices C/D/E (plan execution-order step 9).** Drafted:
  - **Appendix C** (`sections/_appendix_c.qmd`): full Stan listings for the five
    programs referenced by the body code excerpts --- `m_0.stan` (C.1),
    `m_0_sim.stan` (C.2), `m_1.stan` (C.3), `m_1_sim.stan` (C.4), and the
    calibrated-prior `m_01.stan` given as a one-line prior diff (C.5,
    `Lognormal(3.0,0.75)` vs the `Lognormal(0,1)` default). Notation cross-walk
    to §2.6; the repetitive generated-quantities PPC test-statistic boilerplate
    is shown in full for `m_0` and summarized for `m_1`, with bookkeeping arrays
    elided and a pointer to the pinned `models/`.
  - **Appendix D** (`sections/_appendix_d.qmd`): the **canonical** design
    constants. D.0 foundational design (K=3, D=5, R=15, S=15, M,N∈{25,50}, 2–5
    alts, normal features); D.1 software versions (CmdStan 2.37.0, Python 3.10,
    CmdStanPy ≥1.2, env `seu-sensitivity`); D.2 sampler (4 chains, 1000 warmup /
    2000 sampling, NUTS defaults adapt_delta=0.8 / max_treedepth=10, seeds
    12345+i / 54321+i) + diagnostics policy (R̂≤1.01, bulk/tail ESS, divergence
    fraction; ties to C14); D.3 SBC (N_sbc=999, thin=4, single chain, 20 bins,
    L=N_sbc, ECDF/KS bands); D.4 matched A/B/C/D study (n=100 re-run, pilot 30
    superseded; the four-slice protocol, C1–C3); D.5 application scale (M=100×3
    counterbalanced = 300 choices/condition, 5 temps/cell, R=30, D=32 PCA
    insurance, K=3/4, 20 recovery iters/cell, provider temperature grids,
    calibrated α priors incl. K=4 Ellsberg recalibration).
  - **Appendix E** (`sections/_appendix_e.qmd`): E.0 citation/provenance policy
    (pinned commit + Zenodo DOI single source of truth, reference-not-copy, the
    analysis-script/config table); E.1 compute-budget paragraph (SBC at N_sbc=999
    dominates; matched recovery 400 fits + 20/80 application fits cheaper;
    overnight-scale end-to-end, explicitly flagged as order-of-magnitude
    pending a timed clean-room run); E.2 per-figure manifest table (4 `computed`
    rows: Fig B.2, Fig 6.4, §7.5.5 2×2 table, Fig 7.5.2 MDE; 3 `pending`
    foundational-report figures) plus the C11/C14/C15 spike pointers.
  - `paper.qmd` now includes §§1–8 + Appendices A–E; drafting-status comment
    updated to "all sections drafted." Renders clean (pdfLaTeX), 0 unresolved
    refs (`{??}` count = 0), no undefined citations.

- **Paper draft scaffold + formal core (plan execution-order steps 3–4).**
  `paper/` Quarto project (`_quarto.yml`, `paper.qmd`, `references_extra.bib`,
  `sections/`) compiling to `paper/_output/paper.pdf` (pdfLaTeX; 14 pp. for the
  formal core). Drafted:
  - **§2 Abstract model** (`sections/_02_abstract_model.qmd`): softmax rule, the
    three characterizing properties, SEU specialization, consolidated notation
    table, the υ-endpoint convention and the β (additive row-shift) gauge, and
    the ridge-vs-weak-IV terminological demarcation.
  - **§3 m_0 identifiability** (`sections/_03_m0_identifiability.qmd`): Prop 3.1
    (α from η), Prop 3.2 (weak/ridge identifiability of (β,δ), numerically-
    supported proposition), and the §3.5 choice-probability-preserving
    α-separation argument (per the B.2 spike framing).
  - **§5 m_1 identifiability** (`sections/_05_m1_identifiability.qmd`): two-step
    structural argument (β-free risky EU + lottery diversity), Props 5.1–5.3,
    and the **single-α conditional caveat** (gating item; §5.3.3).
  - **Appendix A** (`sections/_appendix_a.qmd`): full proofs of the three
    softmax properties + scale invariance (Thm A.4).
  - **Appendix B** (`sections/_appendix_b.qmd`): B.1 (α-from-η proof), B.2
    (Fisher-curvature proposition with the spike's κ / Schur-ratio /
    choice-prob-sensitivity numbers), B.3 (δ via affine independence — full
    inversion proof), B.4 (β modulo row-shift gauge).
  - Citations resolve cleanly against `reports/references.bib` +
    `references_extra.bib` (adds Modřák 2023, Ellsberg 2001, Staiger–Stock,
    Stock–Wright–Yogo, Andrews–Cheng). No undefined refs/citations on render.
- **Implementation half (plan execution-order step 5).** Drafted:
  - **§4 m_0 implementation** (`sections/_04_m0_implementation.qmd`): data /
    params / priors (Lognormal(0,1) α, with the §7-prior cross-reference), the
    β-gauge handling, prior-predictive SEU-max-rate, α-recovers-/-βδ-don't,
    and the **marginal-SBC demarcation** as a callout (ledger C5–C7).
  - **§6 m_1 implementation** (`sections/_06_m1_implementation.qmd`): the
    matched A/B/C/D design table, the **n=100 re-run** results (ledger C1–C3),
    the single-α conditional callout (§6.4.1 — the remaining gating item, now
    done), and the corrected §6.4.4 two-phenomena lesson + summary table
    (`@tbl-matched`). The pilot's ≈15% / ≈2% figures are explicitly marked
    superseded throughout.
  - `paper.qmd` now includes §§2–6 + Appendices A–B; renders clean (pdfLaTeX),
    no undefined refs/citations.

- **Application (plan execution-order step 6).** Drafted:
  - **§7 illustrative application** (`sections/_07_application.qmd`): the 2×2
    LLM × task design (GPT-4o / Claude 3.5 × insurance / Ellsberg), the `m_01`
    calibrated-prior fit (ledger C8), application-scale validation (recovery,
    SBC-inheritance lemma, MCMC diagnostics C14, PPC C15), the four-cell results
    (C9–C13) with the 2×2 reading table (`@tbl-2x2`), the Claude-null MDE
    callout (C16) and cross-LLM robustness (C11), the construct-validity
    layering, and the §7.6 negative-scope / §7.7 follow-up subsections. Opens
    with the reading-guide + uniform reporting-template callouts per the plan.
  - `paper.qmd` now includes §§2–7 + Appendices A–B; drafting-status comment
    updated (only §§1, 8 and remaining appendices outstanding).

- **Discussion (plan execution-order step 8).** Drafted:
  - **§8 discussion** (`sections/_08_discussion.qmd`): 8.1 restatement of what
    the instrument measures; 8.2 the identifiability mantra with the δ
    (identifiable-not-estimable) and α (precision-tracks-quantity-not-type)
    sides kept apart plus the n=30→n=100 cautionary instance; 8.3 the
    **marginal-SBC demarcation** callout (carries the `#sec-discussion` label
    that §4.4 and §6.5 forward-reference); 8.4 why α is primary, with the
    model-conditional interpretive caveat; 8.5 limitations/extensions
    (δ-optimal design, single-α / `m_2`, functional form, `h_m01`, companion
    work); 8.6 closing. No new numbers; restates §§3–7 findings.
  - `paper.qmd` now includes §§2–8 + Appendices A–B; drafting-status comment
    updated (only §1 and remaining appendices outstanding). Renders clean
    (pdfLaTeX), no undefined refs/citations.

- **Motivation + abstract polish (plan execution-order step 7).** Drafted:
  - **§1 motivation** (`sections/_01_motivation.qmd`): 1.1 external-vs-procedural
    questions; 1.2 why labeled accuracy under-determines decision quality
    (absent / costly / judgment-dependent / luck-confounded labels); 1.3 the
    case for procedural evaluation; 1.4 SEU as the *stated standard* (conditional
    framing, not a defense of SEU); 1.5 why a graded α-measure over a binary
    verdict; 1.6 scope + consolidated negative-scope flags; 1.7 the four-part
    contribution callout (α-from-η + (β,δ) ridge as numerically-supported
    proposition; in-principle-≠-finite-n demonstration with the n=30→n=100
    reversal; marginal-SBC demarcation; honest full-pipeline application); 1.8
    roadmap with a LaTeX dependency-chain display (mermaid avoided for clean
    pdfLaTeX). No new numbers; all figures cited from later sections.
  - **Abstract polish** (`paper.qmd`): tightened the validation clause and
    replaced the abrupt "We report the findings honestly." with a substantive
    finite-sample-caveats sentence; no substantive content change.
  - `paper.qmd` now includes §§1–8 + Appendices A–B; renders clean (pdfLaTeX),
    no undefined refs/citations.

- `REVIEW.md`: review of the writing plan (risks, improvements, venue/structure/sequencing advice).
- `claims_ledger.md` and `figure_manifest.md` provenance tracking documents.
- Migrated `plan_for_methodological_paper.md` from `local/` into this folder.
- `spikes/b2_fisher_block_spike.py` + `spikes/B2_RESULTS.md` + `figures/b2_fisher_spike.png`: the B.2 Fisher-block spike (gating task 1).
- `spikes/report16_mde_spike.py` + `spikes/report16_mde_results.json` + `figures/report16_mde_power_curve.png`: the Claude-null minimum-detectable-effect / power calculation (gating item C16).
- `spikes/report11_cross_llm_spike.py` + `spikes/report11_cross_llm_results.json`: cross-LLM insurance slope comparison, full-grid vs Claude-grid-restricted (ledger C11).
- `spikes/report1415_diagnostics_ppc_spike.py` + `spikes/report1415_diagnostics_ppc_results.json`: MCMC diagnostics + PPC aggregation across all 20 factorial fits (ledger C14, C15).

### Resolved (gating tasks)
- **B.2 Fisher-block / near-kernel spike (gate 1).** Computed at K=3, D=5, M=50
  over 4 representative draws + α sweep. Findings: gauge-fixed (β,δ) block is
  ill-conditioned (κ ≳ 10³); δ profiled/marginal Fisher ratio 0.01–0.37; flat
  directions preserve choice probabilities to 2–8% of a typical direction.
  Decision: B.2/§3.4 = numerically-supported **proposition** (not theorem);
  §3.5 α-separation phrased as **choice-probability-preserving** (not raw-η);
  no strict (β,δ) invariance group claimed. Ledger row C4 → `computed`.
- **Report-14 re-run at n = 100 (gate 2).** Matched-design A/B/C/D recovery via
  `spikes/report14_rerun_analysis.py` (→ `report14_rerun_results.json`,
  `figures/report14_rerun_matched.png`). **⚠ The 30-iteration pilot did not
  replicate:** central B→C α gain is null (−8.1%, 90% CI [−22.0, +6.8], vs the
  pilot's ≈15%); the A→B control *flips* to a +27.1% improvement (CI [+15.0,
  +38.5]); δ CI-width narrows only ≈0.8% (Wilcoxon p≈1.2e-6) vs the pilot's ≈2%.
  Ledger rows C1–C3 → `computed`; §6.4.4 contribution table flagged for
  reframing. C16 (Claude-null MDE) stays pending — separate §7 power calc.
- **Claude-null minimum-detectable-effect (C16).** Computed directly from the
  saved per-condition posterior α draws (`alpha_draws_T*.npz`, 4 000/condition)
  via the application's own draw-wise OLS slope functional
  (`spikes/report16_mde_spike.py` → `report16_mde_results.json`,
  `figures/report16_mde_power_curve.png`, seed 20260618; recomputed slope
  matches `primary_analysis.json` exactly). **MDE ≈ 36 α-units per unit T** at
  P(slope<0) ≥ 0.95 (Gaussian 36.3 / empirical 34.7 / constant-CV MC 36.0;
  ≈ 41–43 at P ≥ 0.975) — ≈ 12× the observed |slope| (≈ 2.9), ≈ 0.53 of
  grand-mean α. The null is *inconclusive at the achievable resolution*; even a
  GPT-4o-sized slope (≈ 31, full-grid ref) sits below the floor. Ledger row
  C16 → `computed`; §7.5.2 MDE sentence + manifest Fig 7.5.2 added. **This was
  the last computational gating item.**
- **Claims-ledger placeholders C11–C15 transcribed to `computed`.**
  - **C11 (cross-LLM, §7.5.2a).** P(GPT-4o slope < Claude slope) = 0.817 on the
    full grids, **0.824** restricting GPT-4o to Claude's T ≤ 1.0 — the
    directional LLM contrast is robust to the unequal-grid confound
    (`spikes/report11_cross_llm_spike.py`; full-grid recomputation matches
    ledger C9 ≈ −31 and the factorial report's ~0.82).
  - **C12 / C13 (Ellsberg cells, §7.5.3–7.5.4).** GPT-4o × Ellsberg slope median
    −38.4 [−72.1, −10.0], P(slope<0) 0.984, P(mono↓) 0.090; Claude × Ellsberg
    slope median −18.8 [−65.3, 24.5], P(slope<0) 0.766, P(mono↓) 0.0085 (no
    pattern) — transcribed from each study's `primary_analysis.json`.
  - **C14 (diagnostics, §7.4.3) — ⚠ plan wording corrected.**
    `spikes/report1415_diagnostics_ppc_spike.py` over all 20 fits: **α never
    R-hat-flagged, ESS satisfactory in every fit**; divergences ≤ 0.15%/fit but
    across 12/20 fits (not just the 2 highest GPT-4o T); R-hat > 1.01 in 4/20
    fits confined to weakly-identified β/δ + per-trial nuisance latents
    (eta/psi/upsilon) in the hard K=4 GPT-4o×Ellsberg fits — which corroborates
    the §3.4/§6.4 β,δ ill-conditioning thesis. §7.4.3 reframed accordingly.
  - **C15 (PPC, §7.4.4).** Holds as stated: 60/60 PPC p-values in [0.317, 0.656]
    ⊂ [0.3, 0.7], mean 0.462.

### Pending (gating tasks before first full draft)
- Per-figure manifest populated as figures are produced.
- Single-α conditional caveat: **§5.3.3 done**, **§6.4.1 done**.
- Pinned commit/tag + Appendix E compute-budget paragraph.
- §1.7(a) contribution claim matched to what B.2 licenses (proposition, not theorem) — pending (§1 not yet drafted).

### Drafting status (plan execution order)
- [x] (1) B.2 spike · [x] (2) Report-14 re-run · [x] (3) §§2,3,5 formal core ·
  [x] (4) Appendix B alongside §§3,5 · [x] (5) §§4,6 implementation ·
  [ ] (6) §7 application · [ ] (7) §1 + abstract · [ ] (8) §8 + references.
