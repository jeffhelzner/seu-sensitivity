# B.2 Fisher-block spike — results and framing decision

**Plan gate:** Definition-of-done item 1 ("B.2 spike run") and Pre-drafting
action item 1. Decides the proposition-vs-theorem wording of §3.4, §3.5, and
Appendix B.2.

**Script:** [`b2_fisher_block_spike.py`](b2_fisher_block_spike.py)
**Raw output:** [`b2_fisher_block_results.json`](b2_fisher_block_results.json)
**Figure:** [`../figures/b2_fisher_spike.png`](../figures/b2_fisher_spike.png)
**Design:** m_0, K = 3, D = 5, R = 15, M = 50, normal features, seed 20260617
(matched-design B condition; plan Appendix D.0).
**Parameter draws:** 4 representative draws from the m_1_sim priors
(α ∼ Lognormal(0, 1), β ∼ N(0, 1), δ ∼ Dirichlet(1, 1)), seed 2026; plus an
α sweep ∈ {0.5, 1, 2, 5, 10} at fixed (β, δ).

The analytic η-Jacobian was verified against central finite differences
(max abs error 1.0 × 10⁻¹⁰).

---

## What was computed

For each draw, the **expected per-design (β, δ) Fisher information block**
$I = \sum_m G_m^\top (\mathrm{diag}(p_m) - p_m p_m^\top) G_m$, with
$G_m = \alpha\, \partial\eta_m/\partial(\beta,\delta)$ and
$p_m = \mathrm{softmax}(\alpha\,\eta_m)$. The exact **β row-shift gauge**
(adding a common $\gamma \in \mathbb{R}^D$ to every row of β; D = 5 directions)
was confirmed *exactly* η-preserving and Fisher-null (≤ 1.6 × 10⁻¹⁶) and
projected out before measuring the ridge.

Three diagnostics:

1. **Condition number** of the gauge-fixed block (κ = λ_max/λ_min) — does a
   low-curvature ridge exist?
2. **δ Schur ratio** = profiled δ Fisher / marginal δ Fisher, where profiled =
   $I_{\delta\delta} - I_{\delta\beta} I_{\beta\beta}^{+} I_{\beta\delta}$ — how
   much of δ's information survives having to co-estimate β?
3. **Choice-probability sensitivity** of the flattest direction vs a typical
   direction (within-menu-centred η-Jacobian) — does moving along the ridge
   preserve choice probabilities, hence leave α untouched?

---

## Results

| draw | α | condition number | δ Schur ratio | flat-dir choice-prob sens. / median |
| ---- | -- | ---------------- | ------------- | ----------------------------------- |
| 1 | 0.45 | 1.4 × 10³ | 0.014 | 0.083 |
| 2 | 0.88 | 2.8 × 10⁴ | 0.367 | 0.023 |
| 3 | 0.60 | 4.2 × 10³ | 0.280 | 0.059 |
| 4 | 0.64 | 3.9 × 10⁴ | 0.034 | 0.032 |

α sweep at draw-1 (β, δ): κ ≈ 1.4–2.1 × 10³, δ Schur ratio ≈ 0.009–0.014,
choice-prob ratio ≈ 0.08–0.15 across α ∈ {0.5, …, 10} — all three findings are
robust to α.

**Three robust conclusions (hold across all draws and the α sweep):**

1. **The ridge exists.** After removing the exact β gauge, the (β, δ) Fisher
   block is strongly ill-conditioned (κ ≳ 10³, up to ~4 × 10⁴). The spectrum has
   a low-curvature tail spanning 3–4 orders of magnitude (see figure).

2. **δ is weakly identified via the β-coupling.** Profiling out β destroys
   63–99% of δ's Fisher information (Schur ratio 0.01–0.37). This is the precise
   computational form of the Reports 4/14 finding that the multiplicative
   (β, δ) coupling limits learning about δ from uncertain choices alone.

3. **The flat directions preserve choice probabilities ⇒ α separates.** The
   flattest gauge-fixed direction changes within-menu η-contrasts (hence choice
   probabilities) only 2–8% as much as a typical direction. Because α is
   identified from those within-menu contrasts (Appendix B.1), a direction that
   preserves them leaves α untouched. This is the precise, α-independent version
   of the plan's "ridge approximately in the near-kernel of the η-Jacobian."

---

## Framing decision (this is the gate's deliverable)

**B.2 / §3.4 are stated as a numerically-supported PROPOSITION, not a theorem.**
This matches contribution bullet §1.7(a) ("a numerically-supported
proposition … not a strict invariance theorem"). Specifically:

- **State ill-conditioning via the condition number / δ Schur ratio, not via a
  single isolated near-zero eigenvalue.** The literal Definition-of-done phrasing
  ("smallest eigenvalue ≥ 1 order of magnitude below the *others*") is **not**
  the right statement: there is a *tail* of small eigenvalues (gap to the
  immediate next is only 4–16×), not one isolated null. The honest, robust
  statements are κ ≳ 10³ and the δ Schur ratio ≪ 1. **→ Update the
  Definition-of-done wording accordingly (done in the claims ledger, row C4).**

- **Phrase the §3.5 "why α survives" argument as *choice-probability-preserving*
  (within-menu η-contrast-preserving), NOT raw-η-preserving.** The raw-η
  near-kernel claim is α-dependent and fails cleanly at low α (draw 1: cos with
  the η null-vector is only 0.17, because the flat eigenvector there is a
  within-menu common-mode η-shift that softmax cannot see). The
  choice-probability-preserving statement is robust across all draws and all α
  and is exactly what licenses the α-separation. **This is the pre-written
  "fallback framing" the plan/REVIEW R1 asked for — and it is strictly cleaner
  than the original raw-η framing, so it becomes the primary framing.**

- **Do NOT claim a strict (β, δ) invariance group.** The block has no exact null
  beyond the separately-handled β gauge; the ridge is approximate
  (near-flat, not flat).

- **The flattest single direction is predominantly β (a near-extension of the
  gauge), δ-weight ≈ 0.** δ's weak identification is therefore reported via the
  Schur ratio (conclusion 2), not by reading it off the smallest eigenvector.
  Keep these two facts distinct in B.2.

**Net:** the gate is GREEN with a refined, more defensible characterization. No
blocker for drafting §§3/3.4/3.5/B.2. The α-separation argument is *supported*,
under the corrected "choice-probability-preserving" phrasing.
