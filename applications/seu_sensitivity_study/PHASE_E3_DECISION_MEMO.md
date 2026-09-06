# Phase E3: Preregistration Decision Memo

Date: 2026-09-06

**Decision, 2026-09-06:** the user approved the complete recommended design
bundle, the `log(1.05)` RQ6 ROPE, Batch collection with a $31 ceiling, and
keeping all commits local. The approved choices are implemented in the tracked
preregistration and frozen configuration.

## Recommended substantive decisions

| Item | Recommendation | Evidence and consequence |
|---|---|---|
| Utility scale | Fix the primary middle utility at 0.50; refit at 0.35 and 0.65 | Removes the validated J=18 ridge without claiming equal spacing is true; six total fits for two pools |
| Eta-gap axis | Keep the embedding axis primary; report the belief axis diagnostically | Preserves the preregistered default; both axes separate venture/hiring perfectly, while insurance improves from AUC 0.472 to 0.870 on the belief axis; the model set defining a belief axis is unresolved |
| RQ4 | Use the cross-pool `sigma_cell` variance component as primary; keep ordering agreement descriptive | Ordering power was 0.294 versus 0.968 for the variance component at three pools; E1 model-effect SDs remain 0.31--0.41 |
| Third estimation pool | Do not author one | G2 uncertainty is not low enough to satisfy the established trigger; an added pool does not rescue ordering agreement and would add design and API risk |
| Insurance | Drop it from production collection | Its embedding R3 is 0.052 versus the 0.30 threshold, so alpha is not identified; a residual diagnostics-only collection adds cost without serving a confirmatory estimand |
| Menu recipes | Adopt variant D for venture and hiring | Two contenders per recipe reduces cross-size eta-gap drift from 0.159 to 0.091 in venture and 0.229 to 0.048 in hiring; current code still implements variant A |
| Menu sizes | Freeze balanced `{2,4,6,8}` | Best validated range; `{2,3,4,6}` remains an approximately 0.80-cost fallback but makes 25% of menus odd-sized and carries an R6 cost |
| Menu counts | Freeze 100 primary-family and 40 matched-family menus per pool | Preserves the validated primary count and current paired-comparator scope: 140 menus and 5,040 calls per pool |

## Recommended rules and disclosures

1. Freeze the current embedding-axis gate thresholds, set
   `provisional=false`, and record `frozen_at`. Re-derive eta-gap thresholds if
   the belief axis is promoted instead.
2. Declare a log-alpha contrast only when its interval excludes zero and its
   magnitude exceeds `log(1.25)`. Keep this rule separate from the RQ6 slope
   rule.
3. Freeze the NA rules at flag above 10% and exclude-with-caveat above 30%,
   applied within cell.
4. State that the OpenAI and Anthropic reasoning contrasts are different
   estimands and are not pooled. The Anthropic flagship and thinking arms share
   the Sonnet 4.5 endpoint.
5. Carry forward verbatim that Phase D power runs used treedepth 10 and were
   truncated; their symmetric comparisons are qualitative. Final J=18 fits use
   treedepth 12. Also disclose that no pinned-model SBC was run and that the
   original free-utility SBC showed mild drift in parameters later implicated
   in the ridge.
6. Put the frozen preregistration at the tracked path
   `applications/seu_sensitivity_study/PREREGISTRATION.md` and commit it with
   the frozen config changes and date.
7. Preserve ordering-agreement output descriptively, but retire its `(k, p0)`
   threshold as a confirmatory decision rule when the variance-component RQ4 is
   adopted.

## Approved values and authorizations

### RQ6 ROPE

The plan requires a separate minimum meaningful change in log alpha per added
alternative, but it never specifies a number. The observed G1 90% interval
widths of 0.135, 0.243, and 0.407 measure recovery resolution; they do not by
themselves define scientific importance. A numeric RQ6 ROPE must therefore be
chosen explicitly rather than reverse-engineered from the smoke result.

Approved value: **`abs(gamma_size) > log(1.05)` per added alternative**.
Across the six-alternative span from menu size 2 to 8, this corresponds to an
alpha ratio of `1.05^6 = 1.34`, close to the 1.25 main-contrast threshold while
remaining interpretable per alternative.

### API mode and ceiling

Approved choice: use provider Batch APIs after an E4 dry run, with a
**$31 choice-collection ceiling**. Keep a **$61 synchronous fallback ceiling**
that requires separate approval before use. The Batch implementation itself
must be validated before its discount is treated as available.

### Push authorization

Keep the current branch local. The blocker diagnosis, pinned-model validation,
recovery, E1 prompt commits, and the E3 freeze must not be pushed without a new
explicit authorization.

## Implementation status

The recipes, thresholds, production pool configuration, decision rules, and
tracked preregistration have been updated. Phase E4 results are recorded in
`applications/seu_sensitivity_study/PHASE_E4_VALIDATION.md`.