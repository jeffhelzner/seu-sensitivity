# Collection Recovery Implementation Approval

Date: 2026-10-10

## Authorization Boundary

The user approved implementation, focused offline tests, and a configured
campaign ceiling of **$125.07**. This note records that later approval without
rewriting the reconciliation plan, preregistration, historical budget snapshots,
or frozen source-bound evidence.

No production transition, ledger/checkpoint mutation, provider request,
submission, preflight, commit, or push is authorized by this implementation.
The checked-in configuration retains `batch_wave_id: null`, an empty
`batch_wave_cell_ids`, and no replacement authorization. No scientific request
parameters, prompts, models, menus, token caps, or estimation rules changed.

## Exact Scope

- Campaign: this package's `results/production_20261010`, not a new root.
- Cell: `gpt_4o_seu_maximizing_venture`.
- Prior intent: `2b2fec7efa3e4968b196bf4ece08efd2`.
- Original wave: `production-20261010-wave01`.
- Request SHA-256: `b0d4403aeed60325b82d467071f7320e205e0b08e7ee3c176f63efbaf643ccb1`.
- Requests: 280; OpenAI `gpt-4o`.
- Prior and replacement reservations: **$2.2945375 each**, both retained.
- Full-campaign accounting including replacement: **$125.06745420158731**.
- Current actual ledger remains two intents totaling **$4.546225**.

The mechanism pins the actual original state, reservation, stage manifest, and
entire ledger snapshot digests. It permits one new 32-character lowercase hex
intent under one new path-safe wave, explicitly supplied in a separately
reviewed authorization. It does not infer permission from remaining budget or
from an ambiguous/failed state. The completed neutral-venture cell is not eligible.

## Future Offline Transition API

The API is `batch_recovery.prepare_replacement_offline(runner,
reconciliation_path=Path(...))`. It is **not invoked by normal collection or
preflight**, and was not executed against production during implementation.
It constructs no SDK client and performs no network access. No generic retry
command was added.

Before any future invocation, obtain separate permission for reconciliation and
the transition itself. A reviewed runtime configuration must select the existing
campaign root, Batch mode, ceiling 125.07, a new wave, and an explicit allowlist.
The intended continuation includes only the 34 unattempted cells plus this one
replacement, never the completed cell.

Set `batch_replacement_authorization` to the exact dictionary accepted by
`validate_authorization`: fixed schema version 1, cell, prior intent/wave,
provider/model, request hash/count, reservation and ceiling; pinned canonical
campaign root and prior ledger/state/reservation/manifest SHA-256 values; plus
the new `submission_id`, `wave_id`, nonempty `operator_note`, and the fresh
`reconciliation_sha256`. Constants in `batch_recovery.py` supply the pinned
values. Do not add an authorization to the default configuration.

Reconciliation JSON must bind `campaign_root`, `provider`, `cell_id`,
`submission_id` (the prior intent), `request_hash`, `request_count`,
`prior_state_sha256`, and `prior_reservation_sha256`. It must contain schema
version 1, timezone-aware `checked_at` no more than one hour before transition,
`read_only: true`, `pagination_complete: true`, empty `matching_batch_ids` and
`conflicts`, and nonempty `evidence_files` mapping ordered relative filenames
to SHA-256 digests. Use zero-padded filenames so JSON key sorting preserves
page order. Each page is an envelope containing `method: "GET"`,
`path: "/v1/batches"`, `request_after` (null on the first page), and `response`
containing the original provider `data`, `has_more`, and pagination `last_id`.
The validator follows the complete cursor chain and rejects any batch naming
the prior intent or target cell, even when the summary claims zero matches.
On a match or conflict, stop and attach/retrieve; never replace automatically.

The original reconciliation snapshot is historical point-in-time evidence, not
a fresh certificate in this schema. Preserve it. Obtain new evidence only after
separate authorization. Hashes verify local consistency, not provider
authenticity or proof that no upload artifact exists.

## Persistence And Failure Semantics

The canonical original-state lock is the stable per-cell lock for both attempts.
Transition takes this lock and then the existing ledger lock, both nonblocking.
Runner path selection resolves the active intent under the same cell lock.
`ProviderBatchClient.process` independently reacquires it and revalidates the
active intent and archive before reconciliation, reservation, submission,
retrieval, or state writes, including direct callers using campaign paths.
A path selected before transition is rejected as superseded after transition;
while either attempt is processing, competing processing and transition fail
on lock contention. The cell lock remains held throughout provider processing.
Replacement processing additionally takes its own state-file lock; original
processing does not nest a second acquisition of the nonreentrant cell lock.
Direct replacement processing without a submission-reservation callback fails
closed; the runner callback retains the exact authorization and preflight checks.

Transition requires the unchanged original ledger snapshot, unchanged
ambiguous state without a batch ID/results/usage, no choice artifact/checkpoint,
the original hash-verified staged request bodies, and exactly matching current
liability. It refuses a used wave, consumed authorization, or insufficient budget.
Perform transition before reserving any further campaign cells; otherwise the
snapshot guard intentionally stops and requires renewed review.

It creates `_checkpoints/venture/batches/attempts/<new-intent>/`, retaining raw
original state, ledger, manifest, requests, reservation, authorization, and fresh
reconciliation evidence. Evidence files are read-only; only the new `state.json`
is mutable. The original ambiguous state file remains byte-identical in place.
After archives and the new reserving state are durable, the API appends and
fsyncs the new linked reservation to the same ledger. The old ledger bytes must
remain its exact prefix. That append consumes the authorization once.

An interrupted transition may leave an orphan directory. It is not deleted,
reset, overwritten, or automatically completed. Missing/tampered new state or
evidence blocks collection. This deliberately favors manual reconciliation over
automatic recovery of a partially written transaction.

Normal collection routes to the prepared state only when the ledger contains
the valid linked attempt. Reservation rechecks exact config authorization,
archive bindings, intent state, and fresh production-stage guards. An ambiguous
new submission reconciles by its own intent and never creates another batch.
Both attempts remain visible in budget reporting; missing usage is unresolved,
not zero. Completed reservations are never released or replaced with observed cost.

## Source Bindings And Next Approvals

The original stage remains frozen and source-bound to its original revision.
Current `config.py`, `study_runner.py`, `batch_client.py`, the production YAML,
and the new `batch_recovery.py` require a new clean-revision stage before future
submission. Tests verify source tampering still blocks the reservation boundary.
Do not regenerate prior-predictive, recovery, offset, or display evidence merely
to make their source digests match this implementation. In particular, realized
recovery verification and offset evidence include `config.py`; those historical
bindings describe their original source, not the amended current source.

Separate approval is still required for commit, fresh reconciliation and offline
transition, new single-use production preflight, and continuation launch. The
one-hour freshness condition applies to the transition; this API does not claim
to establish provider absence at a later launch time. Review evidence again if
launch is delayed. Any new ambiguity or integrity failure stops the workflow.

## Implementation Verification

The focused offline regression command, executed from the repository root:

```sh
/Users/jeffhelzner/miniforge3/envs/seu-sensitivity/bin/python -m pytest applications/seu_sensitivity_study/tests/test_batch_recovery.py applications/seu_sensitivity_study/tests/test_config.py applications/seu_sensitivity_study/tests/test_study_runner.py applications/seu_sensitivity_study/tests/test_client_and_provenance.py -q --tb=short
```

Result after the focused locking repair: **232 passed in 18.31s** (previously
226). The recovery module alone passed **88 tests in 12.18s**. The deterministic
stale-path regression first failed against the pre-repair implementation and
then passed after the repair. Recovery transitions run only in temporary synthetic
campaigns, with network connection and SDK construction blocked. Existing
preflight tests use temporary synthetic inputs; no production preflight ran.
The recovery fixture substitutes a synthetic request hash and snapshot digests;
separate read-only checks verified the real production request hash, original
liability, and pinned state/ledger/reservation/stage digests. Full-campaign
headroom is tested using the recorded original campaign liability, not a new
production render or evidence-generation run.

Coverage includes wrong/absent authorization, reused intent/wave, snapshot and
archive tampering, incomplete/conflicting/stale reconciliation, paginated raw
evidence, malformed and truncated ledgers, retained ledger-prefix validation,
lock contention, one-use consumption, budget overflow, completed refusal,
submission-source guards, missing-state refusal, and idempotent/ambiguous resume.
Six added cases cover a paused stale-path collector across transition, original
reconciliation excluding transition and routing, replacement reservation,
submission and retrieval excluding both competing attempts, and direct
replacement calls without a reserver. Thread coordination uses events rather
than sleeps, and workers are joined. Existing archive-tampering and missing-state
tests also exercise direct `process` calls, without relying on runner routing.
Editor diagnostics and Pylance syntax validation reported no new errors.