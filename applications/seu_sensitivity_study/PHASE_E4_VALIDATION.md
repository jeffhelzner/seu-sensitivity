# Phase E4: Go/No-Go Validation

Date: 2026-09-06

## Frozen-design checks

The explicit production configuration validates with 36 cells across venture
and hiring, 140 menus per pool, two presentations, and 10,080 expected choice
calls. The zero-cost pipeline dry run independently reports 5,040 calls per
pool and 280 observations per cell.

Both retained pools pass under the frozen embedding-axis thresholds and
variant-D recipes:

| Pool | Gate | Worst cross-size eta-gap difference | Cross-pool difference | Threshold |
|---|---|---:|---:|---:|
| Venture | pass | 0.0906 | 0.0601 | 0.25 |
| Hiring | pass | 0.0484 | 0.0601 | 0.25 |

The first E4 validation attempt reported a false venture failure because the
pool-major runner compared venture's new report to hiring's stale pre-freeze
report. Hiring then compared against venture's new report and passed. The
runner now completes every selected pool through validation, refreshes all
selected validation reports after sibling reports are current, and only then
runs later phases. A regression test locks this ordering. With that fix, both
reports independently contain the same 0.0601 cross-pool difference and pass.

The full application test suite passes with 453 tests.

## Batch and usage-persistence gate

The approved production mode is provider Batch collection with a $31 ceiling.
The current client layer only implements synchronous `generate` calls. It
tracks aggregate input/output tokens in memory and writes usage into the
phase-local run summary, but that summary can be overwritten by a later phase.
There is no provider Batch submission/retrieval path and no immutable,
append-only usage artifact.

Therefore the required provider-specific Batch dry run cannot yet be executed.
Implementing Batch is a collection-architecture change, not a flag on the
validated synchronous client, and should receive mocked contract tests before
any paid probe.

## Verdict

**NO-GO for production collection.** The scientific design, frozen gates,
offline call count, and application tests pass. The only blocking condition is
the approved collection mode: Batch submission plus immutable token-usage
persistence must be implemented and tested, followed by a small paid
provider-specific Batch probe. That probe requires a separate API-spend
authorization under the preregistration.

The synchronous path is technically validated, but using it would invoke the
$61 fallback and also requires separate approval. No production calls have
been launched.