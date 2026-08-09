"""
Measure real per-model assessment cost, then extrapolate to the full run.

Makes ONE real assessment call per model through the ACTUAL collector path
(same prompts, same client stack, same token budgets), reads the provider's
reported usage, and prices the remaining work from it.

Exists because the reasoning arms are the cost unknown and estimating them was
how Phase C's forecast went wrong by 1.8x. In particular
`claude-sonnet-4-5-thinking` carries budget_tokens=4096 billed as OUTPUT, which
is a ceiling, not a measurement -- actual thinking usage has to be observed.

~6 calls, a few cents.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from applications.seu_sensitivity_study import pools as pools_module
from applications.seu_sensitivity_study import prompts as prompts_module
from applications.seu_sensitivity_study.assessment_collection import AssessmentCollector
from applications.seu_sensitivity_study.client import build_client
from applications.seu_sensitivity_study.config import (
    MODELS,
    SEUSensitivityStudyConfig,
)
from applications.seu_sensitivity_study.llm_extensions import pricing_for

#: Items per pool in the full run (from the dry run).
POOL_ITEMS = {"insurance": 30, "venture": 60, "hiring": 60}
#: Already collected, so excluded from the remaining-cost figure.
ALREADY_DONE = {("insurance", "gpt-4o"), ("insurance", "claude-sonnet-4-5")}


def probe_model(config: Any, spec: Any, pool_id: str) -> Dict[str, Any]:
    pool = pools_module.load_pool(pool_id)
    one_item_pool = dict(pool)
    one_item_pool["items"] = [pool["items"][0]]

    job = next(
        j for j in config.assessment_jobs().values()
        if j.model_name == spec.name and j.pool_id == pool_id
    )
    client = build_client(job, cache_dir=None, max_retries=2, retry_delay=2.0)

    with tempfile.TemporaryDirectory() as tmp:
        payload = AssessmentCollector(
            pool=one_item_pool,
            prompt_sets=prompts_module.load_prompt_sets(pool_id),
            llm_client=client,
            model_name=spec.name,
            max_tokens=config.max_assessment_tokens,
            temperature=job.temperature,
        ).collect(checkpoint_path=Path(tmp) / "probe.json")

    usage = client.get_usage_summary()
    return {
        "model": spec.name,
        "endpoint": spec.endpoint,
        "tier": spec.tier,
        "input_tokens": usage.get("total_input_tokens", 0),
        "output_tokens": usage.get("total_output_tokens", 0),
        "parsed": bool(payload["assessments"][0]["parse_ok"]),
        "pricing": pricing_for(spec.endpoint, spec.provider),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default="insurance")
    args = parser.parse_args(argv)

    dotenv.load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    config = SEUSensitivityStudyConfig()

    rows = []
    for spec in MODELS:
        try:
            rows.append(probe_model(config, spec, args.pool))
        except Exception as exc:  # pragma: no cover - probe is best effort
            print(f"  {spec.name:<28} FAILED {type(exc).__name__}: {str(exc)[:90]}")

    print("=" * 92)
    print(f"Per-call assessment cost, MEASURED on one {args.pool} item")
    print("=" * 92)
    print(f"{'model':<28}{'tier':<11}{'in':>7}{'out':>8}{'$/call':>10}{'parsed':>8}")
    for r in rows:
        cost = (
            r["input_tokens"] / 1e6 * r["pricing"]["input"]
            + r["output_tokens"] / 1e6 * r["pricing"]["output"]
        )
        r["cost_per_call"] = cost
        print(
            f"{r['model']:<28}{r['tier']:<11}{r['input_tokens']:>7}"
            f"{r['output_tokens']:>8}{cost:>10.4f}{str(r['parsed']):>8}"
        )

    # -- Extrapolate. Insurance items are ~293 chars; venture/hiring ~510, so
    # input scales roughly 1.4x there while output is task-bound and assumed
    # flat. Stated as an assumption, not hidden in the arithmetic.
    input_scale = {"insurance": 1.0, "venture": 1.4, "hiring": 1.4}

    print("\n" + "-" * 92)
    print("Projected cost of the REMAINING assessment work")
    print("  assumption: input tokens scale with item text length "
          f"({input_scale}); output assumed flat")
    total = 0.0
    for pool_id, n_items in POOL_ITEMS.items():
        pool_cost = 0.0
        for r in rows:
            if (pool_id, r["model"]) in ALREADY_DONE:
                continue
            scaled_in = r["input_tokens"] * input_scale[pool_id]
            per_call = (
                scaled_in / 1e6 * r["pricing"]["input"]
                + r["output_tokens"] / 1e6 * r["pricing"]["output"]
            )
            pool_cost += per_call * n_items
        total += pool_cost
        print(f"  {pool_id:<12}{pool_cost:>10.2f}")
    print(f"  {'TOTAL':<12}{total:>10.2f}   (840 calls)")
    print("\n  NB: probe calls above are real and already spent (~6 calls).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
