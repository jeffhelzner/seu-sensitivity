"""
Capability probe for the model set (Phase D blocker triage).

Sends a handful of TINY requests to establish, factually:
  1. whether the configured reasoning-tier treatments actually reach the API
     through the existing client path, and
  2. which Anthropic successor models accept the settings each tier needs.

Cost is a few cents; every request is capped at a small token budget.
Run with --dry to list what it would do without calling anything.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROMPT = "Reply with exactly: OK"

SONNET_CANDIDATES = [
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-6",
    "claude-sonnet-5",
]
HAIKU_CANDIDATES = ["claude-haiku-4-5-20251001"]


def probe_existing_client_path() -> None:
    """Does the configured reasoning treatment survive the client stack?"""
    from applications.seu_sensitivity_study.client import build_client
    from applications.seu_sensitivity_study.config import (
        SEUSensitivityStudyConfig,
        get_model_spec,
    )

    config = SEUSensitivityStudyConfig()
    print("=" * 78)
    print("1. Reasoning-tier treatment through the EXISTING client path")
    print("=" * 78)
    for model_name in ("o3-mini",):
        spec = get_model_spec(model_name)
        print(f"\n[{model_name}] configured request_params={spec.request_params} "
              f"temperature={spec.temperature}")
        job = next(
            j for j in config.assessment_jobs().values() if j.model_name == model_name
        )
        client = build_client(job, cache_dir=None, max_retries=1, retry_delay=0.5)
        try:
            text = client.generate(PROMPT, temperature=job.temperature, max_tokens=16)
            print(f"  RESULT: ok -> {text!r}")
        except Exception as e:
            print(f"  RESULT: FAILED -> {type(e).__name__}: {str(e)[:220]}")


def probe_anthropic_direct() -> None:
    """Which successors accept temperature=0, and which accept thinking?"""
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    print("\n" + "=" * 78)
    print("2. Anthropic successors -- temperature=0.0 (flagship / small tiers)")
    print("=" * 78)
    for model in SONNET_CANDIDATES + HAIKU_CANDIDATES:
        try:
            r = client.messages.create(
                model=model,
                max_tokens=16,
                temperature=0.0,
                messages=[{"role": "user", "content": PROMPT}],
            )
            print(f"  {model:<32} OK   in={r.usage.input_tokens} "
                  f"out={r.usage.output_tokens}")
        except Exception as e:
            print(f"  {model:<32} FAIL {type(e).__name__}: {str(e)[:110]}")

    print("\n" + "=" * 78)
    print("3. Anthropic successors -- extended thinking (reasoning tier)")
    print("=" * 78)
    for model in SONNET_CANDIDATES:
        try:
            r = client.messages.create(
                model=model,
                max_tokens=1300,
                thinking={"type": "enabled", "budget_tokens": 1024},
                messages=[{"role": "user", "content": PROMPT}],
            )
            kinds = [b.type for b in r.content]
            print(f"  {model:<32} OK   blocks={kinds} "
                  f"out={r.usage.output_tokens}")
        except Exception as e:
            print(f"  {model:<32} FAIL {type(e).__name__}: {str(e)[:110]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()

    dotenv.load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    if args.dry:
        print("Would probe: o3-mini via client path; "
              f"{SONNET_CANDIDATES + HAIKU_CANDIDATES} at temperature 0; "
              f"{SONNET_CANDIDATES} with thinking budget 1024.")
        return 0

    probe_existing_client_path()
    probe_anthropic_direct()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
