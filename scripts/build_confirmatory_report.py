"""Build the frozen confirmatory report from completed CmdStan chain CSVs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from applications.seu_sensitivity_study.confirmatory_reporting import (
    build_report_from_manifest,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.fit_manifest.read_text())
    report = build_report_from_manifest(manifest)
    write_report(args.output, report)
    print(f"Wrote confirmatory report to {args.output}")


if __name__ == "__main__":
    main()