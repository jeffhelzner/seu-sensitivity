#!/usr/bin/env python3
"""
Freeze data snapshots for a factorial cell's report.

Copies relevant outputs from the pipeline results directory into the report's
data/ directory so that report .qmd files can load self-contained data.

Usage:
    python scripts/freeze_report_data.py <study_name>

    where <study_name> is the directory name under applications/, e.g.:
        temperature_study
        ellsberg_study
        claude_insurance_study
        gpt4o_ellsberg_study

Examples:
    python scripts/freeze_report_data.py claude_insurance_study
    python scripts/freeze_report_data.py gpt4o_ellsberg_study
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def temp_key(t: float) -> str:
    return f"T{str(t).replace('.', '_')}"


def discover_temperatures(results_dir: Path) -> list[float]:
    """Discover temperatures from fit_T* subdirectories."""
    temps = []
    for fit_dir in sorted(results_dir.glob("fit_T*")):
        tag = fit_dir.name.removeprefix("fit_T")
        temps.append(float(tag.replace("_", ".")))
    return temps


def freeze(study_name: str) -> None:
    app_dir = PROJECT_ROOT / "applications" / study_name
    results_dir = app_dir / "results"
    report_data = PROJECT_ROOT / "reports" / "applications" / study_name / "data"

    if not results_dir.is_dir():
        print(f"Error: {results_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    temperatures = discover_temperatures(results_dir)
    if not temperatures:
        print(f"Error: no fit_T* dirs found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    report_data.mkdir(parents=True, exist_ok=True)
    copied = 0

    print(f"Freezing {study_name} ({len(temperatures)} temperatures: {temperatures})")
    print(f"  Source: {results_dir}")
    print(f"  Dest:   {report_data}\n")

    # Per-temperature files
    for t in temperatures:
        tk = temp_key(t)
        fit_dir = results_dir / f"fit_{tk}"

        for src_name, dst_name in [
            ("alpha_draws.npz", f"alpha_draws_{tk}.npz"),
            ("diagnostics.txt", f"diagnostics_{tk}.txt"),
            ("ppc.json", f"ppc_{tk}.json"),
        ]:
            src = fit_dir / src_name
            dst = report_data / dst_name
            if src.exists():
                shutil.copy2(src, dst)
                print(f"  {dst.name}")
                copied += 1

        # Stan data lives in the top-level results dir
        src = results_dir / f"stan_data_{tk}.json"
        dst = report_data / f"stan_data_{tk}.json"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  {dst.name}")
            copied += 1

    # Aggregate files
    for fname in ["fit_summary.json", "run_summary.json", "primary_analysis.json"]:
        src = results_dir / fname
        dst = report_data / fname
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  {dst.name}")
            copied += 1

    # Study config
    src = app_dir / "configs" / "study_config.yaml"
    dst = report_data / "study_config.yaml"
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  {dst.name}")
        copied += 1

    print(f"\nFroze {copied} files to {report_data}")


def main():
    parser = argparse.ArgumentParser(
        description="Freeze report data for a factorial study cell"
    )
    parser.add_argument(
        "study_name",
        help="Study directory name under applications/",
    )
    args = parser.parse_args()
    freeze(args.study_name)


if __name__ == "__main__":
    main()
