#!/usr/bin/env python3
"""
Cleanup script for temporary files generated during SEU sensitivity analyses.

This script removes temporary files created by:
- Parameter recovery analyses (param_recovery_*)
- SBC analyses (sbc_*)  
- Prior predictive analyses (prior_pred_*)
- Quarto rendering sessions (quarto-session*)
- Stan MCMC chain CSVs in fit directories (m_*-*.csv)

Usage:
    python scripts/cleanup_temp_files.py           # Dry run (shows what would be deleted)
    python scripts/cleanup_temp_files.py --execute # Actually delete the files
    python scripts/cleanup_temp_files.py --help    # Show help
"""

import os
import shutil
import argparse
import glob
from pathlib import Path

# Project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Patterns for temporary directories to clean up
TEMP_PATTERNS = [
    "param_recovery_*",
    "param_recovery_m50_*",
    "sbc_*",
    "prior_pred_*",
    "quarto-session*",
    "quarto-preview-*",
]


def get_temp_dirs():
    """Find all temporary directories matching our patterns."""
    temp_dirs = []
    
    # macOS temp folder location
    macos_temp_base = "/private/var/folders/*/*/T"
    
    # Also check standard /tmp
    temp_locations = ["/tmp"] + glob.glob(macos_temp_base)
    
    for temp_loc in temp_locations:
        for pattern in TEMP_PATTERNS:
            full_pattern = os.path.join(temp_loc, pattern)
            matches = glob.glob(full_pattern)
            temp_dirs.extend(matches)
    
    return temp_dirs


def get_stan_chain_csvs():
    """Find Stan MCMC chain CSV files in fit directories.

    These are the large per-chain output files (m_*-YYYYMMDD*.csv) that
    CmdStanPy writes during sampling.  The important results are already
    extracted into alpha_draws.npz, summary.csv, diagnostics.txt, and
    ppc.json, so the raw chain CSVs can be safely removed.
    """
    chain_csvs = []
    fit_dirs = sorted(PROJECT_ROOT.glob("applications/*/results/fit_T*"))
    for fit_dir in fit_dirs:
        csvs = sorted(fit_dir.glob("m_*-*.csv"))
        chain_csvs.extend(csvs)
    return [str(p) for p in chain_csvs]


def get_size(path):
    """Get total size of a directory in bytes."""
    total = 0
    if os.path.isfile(path):
        return os.path.getsize(path)
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total


def format_size(size_bytes):
    """Format size in human-readable form."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def main():
    parser = argparse.ArgumentParser(
        description="Clean up temporary files from SEU sensitivity analyses.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--execute", "-x",
        action="store_true",
        help="Actually delete the files (default is dry run)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true", 
        help="Only show summary, not individual files"
    )
    args = parser.parse_args()
    
    prefix = "[DRY RUN] " if not args.execute else ""
    total_size = 0
    total_items = 0

    # --- System temp directories ---
    temp_dirs = get_temp_dirs()
    if temp_dirs:
        print(f"{prefix}System temp directories ({len(temp_dirs)}):\n")
        for temp_dir in sorted(temp_dirs):
            size = get_size(temp_dir)
            total_size += size
            total_items += 1
            if not args.quiet:
                print(f"  {format_size(size):>10}  {temp_dir}")
            if args.execute:
                try:
                    if os.path.isdir(temp_dir):
                        shutil.rmtree(temp_dir)
                    else:
                        os.remove(temp_dir)
                except Exception as e:
                    print(f"  ERROR removing {temp_dir}: {e}")
        print()

    # --- Stan chain CSVs ---
    chain_csvs = get_stan_chain_csvs()
    if chain_csvs:
        print(f"{prefix}Stan MCMC chain CSVs ({len(chain_csvs)} files):\n")
        for csv_path in chain_csvs:
            size = get_size(csv_path)
            total_size += size
            total_items += 1
            if not args.quiet:
                # Show path relative to project root for readability
                try:
                    rel = os.path.relpath(csv_path, PROJECT_ROOT)
                except ValueError:
                    rel = csv_path
                print(f"  {format_size(size):>10}  {rel}")
            if args.execute:
                try:
                    os.remove(csv_path)
                except Exception as e:
                    print(f"  ERROR removing {csv_path}: {e}")
        print()

    if total_items == 0:
        print("No temporary files found to clean up.")
        return

    print(f"{'Removed' if args.execute else 'Would remove'}: {total_items} items, {format_size(total_size)} total")
    
    if not args.execute:
        print("\nRun with --execute to actually delete these files.")


if __name__ == "__main__":
    main()
