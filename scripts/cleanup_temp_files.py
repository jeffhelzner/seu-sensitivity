#!/usr/bin/env python3
"""
Cleanup script for temporary files generated during SEU sensitivity analyses.

This script removes temporary files created by:
- Parameter recovery analyses (param_recovery_*)
- SBC analyses (sbc_*)  
- Prior predictive analyses (prior_pred_*)
- Quarto rendering sessions (quarto-session*)

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
    
    temp_dirs = get_temp_dirs()
    
    if not temp_dirs:
        print("No temporary files found to clean up.")
        return
    
    total_size = 0
    
    print(f"{'[DRY RUN] ' if not args.execute else ''}Found {len(temp_dirs)} temporary directories:\n")
    
    for temp_dir in sorted(temp_dirs):
        size = get_size(temp_dir)
        total_size += size
        
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
    
    print(f"\n{'Removed' if args.execute else 'Would remove'}: {len(temp_dirs)} directories, {format_size(total_size)} total")
    
    if not args.execute:
        print("\nRun with --execute to actually delete these files.")


if __name__ == "__main__":
    main()
