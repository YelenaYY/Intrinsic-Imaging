#!/usr/bin/env python3
"""
Author: Yue (Yelena) Yu, Rongfei (Eric) Jin
Purpose: Cleanup script for Intrinsic Imaging project
- Check if every CSV file in logs folder has more than 2 rows
- Check if every subfolder in selected folders has non-empty files
- Clean up invalid files and empty directories
"""

import os
import csv
import shutil
from pathlib import Path
from typing import List, Tuple


def check_csv_files(logs_dir: str) -> List[str]:
    """
    Check CSV files in logs directory for files with only headers (≤2 rows).
    Returns list of files that should be removed.
    """
    invalid_files = []
    
    for root, dirs, files in os.walk(logs_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # Count non-empty lines
                        lines = [line.strip() for line in f.readlines() if line.strip()]
                        if len(lines) <= 1:  # Only header or completely empty
                            invalid_files.append(file_path)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    invalid_files.append(file_path)
    
    return invalid_files


def check_checkpoint_folders(checkpoints_dir: str) -> List[str]:
    """
    Check checkpoint subfolders for empty or incomplete directories.
    Only checks leaf directories (timestamp folders) to avoid removing parent directories
    that contain valid subdirectories.
    Returns list of directories that should be removed.
    """
    empty_dirs = []
    
    # Only check the immediate subdirectories of checkpoints_dir
    # (like composer/, decomposer/, shader/, etc.)
    for main_category in os.listdir(checkpoints_dir):
        category_path = os.path.join(checkpoints_dir, main_category)
        if not os.path.isdir(category_path):
            continue
            
        # Now check each timestamp subdirectory within each category
        for timestamp_dir in os.listdir(category_path):
            timestamp_path = os.path.join(category_path, timestamp_dir)
            if not os.path.isdir(timestamp_path):
                continue
                
            # This is a leaf directory (timestamp folder), check if it's empty or invalid
            try:
                dir_contents = os.listdir(timestamp_path)
                if not dir_contents:
                    # Completely empty directory
                    empty_dirs.append(timestamp_path)
                else:
                    # Check if all files are empty (0 bytes)
                    all_empty = True
                    for item in dir_contents:
                        item_path = os.path.join(timestamp_path, item)
                        if os.path.isfile(item_path) and os.path.getsize(item_path) > 0:
                            all_empty = False
                            break
                    
                    if all_empty:
                        empty_dirs.append(timestamp_path)
            except Exception as e:
                print(f"Error checking {timestamp_path}: {e}")
                # Don't add to empty_dirs if we can't read it - might be a permission issue
    
    return empty_dirs


def clean_files_and_dirs(items_to_remove: List[str], dry_run: bool = True) -> None:
    """
    Remove files and directories. If dry_run is True, only print what would be removed.
    Includes safety checks to prevent removing parent directories with valid content.
    """
    for item in items_to_remove:
        if dry_run:
            print(f"[DRY RUN] Would remove: {item}")
        else:
            try:
                if os.path.isfile(item):
                    os.remove(item)
                    print(f"Removed file: {item}")
                elif os.path.isdir(item):
                    # Safety check: ensure this is a leaf directory (timestamp folder)
                    # and not a parent directory that might contain valid subdirectories
                    parent_dir = os.path.dirname(item)
                    if parent_dir.endswith(('composer', 'decomposer', 'shader', 'shader_variant', 'composer_category', 'composer_shape')):
                        # This is a timestamp folder under a category, safe to remove
                        shutil.rmtree(item)
                        print(f"Removed directory: {item}")
                    else:
                        print(f"SKIPPED: {item} (safety check - not a timestamp folder)")
            except Exception as e:
                print(f"Error removing {item}: {e}")


def main():
    """Main function to run the cleanup process."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(base_dir, "logs")
    checkpoints_dir = os.path.join(base_dir, "checkpoints")
    
    print("=== Intrinsic Imaging Cleanup Script ===\n")
    
    # Check CSV files
    print("1. Checking CSV files in logs directory...")
    invalid_csv_files = check_csv_files(logs_dir)
    print(f"Found {len(invalid_csv_files)} invalid CSV files:")
    for file in invalid_csv_files:
        print(f"  - {file}")
    
    print("\n2. Checking checkpoint directories...")
    empty_checkpoint_dirs = check_checkpoint_folders(checkpoints_dir)
    print(f"Found {len(empty_checkpoint_dirs)} empty checkpoint directories:")
    for dir_path in empty_checkpoint_dirs:
        print(f"  - {dir_path}")
    
    # Summary
    total_items = len(invalid_csv_files) + len(empty_checkpoint_dirs)
    print(f"\n=== SUMMARY ===")
    print(f"Total items to clean: {total_items}")
    print(f"  - Invalid CSV files: {len(invalid_csv_files)}")
    print(f"  - Empty checkpoint directories: {len(empty_checkpoint_dirs)}")
    
    if total_items > 0:
        print(f"\n=== DRY RUN (no files removed) ===")
        all_items = invalid_csv_files + empty_checkpoint_dirs
        clean_files_and_dirs(all_items, dry_run=True)
        
        print(f"\nTo actually remove these files, run:")
        print(f"python clean.py --execute")
    else:
        print("\nNo cleanup needed - all files and directories are valid!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--execute":
        # Actually perform the cleanup
        base_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(base_dir, "logs")
        checkpoints_dir = os.path.join(base_dir, "checkpoints")
        
        print("=== EXECUTING CLEANUP ===\n")
        
        invalid_csv_files = check_csv_files(logs_dir)
        empty_checkpoint_dirs = check_checkpoint_folders(checkpoints_dir)
        
        all_items = invalid_csv_files + empty_checkpoint_dirs
        if all_items:
            print("Items to remove")
            for item in all_items:
                print(f"  - {item}")
            confirm = input(f"Are you sure you want to remove {len(all_items)} items? (yes/no): ")
            if confirm.lower() in ['yes', 'y']:
                clean_files_and_dirs(all_items, dry_run=False)
                print("\nCleanup completed!")
            else:
                print("Cleanup cancelled.")
        else:
            print("No items to clean.")
    else:
        # Default: dry run
        main()
