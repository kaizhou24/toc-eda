#!/usr/bin/env python3
"""
rename_circuit_files.py

Script to rename circuit files from the current confusing naming convention to:
XX_thought_YY_variation.json (where XX = 00-04, YY = 00-04)
"""

import os
import re
from pathlib import Path


def rename_circuit_files(raw_dir: str = "data/raw/test2"):
    """Rename circuit files to cleaner naming convention."""
    raw_path = Path(raw_dir)
    
    if not raw_path.exists():
        print(f"Directory {raw_dir} does not exist")
        return
    
    # Get all current circuit files
    circuit_files = list(raw_path.glob("*_th_thought_*_th_variation.png.json"))
    
    if not circuit_files:
        print("No circuit files found with current naming pattern")
        return
    
    print(f"Found {len(circuit_files)} circuit files to rename")
    
    # Parse current files and group by thought
    thought_groups = {}
    
    for file_path in circuit_files:
        filename = file_path.name
        
        # Parse: {thought}_th_thought_{variation}_th_variation.png.json
        match = re.match(r'(\d+)_th_thought_(\d+)_th_variation\.png\.json', filename)
        
        if match:
            thought_num = int(match.group(1))
            variation_num = int(match.group(2))
            
            if thought_num not in thought_groups:
                thought_groups[thought_num] = []
            
            thought_groups[thought_num].append((variation_num, file_path))
    
    print(f"Grouped files by thought: {sorted(thought_groups.keys())}")
    
    # Rename files with new convention
    rename_count = 0
    
    for thought_num in sorted(thought_groups.keys()):
        variations = sorted(thought_groups[thought_num], key=lambda x: x[0])
        
        for var_idx, (original_var_num, file_path) in enumerate(variations):
            # New naming: XX_thought_YY_variation.json
            new_name = f"{thought_num:02d}_thought_{var_idx:02d}_variation.json"
            new_path = file_path.parent / new_name
            
            print(f"Renaming: {file_path.name} -> {new_name}")
            
            try:
                file_path.rename(new_path)
                rename_count += 1
            except Exception as e:
                print(f"ERROR renaming {file_path.name}: {e}")
    
    print(f"\nSuccessfully renamed {rename_count} files")
    
    # Show final structure
    print(f"\nFinal file structure:")
    new_files = sorted(raw_path.glob("*_thought_*_variation.json"))
    for file_path in new_files:
        size_mb = file_path.stat().st_size / (1024*1024)
        print(f"  {file_path.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    rename_circuit_files()