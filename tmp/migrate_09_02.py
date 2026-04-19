#!/usr/bin/env python3
"""
Temporary script to migrate 09_02 results from old_evaluation_data to results folder
"""
import shutil
from pathlib import Path

# Base paths
OLD_DATA_DIR = Path("old_evaluation_data")
RESULTS_DIR = Path("results")

# Styles to migrate
STYLES = ["cheer", "refl", "sad", "agit"]

def migrate_09_02_results():
    """Migrate 09_02 results to the new structure"""
    
    for style in STYLES:
        # Source files
        ranked_file = OLD_DATA_DIR / f"ranked_09_02_{style}.csv"
        scored_detailed_file = OLD_DATA_DIR / f"scored_09_02_{style}_detailed.csv"
        
        # Check if files exist
        if not ranked_file.exists():
            print(f"⚠️  Skipping {style}: {ranked_file} not found")
            continue
        if not scored_detailed_file.exists():
            print(f"⚠️  Skipping {style}: {scored_detailed_file} not found")
            continue
        
        # Target directory
        target_dir = RESULTS_DIR / f"09_02_{style}"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Target files
        target_ranked = target_dir / "ranked.csv"
        target_scored_detailed = target_dir / "scored_detailed.csv"
        
        # Copy files
        shutil.copy2(ranked_file, target_ranked)
        print(f"✓ Copied {ranked_file} -> {target_ranked}")
        
        shutil.copy2(scored_detailed_file, target_scored_detailed)
        print(f"✓ Copied {scored_detailed_file} -> {target_scored_detailed}")
        
        print(f"✓ Successfully migrated 09_02_{style}\n")

if __name__ == "__main__":
    print("Starting migration of 09_02 results...\n")
    migrate_09_02_results()
    print("Migration complete!")
