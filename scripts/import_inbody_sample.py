#!/usr/bin/env python3
"""
Import InBody sample data into the gym tracker database.

Usage:
    python scripts/import_inbody_sample.py
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from gymtracker.repos import repo_factory


def main():
    # Load sample CSV
    csv_path = Path(__file__).parent.parent / "data" / "sample_inbody.csv"
    
    if not csv_path.exists():
        print(f"❌ Sample CSV not found at {csv_path}")
        return 1
    
    print(f"📂 Loading InBody data from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ Found {len(df)} records")
    
    # Initialize repo
    repo = repo_factory()
    
    # Import each record
    imported = 0
    for _, row in df.iterrows():
        payload = row.to_dict()
        # Clean up any NaN values
        payload = {k: (v if pd.notna(v) else None) for k, v in payload.items()}
        
        try:
            repo.append("InBody", payload)
            print(f"  ✓ Imported {payload['id']} ({payload['date']})")
            imported += 1
        except Exception as e:
            print(f"  ✗ Failed to import {payload.get('id', 'unknown')}: {e}")
    
    print(f"\n🎉 Import complete! {imported}/{len(df)} records imported successfully.")
    print("\n💡 Refresh your Shiny app to see the InBody tab populated with data.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
