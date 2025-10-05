#!/usr/bin/env python3
"""
Prepare data for transformer training - Convert JSON to CSV format
"""

import json
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings

def prepare_data(input_file=None, output_dir=None):
    """Convert JSON data to CSV format for transformer training"""
    print("🔄 Preparing data for transformer training...")
    
    if input_file is None:
        input_file = settings.RAW_DATA_PATH
    if output_dir is None:
        output_dir = settings.PROCESSED_DATA_DIR
    
    # Load JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    rows = []
    for item in data:
        if 'translation' in item:
            rows.append({
                'src': item['translation'].get(settings.SRC_LANG, ''),
                'tgt': item['translation'].get(settings.TGT_LANG, '')
            })
    
    df = pd.DataFrame(rows)
    
    # Remove empty rows
    df = df.dropna()
    df = df[(df['src'].str.len() > 0) & (df['tgt'].str.len() > 0)]
    
    # Split into train/validation
    split_index = int(len(df) * settings.TRAIN_TEST_SPLIT)
    train_df = df[:split_index]
    valid_df = df[split_index:]
    
    # Save as CSV
    train_file = Path(output_dir) / "train.csv"
    valid_file = Path(output_dir) / "valid.csv"
    
    train_df.to_csv(train_file, index=False)
    valid_df.to_csv(valid_file, index=False)
    
    print(f"✅ Data prepared: {len(train_df)} train, {len(valid_df)} validation examples")
    return str(train_file), str(valid_file)

def main():
    """Test data preparation"""
    input_file = settings.RAW_DATA_PATH
    output_dir = settings.PROCESSED_DATA_DIR
    
    if not Path(input_file).exists():
        print(f"❌ Input file not found: {input_file}")
        print("💡 Run: python scripts/download_data.py first")
        return
    
    train_file, valid_file = prepare_data(input_file, output_dir)
    print(f"📁 Train file: {train_file}")
    print(f"📁 Valid file: {valid_file}")

if __name__ == "__main__":
    main()