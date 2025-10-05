#!/usr/bin/env python3
"""
Download English-Assamese parallel corpus from HuggingFace
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_dataset
from huggingface_hub import login
import json
from src.config import settings

def download_dataset(num_rows=10000):
    """Download the BPCC dataset"""
    print("🔐 Logging into HuggingFace Hub...")
    login()
    
    print(f"⬇️ Downloading English→Assamese dataset ({num_rows} examples)...")
    
    try:
        # Use the correct split format for BPCC dataset
        split = f"{settings.TGT_LANG}[:{num_rows}]"
        
        dataset = load_dataset(
            settings.DATASET_NAME,
            name=settings.DATASET_CONFIG,
            split=split
        )
        print(f"✅ Successfully loaded dataset with {len(dataset)} examples")
        return dataset
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        sys.exit(1)

def process_dataset(dataset):
    """Process and filter the dataset"""
    processed_data = []
    skipped_count = 0
    
    print("🔄 Processing dataset...")
    for i, item in enumerate(dataset):
        try:
            # Check language pair and data quality
            if (item["src_lang"] == settings.SRC_LANG and 
                item["tgt_lang"] == settings.TGT_LANG and
                item["src"].strip() and item["tgt"].strip()):
                
                src_len = len(item["src"].split())
                tgt_len = len(item["tgt"].split())
                
                # Filter by sentence length
                if (settings.MIN_SENTENCE_LENGTH <= src_len <= settings.MAX_SENTENCE_LENGTH and
                    settings.MIN_SENTENCE_LENGTH <= tgt_len <= settings.MAX_SENTENCE_LENGTH):
                    
                    processed_data.append({
                        "id": i,
                        "translation": {
                            settings.SRC_LANG: item["src"].strip(),
                            settings.TGT_LANG: item["tgt"].strip()
                        }
                    })
                else:
                    skipped_count += 1
            else:
                skipped_count += 1
                
        except KeyError as e:
            skipped_count += 1
            
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"   Processed {i + 1} examples...")
    
    print(f"📊 Dataset statistics:")
    print(f"   - Valid examples: {len(processed_data)}")
    print(f"   - Skipped examples: {skipped_count}")
    
    return processed_data

def save_dataset(data):
    """Save processed dataset"""
    with open(settings.RAW_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Raw dataset saved to: {settings.RAW_DATA_PATH}")

def main():
    print("=" * 60)
    print("📥 ENGLISH-ASSAMESE DATA DOWNLOADER")
    print("=" * 60)

    dataset = download_dataset(num_rows=10000)
    processed_data = process_dataset(dataset)
    save_dataset(processed_data)
    
    print("✅ Download completed successfully!")
    if processed_data:
        sample = processed_data[0]
        print(f"\n🔍 Sample translation:")
        print(f"   English: {sample['translation'][settings.SRC_LANG]}")
        print(f"   Assamese: {sample['translation'][settings.TGT_LANG]}")

if __name__ == "__main__":
    main()