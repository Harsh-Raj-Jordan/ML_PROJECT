"""
Preprocess and split the dataset into train/test sets
"""

import json
import random
from src.config import settings

def main():
    """Preprocess and split the dataset"""
    print("🔄 Loading raw data...")
    
    # Check if raw data exists
    if not settings.RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"❌ Raw data not found at {settings.RAW_DATA_PATH}. Run download_data.py first!")
    
    # Load raw data
    with open(settings.RAW_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"📊 Loaded {len(data)} examples")
    
    # Shuffle for random distribution
    random.shuffle(data)
    split_index = int(len(data) * settings.TRAIN_TEST_SPLIT)
    train_data = data[:split_index]
    test_data = data[split_index:]

    # Save splits
    with open(settings.TRAIN_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(settings.TEST_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Preprocessing completed!")
    print(f"   - Training examples: {len(train_data)}")
    print(f"   - Testing examples: {len(test_data)}")
    print(f"   - Total examples: {len(data)}")
    
    # Show sample
    if train_data:
        sample = train_data[0]
        print(f"\n🔍 Sample training example:")
        print(f"   English: {sample['translation'][settings.SRC_LANG]}")
        print(f"   Assamese: {sample['translation'][settings.TGT_LANG]}")

if __name__ == "__main__":
    main()