"""
Dataset loader for English-Assamese parallel corpus
"""

from datasets import load_dataset
from src.config import settings

def load_eng_asm_dataset(num_rows=None):
    """
    Load English-Assamese parallel dataset from BPCC
    
    Args:
        num_rows (int, optional): Number of examples to load. If None, loads all.
        
    Returns:
        Dataset: HuggingFace dataset object
    """
    try:
        # Build split string
        if num_rows:
            split = f"{settings.TGT_LANG}[:{num_rows}]"
        else:
            split = settings.TGT_LANG

        dataset = load_dataset(
            settings.DATASET_NAME,
            name=settings.DATASET_CONFIG,
            split=split
        )
        
        print(f"✅ Loaded {len(dataset)} examples from BPCC dataset")
        return dataset
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        raise

def load_processed_data():
    """
    Load processed train/test data from local files
    """
    import json
    
    if not settings.TRAIN_DATA_PATH.exists() or not settings.TEST_DATA_PATH.exists():
        raise FileNotFoundError("Processed data not found. Run preprocessing first!")
    
    with open(settings.TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open(settings.TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"📖 Loaded {len(train_data)} train and {len(test_data)} test examples")
    return train_data, test_data

def get_dataset_stats(dataset):
    """
    Get statistics about the dataset
    """
    if hasattr(dataset, '__len__'):
        total_examples = len(dataset)
    else:
        total_examples = 'Unknown (streaming)'
    
    # Sample some examples to get stats
    sample_size = min(1000, len(dataset) if hasattr(dataset, '__len__') else 1000)
    sample = dataset[:sample_size] if hasattr(dataset, '__len__') else list(dataset.take(sample_size))
    
    # Handle different dataset formats
    if sample and 'src' in sample[0] and 'tgt' in sample[0]:
        eng_lengths = [len(ex['src'].split()) for ex in sample if 'src' in ex]
        asm_lengths = [len(ex['tgt'].split()) for ex in sample if 'tgt' in ex]
    elif sample and 'translation' in sample[0]:
        eng_lengths = [len(ex['translation'][settings.SRC_LANG].split()) for ex in sample]
        asm_lengths = [len(ex['translation'][settings.TGT_LANG].split()) for ex in sample]
    else:
        eng_lengths, asm_lengths = [], []
    
    stats = {
        'total_examples': total_examples,
        'avg_english_length': sum(eng_lengths) / len(eng_lengths) if eng_lengths else 0,
        'avg_assamese_length': sum(asm_lengths) / len(asm_lengths) if asm_lengths else 0,
        'max_english_length': max(eng_lengths) if eng_lengths else 0,
        'max_assamese_length': max(asm_lengths) if asm_lengths else 0,
    }
    
    return stats

def main():
    """Test the dataset loader"""
    print("🧪 Testing Dataset Loader")
    print("=" * 40)
    
    try:
        # Test loading from HuggingFace
        dataset = load_eng_asm_dataset(num_rows=100)
        stats = get_dataset_stats(dataset)
        
        print("📊 Dataset Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Show a few examples
        print("\n🔍 Sample examples:")
        for i in range(min(3, len(dataset))):
            example = dataset[i]
            print(f"   Example {i + 1}:")
            if 'src' in example and 'tgt' in example:
                print(f"      English: {example['src']}")
                print(f"      Assamese: {example['tgt']}")
                print(f"      Source Lang: {example.get('src_lang', 'N/A')}")
                print(f"      Target Lang: {example.get('tgt_lang', 'N/A')}")
            elif 'translation' in example:
                print(f"      English: {example['translation'][settings.SRC_LANG]}")
                print(f"      Assamese: {example['translation'][settings.TGT_LANG]}")
            print()
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main()