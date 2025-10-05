#!/usr/bin/env python3
"""
Streamlined Pipeline for Transformer-Based English-Assamese Translation
"""

import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def print_banner(text):
    """Simple banner printer"""
    print("\n" + "=" * 60)
    print(f"🚀 {text}")
    print("=" * 60)

def check_environment():
    """Check if required dependencies are available"""
    try:
        import transformers
        import sentencepiece
        import sacrebleu
        import pandas
        print("✅ All dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Run: pip install -r requirements.txt")
        return False

def download_data():
    """Download dataset from HuggingFace"""
    print_banner("DOWNLOADING DATASET")
    try:
        from scripts.download_data import main
        main()
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def prepare_data():
    """Prepare data for transformer training"""
    print_banner("PREPARING DATA")
    try:
        from scripts.prepare_data import main
        main()
        return True
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        return False

def train_transformer():
    """Train the transformer model"""
    print_banner("TRAINING TRANSFORMER")
    try:
        from scripts.train_transformer import main
        main()
        return True
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def interactive_translation():
    """Start interactive translation session"""
    print_banner("INTERACTIVE TRANSLATION")
    try:
        from scripts.interactive_translator import main
        main()
        return True
    except Exception as e:
        print(f"❌ Interactive translation failed: {e}")
        return False

def evaluate_model():
    """Evaluate the trained model"""
    print_banner("EVALUATING MODEL")
    try:
        from src.evaluation.transformer_evaluate import TransformerEvaluator
        from src.training.transformer_trainer import TransformerTrainer
        from src.models.transformer import DirectMTModel
        from src.config import settings
        from transformers import BertTokenizer
        import sentencepiece as spm
        import torch
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizers
        bert_tokenizer = BertTokenizer.from_pretrained(settings.TRANSFORMER_CONFIG['model_name'])
        sp_tokenizer = spm.SentencePieceProcessor()
        sp_tokenizer.load(str(settings.SP_MODEL_PREFIX) + ".model")
        
        # Load model
        model = DirectMTModel(
            bert_model_name=settings.TRANSFORMER_CONFIG['model_name'],
            tgt_vocab_size=sp_tokenizer.get_piece_size()
        ).to(device)
        
        trainer = TransformerTrainer(model, device)
        trainer.load_model(settings.TRANSFORMER_MODEL_SAVE_PATH)
        
        # Evaluate
        evaluator = TransformerEvaluator(trainer, bert_tokenizer, sp_tokenizer, device)
        test_sentences = [
            "hello world",
            "good morning",
            "how are you", 
            "thank you",
            "where is the house"
        ]
        
        print("🧪 Testing on sample sentences:")
        for sentence in test_sentences:
            translation = evaluator.generate_translation(sentence)
            print(f"   '{sentence}' → '{translation}'")
            
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False

def run_complete_pipeline():
    """Run the complete transformer pipeline"""
    print_banner("STARTING COMPLETE TRANSFORMER PIPELINE")
    
    if not check_environment():
        return False
    
    steps = [
        ("Downloading data", download_data),
        ("Preparing data", prepare_data), 
        ("Training transformer", train_transformer),
        ("Evaluating model", evaluate_model)
    ]
    
    for step_name, step_function in steps:
        print(f"\n📋 Step: {step_name}")
        if not step_function():
            print(f"❌ Pipeline failed at: {step_name}")
            return False
    
    print_banner("PIPELINE COMPLETED SUCCESSFULLY!")
    print("\n🎯 Next steps:")
    print("   python run_pipeline.py interactive  - For live translation")
    print("   python run_pipeline.py evaluate     - To test the model")
    return True

def show_help():
    """Show brief help information"""
    print_banner("TRANSFORMER ENGLISH-ASSAMESE TRANSLATION")
    print("A neural machine translation system using Transformer architecture")
    
    print("\n📋 USAGE:")
    print("  python run_pipeline.py              # Run complete pipeline")
    print("  python run_pipeline.py <command>    # Run specific command")
    
    print("\n🔧 COMMANDS:")
    print("  download     - Download dataset from HuggingFace")
    print("  prepare      - Prepare data for training") 
    print("  train        - Train transformer model")
    print("  interactive  - Start interactive translation")
    print("  evaluate     - Evaluate trained model")
    print("  help         - Show this help")
    
    print("\n💡 EXAMPLE:")
    print("  python run_pipeline.py              # Full setup")
    print("  python run_pipeline.py interactive  # Translate sentences")

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'help':
            show_help()
        elif command == 'download':
            download_data()
        elif command == 'prepare':
            prepare_data()
        elif command == 'train':
            train_transformer()
        elif command == 'interactive':
            interactive_translation()
        elif command == 'evaluate':
            evaluate_model()
        else:
            print(f"❌ Unknown command: {command}")
            show_help()
    else:
        # No arguments - run complete pipeline
        if not run_complete_pipeline():
            sys.exit(1)

if __name__ == "__main__":
    main()