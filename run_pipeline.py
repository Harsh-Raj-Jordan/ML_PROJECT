#!/usr/bin/env python3
"""
Main pipeline runner for Dictionary-Based English-Assamese Translation
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path - FIXED PATH HANDLING
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

def print_banner(text):
    """Simple banner printer"""
    print("\n" + "=" * 60)
    print(f"🎯 {text}")
    print("=" * 60)

def setup_environment():
    """Setup environment and check dependencies"""
    print("🔧 Setting up environment...")
    
    # Check if required directories exist
    required_dirs = ['data/raw', 'data/processed', 'data/dictionary']
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Check if NLTK data is available
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
        print("✅ NLTK data found")
    except LookupError:
        print("⚠️  NLTK punkt data not found. Run: python -c \"import nltk; nltk.download('punkt')\"")
    
    return True

def run_complete_pipeline():
    """Run the entire translation pipeline"""
    print_banner("🚀 STARTING COMPLETE TRANSLATION PIPELINE")
    
    if not setup_environment():
        print("❌ Environment setup failed!")
        return
    
    try:
        # 1. Download Data
        print_banner("STEP 1: Downloading English-Assamese Dataset")
        try:
            from scripts.download_data import main as download_main
            download_main()
        except Exception as e:
            print(f"❌ Download failed: {e}")
            print("⚠️  Continuing with existing data if available...")
        
        # 2. Preprocess Data
        print_banner("STEP 2: Preprocessing Data")
        try:
            from src.data.preprocess import main as preprocess_main
            preprocess_main()
        except Exception as e:
            print(f"❌ Preprocessing failed: {e}")
            return
        
        # 3. Build Dictionary
        print_banner("STEP 3: Building Bilingual Dictionary")
        try:
            from src.data.dictionary_builder import build_dictionary
            dictionary = build_dictionary()
            
            # Check if dictionary was built successfully
            if not dictionary or len(dictionary) < 10:
                print("⚠️  Dictionary is very small. Results may be poor.")
        except Exception as e:
            print(f"❌ Dictionary building failed: {e}")
            return
        
        # 4. Test Baseline Model
        print_banner("STEP 4: Testing Baseline Dictionary Model")
        try:
            from src.models.baseline_dictionary import test_baseline
            test_baseline()
        except Exception as e:
            print(f"⚠️  Baseline test failed: {e}")
        
        # 5. Test Advanced Dictionary Model
        print_banner("STEP 5: Testing Advanced Dictionary Translator")
        try:
            from src.models.dictionary_translator import main as translator_main
            translator_main()
        except Exception as e:
            print(f"⚠️  Translator test failed: {e}")
        
        # 6. Evaluate System
        print_banner("STEP 6: Evaluating Translation Quality")
        try:
            from src.evaluation.evaluate import main as evaluate_main
            evaluate_main()
        except Exception as e:
            print(f"⚠️  Evaluation failed: {e}")
        
        print_banner("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("\n📊 Next steps:")
        print("   - Check evaluation results above")
        print("   - Improve dictionary coverage by downloading more data")
        print("   - Review sample translations for quality")
        print("   - Consider adding external dictionary resources")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def run_individual_step(step_name):
    """Run individual pipeline steps with better error handling"""
    print_banner(f"RUNNING: {step_name.upper()}")
    
    if not setup_environment():
        print("❌ Environment setup failed!")
        return
    
    try:
        if step_name == 'download':
            from scripts.download_data import main
            main()
            
        elif step_name == 'preprocess':
            # Check if raw data exists first
            raw_path = Path("data/raw/eng_asm.json")
            if not raw_path.exists():
                print("❌ Raw data not found. Run 'download' step first.")
                return
            from src.data.preprocess import main
            main()
            
        elif step_name == 'dictionary':
            # Check if processed data exists
            train_path = Path("data/processed/train.json")
            if not train_path.exists():
                print("❌ Training data not found. Run 'preprocess' step first.")
                return
            from src.data.dictionary_builder import build_dictionary
            result = build_dictionary()
            if result and len(result) > 0:
                print(f"✅ Dictionary built with {len(result)} words")
            else:
                print("⚠️  Dictionary is empty or very small")
                
        elif step_name == 'baseline':
            # Check if dictionary exists
            dict_path = Path("data/dictionary/eng_asm_dict.json")
            if not dict_path.exists():
                print("❌ Dictionary not found. Run 'dictionary' step first.")
                return
            from src.models.baseline_dictionary import test_baseline
            test_baseline()
            
        elif step_name == 'translate':
            # Check if dictionary exists
            dict_path = Path("data/dictionary/eng_asm_dict.json")
            if not dict_path.exists():
                print("❌ Dictionary not found. Run 'dictionary' step first.")
                return
            from src.models.dictionary_translator import main
            main()
            
        elif step_name == 'evaluate':
            # Check if dictionary and test data exist
            dict_path = Path("data/dictionary/eng_asm_dict.json")
            test_path = Path("data/processed/test.json")
            if not dict_path.exists():
                print("❌ Dictionary not found. Run 'dictionary' step first.")
                return
            if not test_path.exists():
                print("❌ Test data not found. Run 'preprocess' step first.")
                return
            from src.evaluation.evaluate import main
            main()
            
        else:
            print(f"❌ Unknown step: {step_name}")
            print("\n📋 Available steps:")
            print("   download    - Download dataset from HuggingFace")
            print("   preprocess  - Split data into train/test sets")
            print("   dictionary  - Build bilingual dictionary")
            print("   baseline    - Test baseline dictionary model")
            print("   translate   - Test advanced dictionary translator")
            print("   evaluate    - Evaluate translation quality")
            print("\n💡 Run without arguments for complete pipeline")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Step '{step_name}' failed: {e}")
        import traceback
        traceback.print_exc()

def show_help():
    """Show help information"""
    print_banner("ENGLISH-ASSAMESE TRANSLATION PIPELINE")
    print("A dictionary-based machine translation system for low-resource languages")
    print("\n📋 USAGE:")
    print("  python run_pipeline.py              # Run complete pipeline")
    print("  python run_pipeline.py <step>       # Run individual step")
    print("  python run_pipeline.py help         # Show this help")
    print("\n🔧 STEPS:")
    print("  download    - Download dataset from HuggingFace")
    print("  preprocess  - Split data into train/test sets") 
    print("  dictionary  - Build bilingual dictionary")
    print("  baseline    - Test baseline dictionary model")
    print("  translate   - Test advanced dictionary translator")
    print("  evaluate    - Evaluate translation quality")
    print("\n📁 OUTPUT:")
    print("  data/raw/eng_asm.json          - Raw downloaded data")
    print("  data/processed/train.json      - Training data")
    print("  data/processed/test.json       - Test data")
    print("  data/dictionary/eng_asm_dict.json - Bilingual dictionary")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] in ['help', '-h', '--help']:
            show_help()
        else:
            run_individual_step(sys.argv[1])
    else:
        # Run complete pipeline
        run_complete_pipeline()