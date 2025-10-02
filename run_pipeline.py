#!/usr/bin/env python3
"""
Enhanced Pipeline Runner for Dictionary-Based English-Assamese Translation
With session management, quality metrics, and advanced analytics
"""

import sys
import os
import json
import traceback
from pathlib import Path

# Add the project root to Python path - FIXED PATH HANDLING
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

def print_banner(text):
    """Enhanced banner printer"""
    print("\n" + "=" * 60)
    print(f"🎯 {text}")
    print("=" * 60)

def setup_environment():
    """Setup environment and check dependencies"""
    print("🔧 Setting up environment...")
    
    # Check if required directories exist
    required_dirs = [
        'data/raw', 
        'data/processed', 
        'data/dictionary', 
        'results', 
        'logs',
        'results/translation_sessions',
        'experiments'
    ]
    for dir_path in required_dirs:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        if not dir_path.exists():
            print(f"⚠️  Could not create directory: {dir_path}")
    
    # Check if NLTK data is available
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
        print("✅ NLTK data found")
    except LookupError:
        print("⚠️  NLTK punkt data not found. Run: python -c \"import nltk; nltk.download('punkt')\"")
        try:
            nltk.download('punkt', quiet=True)
            print("✅ NLTK punkt downloaded successfully")
        except:
            print("❌ Failed to download NLTK punkt")
    
    # Check for required packages
    try:
        import datasets
        import tqdm
        import numpy as np
        print("✅ Required packages found")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("💡 Install required packages: pip install -r requirements.txt")
        return False
    
    return True

def run_complete_pipeline():
    """Run the entire translation pipeline with enhanced error handling"""
    print_banner("🚀 STARTING COMPLETE TRANSLATION PIPELINE")
    
    if not setup_environment():
        print("❌ Environment setup failed!")
        return False
    
    try:
        # 1. Download Data
        print_banner("STEP 1: Downloading English-Assamese Dataset")
        try:
            from scripts.download_data import main as download_main
            success = download_main()
            if not success:
                print("⚠️  Download may have issues, but continuing...")
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
            return False
        
        # 3. Build Dictionary
        print_banner("STEP 3: Building Bilingual Dictionary")
        try:
            from src.data.dictionary_builder import build_dictionary
            dictionary = build_dictionary()
            
            # Check if dictionary was built successfully
            if not dictionary:
                print("❌ Dictionary building returned None")
                return False
                
            dict_size = len(dictionary) if isinstance(dictionary, dict) else 0
            
            if dict_size < 100:
                print("⚠️  Dictionary is very small. Results may be poor.")
            else:
                print(f"✅ Dictionary built with {dict_size} words")
                # Generate dictionary report
                try:
                    from src.data.dictionary_builder import DictionaryBuilder
                    builder = DictionaryBuilder()
                    builder.dictionary = dictionary
                    builder.export_dictionary_report()
                except Exception as e:
                    print(f"⚠️  Could not generate dictionary report: {e}")
                
        except Exception as e:
            print(f"❌ Dictionary building failed: {e}")
            return False
        
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
        print("   - Run 'interactive' for live translation")
        print("   - Run 'analyze' for detailed dictionary analysis")
        print("   - Run 'session-stats' to view translation history")
        print("   - Run 'vocabulary-gaps' to identify missing words")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        traceback.print_exc()
        return False

def run_individual_step(step_name):
    """Run individual pipeline steps with enhanced error handling"""
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
            
            try:
                from src.data.dictionary_builder import build_dictionary
                result = build_dictionary()
                
                if result and len(result) > 0:
                    print(f"✅ Dictionary built with {len(result)} words")
                    # Auto-generate dictionary report
                    try:
                        from src.data.dictionary_builder import DictionaryBuilder
                        builder = DictionaryBuilder()
                        builder.dictionary = result
                        builder.export_dictionary_report()
                    except Exception as e:
                        print(f"⚠️  Could not generate dictionary report: {e}")
                else:
                    print("⚠️  Dictionary is empty or very small")
                    
            except Exception as e:
                print(f"❌ Dictionary building failed: {e}")
                traceback.print_exc()
                
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
            
            try:
                from src.models.dictionary_translator import main
                main()
            except Exception as e:
                print(f"❌ Translator test failed: {e}")
                traceback.print_exc()
            
        elif step_name == 'interactive':
            # Check if dictionary exists
            dict_path = Path("data/dictionary/eng_asm_dict.json")
            if not dict_path.exists():
                print("❌ Dictionary not found. Run 'dictionary' step first.")
                return
            
            try:
                from scripts.interactive_translator import interactive_translator
                interactive_translator()
            except Exception as e:
                print(f"❌ Interactive translator failed: {e}")
                traceback.print_exc()
            
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
            
            try:
                from src.evaluation.evaluate import main
                main()
            except Exception as e:
                print(f"❌ Evaluation failed: {e}")
                traceback.print_exc()
            
        elif step_name == 'analyze':
            # Analyze dictionary coverage and generate report
            dict_path = Path("data/dictionary/eng_asm_dict.json")
            if not dict_path.exists():
                print("❌ Dictionary not found. Run 'dictionary' step first.")
                return
            
            try:
                from src.data.dictionary_builder import DictionaryBuilder
                builder = DictionaryBuilder()
                # Load existing dictionary
                with open(dict_path, 'r', encoding='utf-8') as f:
                    dictionary_data = json.load(f)
                
                # Handle different dictionary formats
                if isinstance(dictionary_data, dict):
                    builder.dictionary = dictionary_data
                else:
                    print("⚠️  Unexpected dictionary format")
                    builder.dictionary = {}
                
                coverage_stats = builder.analyze_dictionary_coverage()
                
                print("\n📊 DICTIONARY ANALYSIS REPORT:")
                print("=" * 40)
                print(f"Total words: {len(builder.dictionary)}")
                print(f"Common words coverage: {coverage_stats['coverage_percentage']:.1f}%")
                print(f"Covered: {coverage_stats['covered']}/{coverage_stats['total_common_words']}")
                if coverage_stats.get('missing'):
                    print(f"Missing: {coverage_stats['missing'][:10]}...")  # Show top 10 missing
                
                # Export full report
                builder.export_dictionary_report()
                
            except Exception as e:
                print(f"❌ Dictionary analysis failed: {e}")
                traceback.print_exc()
            
        elif step_name == 'session-stats':
            # Show recent session statistics
            sessions_dir = Path("results/translation_sessions")
            if not sessions_dir.exists():
                print("📭 No session history found. Run 'interactive' first to create sessions.")
                return
                
            session_files = list(sessions_dir.glob("*.json"))
            if not session_files:
                print("📭 No session history found. Run 'interactive' first to create sessions.")
                return
                
            latest_session = max(session_files, key=lambda x: x.stat().st_mtime)
            try:
                with open(latest_session, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                    
                print(f"\n📈 LATEST SESSION: {latest_session.name}")
                print("=" * 40)
                stats = session_data.get('session_info', {})
                if stats:
                    for key, value in stats.items():
                        print(f"{key.replace('_', ' ').title()}: {value}")
                else:
                    print("No session statistics available")
                    
            except Exception as e:
                print(f"❌ Failed to load session: {e}")
                
        elif step_name == 'vocabulary-gaps':
            # Analyze vocabulary gaps across sessions
            sessions_dir = Path("results/translation_sessions")
            if not sessions_dir.exists():
                print("📭 No session history found. Run 'interactive' first to create sessions.")
                return
                
            session_files = list(sessions_dir.glob("*.json"))
            if not session_files:
                print("📭 No session history found. Run 'interactive' first to create sessions.")
                return
                
            all_missing_words = set()
            for session_file in session_files[-5:]:  # Last 5 sessions
                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                    for translation in session_data.get('translations', []):
                        analysis = translation.get('word_analysis', {})
                        missing_words = analysis.get('missing', [])
                        if missing_words:
                            all_missing_words.update(missing_words)
                except Exception as e:
                    print(f"⚠️  Could not read session {session_file}: {e}")
            
            if all_missing_words:
                print(f"\n📝 VOCABULARY GAPS ({len(all_missing_words)} words):")
                print("=" * 40)
                sorted_words = sorted(list(all_missing_words))
                for i, word in enumerate(sorted_words[:20]):  # Show top 20
                    print(f"  {i+1:2d}. {word}")
                if len(sorted_words) > 20:
                    print(f"  ... and {len(sorted_words) - 20} more")
                    
                # Save to file
                gaps_file = Path("results/vocabulary_gaps.txt")
                try:
                    with open(gaps_file, 'w', encoding='utf-8') as f:
                        f.write("Vocabulary Gaps Analysis\n")
                        f.write("=" * 40 + "\n")
                        for word in sorted_words:
                            f.write(f"{word}\n")
                    print(f"\n💾 Full list saved to: {gaps_file}")
                except Exception as e:
                    print(f"⚠️  Could not save vocabulary gaps: {e}")
            else:
                print("✅ No vocabulary gaps found in recent sessions!")
                
        elif step_name == 'quality-report':
            # Generate quality report from evaluation
            try:
                from src.evaluation.quality_metrics import QualityMetrics
                metrics = QualityMetrics()
                report = metrics.get_session_quality_report()
                
                print("\n🎯 TRANSLATION QUALITY REPORT:")
                print("=" * 35)
                if report:
                    for key, value in report.items():
                        print(f"   {key.replace('_', ' ').title()}: {value}")
                else:
                    print("   No quality data available yet.")
                    print("   Run 'interactive' or 'evaluate' to generate quality metrics.")
                    
            except Exception as e:
                print(f"❌ Quality report failed: {e}")
                traceback.print_exc()
            
        elif step_name == 'test-all':
            # Test all components
            test_steps = ['download', 'preprocess', 'dictionary', 'translate', 'interactive']
            for test_step in test_steps:
                run_individual_step(test_step)
                input("\n⏎ Press Enter to continue to next test...")
            
        else:
            print(f"❌ Unknown step: {step_name}")
            show_available_steps()
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
        traceback.print_exc()
    except Exception as e:
        print(f"❌ Step '{step_name}' failed: {e}")
        traceback.print_exc()

def show_available_steps():
    """Show available pipeline steps"""
    print("\n📋 AVAILABLE STEPS:")
    print("   download        - Download dataset from HuggingFace")
    print("   preprocess      - Split data into train/test sets")
    print("   dictionary      - Build bilingual dictionary")
    print("   baseline        - Test baseline dictionary model")
    print("   translate       - Test advanced dictionary translator")
    print("   interactive     - 🆕 Interactive translation session")
    print("   evaluate        - Evaluate translation quality")
    print("   analyze         - 📊 Analyze dictionary coverage")
    print("   session-stats   - 📈 Show session statistics")
    print("   vocabulary-gaps - 📝 Identify missing words")
    print("   quality-report  - 🎯 Generate quality report")
    print("   test-all        - 🧪 Test all components sequentially")
    print("\n💡 Run without arguments for complete pipeline")

def show_help():
    """Show enhanced help information"""
    print_banner("ENGLISH-ASSAMESE TRANSLATION PIPELINE")
    print("A dictionary-based machine translation system for low-resource languages")
    print("\n📋 USAGE:")
    print("  python run_pipeline.py              # Run complete pipeline")
    print("  python run_pipeline.py <step>       # Run individual step")
    print("  python run_pipeline.py help         # Show this help")
    print("\n🔧 CORE STEPS:")
    print("  download    - Download dataset from HuggingFace")
    print("  preprocess  - Split data into train/test sets") 
    print("  dictionary  - Build bilingual dictionary")
    print("  baseline    - Test baseline dictionary model")
    print("  translate   - Test advanced dictionary translator")
    print("  interactive - 🆕 Interactive translation session")
    print("  evaluate    - Evaluate translation quality")
    print("\n📊 ANALYTICS STEPS:")
    print("  analyze         - Dictionary coverage analysis")
    print("  session-stats   - Translation session statistics") 
    print("  vocabulary-gaps - Identify missing vocabulary")
    print("  quality-report  - Translation quality metrics")
    print("  test-all        - Test all components sequentially")
    print("\n📁 OUTPUT FILES:")
    print("  data/raw/eng_asm.json          - Raw downloaded data")
    print("  data/processed/train.json      - Training data")
    print("  data/processed/test.json       - Test data")
    print("  data/dictionary/eng_asm_dict.json - Bilingual dictionary")
    print("  results/dictionary_report_*.json - Dictionary analysis")
    print("  results/translation_sessions/   - Session history")
    print("  results/vocabulary_gaps.txt     - Missing words list")
    print("\n💡 TIPS:")
    print("  - Start with 'python run_pipeline.py' for complete setup")
    print("  - Use 'interactive' for live translation practice")
    print("  - Check 'analyze' to improve dictionary coverage")
    print("  - Review 'vocabulary-gaps' to identify missing words")
    print("  - Use 'test-all' to verify all components work")

def main():
    """Main entry point with enhanced error handling"""
    try:
        if len(sys.argv) > 1:
            if sys.argv[1] in ['help', '-h', '--help']:
                show_help()
            else:
                run_individual_step(sys.argv[1])
        else:
            # Run complete pipeline
            success = run_complete_pipeline()
            if not success:
                print("\n❌ Pipeline completed with errors.")
                print("💡 Try running individual steps to identify the issue:")
                print("   python run_pipeline.py download")
                print("   python run_pipeline.py preprocess")
                print("   python run_pipeline.py dictionary")
                sys.exit(1)
                
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()