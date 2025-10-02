#!/usr/bin/env python3
"""
Enhanced Interactive English-Assamese Translator
With session management and quality metrics
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.dictionary_translator import DictionaryTranslator
from src.config import settings
from src.utils.session_manager import SessionManager
from src.evaluation.quality_metrics import QualityMetrics

def interactive_translator():
    """Run enhanced interactive translation session"""
    print("=" * 70)
    print("🔤 ENHANCED INTERACTIVE ENGLISH-ASSAMESE TRANSLATOR")
    print("=" * 70)
    print("Type English sentences and get Assamese translations")
    print("Type 'quit', 'exit', or 'q' to stop")
    print("Type 'help' for commands")
    print("-" * 70)
    
    try:
        # Load translator and utilities
        translator = DictionaryTranslator()
        session_manager = SessionManager()
        quality_metrics = QualityMetrics()
        
        print(f"✅ Dictionary loaded with {len(translator.dictionary)} words")
        
        # Show some sample translations
        print("\n🧪 Sample translations:")
        samples = [
            "hello world",
            "good morning", 
            "how are you",
            "thank you"
        ]
        for sample in samples:
            translation = translator.translate_sentence(sample)
            print(f"   '{sample}' → '{translation}'")
        
        print("\n" + "-" * 70)
        
        # Main translation loop
        session_count = 0
        while True:
            try:
                # Get user input
                user_input = input("\n🗣️  Enter English sentence: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    end_session(session_manager, quality_metrics, session_count)
                    break
                
                # Check for special commands
                if handle_special_commands(user_input, session_manager, quality_metrics, translator):
                    continue
                
                # Check for empty input
                if not user_input:
                    print("⚠️  Please enter a sentence to translate.")
                    continue
                
                # Check sentence length
                if len(user_input.split()) > settings.MAX_SENTENCE_LENGTH_TRANSLATE:
                    print(f"⚠️  Sentence too long. Maximum {settings.MAX_SENTENCE_LENGTH_TRANSLATE} words allowed.")
                    continue
                
                # Translate the sentence
                translation = translator.translate_sentence(user_input)
                session_count += 1
                
                # Analyze translation
                word_analysis = show_word_analysis(user_input, translator)
                quality_info = quality_metrics.calculate_sentence_quality(user_input, translation)
                
                # Display results
                print("\n" + "=" * 50)
                print("🔄 TRANSLATION RESULT:")
                print("=" * 50)
                print(f"📥 English: {user_input}")
                print(f"📤 Assamese: {translation}")
                if settings.SHOW_COVERAGE_PERCENTAGE:
                    print(f"📊 Quality Score: {quality_info['quality_score']:.1f}/100")
                print("=" * 50)
                
                # Save to session
                session_manager.add_translation(user_input, translation, {
                    'coverage': word_analysis['coverage'],
                    'found': word_analysis['found'],
                    'missing': word_analysis['missing'],
                    'quality_score': quality_info['quality_score']
                })
                
            except KeyboardInterrupt:
                print(f"\n\n🛑 Session interrupted. Translated {session_count} sentences.")
                end_session(session_manager, quality_metrics, session_count)
                break
            except Exception as e:
                print(f"❌ Translation error: {e}")
                continue
                
    except Exception as e:
        print(f"❌ Failed to load translator: {e}")
        print("💡 Make sure to run 'python run_pipeline.py dictionary' first to build the dictionary")

def handle_special_commands(command, session_manager, quality_metrics, translator):
    """Handle special commands in interactive mode"""
    command = command.lower()
    
    if command == 'help':
        show_help()
        return True
        
    elif command == 'stats':
        show_session_stats(session_manager, quality_metrics)
        return True
        
    elif command == 'save':
        session_manager.save_session()
        return True
        
    elif command == 'export':
        session_manager.export_word_gaps()
        return True
        
    elif command == 'coverage':
        show_dictionary_coverage(translator)
        return True
        
    elif command == 'quality':
        show_quality_report(quality_metrics)
        return True
        
    return False

def show_word_analysis(sentence, translator):
    """Enhanced word analysis with coverage calculation"""
    words = sentence.split()
    found_words = []
    missing_words = []
    
    for word in words:
        clean_word = ''.join(char for char in word if char.isalnum()).lower()
        if clean_word and clean_word in translator.dictionary:
            found_words.append(word)
        elif clean_word:
            missing_words.append(word)
    
    # Calculate coverage
    total_clean_words = len([w for w in words if ''.join(char for char in w if char.isalnum())])
    coverage = (len(found_words) / total_clean_words * 100) if total_clean_words > 0 else 0
    
    if settings.SHOW_WORD_ANALYSIS:
        print(f"\n📊 Word Analysis:")
        print(f"   ✅ Translated: {', '.join(found_words) if found_words else 'None'}")
        print(f"   ❌ Not Found: {', '.join(missing_words) if missing_words else 'None'}")
        if settings.SHOW_COVERAGE_PERCENTAGE:
            print(f"   📈 Coverage: {coverage:.1f}%")
    
    return {
        'found': found_words,
        'missing': missing_words,
        'coverage': coverage
    }

def show_session_stats(session_manager, quality_metrics):
    """Show current session statistics"""
    stats = session_manager.get_session_stats()
    quality_report = quality_metrics.get_session_quality_report()
    
    print("\n📈 SESSION STATISTICS:")
    print("=" * 30)
    for key, value in stats.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    if quality_report:
        print("\n🎯 QUALITY METRICS:")
        for key, value in quality_report.items():
            print(f"   {key.replace('_', ' ').title()}: {value}")

def show_dictionary_coverage(translator):
    """Show dictionary coverage information"""
    from src.data.dictionary_builder import DictionaryBuilder
    
    builder = DictionaryBuilder()
    builder.dictionary = translator.dictionary  # Reuse existing dictionary
    
    coverage_stats = builder.analyze_dictionary_coverage()
    
    print("\n📚 DICTIONARY COVERAGE:")
    print("=" * 30)
    print(f"   Common Words: {coverage_stats['covered']}/{coverage_stats['total_common_words']}")
    print(f"   Coverage: {coverage_stats['coverage_percentage']:.1f}%")
    
    if coverage_stats['missing']:
        print(f"   Top Missing: {', '.join(coverage_stats['missing'][:10])}")

def show_quality_report(quality_metrics):
    """Show translation quality report"""
    report = quality_metrics.get_session_quality_report()
    
    print("\n🎯 TRANSLATION QUALITY REPORT:")
    print("=" * 35)
    for key, value in report.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")

def end_session(session_manager, quality_metrics, session_count):
    """Clean up and save session when ending"""
    print(f"\n👋 Goodbye! Translated {session_count} sentences.")
    
    if session_count > 0 and settings.SAVE_SESSION_HISTORY:
        save = input("💾 Save session history? (y/n): ").lower().strip()
        if save in ['y', 'yes']:
            session_manager.save_session()
            session_manager.export_word_gaps()
    
    # Show final statistics
    if session_count > 0:
        show_session_stats(session_manager, quality_metrics)

def show_help():
    """Show enhanced help information"""
    print("\n📋 AVAILABLE COMMANDS:")
    print("   [any sentence] - Translate English to Assamese")
    print("   stats          - Show session statistics")
    print("   save           - Save current session")
    print("   export         - Export vocabulary gaps")
    print("   coverage       - Show dictionary coverage")
    print("   quality        - Show quality report")
    print("   help           - Show this help message")
    print("   quit/exit/q    - Exit the translator")
    print("\n💡 TIPS:")
    print("   - Use simple, clear sentences for better results")
    print("   - Common words have better translation coverage")
    print("   - Check 'coverage' to see dictionary limitations")
    print("   - Save session to keep translation history")

def main():
    """Main function"""
    try:
        interactive_translator()
    except KeyboardInterrupt:
        print("\n\n👋 Session ended by user.")
    except Exception as e:
        print(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    main()