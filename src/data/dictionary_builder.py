"""
Enhanced Bilingual Dictionary Builder for English-Assamese Translation
Processes large datasets efficiently to build comprehensive dictionary
"""

import sys
import os
import json
import re
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize

# Add project root to path to fix src imports when running directly
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings

class DictionaryBuilder:
    def __init__(self):
        self.word_pairs = defaultdict(Counter)
        self.dictionary = {}
        self.processed_count = 0
        self.total_pairs = 0
        
        # Expanded common English words list
        self.common_english_words = {
            'the', 'and', 'of', 'to', 'in', 'is', 'it', 'that', 'for', 'with', 'on', 'as', 'was', 
            'are', 'be', 'this', 'have', 'from', 'or', 'by', 'but', 'not', 'they', 'which', 'at',
            'all', 'their', 'an', 'we', 'has', 'been', 'were', 'will', 'would', 'there', 'what',
            'so', 'if', 'no', 'out', 'up', 'when', 'who', 'them', 'some', 'could', 'her', 'than',
            'its', 'then', 'also', 'two', 'more', 'these', 'may', 'like', 'other', 'any', 'new',
            'very', 'should', 'now', 'most', 'even', 'only', 'well', 'where', 'such', 'because',
            'over', 'many', 'those', 'through', 'into', 'down', 'off', 'under', 'before', 'between',
            'after', 'above', 'below', 'since', 'during', 'without', 'within', 'upon', 'among',
            'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
            'his', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs', 'this', 'that',
            'these', 'those', 'who', 'whom', 'whose', 'which', 'what', 'where', 'when', 'why', 'how',
            'a', 'an', 'the', 'and', 'but', 'or', 'nor', 'for', 'yet', 'so', 'both', 'either', 'neither',
            'not', 'no', 'yes', 'please', 'thank', 'sorry', 'hello', 'goodbye', 'welcome', 'okay'
        }
    
    def load_training_data(self):
        """Load training data for dictionary building"""
        if not settings.TRAIN_DATA_PATH.exists():
            raise FileNotFoundError(f"Training data not found at {settings.TRAIN_DATA_PATH}")
        
        print(f"📖 Loading training data from: {settings.TRAIN_DATA_PATH}")
        with open(settings.TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"✅ Loaded {len(data)} training examples")
        return data
    
    def clean_word(self, word):
        """Clean and normalize a word - less aggressive cleaning"""
        if not word or not isinstance(word, str):
            return None
            
        # Convert to lowercase and remove extra spaces
        cleaned = word.lower().strip()
        
        # Remove punctuation from start/end but keep internal hyphens and apostrophes
        cleaned = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', cleaned)
        
        return cleaned if cleaned else None
    
    def extract_word_pairs_enhanced(self, data):
        """Enhanced word pair extraction with improved alignment strategies"""
        print("🔄 Extracting word pairs using improved alignment strategies...")
        
        alignment_stats = {
            'exact_length': 0,
            'single_word': 0,
            'common_word': 0,
            'positional': 0,
            'failed': 0
        }
        
        for item_idx, item in enumerate(data):
            try:
                if "translation" not in item:
                    continue
                    
                translation = item["translation"]
                src_sentence = translation.get(settings.SRC_LANG, "").strip()
                tgt_sentence = translation.get(settings.TGT_LANG, "").strip()
                
                if not src_sentence or not tgt_sentence:
                    continue
                
                # Clean sentences more effectively
                src_clean = re.sub(r'[^\w\s]', ' ', src_sentence.lower())
                tgt_clean = tgt_sentence.strip()
                
                src_words = [w.strip() for w in src_clean.split() if len(w.strip()) > 0]
                tgt_words = [w.strip() for w in tgt_clean.split() if w.strip()]
                
                if not src_words or not tgt_words:
                    continue
                
                # STRATEGY 1: Single word sentences (highest confidence)
                if len(src_words) == 1 and len(tgt_words) == 1:
                    src_word = src_words[0]
                    tgt_word = tgt_words[0]
                    if len(src_word) > 1 and len(tgt_word) > 0:
                        self.word_pairs[src_word][tgt_word] += 10.0
                        self.total_pairs += 1
                        alignment_stats['single_word'] += 1
                
                # STRATEGY 2: Exact length alignment
                elif len(src_words) == len(tgt_words) and len(src_words) <= 8:
                    for src_word, tgt_word in zip(src_words, tgt_words):
                        if len(src_word) > 1 and len(tgt_word) > 0:
                            self.word_pairs[src_word][tgt_word] += 5.0
                            self.total_pairs += 1
                    alignment_stats['exact_length'] += 1
                
                # STRATEGY 3: Common word positional matching
                for common_word in self.common_english_words:
                    if common_word in src_words:
                        src_idx = src_words.index(common_word)
                        if src_idx < len(tgt_words) and len(tgt_words[src_idx]) > 0:
                            self.word_pairs[common_word][tgt_words[src_idx]] += 4.0
                            self.total_pairs += 1
                            alignment_stats['common_word'] += 1
                
                # STRATEGY 4: Position-based mapping for short sentences
                if len(src_words) <= 6 and len(tgt_words) <= 8:
                    # First word alignment
                    if len(src_words[0]) > 1 and len(tgt_words[0]) > 0:
                        self.word_pairs[src_words[0]][tgt_words[0]] += 3.0
                        self.total_pairs += 1
                    
                    # Last word alignment if different from first
                    if len(src_words) > 1 and len(tgt_words) > 1:
                        if len(src_words[-1]) > 1 and len(tgt_words[-1]) > 0:
                            self.word_pairs[src_words[-1]][tgt_words[-1]] += 3.0
                            self.total_pairs += 1
                    
                    alignment_stats['positional'] += 1
                
                # Progress indicator
                self.processed_count += 1
                if self.processed_count % settings.PROCESSING_BATCH_SIZE == 0:
                    print(f"   Processed {self.processed_count} sentences, found {int(self.total_pairs)} pairs...")
                    
            except Exception as e:
                alignment_stats['failed'] += 1
                continue
        
        print(f"✅ Extraction complete: {self.processed_count} sentences, {int(self.total_pairs)} word pairs")
        print(f"📊 Alignment Statistics:")
        print(f"   - Single word sentences: {alignment_stats['single_word']}")
        print(f"   - Exact length matches: {alignment_stats['exact_length']}")
        print(f"   - Common word matches: {alignment_stats['common_word']}")
        print(f"   - Positional matches: {alignment_stats['positional']}")
        print(f"   - Failed alignments: {alignment_stats['failed']}")
    
    def build_dictionary_enhanced(self):
        """Build dictionary with improved filtering and validation"""
        print("🔨 Building enhanced dictionary...")
        
        total_possible_words = len(self.word_pairs)
        words_with_translations = 0
        
        for src_word, translations in self.word_pairs.items():
            # Calculate total frequency for this word
            total_freq = sum(translations.items())
            
            # Dynamic threshold based on word type
            if src_word in self.common_english_words:
                min_freq = 0.3  # Very low for common words
            else:
                min_freq = settings.MIN_WORD_FREQUENCY
            
            # Get valid translations with frequency threshold
            valid_translations = [
                (tgt_word, count) for tgt_word, count in translations.items() 
                if count >= min_freq and len(tgt_word.strip()) > 0
            ]
            
            if valid_translations:
                # Sort by frequency and take top translations
                valid_translations.sort(key=lambda x: x[1], reverse=True)
                top_translations = [
                    tgt_word for tgt_word, count in 
                    valid_translations[:settings.MAX_TRANSLATIONS_PER_WORD]
                ]
                
                # Only include if we have reasonable translations
                if len(top_translations) > 0:
                    self.dictionary[src_word] = {
                        'translations': top_translations,
                        'frequency': total_freq,
                        'translation_count': len(valid_translations)
                    }
                    words_with_translations += 1
        
        # Add comprehensive default translations
        self._add_comprehensive_default_translations()
        
        print(f"📊 Dictionary stats: {words_with_translations}/{total_possible_words} words with translations")
        
        # Validate dictionary quality
        self._validate_dictionary_quality()
        
        return self.dictionary
    
    def _validate_dictionary_quality(self):
        """Validate dictionary quality and coverage"""
        print("\n🔍 Validating dictionary quality...")
        
        # Check coverage of essential words
        essential_words = ['the', 'is', 'and', 'to', 'in', 'of', 'a', 'that', 'it', 'for', 'you', 'i', 'he', 'she']
        covered_essential = sum(1 for word in essential_words if word in self.dictionary)
        
        # Check average translations per word
        translation_counts = []
        for word_data in self.dictionary.values():
            if isinstance(word_data, dict) and 'translations' in word_data:
                translation_counts.append(len(word_data['translations']))
        
        avg_translations = np.mean(translation_counts) if translation_counts else 0
        
        print(f"📈 Quality Metrics:")
        print(f"   - Essential words coverage: {covered_essential}/{len(essential_words)}")
        print(f"   - Average translations per word: {avg_translations:.2f}")
        print(f"   - Total dictionary size: {len(self.dictionary)} words")
        
        # Warn if dictionary is too small
        if len(self.dictionary) < settings.MIN_DICTIONARY_SIZE:
            print(f"⚠️  Warning: Dictionary size ({len(self.dictionary)}) is below minimum threshold ({settings.MIN_DICTIONARY_SIZE})")
    
    def _add_comprehensive_default_translations(self):
        """Add comprehensive default translations for common words"""
        default_translations = {
            'hello': ['নমস্কাৰ'],
            'good': ['ভাল'],
            'morning': ['ৰাতিপুৱা'],
            'thank': ['ধন্যবাদ'],
            'you': ['আপুনি'],
            'how': ['কেনেকৈ'],
            'are': ['আছে'],
            'what': ['কি'],
            'where': ['ক\'ত'],
            'when': ['কেতিয়া'],
            'why': ['কিয়'],
            'who': ['যি'],
            'yes': ['হয়'],
            'no': ['নহয়'],
            'please': ['অনুগ্ৰহ কৰি'],
            'sorry': ['ক্ষমা কৰিব'],
            'water': ['পানী'],
            'food': ['খাদ্য'],
            'house': ['ঘৰ'],
            'day': ['দিন'],
            'night': ['ৰাতি'],
            'man': ['মানুহ'],
            'woman': ['মহিলা'],
            'child': ['ল\'ৰা'],
            'school': ['বিদ্যালয়'],
            'book': ['কিতাপ'],
            'friend': ['বন্ধু'],
            'love': ['মৰম'],
            'work': ['কাম'],
            'time': ['সময়'],
            'year': ['বছৰ'],
            'people': ['মানুহ'],
            'country': ['দেশ'],
            'city': ['চহৰ'],
            'language': ['ভাষা'],
            'name': ['নাম'],
            'family': ['পৰিয়াল'],
            'mother': ['মা'],
            'father': ['দেউতা'],
            'brother': ['ভাই'],
            'sister': ['ভনী'],
            'this': ['এই'],
            'that': ['সেই'],
            'these': ['এইবোৰ'],
            'those': ['সেইবোৰ'],
            'my': ['মোৰ'],
            'your': ['আপোনাৰ'],
            'his': ['তাৰ'],
            'her': ['তাৰ'],
            'our': ['আমাৰ'],
            'their': ['সিহঁতৰ']
        }
        
        added_count = 0
        for word, translations in default_translations.items():
            if word not in self.dictionary:
                self.dictionary[word] = {
                    'translations': translations,
                    'frequency': 10.0,  # High frequency for defaults
                    'translation_count': len(translations)
                }
                added_count += 1
            else:
                # Merge with existing translations, prioritizing defaults
                existing = self.dictionary[word]['translations']
                for trans in translations:
                    if trans not in existing:
                        existing.insert(0, trans)  # Add defaults at beginning
                # Keep only top translations
                self.dictionary[word]['translations'] = existing[:settings.MAX_TRANSLATIONS_PER_WORD]
        
        print(f"📝 Added {added_count} default translations")
    
    def save_dictionary(self):
        """Save dictionary to file"""
        # Convert to simple format for compatibility
        simple_dict = {}
        for word, data in self.dictionary.items():
            if isinstance(data, dict) and 'translations' in data:
                simple_dict[word] = data['translations']
            else:
                simple_dict[word] = data
        
        with open(settings.DICTIONARY_PATH, "w", encoding="utf-8") as f:
            json.dump(simple_dict, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Dictionary saved to: {settings.DICTIONARY_PATH}")
        return settings.DICTIONARY_PATH
    
    def get_statistics(self):
        """Get comprehensive dictionary statistics"""
        total_words = len(self.dictionary)
        total_translations = 0
        
        for word_data in self.dictionary.values():
            if isinstance(word_data, dict) and 'translations' in word_data:
                total_translations += len(word_data['translations'])
            elif isinstance(word_data, list):
                total_translations += len(word_data)
        
        avg_translations = total_translations / total_words if total_words > 0 else 0
        
        # Count common words in dictionary
        common_words_in_dict = len([w for w in self.common_english_words if w in self.dictionary])
        
        return {
            "total_words": total_words,
            "total_translations": total_translations,
            "avg_translations_per_word": round(avg_translations, 2),
            "common_words_coverage": f"{common_words_in_dict}/{len(self.common_english_words)}",
            "coverage_percentage": f"{(common_words_in_dict/len(self.common_english_words))*100:.1f}%",
            "processed_sentences": self.processed_count,
            "total_word_pairs": int(self.total_pairs)
        }

    def analyze_dictionary_coverage(self, common_words_list=None):
        """Analyze how well the dictionary covers common words"""
        if common_words_list is None:
            common_words_list = self.common_english_words
        
        covered = [word for word in common_words_list if word in self.dictionary]
        missing = [word for word in common_words_list if word not in self.dictionary]
        
        coverage_stats = {
            'total_common_words': len(common_words_list),
            'covered': len(covered),
            'missing': missing,
            'coverage_percentage': (len(covered) / len(common_words_list)) * 100,
            'top_missing_words': missing[:20]  # Show top 20 missing words
        }
        
        return coverage_stats

    def export_dictionary_report(self):
        """Export comprehensive dictionary report"""
        # Get statistics
        stats = self.get_statistics()
        coverage_stats = self.analyze_dictionary_coverage()
        
        # Get most frequent words
        word_frequencies = []
        for word, data in self.dictionary.items():
            if isinstance(data, dict) and 'frequency' in data:
                word_frequencies.append((word, data['frequency']))
            else:
                word_frequencies.append((word, 1))
        
        most_frequent_words = sorted(word_frequencies, key=lambda x: x[1], reverse=True)[:20]
        
        # Get sample entries
        all_words = list(self.dictionary.keys())
        common_words_samples = [w for w in all_words if w in self.common_english_words][:5]
        other_words_samples = [w for w in all_words if w not in self.common_english_words][:5]
        
        report = {
            'build_timestamp': datetime.now().isoformat(),
            'statistics': stats,
            'coverage_analysis': coverage_stats,
            'most_frequent_words': most_frequent_words,
            'sample_common_words': {word: self.dictionary[word] for word in common_words_samples},
            'sample_other_words': {word: self.dictionary[word] for word in other_words_samples},
            'dictionary_size_bytes': len(json.dumps(self.dictionary, ensure_ascii=False)),
            'processing_metrics': {
                'sentences_processed': self.processed_count,
                'word_pairs_collected': self.total_pairs
            }
        }
        
        # Ensure reports directory exists
        settings.DICTIONARY_DIR.mkdir(parents=True, exist_ok=True)
        
        report_file = settings.DICTIONARY_DIR / f"dictionary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"📊 Dictionary report exported: {report_file}")
        return report_file

def build_dictionary():
    """Main function to build dictionary"""
    print("=" * 60)
    print("🚀 ENHANCED BILINGUAL DICTIONARY BUILDER")
    print("=" * 60)
    
    builder = DictionaryBuilder()
    
    try:
        # Load training data
        training_data = builder.load_training_data()
        
        # Extract word pairs and build dictionary
        builder.extract_word_pairs_enhanced(training_data)
        builder.build_dictionary_enhanced()
        
        # Save dictionary
        builder.save_dictionary()
        
        # Print comprehensive statistics
        stats = builder.get_statistics()
        print(f"\n📊 DICTIONARY STATISTICS:")
        print(f"   - Total English words: {stats['total_words']}")
        print(f"   - Total translations: {stats['total_translations']}")
        print(f"   - Avg translations per word: {stats['avg_translations_per_word']}")
        print(f"   - Common words coverage: {stats['common_words_coverage']} ({stats['coverage_percentage']})")
        print(f"   - Processed sentences: {stats['processed_sentences']}")
        print(f"   - Word pairs collected: {stats['total_word_pairs']}")
        
        # Export dictionary report
        builder.export_dictionary_report()
        
        print(f"\n✅ Dictionary building completed successfully!")
        return builder.dictionary
        
    except Exception as e:
        print(f"❌ Dictionary building failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    build_dictionary()