"""
Enhanced Bilingual Dictionary Builder for English-Assamese Translation
Processes large datasets efficiently to build comprehensive dictionary
"""

import sys
import os
import json
import re
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
    
    def tokenize_english(self, text):
        """Enhanced English tokenization that preserves more words"""
        if not text:
            return []
        
        # Use multiple tokenization strategies
        words = []
        
        # Strategy 1: Simple split with punctuation handling
        simple_words = re.findall(r"[a-zA-Z]+(?:['-][a-zA-Z]+)*", text.lower())
        words.extend(simple_words)
        
        # Strategy 2: NLTK tokenization if available
        try:
            nltk_words = word_tokenize(text.lower())
            # Filter to keep only words with letters
            nltk_words = [w for w in nltk_words if re.search(r'[a-zA-Z]', w)]
            words.extend(nltk_words)
        except:
            pass
        
        # Remove duplicates and clean
        unique_words = list(set(words))
        cleaned_words = [self.clean_word(w) for w in unique_words]
        return [w for w in cleaned_words if w and len(w) > 0]
    
    def tokenize_assamese(self, text):
        """Tokenize Assamese text - keep all meaningful characters"""
        if not text:
            return []
        
        # Split by spaces and filter empty strings
        words = text.split()
        return [w.strip() for w in words if w.strip()]
    
    def extract_word_pairs_enhanced(self, data):
        """Enhanced word pair extraction with multiple alignment strategies"""
        print("🔄 Extracting word pairs using enhanced strategies...")
        
        for item_idx, item in enumerate(data):
            try:
                if "translation" not in item:
                    continue
                    
                translation = item["translation"]
                src_sentence = translation.get(settings.SRC_LANG, "").strip()
                tgt_sentence = translation.get(settings.TGT_LANG, "").strip()
                
                if not src_sentence or not tgt_sentence:
                    continue
                
                # Tokenize sentences
                src_words = self.tokenize_english(src_sentence)
                tgt_words = self.tokenize_assamese(tgt_sentence)
                
                if not src_words or not tgt_words:
                    continue
                
                # STRATEGY 1: Direct 1-to-1 mapping for same length
                if len(src_words) == len(tgt_words):
                    for src_word, tgt_word in zip(src_words, tgt_words):
                        self.word_pairs[src_word][tgt_word] += 2.0  # Higher weight
                        self.total_pairs += 1
                
                # STRATEGY 2: Common word positional matching
                for common_word in self.common_english_words:
                    if common_word in src_words:
                        src_idx = src_words.index(common_word)
                        if src_idx < len(tgt_words):
                            self.word_pairs[common_word][tgt_words[src_idx]] += 3.0  # Even higher weight
                            self.total_pairs += 1
                
                # STRATEGY 3: All-to-all mapping with distance weighting
                self._all_to_all_mapping(src_words, tgt_words)
                
                # STRATEGY 4: Position-based mapping (first/last words)
                self._positional_mapping(src_words, tgt_words)
                
                # STRATEGY 5: Substring matching for compound words
                self._substring_mapping(src_sentence, tgt_sentence)
                
                # Progress indicator
                self.processed_count += 1
                if self.processed_count % 10000 == 0:
                    print(f"   Processed {self.processed_count} sentences, found {self.total_pairs} pairs...")
                    
            except Exception as e:
                # Continue processing even if one example fails
                continue
        
        print(f"✅ Extraction complete: {self.processed_count} sentences, {self.total_pairs} word pairs")
    
    def _all_to_all_mapping(self, src_words, tgt_words):
        """All-to-all word mapping with distance-based weighting"""
        for src_word in src_words:
            for tgt_word in tgt_words:
                # Only add if both words are meaningful
                if len(src_word) >= 2 and len(tgt_word) >= 1:
                    # Give higher weight to shorter sentences (likely better alignment)
                    weight = 1.0 / (len(src_words) * len(tgt_words))
                    self.word_pairs[src_word][tgt_word] += weight
                    self.total_pairs += weight
    
    def _positional_mapping(self, src_words, tgt_words):
        """Position-based word mapping"""
        if len(src_words) >= 2 and len(tgt_words) >= 2:
            # Map first words with high confidence
            self.word_pairs[src_words[0]][tgt_words[0]] += 2.0
            self.total_pairs += 1
            
            # Map last words with high confidence  
            self.word_pairs[src_words[-1]][tgt_words[-1]] += 2.0
            self.total_pairs += 1
            
            # Map middle words with medium confidence
            if len(src_words) >= 3 and len(tgt_words) >= 3:
                mid_src = len(src_words) // 2
                mid_tgt = len(tgt_words) // 2
                self.word_pairs[src_words[mid_src]][tgt_words[mid_tgt]] += 1.5
                self.total_pairs += 1
    
    def _substring_mapping(self, src_sentence, tgt_sentence):
        """Handle compound words and substring matches"""
        src_lower = src_sentence.lower()
        
        # Look for common multi-word patterns
        common_patterns = [
            'good morning', 'good night', 'thank you', 'how are', 'what is',
            'i am', 'you are', 'he is', 'she is', 'we are', 'they are'
        ]
        
        for pattern in common_patterns:
            if pattern in src_lower:
                # Add the pattern as a single entry
                clean_pattern = self.clean_word(pattern)
                if clean_pattern:
                    self.word_pairs[clean_pattern][tgt_sentence] += 1.0
                    self.total_pairs += 1
    
    def build_dictionary_enhanced(self):
        """Build dictionary with enhanced filtering and ranking"""
        print("🔨 Building enhanced dictionary...")
        
        total_possible_words = len(self.word_pairs)
        words_with_translations = 0
        
        for src_word, translations in self.word_pairs.items():
            # Calculate total frequency for this word
            total_freq = sum(translations.values())
            
            # Dynamic threshold based on word type
            if src_word in self.common_english_words:
                min_freq = max(0.5, settings.MIN_WORD_FREQUENCY - 0.5)  # Lower for common words
            else:
                min_freq = settings.MIN_WORD_FREQUENCY
            
            # Get valid translations
            valid_translations = [
                (tgt_word, count) for tgt_word, count in translations.items() 
                if count >= min_freq
            ]
            
            if valid_translations:
                # Sort by frequency and take top translations
                valid_translations.sort(key=lambda x: x[1], reverse=True)
                top_translations = [
                    tgt_word for tgt_word, count in 
                    valid_translations[:settings.MAX_TRANSLATIONS_PER_WORD]
                ]
                
                # Store with metadata
                self.dictionary[src_word] = {
                    'translations': top_translations,
                    'frequency': total_freq,
                    'translation_count': len(valid_translations)
                }
                words_with_translations += 1
        
        # Add default translations
        self._add_comprehensive_default_translations()
        
        print(f"📊 Dictionary stats: {words_with_translations}/{total_possible_words} words with translations")
        return self.dictionary
    
    def _add_comprehensive_default_translations(self):
        """Add comprehensive default translations for common words"""
        default_translations = {
            'hello': ['নমস্কাৰ', 'হেল'],
            'good': ['ভাল', 'উত্তম'],
            'morning': ['ৰাতিপুৱা', 'সকাল'],
            'thank': ['ধন্যবাদ'],
            'you': ['আপুনি', 'তুমি'],
            'how': ['কেনেকৈ', 'কিমান'],
            'are': ['আছে', 'হয়'],
            'what': ['কি', 'কোন'],
            'where': ['ক\'ত', 'কোনঠাইত'],
            'when': ['কেতিয়া', 'কিমান'],
            'why': ['কিয়', 'কাৰণ'],
            'who': ['যি', 'কোন'],
            'yes': ['হয়', 'হ'],
            'no': ['নহয়', 'না'],
            'please': ['অনুগ্ৰহ কৰি', 'দয়া কৰি'],
            'sorry': ['ক্ষমা কৰিব', 'দুঃখিত'],
            'water': ['পানী', 'জল'],
            'food': ['খাদ্য', 'খানা'],
            'house': ['ঘৰ', 'গৃহ'],
            'day': ['দিন', 'দিবস'],
            'night': ['ৰাতি', 'নিশা'],
            'man': ['মানুহ', 'পুৰুষ'],
            'woman': ['মহিলা', 'স্ত্রী'],
            'child': ['ল\'ৰা', 'শিশু'],
            'school': ['বিদ্যালয়', 'স্কুল'],
            'book': ['কিতাপ', 'পুস্তক'],
            'friend': ['বন্ধু', 'সখা'],
            'love': ['মৰম', 'প্রেম'],
            'work': ['কাম', 'কাৰ্য'],
            'time': ['সময়', 'বেলা'],
            'year': ['বছৰ', 'বৰ্ষ'],
            'people': ['মানুহ', 'লোক'],
            'country': ['দেশ', 'ৰাষ্ট্র'],
            'city': ['চহৰ', 'নগৰ'],
            'language': ['ভাষা', 'বোল'],
            'name': ['নাম', 'নাম'],
            'family': ['পৰিয়াল', 'পরিবার'],
            'mother': ['মা', 'মাতৃ'],
            'father': ['দেউতা', 'পিতা'],
            'brother': ['ভাই', 'ভ্রাতা'],
            'sister': ['ভনী', 'ভগ্নী']
        }
        
        for word, translations in default_translations.items():
            if word not in self.dictionary:
                self.dictionary[word] = {
                    'translations': translations,
                    'frequency': 1.0,
                    'translation_count': len(translations)
                }
            else:
                # Merge with existing translations
                existing = self.dictionary[word]['translations']
                for trans in translations:
                    if trans not in existing:
                        existing.append(trans)
                # Keep only top translations
                self.dictionary[word]['translations'] = existing[:settings.MAX_TRANSLATIONS_PER_WORD]
    
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
            'missing': len(missing),
            'coverage_percentage': (len(covered) / len(common_words_list)) * 100,
            'top_missing_words': missing[:20]  # Show top 20 missing words
        }
        
        return coverage_stats

    def export_dictionary_report(self):
        """Export comprehensive dictionary report"""
        # Get statistics
        stats = self.get_statistics()
        coverage_stats = self.analyze_dictionary_coverage()
        
        # Get most translated words
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