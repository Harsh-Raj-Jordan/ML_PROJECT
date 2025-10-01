"""
Build bilingual dictionary from training data with improved alignment strategies
"""

import sys
import os
import json
import re
from pathlib import Path
from collections import defaultdict, Counter

# Add project root to path to fix src imports when running directly
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings

class DictionaryBuilder:
    def __init__(self):
        self.word_pairs = defaultdict(Counter)
        self.dictionary = {}
        self.common_english_words = {
            'the', 'and', 'of', 'to', 'in', 'is', 'it', 'that', 'for', 'with', 'on', 'as', 'was', 
            'are', 'be', 'this', 'have', 'from', 'or', 'by', 'but', 'not', 'they', 'which', 'at',
            'all', 'their', 'an', 'we', 'has', 'been', 'were', 'will', 'would', 'there', 'what',
            'so', 'if', 'no', 'out', 'up', 'when', 'who', 'them', 'some', 'could', 'her', 'than',
            'its', 'then', 'also', 'two', 'more', 'these', 'may', 'like', 'other', 'any', 'new',
            'very', 'should', 'now', 'most', 'even', 'only', 'well', 'where', 'such', 'because',
            'over', 'many', 'those', 'through', 'into', 'down', 'off', 'under', 'before', 'between',
            'after', 'above', 'below', 'since', 'during', 'without', 'within', 'upon', 'among'
        }
    
    def load_training_data(self):
        """Load training data for dictionary building"""
        if not settings.TRAIN_DATA_PATH.exists():
            raise FileNotFoundError(f"Training data not found at {settings.TRAIN_DATA_PATH}")
        
        with open(settings.TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def clean_word(self, word):
        """Clean and normalize a word"""
        # Remove punctuation and convert to lowercase
        cleaned = re.sub(r'[^\w\s]', '', word.lower().strip())
        return cleaned if cleaned else None
    
    def extract_word_pairs(self, data):
        """Extract word alignment pairs using multiple strategies"""
        print("🔄 Extracting word pairs from training data...")
        
        total_pairs = 0
        for item_idx, item in enumerate(data):
            try:
                translation = item["translation"]
                src_sentence = translation[settings.SRC_LANG]
                tgt_sentence = translation[settings.TGT_LANG]
                
                src_words = [self.clean_word(w) for w in src_sentence.split()]
                tgt_words = [self.clean_word(w) for w in tgt_sentence.split()]
                
                # Remove None values (empty after cleaning)
                src_words = [w for w in src_words if w]
                tgt_words = [w for w in tgt_words if w]
                
                if not src_words or not tgt_words:
                    continue
                
                # STRATEGY 1: 1-to-1 mapping for same length sentences
                if len(src_words) == len(tgt_words):
                    for src_word, tgt_word in zip(src_words, tgt_words):
                        self.word_pairs[src_word][tgt_word] += 1
                        total_pairs += 1
                
                # STRATEGY 2: Common word mapping with positional matching
                for common_word in self.common_english_words:
                    if common_word in src_words:
                        src_idx = src_words.index(common_word)
                        if src_idx < len(tgt_words):
                            self.word_pairs[common_word][tgt_words[src_idx]] += 1
                            total_pairs += 1
                
                # STRATEGY 3: All-to-all mapping for short sentences
                if len(src_words) <= 8 and len(tgt_words) <= 10:
                    for src_word in src_words:
                        for tgt_word in tgt_words:
                            # Only add if both words are meaningful (not too short)
                            if len(src_word) > 2 and len(tgt_word) > 1:
                                self.word_pairs[src_word][tgt_word] += 0.5  # Lower weight
                                total_pairs += 0.5
                
                # STRATEGY 4: First and last word mapping
                if len(src_words) >= 2 and len(tgt_words) >= 2:
                    # Map first words
                    self.word_pairs[src_words[0]][tgt_words[0]] += 1
                    total_pairs += 1
                    # Map last words  
                    self.word_pairs[src_words[-1]][tgt_words[-1]] += 1
                    total_pairs += 1
                
                # Progress indicator
                if (item_idx + 1) % 500 == 0:
                    print(f"   Processed {item_idx + 1} sentences, found {int(total_pairs)} pairs...")
                    
            except (KeyError, IndexError) as e:
                continue
        
        print(f"   Total word pairs collected: {int(total_pairs)}")
    
    def build_dictionary(self):
        """Build final dictionary with improved filtering"""
        print("🔨 Building dictionary...")
        
        total_possible_words = len(self.word_pairs)
        words_with_translations = 0
        
        for src_word, translations in self.word_pairs.items():
            # Lower frequency threshold for common words
            if src_word in self.common_english_words:
                min_freq = max(1, settings.MIN_WORD_FREQUENCY - 1)
            else:
                min_freq = settings.MIN_WORD_FREQUENCY
            
            # Get translations meeting frequency threshold
            valid_translations = [
                (tgt_word, count) for tgt_word, count in translations.items() 
                if count >= min_freq
            ]
            
            if valid_translations:
                # Sort by frequency and take top translations
                valid_translations.sort(key=lambda x: x[1], reverse=True)
                top_translations = [tgt_word for tgt_word, count in 
                                  valid_translations[:settings.MAX_TRANSLATIONS_PER_WORD]]
                self.dictionary[src_word] = top_translations
                words_with_translations += 1
        
        # Add some default translations for very common words
        self._add_default_translations()
        
        print(f"   Words with translations: {words_with_translations}/{total_possible_words}")
    
    def _add_default_translations(self):
        """Add some default translations for common words"""
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
            'friend': ['বন্ধু', 'সখা']
        }
        
        for word, translations in default_translations.items():
            if word not in self.dictionary:
                self.dictionary[word] = translations
            else:
                # Merge with existing translations
                existing = self.dictionary[word]
                for trans in translations:
                    if trans not in existing:
                        existing.append(trans)
                self.dictionary[word] = existing[:settings.MAX_TRANSLATIONS_PER_WORD]
    
    def save_dictionary(self):
        """Save dictionary to file"""
        with open(settings.DICTIONARY_PATH, "w", encoding="utf-8") as f:
            json.dump(self.dictionary, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Dictionary saved to: {settings.DICTIONARY_PATH}")
    
    def get_statistics(self):
        """Get comprehensive dictionary statistics"""
        total_words = len(self.dictionary)
        total_translations = sum(len(translations) for translations in self.dictionary.values())
        avg_translations = total_translations / total_words if total_words > 0 else 0
        
        # Count common words in dictionary
        common_words_in_dict = len([w for w in self.common_english_words if w in self.dictionary])
        
        return {
            "total_words": total_words,
            "total_translations": total_translations,
            "avg_translations_per_word": avg_translations,
            "common_words_coverage": f"{common_words_in_dict}/{len(self.common_english_words)}",
            "coverage_percentage": f"{(common_words_in_dict/len(self.common_english_words))*100:.1f}%"
        }

def build_dictionary():
    """Main function to build dictionary"""
    print("=" * 60)
    print("📚 IMPROVED BILINGUAL DICTIONARY BUILDER")
    print("=" * 60)
    
    builder = DictionaryBuilder()
    
    # Load training data
    training_data = builder.load_training_data()
    print(f"📖 Loaded {len(training_data)} training examples")
    
    # Extract word pairs and build dictionary
    builder.extract_word_pairs(training_data)
    builder.build_dictionary()
    
    # Save dictionary
    builder.save_dictionary()
    
    # Print comprehensive statistics
    stats = builder.get_statistics()
    print(f"\n📊 DICTIONARY STATISTICS:")
    print(f"   - Total English words: {stats['total_words']}")
    print(f"   - Total translations: {stats['total_translations']}")
    print(f"   - Avg translations per word: {stats['avg_translations_per_word']:.2f}")
    print(f"   - Common words coverage: {stats['common_words_coverage']} ({stats['coverage_percentage']})")
    
    # Show sample entries from different categories
    all_words = list(builder.dictionary.keys())
    common_words = [w for w in all_words if w in builder.common_english_words][:3]
    other_words = [w for w in all_words if w not in builder.common_english_words][:3]
    
    print(f"\n🔍 SAMPLE COMMON WORDS:")
    for word in common_words:
        print(f"   '{word}': {builder.dictionary[word]}")
    
    print(f"\n🔍 SAMPLE OTHER WORDS:")
    for word in other_words:
        print(f"   '{word}': {builder.dictionary[word]}")
    
    print(f"\n✅ Dictionary building completed successfully!")
    return builder.dictionary

if __name__ == "__main__":
    build_dictionary()