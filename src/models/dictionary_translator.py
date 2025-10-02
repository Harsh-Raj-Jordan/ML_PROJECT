"""
Enhanced Dictionary-based translator for English to Assamese
With improved word handling, fallback strategies, and better compatibility
"""

import json
import re
from pathlib import Path
from src.config import settings

class DictionaryTranslator:
    def __init__(self, dictionary_path=None):
        self.dictionary_path = dictionary_path or settings.DICTIONARY_PATH
        self.dictionary = {}
        self.load_dictionary()
        
        # Common word mappings for better translations
        self.common_phrases = {
            'how are you': 'আপোনাৰ কেনে?',
            'thank you': 'ধন্যবাদ',
            'good morning': 'শুভ ৰাতিপুৱা',
            'good night': 'শুভ ৰাতি',
            'i love you': 'মই আপোনাক ভাল পাওঁ',
            'what is your name': 'আপোনাৰ নাম কি?',
            'my name is': 'মোৰ নাম',
            'how old are you': 'আপোনাৰ বয়স কিমান?',
            'where are you from': "আপুনি ক'ৰ পৰা আহিছে?",
            'nice to meet you': 'আপোনাক লগ পাই বৰ ভাল লাগিল'
        }
        
        # Pronouns mapping
        self.pronouns = {
            'i': 'মই',
            'you': 'আপুনি',
            'he': 'তেওঁ',
            'she': 'তেওঁ',
            'we': 'আমি',
            'they': 'সিহঁত',
            'me': 'মোক',
            'him': 'তেওঁক',
            'her': 'তেওঁক',
            'us': 'আমাক',
            'them': 'সিহঁতক',
            'my': 'মোৰ',
            'your': 'আপোনাৰ',
            'his': 'তেওঁৰ',
            'her': 'তেওঁৰ',
            'our': 'আমাৰ',
            'their': 'সিহঁতৰ'
        }
    
    def load_dictionary(self):
        """Load bilingual dictionary with enhanced compatibility"""
        try:
            if not self.dictionary_path.exists():
                raise FileNotFoundError(f"Dictionary not found at {self.dictionary_path}")
            
            print(f"📖 Loading dictionary from: {self.dictionary_path}")
            with open(self.dictionary_path, "r", encoding="utf-8") as f:
                dictionary_data = json.load(f)
            
            # Handle both simple and enhanced dictionary formats
            self.dictionary = {}
            for word, translations in dictionary_data.items():
                if isinstance(translations, list):
                    self.dictionary[word] = translations
                elif isinstance(translations, dict) and 'translations' in translations:
                    self.dictionary[word] = translations['translations']
                elif isinstance(translations, str):
                    self.dictionary[word] = [translations]
                else:
                    self.dictionary[word] = []
            
            print(f"✅ Loaded dictionary with {len(self.dictionary)} words")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load dictionary: {e}")
            # Create empty dictionary as fallback
            self.dictionary = {}
            return False
    
    def clean_word(self, word):
        """Clean word while preserving important characters"""
        if not word:
            return ""
        
        # Remove punctuation from start/end but keep internal hyphens and apostrophes
        cleaned = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', word.lower())
        return cleaned.strip()
    
    def get_best_translation(self, word, context_words=None):
        """Get the best translation for a word considering context"""
        clean_word = self.clean_word(word)
        
        # Check common phrases first
        if context_words and len(context_words) >= 2:
            # Check for 2-word phrases
            if len(context_words) >= 2:
                two_word_phrase = f"{self.clean_word(context_words[-2])} {clean_word}"
                if two_word_phrase in self.common_phrases:
                    return self.common_phrases[two_word_phrase]
            
            # Check for 3-word phrases  
            if len(context_words) >= 3:
                three_word_phrase = f"{self.clean_word(context_words[-3])} {self.clean_word(context_words[-2])} {clean_word}"
                if three_word_phrase in self.common_phrases:
                    return self.common_phrases[three_word_phrase]
        
        # Check single word in dictionary
        if clean_word in self.dictionary:
            translations = self.dictionary[clean_word]
            if translations:
                return translations[0]  # Return most common translation
        
        # Check pronouns
        if clean_word in self.pronouns:
            return self.pronouns[clean_word]
        
        # Return original word if no translation found
        return word
    
    def translate_word(self, word, context_words=None):
        """Translate a single word with context awareness"""
        # Handle empty words
        if not word or not word.strip():
            return word
        
        # Check if it's a common phrase (multiple words)
        if ' ' in word.strip():
            phrase_lower = word.lower().strip()
            if phrase_lower in self.common_phrases:
                return self.common_phrases[phrase_lower]
        
        # Get translation
        translation = self.get_best_translation(word, context_words)
        
        # Preserve original capitalization for proper nouns
        if word and word[0].isupper() and translation:
            translation = translation.capitalize()
        
        return translation
    
    def translate_sentence(self, sentence):
        """Translate a complete sentence with enhanced processing"""
        if not sentence or not sentence.strip():
            return sentence
        
        # Check for exact common phrases first
        sentence_lower = sentence.lower().strip()
        if sentence_lower in self.common_phrases:
            return self.common_phrases[sentence_lower]
        
        # Tokenize while preserving punctuation positions
        words = []
        current_word = ""
        for char in sentence:
            if char.isspace() or char in ',.!?;:"':
                if current_word:
                    words.append(current_word)
                    current_word = ""
                if char.isspace():
                    words.append(' ')
                else:
                    words.append(char)
            else:
                current_word += char
        
        if current_word:
            words.append(current_word)
        
        # Translate words with context
        translated_words = []
        context_window = []
        
        for i, word in enumerate(words):
            if word.strip() and not word.isspace() and word not in ',.!?;:"':
                # Get context (previous 2 words)
                context = context_window[-2:] if len(context_window) >= 2 else context_window
                translated_word = self.translate_word(word, context)
                translated_words.append(translated_word)
                context_window.append(self.clean_word(word))
            else:
                translated_words.append(word)
                # Don't add punctuation to context
        
        # Join translated words
        translation = ''.join(translated_words)
        
        # Post-processing: fix common issues
        translation = self.post_process_translation(translation)
        
        return translation
    
    def post_process_translation(self, translation):
        """Post-process translation to fix common issues"""
        if not translation:
            return translation
        
        # Fix spacing around punctuation
        translation = re.sub(r'\s+([,.!?;:])', r'\1', translation)
        translation = re.sub(r'([,.!?;:])(\w)', r'\1 \2', translation)
        
        # Fix multiple spaces
        translation = re.sub(r'\s+', ' ', translation)
        
        # Ensure proper capitalization for sentences
        translation = translation.strip()
        if translation and translation[0].isalpha():
            translation = translation[0].upper() + translation[1:]
        
        return translation
    
    def batch_translate(self, sentences):
        """Translate multiple sentences"""
        return [self.translate_sentence(sentence) for sentence in sentences]
    
    def get_dictionary_stats(self):
        """Get dictionary statistics"""
        total_words = len(self.dictionary)
        total_translations = sum(len(translations) for translations in self.dictionary.values())
        words_with_translations = sum(1 for translations in self.dictionary.values() if translations)
        
        return {
            'total_words': total_words,
            'total_translations': total_translations,
            'words_with_translations': words_with_translations,
            'coverage_ratio': words_with_translations / total_words if total_words > 0 else 0
        }
    
    def find_similar_words(self, word, max_results=5):
        """Find similar words in dictionary (fuzzy matching)"""
        clean_word = self.clean_word(word)
        similar = []
        
        for dict_word in self.dictionary.keys():
            # Simple similarity: shared prefix or edit distance
            if (clean_word in dict_word or dict_word in clean_word or 
                self.levenshtein_distance(clean_word, dict_word) <= 2):
                similar.append((dict_word, self.dictionary[dict_word]))
                if len(similar) >= max_results:
                    break
        
        return similar
    
    def levenshtein_distance(self, s1, s2):
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

def main():
    """Test the enhanced dictionary translator"""
    print("=" * 60)
    print("🔤 ENHANCED DICTIONARY TRANSLATOR")
    print("=" * 60)
    
    try:
        translator = DictionaryTranslator()
        stats = translator.get_dictionary_stats()
        
        print(f"📊 Dictionary Statistics:")
        print(f"   - Total words: {stats['total_words']}")
        print(f"   - Total translations: {stats['total_translations']}")
        print(f"   - Words with translations: {stats['words_with_translations']}")
        print(f"   - Coverage ratio: {stats['coverage_ratio']:.2%}")
        
        # Test sentences covering various scenarios
        test_sentences = [
            "hello world",
            "good morning",
            "how are you",
            "thank you very much",
            "this is a test sentence",
            "I love programming!",
            "What is your name?",
            "My name is John.",
            "Where are you from?",
            "I am from India.",
            "How old are you?",
            "Nice to meet you!",
            "The weather is good today.",
            "I want to learn Assamese.",
            "This is very helpful.",
            "Can you help me?",
            "I don't understand.",
            "Please speak slowly.",
            "What does this mean?",
            "See you tomorrow!"
        ]
        
        print(f"\n🧪 Test Translations:")
        print("-" * 50)
        
        for i, sentence in enumerate(test_sentences, 1):
            translation = translator.translate_sentence(sentence)
            print(f"{i:2d}. English: {sentence}")
            print(f"    Assamese: {translation}")
            
            # Show word analysis for complex sentences
            if len(sentence.split()) > 3:
                words = sentence.split()
                found = []
                missing = []
                for word in words:
                    clean_word = translator.clean_word(word)
                    if clean_word and clean_word in translator.dictionary:
                        found.append(word)
                    elif clean_word and len(clean_word) > 1:
                        missing.append(word)
                
                if missing:
                    print(f"    📊 Analysis: ✅ {len(found)} translated, ❌ {len(missing)} missing")
                    # Show similar words for missing ones
                    for word in missing[:2]:  # Show up to 2 missing words
                        similar = translator.find_similar_words(word, 2)
                        if similar:
                            print(f"       Similar to '{word}': {[s[0] for s in similar]}")
            
            print()
        
        # Test word similarity search
        print(f"\n🔍 Word Similarity Test:")
        test_words = ['helo', 'thnk', 'gud', 'mornin']
        for word in test_words:
            similar = translator.find_similar_words(word, 3)
            if similar:
                print(f"   '{word}' → Similar: {[s[0] for s in similar]}")
            else:
                print(f"   '{word}' → No similar words found")
        
    except Exception as e:
        print(f"❌ Error in translator: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()