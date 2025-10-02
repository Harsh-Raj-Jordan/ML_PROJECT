"""
Dictionary-based translator for English to Assamese
"""

import json
from src.config import settings

class DictionaryTranslator:
    def __init__(self, dictionary_path=None):
        self.dictionary_path = dictionary_path or settings.DICTIONARY_PATH
        self.dictionary = self.load_dictionary()
    
    def load_dictionary(self):
        """Load bilingual dictionary"""
        if not self.dictionary_path.exists():
            raise FileNotFoundError(f"Dictionary not found at {self.dictionary_path}")
        
        with open(self.dictionary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def translate_word(self, word):
        """Translate a single word"""
        word_lower = word.lower()
        if word_lower in self.dictionary:
            translations = self.dictionary[word_lower]
            return translations[0] if translations else word  # Return first translation
        return word  # Return original word if not found
    
    def translate_sentence(self, sentence):
        """Translate a complete sentence"""
        words = sentence.split()
        translated_words = []
        
        for word in words:
            # Handle punctuation
            clean_word = ''.join(char for char in word if char.isalnum())
            if clean_word:
                translated_word = self.translate_word(clean_word)
                # Preserve original capitalization for proper nouns
                if word[0].isupper():
                    translated_word = translated_word.capitalize()
                translated_words.append(translated_word)
            else:
                translated_words.append(word)
        
        return ' '.join(translated_words)
    
    def batch_translate(self, sentences):
        """Translate multiple sentences"""
        return [self.translate_sentence(sentence) for sentence in sentences]

def main():
    """Test the dictionary translator"""
    print("=" * 60)
    print("🔤 DICTIONARY TRANSLATOR")
    print("=" * 60)
    
    try:
        translator = DictionaryTranslator()
        print(f"✅ Loaded dictionary with {len(translator.dictionary)} words")
        
        # Test sentences
        test_sentences = [
            "hello world",
            "good morning",
            "how are you",
            "thank you very much",
            "this is a test sentence",
            "I love you Miley Bobby Brown!",
            "Hindustan is blessed country.",
            "Satyam is a good boy!",
            "hello you"
        ]
        
        print("\n🧪 Test Translations:")
        for sentence in test_sentences:
            translation = translator.translate_sentence(sentence)
            print(f"   English: {sentence}")
            print(f"   Assamese: {translation}")
            print()
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()