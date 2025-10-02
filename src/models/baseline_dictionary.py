"""
Baseline dictionary-based translation model
"""

import json
from pathlib import Path
from src.config import settings

class BaselineDictionary:
    """Simple dictionary lookup translator"""
    
    def __init__(self, dictionary_path=None):
        self.dictionary_path = dictionary_path or settings.DICTIONARY_PATH
        self.dictionary = self._load_dictionary()
    
    def _load_dictionary(self):
        """Load the bilingual dictionary"""
        if not Path(self.dictionary_path).exists():
            raise FileNotFoundError(f"Dictionary not found at {self.dictionary_path}")
        
        with open(self.dictionary_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def translate_word(self, word):
        """Translate a single word"""
        word_lower = word.lower().strip()
        return self.dictionary.get(word_lower, [word])[0]  # Return first translation or original
    
    def translate_sentence(self, sentence):
        """Translate a complete sentence word by word"""
        words = sentence.split()
        translated_words = []
        
        for word in words:
            # Simple word translation
            translated_word = self.translate_word(word)
            translated_words.append(translated_word)
        
        return ' '.join(translated_words)
    
    def batch_translate(self, sentences):
        """Translate multiple sentences"""
        return [self.translate_sentence(sentence) for sentence in sentences]

def test_baseline():
    """Test the baseline dictionary model"""
    try:
        model = BaselineDictionary()
        test_sentences = [
            "hello world",
            "good morning",
            "how are you",
            "thank you"
        ]
        
        print("🧪 Baseline Dictionary Model Test:")
        for sentence in test_sentences:
            translation = model.translate_sentence(sentence)
            print(f"   '{sentence}' → '{translation}'")
            
    except Exception as e:
        print(f"❌ Baseline test failed: {e}")

if __name__ == "__main__":
    test_baseline()