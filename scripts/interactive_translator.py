#!/usr/bin/env python3
"""
Enhanced Interactive English-Assamese Translator - TRANSFORMER ONLY
"""

import sys
import torch
from pathlib import Path
from transformers import BertTokenizer
import sentencepiece as spm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.transformer import DirectMTModel
from src.training.transformer_trainer import TransformerTrainer
from src.config import settings

class InteractiveTransformerTranslator:
    def __init__(self, model_path, sp_model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📱 Using device: {self.device}")
        
        # Load tokenizers
        self.bert_tokenizer = BertTokenizer.from_pretrained(settings.TRANSFORMER_CONFIG['model_name'])
        self.sp_tokenizer = spm.SentencePieceProcessor()
        self.sp_tokenizer.load(sp_model_path)
        
        # Load model
        self.model = DirectMTModel(
            bert_model_name=settings.TRANSFORMER_CONFIG['model_name'],
            tgt_vocab_size=self.sp_tokenizer.get_piece_size()
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.trainer = TransformerTrainer(self.model, self.device)
    
    def translate(self, text):
        src_encoding = self.bert_tokenizer(
            text, 
            truncation=True, 
            padding="max_length", 
            max_length=128, 
            return_tensors="pt"
        )
        
        output_ids = self.trainer.beam_search_decode(
            src_encoding["input_ids"].to(self.device), 
            src_encoding["attention_mask"].to(self.device), 
            self.sp_tokenizer
        )
        
        filtered_ids = [id for id in output_ids if id not in [
            self.sp_tokenizer.bos_id(), 
            self.sp_tokenizer.eos_id(), 
            self.sp_tokenizer.pad_id()
        ]]
        
        return self.sp_tokenizer.decode_ids(filtered_ids)
    
    def interactive_session(self):
        print("🤖 Transformer-based English to Assamese Translator")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            text = input("\n🗣️  Enter English text: ").strip()
            if text.lower() in ['quit', 'exit', 'q']:
                break
                
            if not text:
                print("⚠️  Please enter some text.")
                continue
                
            try:
                translation = self.translate(text)
                print(f"📤 Assamese: {translation}")
            except Exception as e:
                print(f"❌ Translation error: {e}")

def main():
    """Main function"""
    print("=" * 70)
    print("🔤 TRANSFORMER ENGLISH-ASSAMESE TRANSLATOR")
    print("=" * 70)
    
    try:
        translator = InteractiveTransformerTranslator(
            model_path=settings.TRANSFORMER_MODEL_SAVE_PATH,
            sp_model_path=str(settings.SP_MODEL_PREFIX) + ".model"
        )
        translator.interactive_session()
    except FileNotFoundError:
        print("❌ Transformer model not found!")
        print("💡 Please run: python scripts/train_transformer.py first")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure the transformer model is trained first")

if __name__ == "__main__":
    main()