"""
Transformer Evaluation with BLEU and ROUGE
"""

import sacrebleu
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer
import sentencepiece as spm

class TransformerEvaluator:
    def __init__(self, trainer, bert_tokenizer, sp_tokenizer, device, max_len=128, beam_size=3):
        self.trainer = trainer
        self.bert_tokenizer = bert_tokenizer
        self.sp_tokenizer = sp_tokenizer
        self.device = device
        self.max_len = max_len
        self.beam_size = beam_size
    
    def generate_translation(self, sentence):
        src_encoding = self.bert_tokenizer(
            sentence, 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_len, 
            return_tensors="pt"
        )
        output_ids = self.trainer.beam_search_decode(
            src_encoding["input_ids"].to(self.device), 
            src_encoding["attention_mask"].to(self.device), 
            self.sp_tokenizer,
            self.beam_size,
            self.max_len
        )
        filtered_ids = [id for id in output_ids if id not in [self.sp_tokenizer.bos_id(), self.sp_tokenizer.eos_id(), self.sp_tokenizer.pad_id()]]
        return self.sp_tokenizer.decode_ids(filtered_ids)
    
    def evaluate(self, validation_file):
        self.trainer.model.eval()
        hypotheses = []
        references = []
        
        val_df = pd.read_csv(validation_file).dropna()
        for _, row in tqdm(val_df.iterrows(), desc="Evaluating", total=len(val_df)):
            src_text = str(row['src'])
            ref_text = str(row['tgt'])
            hyp_text = self.generate_translation(src_text)
            hypotheses.append(hyp_text)
            references.append(ref_text)
            
        bleu = sacrebleu.corpus_bleu(hypotheses, [references])
        return bleu.score, hypotheses, references

    def evaluate_sample_sentences(self, test_sentences):
        """Evaluate on specific test sentences"""
        print("\n🧪 Testing Transformer on Sample Sentences:")
        results = []
        for sentence in test_sentences:
            translation = self.generate_translation(sentence)
            results.append((sentence, translation))
            print(f"   '{sentence}' → '{translation}'")
        return results