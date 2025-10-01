"""
Evaluate dictionary-based translation using BLEU and other metrics
"""

import json
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from src.config import settings
from src.models.dictionary_translator import DictionaryTranslator

class TranslationEvaluator:
    def __init__(self):
        self.translator = DictionaryTranslator()
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def load_test_data(self):
        """Load test data for evaluation"""
        if not settings.TEST_DATA_PATH.exists():
            raise FileNotFoundError(f"Test data not found at {settings.TEST_DATA_PATH}")
        
        with open(settings.TEST_DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def evaluate_corpus(self, test_data=None, max_examples=1000):
        """Evaluate translation on the test corpus"""
        if test_data is None:
            test_data = self.load_test_data()
        
        # Limit for faster evaluation
        test_data = test_data[:max_examples]
        
        references = []
        hypotheses = []
        src_sentences = []
        
        print(f"🔄 Evaluating {len(test_data)} test examples...")
        for item in test_data:
            try:
                translation = item["translation"]
                src_sentence = translation[settings.SRC_LANG]
                tgt_sentence = translation[settings.TGT_LANG]
                
                # Translate source sentence
                translated = self.translator.translate_sentence(src_sentence)
                
                src_sentences.append(src_sentence)
                references.append([tgt_sentence.split()])  # BLEU expects list of references
                hypotheses.append(translated.split())
            except KeyError as e:
                print(f"⚠️ Skipping item due to missing key: {e}")
                continue
        
        if not references:
            raise ValueError("No valid test examples found for evaluation!")
        
        # Calculate BLEU score
        bleu_score = corpus_bleu(references, hypotheses, weights=settings.BLEU_WEIGHTS)
        
        # Calculate ROUGE scores
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for i in range(len(references)):
            ref_text = ' '.join(references[i][0])
            hyp_text = ' '.join(hypotheses[i])
            scores = self.scorer.score(ref_text, hyp_text)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
        avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
        avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)
        
        return {
            "bleu": bleu_score,
            "rouge1": avg_rouge1,
            "rouge2": avg_rouge2,
            "rougeL": avg_rougeL,
            "test_size": len(test_data),
            "evaluated_size": len(references),
            "sample_examples": list(zip(src_sentences[:3], 
                                      [' '.join(ref[0]) for ref in references[:3]], 
                                      [' '.join(hyp) for hyp in hypotheses[:3]]))
        }

def evaluate_translation():
    """Convenience function for easy importing"""
    return main()

def main():
    print("=" * 60)
    print("📊 TRANSLATION EVALUATION")
    print("=" * 60)
    
    try:
        evaluator = TranslationEvaluator()
        results = evaluator.evaluate_corpus(max_examples=500)  # Limit for speed
        
        print(f"\n📈 Evaluation Results:")
        print(f"   Test set size: {results['test_size']}")
        print(f"   Evaluated examples: {results['evaluated_size']}")
        print(f"   BLEU Score: {results['bleu']:.4f}")
        print(f"   ROUGE-1 F1: {results['rouge1']:.4f}")
        print(f"   ROUGE-2 F1: {results['rouge2']:.4f}")
        print(f"   ROUGE-L F1: {results['rougeL']:.4f}")
        
        print(f"\n🔍 Sample Translations:")
        for i, (src, ref, hyp) in enumerate(results['sample_examples']):
            print(f"   Example {i + 1}:")
            print(f"      Source: {src}")
            print(f"      Reference: {ref}")
            print(f"      Hypothesis: {hyp}")
            print()
        
        print("✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Evaluation error: {e}")

if __name__ == "__main__":
    main()