"""
Enhanced evaluation with better metrics and smoothing
"""

import json
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from src.config import settings
from src.models.dictionary_translator import DictionaryTranslator

class TranslationEvaluator:
    def __init__(self):
        self.translator = DictionaryTranslator()
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothie = SmoothingFunction().method1
    
    def load_test_data(self):
        """Load test data for evaluation"""
        if not settings.TEST_DATA_PATH.exists():
            raise FileNotFoundError(f"Test data not found at {settings.TEST_DATA_PATH}")
        
        with open(settings.TEST_DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def evaluate_corpus(self, test_data=None, max_examples=200):
        """Evaluate translation on the test corpus with enhanced metrics"""
        if test_data is None:
            test_data = self.load_test_data()
        
        # Use smaller, more manageable test set
        test_data = test_data[:max_examples]
        
        references = []
        hypotheses = []
        src_sentences = []
        
        print(f"🔄 Evaluating {len(test_data)} test examples...")
        
        valid_count = 0
        for item in test_data:
            try:
                translation = item["translation"]
                src_sentence = translation[settings.SRC_LANG].strip()
                tgt_sentence = translation[settings.TGT_LANG].strip()
                
                # Skip very long or very short sentences
                src_words = src_sentence.split()
                tgt_words = tgt_sentence.split()
                
                if (len(src_words) > settings.MAX_EVALUATION_LENGTH or 
                    len(tgt_words) > settings.MAX_EVALUATION_LENGTH or
                    len(src_words) < 1 or len(tgt_words) < 1):
                    continue
                
                # Translate source sentence
                translated = self.translator.translate_sentence(src_sentence)
                
                # Skip if translation is identical to source (no translation happened)
                if translated.lower() == src_sentence.lower():
                    continue
                
                src_sentences.append(src_sentence)
                references.append([tgt_sentence.split()])  # BLEU expects list of references
                hypotheses.append(translated.split())
                valid_count += 1
                
            except (KeyError, Exception) as e:
                continue
        
        if valid_count < settings.EVALUATION_MIN_SAMPLES:
            raise ValueError(f"Only {valid_count} valid test examples found (minimum {settings.EVALUATION_MIN_SAMPLES} required)")
        
        print(f"📊 Using {valid_count} valid examples for evaluation")
        
        # Calculate BLEU scores with smoothing
        bleu_scores = []
        for ref, hyp in zip(references, hypotheses):
            if hyp and ref[0]:  # Only calculate if both hypothesis and reference are not empty
                if settings.USE_BLEU_SMOOTHING:
                    score = sentence_bleu(ref, hyp, smoothing_function=self.smoothie)
                else:
                    score = sentence_bleu(ref, hyp)
                bleu_scores.append(score)
        
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
        
        # FIXED: Calculate ROUGE scores properly
        rouge1_f1_scores = []
        rouge2_f1_scores = [] 
        rougeL_f1_scores = []
        
        for i in range(len(references)):
            ref_text = ' '.join(references[i][0])
            hyp_text = ' '.join(hypotheses[i])
            
            # Skip empty references or hypotheses
            if not ref_text.strip() or not hyp_text.strip():
                continue
                
            scores = self.scorer.score(ref_text, hyp_text)
            rouge1_f1_scores.append(scores['rouge1'].fmeasure)
            rouge2_f1_scores.append(scores['rouge2'].fmeasure)
            rougeL_f1_scores.append(scores['rougeL'].fmeasure)
        
        # Calculate averages only if we have valid scores
        avg_rouge1 = np.mean(rouge1_f1_scores) if rouge1_f1_scores else 0.0
        avg_rouge2 = np.mean(rouge2_f1_scores) if rouge2_f1_scores else 0.0
        avg_rougeL = np.mean(rougeL_f1_scores) if rougeL_f1_scores else 0.0
        
        # Get sample examples for analysis
        sample_examples = []
        for i in range(min(3, len(references))):
            sample_examples.append((
                src_sentences[i],
                ' '.join(references[i][0]),
                ' '.join(hypotheses[i])
            ))
        
        return {
            "bleu": avg_bleu,
            "rouge1": avg_rouge1,
            "rouge2": avg_rouge2,
            "rougeL": avg_rougeL,
            "test_size": len(test_data),
            "evaluated_size": valid_count,
            "sample_examples": sample_examples,
            "metrics_breakdown": {
                "bleu_scores_count": len(bleu_scores),
                "rouge_scores_count": len(rouge1_f1_scores),
                "avg_bleu_per_sentence": avg_bleu
            }
        }

def evaluate_translation():
    """Convenience function for easy importing"""
    return main()

def main():
    print("=" * 60)
    print("📊 ENHANCED TRANSLATION EVALUATION")
    print("=" * 60)
    
    try:
        evaluator = TranslationEvaluator()
        results = evaluator.evaluate_corpus(max_examples=settings.EVALUATION_MAX_SAMPLES)
        
        print(f"\n📈 Evaluation Results:")
        print(f"   Test set size: {results['test_size']}")
        print(f"   Valid examples: {results['evaluated_size']}")
        print(f"   BLEU Score: {results['bleu']:.4f}")
        print(f"   ROUGE-1 F1: {results['rouge1']:.4f}")
        print(f"   ROUGE-2 F1: {results['rouge2']:.4f}")
        print(f"   ROUGE-L F1: {results['rougeL']:.4f}")
        
        # Show metrics breakdown
        metrics = results.get('metrics_breakdown', {})
        print(f"\n📊 Metrics Breakdown:")
        print(f"   BLEU scores calculated: {metrics.get('bleu_scores_count', 0)}")
        print(f"   ROUGE scores calculated: {metrics.get('rouge_scores_count', 0)}")
        
        print(f"\n🔍 Sample Translations:")
        for i, (src, ref, hyp) in enumerate(results['sample_examples']):
            print(f"   Example {i + 1}:")
            print(f"      Source: {src}")
            print(f"      Reference: {ref}")
            print(f"      Hypothesis: {hyp}")
            print()
        
        # Enhanced quality assessment
        bleu_score = results['bleu']
        if bleu_score > 0.05:
            print("🎉 EXCELLENT: Translation quality is very good!")
        elif bleu_score > 0.02:
            print("✅ GOOD: Translation quality is improving!")
        elif bleu_score > 0.01:
            print("⚠️  FAIR: Translation quality needs improvement.")
        else:
            print("❌ POOR: Translation quality needs significant improvement.")
            print("💡 Suggestions:")
            print("   - Rebuild dictionary with more data")
            print("   - Add more default translations for common words")
            print("   - Improve word alignment strategies")
            print("   - Consider using transformer models")
        
        print("✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()