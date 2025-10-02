"""
Real-time translation quality metrics
"""

import numpy as np
from src.config import settings

class QualityMetrics:
    def __init__(self):
        self.metrics_history = []
    
    def calculate_sentence_quality(self, source, translation, reference=None):
        """Calculate quality metrics for a single translation"""
        source_words = source.split()
        translation_words = translation.split()
        
        # Basic metrics
        word_count_ratio = len(translation_words) / len(source_words) if source_words else 1
        unchanged_words = len(set(source_words) & set(translation_words))
        preservation_ratio = unchanged_words / len(source_words) if source_words else 0
        
        # Dictionary coverage (approximate)
        dictionary_coverage = self.estimate_coverage(translation)
        
        metrics = {
            'word_count_ratio': word_count_ratio,
            'preservation_ratio': preservation_ratio,
            'dictionary_coverage': dictionary_coverage,
            'quality_score': self.calculate_quality_score(
                word_count_ratio, preservation_ratio, dictionary_coverage
            )
        }
        
        if reference:
            metrics['exact_match'] = translation == reference
            metrics['word_overlap'] = len(set(translation_words) & set(reference.split())) / len(set(reference.split()))
        
        self.metrics_history.append(metrics)
        return metrics
    
    def estimate_coverage(self, translation):
        """Estimate what percentage of words came from dictionary vs remained unchanged"""
        # This is a simplified estimation - you could enhance this
        words = translation.split()
        if not words:
            return 0
        
        # Count words that look like they might be Assamese (non-ASCII)
        assamese_words = sum(1 for word in words if any(ord(char) > 127 for char in word))
        return (assamese_words / len(words)) * 100
    
    def calculate_quality_score(self, word_ratio, preservation_ratio, coverage):
        """Calculate overall quality score (0-100)"""
        # Weights for different factors
        weights = {
            'word_ratio': 0.3,      # Balanced sentence length
            'preservation': 0.2,    # Not too many unchanged words
            'coverage': 0.5         # High dictionary coverage
        }
        
        # Normalize word ratio (ideal is around 1.0)
        word_score = 100 * (1 - abs(1 - word_ratio))
        
        # Preservation should be low (we want translation, not copying)
        preservation_score = 100 * (1 - preservation_ratio)
        
        # Coverage should be high
        coverage_score = coverage
        
        quality_score = (
            weights['word_ratio'] * word_score +
            weights['preservation'] * preservation_score +
            weights['coverage'] * coverage_score
        )
        
        return min(100, max(0, quality_score))
    
    def get_session_quality_report(self):
        """Get quality report for entire session"""
        if not self.metrics_history:
            return {
                'average_quality': 0,
                'total_translations': 0,
                'quality_trend': 'stable'
            }
        
        avg_quality = np.mean([m['quality_score'] for m in self.metrics_history])
        avg_coverage = np.mean([m['dictionary_coverage'] for m in self.metrics_history])
        
        return {
            'total_translations': len(self.metrics_history),
            'average_quality_score': f"{avg_quality:.1f}%",
            'average_coverage': f"{avg_coverage:.1f}%",
            'quality_trend': 'improving' if len(self.metrics_history) > 1 and 
                            self.metrics_history[-1]['quality_score'] > self.metrics_history[0]['quality_score'] else 'stable'
        }