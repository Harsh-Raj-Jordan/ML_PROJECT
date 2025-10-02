"""
Session management for translation history and analytics
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from src.config import settings

class SessionManager:
    def __init__(self):
        self.session_data = []
        self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Ensure session directory exists
        settings.SESSION_HISTORY_PATH.mkdir(parents=True, exist_ok=True)
    
    def add_translation(self, input_text, output_text, word_analysis=None):
        """Add a translation to session history"""
        translation_record = {
            'timestamp': datetime.now().isoformat(),
            'input': input_text,
            'output': output_text,
            'word_analysis': word_analysis or {},
            'session_id': self.current_session_id
        }
        self.session_data.append(translation_record)
        return translation_record
    
    def get_session_stats(self):
        """Get statistics for current session"""
        if not self.session_data:
            return {}
        
        total_translations = len(self.session_data)
        total_words = sum(len(record['input'].split()) for record in self.session_data)
        successful_translations = sum(1 for record in self.session_data 
                                    if record['output'] != record['input'])
        
        # Calculate average coverage
        coverages = [analysis.get('coverage', 0) for record in self.session_data 
                    if (analysis := record.get('word_analysis'))]
        avg_coverage = sum(coverages) / len(coverages) if coverages else 0
        
        return {
            'total_translations': total_translations,
            'total_words': total_words,
            'successful_translations': successful_translations,
            'success_rate': (successful_translations / total_translations) * 100,
            'average_coverage': avg_coverage,
            'session_duration': f"Started at {self.session_data[0]['timestamp']}"
        }
    
    def save_session(self, format='json'):
        """Save session data to file"""
        if not self.session_data:
            print("📭 No session data to save.")
            return None
        
        filename = f"translation_session_{self.current_session_id}"
        
        if format == 'json':
            filepath = settings.SESSION_HISTORY_PATH / f"{filename}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'session_info': self.get_session_stats(),
                    'translations': self.session_data
                }, f, ensure_ascii=False, indent=2)
        
        elif format == 'csv':
            filepath = settings.SESSION_HISTORY_PATH / f"{filename}.csv"
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Input', 'Output', 'Coverage'])
                for record in self.session_data:
                    coverage = record.get('word_analysis', {}).get('coverage', 0)
                    writer.writerow([
                        record['timestamp'],
                        record['input'],
                        record['output'],
                        f"{coverage:.1f}%"
                    ])
        
        print(f"💾 Session saved to: {filepath}")
        return filepath
    
    def export_word_gaps(self):
        """Export words that couldn't be translated for dictionary improvement"""
        missing_words = set()
        for record in self.session_data:
            analysis = record.get('word_analysis', {})
            missing_words.update(analysis.get('missing', []))
        
        if missing_words:
            gaps_file = settings.SESSION_HISTORY_PATH / f"vocabulary_gaps_{self.current_session_id}.txt"
            with open(gaps_file, 'w', encoding='utf-8') as f:
                f.write("Vocabulary Gaps Found in Session\n")
                f.write("=" * 40 + "\n")
                for word in sorted(missing_words):
                    f.write(f"{word}\n")
            print(f"📝 Vocabulary gaps exported: {gaps_file}")
            return list(missing_words)
        else:
            print("✅ No vocabulary gaps found in this session!")
            return []