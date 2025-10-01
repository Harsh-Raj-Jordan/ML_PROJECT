"""
Utility functions for the translation project
"""

import json
import logging
from datetime import datetime
from pathlib import Path

def setup_logging(log_dir="logs"):
    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def save_results(results, filepath):
    """Save evaluation results to JSON"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Results saved to: {filepath}")

def load_results(filepath):
    """Load evaluation results from JSON"""
    filepath = Path(filepath)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def print_banner(text):
    """Print a formatted banner"""
    print("\n" + "=" * 60)
    print(f"🎯 {text}")
    print("=" * 60)

def format_metrics(metrics):
    """Format evaluation metrics for display"""
    formatted = []
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted.append(f"{key}: {value:.4f}")
        else:
            formatted.append(f"{key}: {value}")
    return " | ".join(formatted)