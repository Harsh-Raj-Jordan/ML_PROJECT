"""
Transformer-Based Machine Translation for English-Assamese
Neural Machine Translation System using Transformer Architecture
"""

__version__ = "2.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main components
from src.config import settings

# Import transformer modules
from src.data.transformer_dataset import TransformerTranslationDataset
from src.models.transformer import DirectMTModel
from src.training.transformer_trainer import TransformerTrainer
from src.evaluation.transformer_evaluate import TransformerEvaluator

# Define what gets imported with "from src import *"
__all__ = [
    'settings',
    'TransformerTranslationDataset',
    'DirectMTModel', 
    'TransformerTrainer',
    'TransformerEvaluator'
]