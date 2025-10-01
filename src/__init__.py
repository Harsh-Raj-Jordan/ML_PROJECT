"""
Dictionary-Based Machine Translation for Low-Resource Languages
English to Assamese Translation System
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main components
from src.config import settings

# Import data modules
from src.data.dataset_loader import load_eng_asm_dataset, load_processed_data
from src.data.dictionary_builder import DictionaryBuilder, build_dictionary
from src.data.preprocess import main as preprocess_data

# Import models
from src.models.baseline_dictionary import BaselineDictionary
from src.models.dictionary_translator import DictionaryTranslator
from src.models.transformer import TransformerModel

# Import evaluation
from src.evaluation.evaluate import TranslationEvaluator

# Import training and utils
from src.training.trainer import ModelTrainer
from src.utils.helpers import setup_logging, save_results, load_results

# Define what gets imported with "from src import *"
__all__ = [
    'settings',
    'load_eng_asm_dataset',
    'load_processed_data',
    'DictionaryBuilder',
    'build_dictionary',
    'preprocess_data',
    'BaselineDictionary',
    'DictionaryTranslator',
    'TransformerModel',
    'TranslationEvaluator',
    'ModelTrainer',
    'setup_logging',
    'save_results',
    'load_results'
]