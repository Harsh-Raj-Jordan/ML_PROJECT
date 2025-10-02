import os
from pathlib import Path

# Project root directory (fixed to work from any location)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DICTIONARY_DIR = DATA_DIR / "dictionary"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
LOGS_DIR = PROJECT_ROOT / "logs"

# File paths
RAW_DATA_PATH = RAW_DATA_DIR / "eng_asm.json"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.json"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.json"
DICTIONARY_PATH = DICTIONARY_DIR / "eng_asm_dict.json"

# Dataset parameters
SRC_LANG = "eng_Latn"
TGT_LANG = "asm_Beng"
TRAIN_TEST_SPLIT = 0.8
MIN_SENTENCE_LENGTH = 3
MAX_SENTENCE_LENGTH = 50

# Dictionary parameters
MIN_WORD_FREQUENCY = 1
MAX_TRANSLATIONS_PER_WORD = 3

# Evaluation parameters
BLEU_WEIGHTS = (0.25, 0.25, 0.25, 0.25)  # Uniform weights for BLEU-4

# Interactive Translation Settings
MAX_SENTENCE_LENGTH_TRANSLATE = 20  # Maximum words for interactive translation
SHOW_COVERAGE_PERCENTAGE = True     # Show coverage percentage in analysis
SHOW_WORD_ANALYSIS = True          # Show word-by-word analysis
SAVE_SESSION_HISTORY = True        # Save translation sessions

# Session Management
SESSION_HISTORY_PATH = PROJECT_ROOT / "results" / "translation_sessions"

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, DICTIONARY_DIR, EXPERIMENTS_DIR, LOGS_DIR, SESSION_HISTORY_PATH]:
    directory.mkdir(parents=True, exist_ok=True)

# HuggingFace dataset config
DATASET_NAME = "ai4bharat/BPCC"
DATASET_CONFIG = "bpcc-seed-latest"