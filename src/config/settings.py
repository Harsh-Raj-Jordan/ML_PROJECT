from pathlib import Path

# Define the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# File paths
RAW_DATA_PATH = RAW_DATA_DIR / "eng_asm.json"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.json"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.json"

# Dataset parameters (needed for download_data.py)
SRC_LANG = "eng_Latn"
TGT_LANG = "asm_Beng"
DATASET_NAME = "ai4bharat/BPCC"
DATASET_CONFIG = "bpcc-seed-latest"
MIN_SENTENCE_LENGTH = 1
MAX_SENTENCE_LENGTH = 30
TRAIN_TEST_SPLIT = 0.9

# Transformer Model Configuration
TRANSFORMER_CONFIG = {
    'model_name': 'bert-base-multilingual-cased',
    'vocab_size': 16000,
    'batch_size': 16,
    'num_epochs': 5,
    'learning_rate': 5e-5,
    'max_len': 128,
    'decoder_layers': 4,
    'decoder_heads': 8,
    'decoder_ff_dim': 2048,
    'dropout': 0.1,
    'beam_size': 3,
    'seed': 42,
}

# Transformer Paths
TRANSFORMER_MODEL_PATH = PROJECT_ROOT / "models" / "transformer_model"
TRANSFORMER_MODEL_SAVE_PATH = TRANSFORMER_MODEL_PATH / "direct_en_bn_model.pt"
SP_MODEL_PREFIX = TRANSFORMER_MODEL_PATH / "spm_assamese"
TRANSFORMER_OUTPUT_DIR = TRANSFORMER_MODEL_PATH / "output"

# Ensure all required directories exist
for directory in [TRANSFORMER_MODEL_PATH, TRANSFORMER_OUTPUT_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)