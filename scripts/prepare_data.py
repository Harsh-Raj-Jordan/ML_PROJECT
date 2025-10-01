import json
from src.data.preprocess import preprocess
from src.models.dictionary_translator import build_dictionary

RAW_PATH = "data/raw/eng_asm.json"

def main():
    print("📥 Loading raw data...")
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print("🧹 Preprocessing...")
    train, test = preprocess(raw_data)

    print("📖 Building dictionary...")
    build_dictionary(train)

    print("✅ Data pipeline completed!")

if __name__ == "__main__":
    main()
