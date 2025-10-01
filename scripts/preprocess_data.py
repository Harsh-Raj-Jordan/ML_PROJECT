# scripts/download_data.py

#!/usr/bin/env python3
"""
Preprocess and split the dataset into train/test sets
"""

import sys
from pathlib import Path

# Add src to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocess import main as preprocess_main

def main():
    print("=" * 60)
    print("🔄 DATA PREPROCESSING SCRIPT")
    print("=" * 60)
    
    preprocess_main()

if __name__ == "__main__":
    main()