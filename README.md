# 🌐 Machine Translation Project

> A modular and scalable machine translation system for English-Assamese translation with comprehensive evaluation metrics.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 📋 Table of Contents

- [Overview](#-overview) 
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project implements a **dictionary-based machine translation system** with a focus on English to Assamese translation. It features a modular architecture that separates data processing, model implementation, training, and evaluation into distinct components for easy maintenance and extensibility.

### Key Highlights

- **Modular Architecture**: Clean separation of concerns with dedicated modules for each functionality
- **Comprehensive Evaluation**: Multiple metrics including BLEU, METEOR, chrF, and TER
- **Flexible Pipeline**: Run individual components or the entire pipeline with simple commands
- **Production-Ready**: Includes testing, linting, and code formatting standards

---

## ✨ Features

- 📊 **Data Processing Pipeline**: Automated download, preprocessing, and train-test splitting
- 📖 **Dictionary Builder**: Intelligent bilingual dictionary generation from parallel corpora
- 🔄 **Translation Engine**: Dictionary-based translation with fallback mechanisms
- 📈 **Evaluation Suite**: Multi-metric evaluation (BLEU, METEOR, chrF, TER)
- 🧪 **Testing Framework**: Comprehensive unit tests for all components
- 📝 **Code Quality**: Black formatting and Flake8 linting integration

---

## 📁 Project Structure

```
ML_Project/
│
├── 📁 data/                    # Data storage
│   ├── raw/                    # Original downloaded datasets
│   ├── processed/              # Preprocessed train/test splits
│   └── dictionary/             # Generated bilingual dictionaries
│
├── 📁 src/                     # Source code
│   ├── config/                 # Configuration management
│   ├── data/                   # Data loading and processing
│   ├── models/                 # Translation models
│   ├── evaluation/             # Evaluation metrics
│   ├── training/               # Training utilities
│   └── utils/                  # Helper functions
│
├── 📁 scripts/                 # Automation scripts
├── 📁 experiments/             # Experiment logs and results
├── 📁 notebooks/               # Jupyter notebooks for analysis
├── 📁 tests/                   # Unit and integration tests
│
├── 🚀 run_pipeline.py          # Main pipeline orchestrator
├── 📋 requirements.txt         # Python dependencies
└── 📖 README.md                # Project documentation
```

---

## ⚡ Quick Start

### Prerequisites

Ensure you have the following installed on your system:

- **Python** 3.8 or higher
- **pip** package manager
- **Git** version control

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/Harsh-Raj-Jordan/ML_PROJECT.git
cd ML_Project
```

**2. Set up virtual environment**

<details>
<summary><b>🪟 Windows</b></summary>

```bash
python -m venv .venv
.venv\Scripts\activate
```

Or with PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

</details>

<details>
<summary><b>🐧 Linux / 🍎 macOS</b></summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
```

</details>

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Download required NLTK data**

```bash
python -c "import nltk; nltk.download('punkt')"
```

---

## 🚀 Usage

### Running the Complete Pipeline

Execute the entire translation pipeline with a single command:

```bash
python run_pipeline.py
```

### Running Individual Components

Run specific pipeline steps independently:

| Command | Description |
|---------|-------------|
| `python run_pipeline.py download` | Download raw datasets |
| `python run_pipeline.py preprocess` | Preprocess and split data |
| `python run_pipeline.py dictionary` | Build bilingual dictionary |
| `python run_pipeline.py translate` | Test translation system |
| `python run_pipeline.py evaluate` | Run evaluation metrics |
| `python run_pipeline.py interactive` | Live translation |
| `python run_pipeline.py analyze` | Dictionary analysis |
| `python run_pipeline.py baseline` | Test baseline model |
| `python run_pipeline.py help` | Show all available commands |

### Direct Script Execution

For more granular control, you can run scripts directly:

```bash
# Download data
python scripts/download_data.py

# Preprocess data
python scripts/preprocess_data.py

# Build dictionary
python -c "from src.data.dictionary_builder import build_dictionary; build_dictionary()"

# Test translator
python -c "from src.models.dictionary_translator import main; main()"

# Run evaluation
python -c "from src.evaluation.evaluate import main; main()"
```

---

## 🧪 Testing

### Run Test Suite

Execute all unit tests with verbose output:

```bash
python -m pytest tests/ -v
```

### Code Quality Checks

**Format checking with Black:**

```bash
python -m black src/ scripts/ --check
```

**Linting with Flake8:**

```bash
python -m flake8 src/ scripts/
```

### Sanity Checks

**Verify virtual environment:**

```bash
which python  # Linux/macOS
where python  # Windows
```

**Check installed packages:**

```bash
pip list
```

**Verify data files:**

```bash
python -c "from pathlib import Path; files = ['data/raw/eng_asm.json', 'data/processed/train.json', 'data/processed/test.json', 'data/dictionary/eng_asm_dict.json']; [print(f'✅ {f}' if Path(f).exists() else f'❌ {f}') for f in files]"
```

**Test module imports:**

```bash
# Test dictionary builder
python -c "from src.data.dictionary_builder import build_dictionary; print('✅ Dictionary builder works')"

# Test translator
python -c "from src.models.dictionary_translator import DictionaryTranslator; print('✅ Translator works')"

# Test evaluator
python -c "from src.evaluation.evaluate import TranslationEvaluator; print('✅ Evaluator works')"
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure your code follows the project's coding standards (Black formatting, Flake8 compliance).

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Harsh Raj Jordan** - [@Harsh-Raj-Jordan](https://github.com/Harsh-Raj-Jordan)

---

## 🙏 Acknowledgments

- NLTK for natural language processing utilities
- SacreBLEU for evaluation metrics
- The open-source community for various tools and libraries

---

<div align="center">

**[⬆ Back to Top](#-machine-translation-project)**

Made with ❤️ by [Harsh Raj Jordan](https://github.com/Harsh-Raj-Jordan)

</div>