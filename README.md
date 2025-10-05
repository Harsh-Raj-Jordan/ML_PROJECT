🌐 Transformer-Based English-Assamese Machine Translation

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

*A state-of-the-art neural machine translation system using Transformer architecture for high-quality English to Assamese translation.*

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Performance](#-performance) • [Documentation](#-documentation)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Highlights](#-key-highlights)
- [Features](#-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Performance](#-performance)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project implements a **Transformer-based neural machine translation system** specifically designed for English to Assamese translation. It leverages state-of-the-art transformer architecture with BERT encoder and custom decoder to achieve high-quality translations with excellent BLEU and ROUGE scores.

### Why This Project?

- 🚀 **Modern Architecture**: Uses cutting-edge Transformer technology
- 📈 **High Accuracy**: Significantly outperforms traditional dictionary-based approaches
- 🛠️ **Easy to Use**: Simple commands to train, evaluate, and translate
- 🔧 **Customizable**: Flexible configuration for different use cases

---

## 🌟 Key Highlights

| Feature | Description |
|---------|-------------|
| **🤖 Transformer Architecture** | BERT encoder + Transformer decoder for superior quality |
| **📊 High Performance** | Achieves excellent BLEU scores (15-25+) |
| **🎯 Beam Search** | Advanced decoding for better translation quality |
| **💬 Interactive Mode** | Real-time translation interface |
| **🔄 Modular Design** | Clean, maintainable codebase |
| **🚀 Production Ready** | Comprehensive evaluation and optimization |

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🧠 Core Features
- ✅ BERT-based encoder
- ✅ Custom Transformer decoder
- ✅ SentencePiece tokenization
- ✅ Teacher forcing training
- ✅ Beam search decoding

</td>
<td width="50%">

### 🔧 Utilities
- ✅ Comprehensive evaluation metrics
- ✅ Interactive translation CLI
- ✅ Progress tracking
- ✅ Model checkpointing
- ✅ Easy configuration

</td>
</tr>
</table>

---

## 🏗️ Architecture

```mermaid
graph LR
    A[English Input] --> B[BERT Tokenizer]
    B --> C[BERT Encoder]
    C --> D[Transformer Decoder]
    D --> E[SentencePiece Decoder]
    E --> F[Assamese Output]
```

### Model Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Encoder** | BERT-base-multilingual | Contextual understanding of English |
| **Decoder** | Custom Transformer | Generate Assamese translation |
| **Tokenizer (EN)** | BERT WordPiece | English text tokenization |
| **Tokenizer (AS)** | SentencePiece | Assamese text tokenization |
| **Training** | Cross-entropy + Teacher forcing | Model optimization |
| **Inference** | Beam search (size=3) | High-quality decoding |

---

## 📁 Project Structure

```
ML_PROJECT/
│
├── 🚀 run_pipeline.py              # Main pipeline orchestrator
├── 📋 requirements.txt             # Python dependencies
├── 📖 README.md                    # This file
│
├── 📁 scripts/                     # Automation scripts
│   ├── download_data.py            # Download dataset from HuggingFace
│   ├── prepare_data.py             # Convert data to training format
│   ├── train_transformer.py        # Train transformer model
│   └── interactive_translator.py  # Interactive translation interface
│
├── 📁 src/                         # Source code
│   ├── config/
│   │   └── settings.py             # Configuration and paths
│   ├── data/
│   │   └── transformer_dataset.py # PyTorch dataset
│   ├── models/
│   │   └── transformer.py          # Model architecture
│   ├── training/
│   │   └── transformer_trainer.py # Training utilities
│   └── evaluation/
│       └── transformer_evaluate.py # Evaluation metrics
│
├── 📁 data/                        # Data storage
│   ├── raw/                        # Original datasets
│   └── processed/                  # Preprocessed data
│
└── 📁 models/                      # Model storage
    └── transformer_model/          # Trained models and tokenizers
```

---

## 🛠️ Installation

### Prerequisites

Before you begin, ensure you have:

- ✅ **Python 3.8+** installed
- ✅ **8GB+ RAM** (recommended)
- ✅ **GPU** (optional, but recommended for training)
- ✅ **Git** installed

### Step-by-Step Setup

#### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Harsh-Raj-Jordan/ML_PROJECT.git
cd ML_PROJECT
```

#### 2️⃣ Create Virtual Environment

<details>
<summary><b>🪟 Windows (PowerShell)</b></summary>

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

</details>

<details>
<summary><b>🪟 Windows (Command Prompt)</b></summary>

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

</details>

<details>
<summary><b>🐧 Linux / 🍎 macOS</b></summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
```

</details>

#### 3️⃣ Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4️⃣ Set Up HuggingFace Token

```bash
huggingface-cli login
```

> **Note**: Get your token from [HuggingFace Settings](https://huggingface.co/settings/tokens)

---

## 🚀 Usage

### Option 1: Complete Pipeline (Recommended)

Run everything with a single command:

```bash
python run_pipeline.py
```

This will:
1. 📥 Download dataset (2-5 min)
2. 🔄 Prepare data and tokenizers (1-2 min)
3. 🤖 Train model (30 min - 2 hrs)
4. 🧪 Evaluate performance (1-2 min)

---

### Option 2: Step-by-Step Execution

Run specific components as needed:

| Command | Action | Time | Description |
|---------|--------|------|-------------|
| `python run_pipeline.py download` | 📥 Download | 2-5 min | Fetch dataset from HuggingFace |
| `python run_pipeline.py prepare` | 🔄 Prepare | 1-2 min | Process data & build tokenizers |
| `python run_pipeline.py train` | 🤖 Train | 30m-2h | Train the transformer model |
| `python run_pipeline.py evaluate` | 🧪 Evaluate | 1-2 min | Test model performance |
| `python run_pipeline.py interactive` | 💬 Translate | Instant | Live translation interface |

---

### Option 3: Direct Script Execution

For advanced users:

```bash
# Download dataset
python scripts/download_data.py

# Prepare data
python scripts/prepare_data.py

# Train model
python scripts/train_transformer.py

# Interactive translation
python scripts/interactive_translator.py
```

---

### 💬 Interactive Translation

Start the interactive translator:

```bash
python run_pipeline.py interactive
```

Example session:

```
🌐 Transformer-Based English-Assamese Translator
================================================

Enter English text (or 'quit' to exit): hello how are you
Translation: নমস্কাৰ আপুনি কেনে আছে

Enter English text (or 'quit' to exit): where is the hospital
Translation: চিকিৎসালয় ক'ত আছে
```

---

## ⚙️ Configuration

### Model Settings

Edit `src/config/settings.py` to customize:

```python
TRANSFORMER_CONFIG = {
    'model_name': 'bert-base-multilingual-cased',
    'vocab_size': 16000,              # Assamese vocabulary size
    'batch_size': 16,                 # Reduce if GPU memory limited
    'num_epochs': 5,                  # More epochs = better quality
    'learning_rate': 5e-5,            # Adjust if training unstable
    'max_len': 128,                   # Maximum sequence length
    'decoder_layers': 4,              # Number of decoder layers
    'decoder_heads': 8,               # Number of attention heads
    'decoder_ff_dim': 2048,           # Feed-forward dimension
    'dropout': 0.1,                   # Dropout rate
    'beam_size': 3,                   # Beam search width
}
```

### Training Tips

| Parameter | Recommendation | Effect |
|-----------|---------------|--------|
| **batch_size** | 8-32 | Higher = faster but more memory |
| **num_epochs** | 5-10 | More = better quality (diminishing returns) |
| **learning_rate** | 1e-5 to 1e-4 | Lower = more stable training |
| **decoder_layers** | 4-6 | More = better capacity but slower |

---

## 📊 Performance

### Expected Results

| Metric | Score Range | Interpretation |
|--------|-------------|----------------|
| **BLEU** | 15-25+ | Translation quality (higher is better) |
| **ROUGE-1** | 0.4-0.6 | Word overlap with reference |
| **ROUGE-2** | 0.3-0.5 | Bigram overlap |
| **Training Loss** | < 2.0 | Model convergence |

### Sample Translations

| English Input | Assamese Output | Quality |
|---------------|----------------|---------|
| "hello how are you today" | "নমস্কাৰ আপুনি আজি কেনে আছে" | ⭐⭐⭐⭐⭐ |
| "where is the nearest hospital" | "সৰ্বাধিক চিকিৎসালয় ক'ত আছে" | ⭐⭐⭐⭐⭐ |
| "thank you very much" | "বহুত ধন্যবাদ" | ⭐⭐⭐⭐⭐ |

---

## 🐛 Troubleshooting

### Common Issues & Solutions

<details>
<summary><b>❌ "HuggingFace token not found"</b></summary>

**Solution:**
```bash
huggingface-cli login
```
Then enter your token from [here](https://huggingface.co/settings/tokens)

</details>

<details>
<summary><b>❌ "CUDA out of memory"</b></summary>

**Solutions:**
1. Reduce `batch_size` in `src/config/settings.py`
2. Use CPU instead: Set `device='cpu'`
3. Clear GPU cache:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

</details>

<details>
<summary><b>❌ "Module not found"</b></summary>

**Solution:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

</details>

<details>
<summary><b>⏱️ Training is too slow</b></summary>

**Solutions:**
1. Enable GPU if available
2. Reduce `batch_size` or `max_len`
3. Use smaller dataset for testing
4. Reduce `num_epochs`

</details>

### Verification Commands

Check your installation:

```bash
# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Check if data exists
python -c "from pathlib import Path; print('Data:', Path('data/raw/eng_asm.json').exists())"

# Check if model exists
python -c "from pathlib import Path; print('Model:', Path('models/transformer_model').exists())"
```

---

## 🔮 Future Enhancements

- [ ] Model quantization for faster inference
- [ ] Web interface for translation
- [ ] Batch translation of documents
- [ ] REST API deployment
- [ ] Multi-language support
- [ ] Domain-specific fine-tuning
- [ ] Mobile app integration

---

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Development Setup

```bash
# Fork and clone
git clone https://github.com/your-username/ML_PROJECT.git
cd ML_PROJECT

# Install dependencies
pip install -r requirements.txt

# Make changes and test
python run_pipeline.py

# Submit PR
git add .
git commit -m "Your descriptive commit message"
git push origin feature/your-feature
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👥 Author

**Harsh Raj Jordan**
- GitHub: [@Harsh-Raj-Jordan](https://github.com/Harsh-Raj-Jordan)
- Project Link: [ML_PROJECT](https://github.com/Harsh-Raj-Jordan/ML_PROJECT)

---

## 🙏 Acknowledgments

- [HuggingFace](https://huggingface.co/) for transformers library and datasets
- [Google Research](https://research.google/) for Transformer architecture
- [AI4Bharat](https://ai4bharat.org/) for the BPCC dataset
- [PyTorch](https://pytorch.org/) team for the deep learning framework

---

## 📞 Support

Need help? Here's what to do:

1. 📖 Check the [Troubleshooting](#-troubleshooting) section
2. 🔍 Search [existing issues](https://github.com/Harsh-Raj-Jordan/ML_PROJECT/issues)
3. 🆕 Create a [new issue](https://github.com/Harsh-Raj-Jordan/ML_PROJECT/issues/new) with details

---

<div align="center">

Made with ❤️ by Harsh Raj Jordan

[⬆ Back to Top](#-transformer-based-english-assamese-machine-translation)

</div>
