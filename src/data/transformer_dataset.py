"""
Transformer Dataset for English-Assamese Translation
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import sentencepiece as spm

class TransformerTranslationDataset(Dataset):
    def __init__(self, file_path, src_tokenizer, tgt_tokenizer, max_len=128):
        df = pd.read_csv(file_path).dropna()
        self.src_texts = df["src"].tolist()
        self.tgt_texts = df["tgt"].tolist()
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = str(self.src_texts[idx])
        tgt_text = str(self.tgt_texts[idx])

        # Tokenize source with BERT tokenizer
        src_encoding = self.src_tokenizer(
            src_text, 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_len, 
            return_tensors="pt"
        )

        # Tokenize target with SentencePiece
        tgt_ids = [self.tgt_tokenizer.bos_id()] + self.tgt_tokenizer.encode_as_ids(tgt_text) + [self.tgt_tokenizer.eos_id()]
        padding_len = self.max_len - len(tgt_ids)
        tgt_ids = (tgt_ids + [self.tgt_tokenizer.pad_id()] * padding_len)[:self.max_len]

        return {
            "src_input_ids": src_encoding["input_ids"].squeeze(0),
            "src_attention_mask": src_encoding["attention_mask"].squeeze(0),
            "tgt_input_ids": torch.tensor(tgt_ids, dtype=torch.long)
        }