#!/usr/bin/env python3
"""
Train Transformer Model for English-Assamese Translation
"""

import sys
from pathlib import Path
import torch
from transformers import BertTokenizer
import sentencepiece as spm
from src.config import settings
from src.data.transformer_dataset import TransformerTranslationDataset
from src.models.transformer import DirectMTModel
from src.training.transformer_trainer import TransformerTrainer
from src.evaluation.transformer_evaluate import TransformerEvaluator
from scripts.prepare_data import prepare_data
from torch.utils.data import DataLoader

def train_sentencepiece(file_path, model_prefix, vocab_size=16000):
    """Train SentencePiece tokenizer for Assamese"""
    print("\n--- Training SentencePiece Tokenizer for Assamese ---")
    import pandas as pd
    import sentencepiece as spm
    
    df = pd.read_csv(file_path).dropna()
    tgt_text_file = settings.TRANSFORMER_MODEL_PATH / "temp_tgt_for_spm.txt"
    df['tgt'].to_csv(tgt_text_file, header=False, index=False, sep='\t', quotechar='"')
    
    spm.SentencePieceTrainer.Train(
        f"--input={tgt_text_file} --model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} --model_type=bpe "
        f"--pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3"
    )
    print(f"✅ SentencePiece model trained with vocab size: {vocab_size}")

def main():
    print("=" * 70)
    print("🚀 TRANSFORMER ENGLISH-ASSAMESE TRANSLATION TRAINING")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Using device: {device}")
    
    try:
        # Step 1: Prepare data (convert JSON to CSV format)
        print("\n📥 Preparing data...")
        train_file, valid_file = prepare_data()
        
        # Step 2: Train SentencePiece tokenizer
        train_sentencepiece(train_file, settings.SP_MODEL_PREFIX, settings.TRANSFORMER_CONFIG['vocab_size'])
        
        # Step 3: Load tokenizers
        bert_tokenizer = BertTokenizer.from_pretrained(settings.TRANSFORMER_CONFIG['model_name'])
        sp_tokenizer = spm.SentencePieceProcessor()
        sp_tokenizer.load(str(settings.SP_MODEL_PREFIX) + ".model")
        
        # Step 4: Create datasets
        print("\n📚 Creating datasets...")
        train_dataset = TransformerTranslationDataset(
            train_file, bert_tokenizer, sp_tokenizer, 
            max_len=settings.TRANSFORMER_CONFIG['max_len']
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=settings.TRANSFORMER_CONFIG['batch_size'], 
            shuffle=True
        )
        
        # Step 5: Initialize model
        print("🤖 Initializing transformer model...")
        model = DirectMTModel(
            bert_model_name=settings.TRANSFORMER_CONFIG['model_name'],
            tgt_vocab_size=sp_tokenizer.get_piece_size(),
            max_len=settings.TRANSFORMER_CONFIG['max_len'],
            decoder_layers=settings.TRANSFORMER_CONFIG['decoder_layers'],
            decoder_heads=settings.TRANSFORMER_CONFIG['decoder_heads'],
            decoder_ff_dim=settings.TRANSFORMER_CONFIG['decoder_ff_dim'],
            dropout=settings.TRANSFORMER_CONFIG['dropout']
        ).to(device)
        
        # Step 6: Train model
        trainer = TransformerTrainer(model, device, learning_rate=settings.TRANSFORMER_CONFIG['learning_rate'])
        criterion = torch.nn.CrossEntropyLoss(ignore_index=sp_tokenizer.pad_id())
        
        print(f"\n🔥 Starting training for {settings.TRANSFORMER_CONFIG['num_epochs']} epochs...")
        for epoch in range(settings.TRANSFORMER_CONFIG['num_epochs']):
            train_loss = trainer.train_one_epoch(train_loader, criterion)
            print(f"📈 Epoch {epoch+1}/{settings.TRANSFORMER_CONFIG['num_epochs']} | Training Loss: {train_loss:.4f}")
        
        # Step 7: Evaluate
        print("\n📊 Evaluating model...")
        evaluator = TransformerEvaluator(trainer, bert_tokenizer, sp_tokenizer, device)
        bleu_score, hypotheses, references = evaluator.evaluate(valid_file)
        print(f"🎯 Validation BLEU Score: {bleu_score:.2f}")
        
        # Step 8: Save model
        trainer.save_model(settings.TRANSFORMER_MODEL_SAVE_PATH)
        
        # Step 9: Test on sample sentences
        test_sentences = [
            "hello world",
            "good morning", 
            "how are you",
            "thank you",
            "where is the house"
        ]
        evaluator.evaluate_sample_sentences(test_sentences)
        
        print(f"\n✅ Transformer training completed successfully!")
        print(f"💾 Model saved to: {settings.TRANSFORMER_MODEL_SAVE_PATH}")
        
    except Exception as e:
        print(f"❌ Transformer training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()