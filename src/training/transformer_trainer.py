"""
Transformer Trainer for English-Assamese Translation
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import math

class TransformerTrainer:
    def __init__(self, model, device, learning_rate=5e-5):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
    def train_one_epoch(self, dataloader, criterion):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc="Training Epoch"):
            self.optimizer.zero_grad()
            
            src_input_ids = batch["src_input_ids"].to(self.device)
            src_mask = batch["src_attention_mask"].to(self.device)
            tgt_ids = batch["tgt_input_ids"].to(self.device)
            
            decoder_input = tgt_ids[:, :-1]
            labels = tgt_ids[:, 1:]
            
            logits = self.model(src_input_ids, src_mask, decoder_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    def beam_search_decode(self, src_input_ids, src_attention_mask, sp_tokenizer, beam_size=3, max_len=128):
        self.model.eval()
        with torch.no_grad():
            memory = self.model.encoder(input_ids=src_input_ids, attention_mask=src_attention_mask).last_hidden_state
            memory_key_padding_mask = (src_attention_mask == 0).repeat_interleave(beam_size, dim=0)
            memory = memory.repeat_interleave(beam_size, dim=0)
            
            sequences = torch.full((beam_size, 1), sp_tokenizer.bos_id(), dtype=torch.long, device=self.device)
            scores = torch.zeros(beam_size, device=self.device)
            
            for _ in range(max_len - 1):
                tgt_emb = self.model.pos_encoder(self.model.tgt_embedding(sequences) * math.sqrt(self.model.d_model))
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(sequences.size(1)).to(self.device)
                
                output = self.model.transformer_decoder(tgt_emb, memory, tgt_mask, memory_key_padding_mask=memory_key_padding_mask)
                logits = self.model.output_projection(output[:, -1, :])
                log_probs = torch.log_softmax(logits, dim=-1) + scores.unsqueeze(1)
                
                scores, indices = torch.topk(log_probs.view(-1), beam_size)
                beam_indices = torch.div(indices, log_probs.size(1), rounding_mode='floor')
                token_indices = indices % log_probs.size(1)
                
                new_sequences = [torch.cat([sequences[beam_idx], token_idx.view(1)]) for beam_idx, token_idx in zip(beam_indices, token_indices)]
                sequences = torch.stack(new_sequences)
                
                if (sequences[:, -1] == sp_tokenizer.eos_id()).any(): 
                    break
                    
            best_seq_idx = scores.argmax()
            return sequences[best_seq_idx].tolist()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"💾 Model saved to: {path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"📥 Model loaded from: {path}")