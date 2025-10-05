"""
Transformer model for English-Assamese Translation - REAL IMPLEMENTATION
"""

import torch
import torch.nn as nn
import math
from transformers import BertModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    
    def forward(self, x): 
        return x + self.pe[:, :x.size(1), :]

class DirectMTModel(nn.Module):
    def __init__(self, bert_model_name, tgt_vocab_size, max_len=128, decoder_layers=4, decoder_heads=8, decoder_ff_dim=2048, dropout=0.1):
        super().__init__()
        self.encoder = BertModel.from_pretrained(bert_model_name)
        self.d_model = self.encoder.config.hidden_size
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, self.d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=max_len)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model, 
            nhead=decoder_heads, 
            dim_feedforward=decoder_ff_dim, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.output_projection = nn.Linear(self.d_model, tgt_vocab_size)

    def forward(self, src_input_ids, src_attention_mask, tgt_input_ids):
        memory = self.encoder(input_ids=src_input_ids, attention_mask=src_attention_mask).last_hidden_state
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt_input_ids) * math.sqrt(self.d_model))
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input_ids.size(1)).to(src_input_ids.device)
        memory_key_padding_mask = (src_attention_mask == 0)
        tgt_key_padding_mask = (tgt_input_ids == 0)
        
        decoder_output = self.transformer_decoder(
            tgt=tgt_emb, 
            memory=memory, 
            tgt_mask=tgt_mask, 
            tgt_key_padding_mask=tgt_key_padding_mask, 
            memory_key_padding_mask=memory_key_padding_mask
        )
        return self.output_projection(decoder_output)

def main():
    """Demo the transformer model structure"""
    print("🤖 Transformer Model - REAL IMPLEMENTATION")
    model = DirectMTModel(bert_model_name='bert-base-multilingual-cased', tgt_vocab_size=16000)
    print(f"✅ Transformer model created with {sum(p.numel() for p in model.parameters()):,} parameters")

if __name__ == "__main__":
    main()