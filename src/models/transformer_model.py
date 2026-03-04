"""
Model 2: Transformer with Multi-Head Attention

Architecture:
- Learned token embeddings + positional encodings
- Transformer Encoder (4-6 layers, multi-head self-attention)
- Intent classification head (pooled representation)
- Slot filling head (per-token output)
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer.
    """
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerNLU(nn.Module):
    """
    Joint Intent Classification and Slot Filling model using
    Transformer Encoder architecture.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 2048,
        num_intents: int = 7,
        num_slots: int = 73,
        max_len: int = 50,
        dropout: float = 0.1,
        pad_idx: int = 0
    ):
        """
        Args:
            vocab_size: Size of the word vocabulary
            d_model: Model dimension (embedding size)
            nhead: Number of attention heads
            num_encoder_layers: Number of Transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            num_intents: Number of intent classes
            num_slots: Number of slot labels
            max_len: Maximum sequence length
            dropout: Dropout probability
            pad_idx: Padding token index
        """
        super(TransformerNLU, self).__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # Token embedding
        self.embedding = nn.Embedding(
            vocab_size, 
            d_model, 
            padding_idx=pad_idx
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Intent classification head (using mean pooling)
        self.intent_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_intents)
        )
        
        # Slot filling head
        self.slot_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_slots)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.embedding.weight, mean=0, std=self.d_model ** -0.5)
        if self.pad_idx is not None:
            nn.init.zeros_(self.embedding.weight[self.pad_idx])
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token indices, shape (batch_size, seq_len)
            attention_mask: Attention mask, shape (batch_size, seq_len)
                            1 for valid tokens, 0 for padding
            
        Returns:
            intent_logits: Shape (batch_size, num_intents)
            slot_logits: Shape (batch_size, seq_len, num_slots)
        """
        batch_size, seq_len = input_ids.shape
        
        # Embed tokens and add positional encoding
        embedded = self.embedding(input_ids) * math.sqrt(self.d_model)
        embedded = self.positional_encoding(embedded)  # (batch, seq_len, d_model)
        
        # Create attention mask for Transformer
        # PyTorch Transformer uses True for positions to mask (opposite of our convention)
        if attention_mask is not None:
            # Convert: 1 (valid) -> False (don't mask), 0 (pad) -> True (mask)
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        # Pass through Transformer encoder
        encoder_output = self.transformer_encoder(
            embedded,
            src_key_padding_mask=src_key_padding_mask
        )
        # encoder_output: (batch, seq_len, d_model)
        
        # Intent classification using mean pooling over valid tokens
        if attention_mask is not None:
            # Mask out padding positions
            mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            sum_embeddings = (encoder_output * mask_expanded).sum(dim=1)  # (batch, d_model)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)  # (batch, 1)
            pooled_output = sum_embeddings / sum_mask  # (batch, d_model)
        else:
            pooled_output = encoder_output.mean(dim=1)  # (batch, d_model)
        
        intent_logits = self.intent_classifier(pooled_output)  # (batch, num_intents)
        
        # Slot classification
        slot_logits = self.slot_classifier(encoder_output)  # (batch, seq_len, num_slots)
        
        return intent_logits, slot_logits
