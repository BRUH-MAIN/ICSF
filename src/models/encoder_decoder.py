"""
Model 1: Encoder-Decoder with Learned Embeddings

Architecture:
- Learned nn.Embedding layer
- Bidirectional GRU Encoder
- Unidirectional GRU Decoder
- Intent classification head (from encoder final state)
- Slot filling head (from decoder outputs)
"""

import torch
import torch.nn as nn
from typing import Tuple


class EncoderDecoderNLU(nn.Module):
    """
    Joint Intent Classification and Slot Filling model using
    Encoder-Decoder architecture with learned embeddings.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 128,
        num_intents: int = 7,
        num_slots: int = 73,
        n_layers: int = 1,
        dropout: float = 0.3,
        pad_idx: int = 0
    ):
        """
        Args:
            vocab_size: Size of the word vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension of GRU layers
            num_intents: Number of intent classes
            num_slots: Number of slot labels
            n_layers: Number of GRU layers
            dropout: Dropout probability
            pad_idx: Padding token index for embedding layer
        """
        super(EncoderDecoderNLU, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=pad_idx
        )
        
        # Encoder: Bidirectional GRU
        self.encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # Linear layer to project encoder hidden state for decoder initialization
        # Encoder produces hidden_dim * 2 (bidirectional), decoder needs hidden_dim
        self.encoder_to_decoder = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Decoder: Unidirectional GRU
        self.decoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # Intent classification head (from encoder final state)
        self.intent_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_intents)
        )
        
        # Slot filling head (from decoder outputs)
        self.slot_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_slots)
        )
        
        self.dropout = nn.Dropout(dropout)
    
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
            
        Returns:
            intent_logits: Shape (batch_size, num_intents)
            slot_logits: Shape (batch_size, seq_len, num_slots)
        """
        batch_size, seq_len = input_ids.shape
        
        # Embed input tokens
        embedded = self.dropout(self.embedding(input_ids))  # (batch, seq_len, embed_dim)
        
        # Encode
        encoder_outputs, encoder_hidden = self.encoder(embedded)
        # encoder_outputs: (batch, seq_len, hidden_dim * 2)
        # encoder_hidden: (n_layers * 2, batch, hidden_dim)
        
        # Get final encoder hidden state for intent classification
        # Concatenate forward and backward final hidden states
        forward_hidden = encoder_hidden[-2, :, :]  # (batch, hidden_dim)
        backward_hidden = encoder_hidden[-1, :, :]  # (batch, hidden_dim)
        encoder_final = torch.cat([forward_hidden, backward_hidden], dim=1)  # (batch, hidden_dim * 2)
        
        # Intent classification
        intent_logits = self.intent_classifier(encoder_final)  # (batch, num_intents)
        
        # Prepare decoder initial hidden state
        # Project encoder final state and reshape for decoder
        decoder_hidden = self.encoder_to_decoder(encoder_final)  # (batch, hidden_dim)
        decoder_hidden = decoder_hidden.unsqueeze(0)  # (1, batch, hidden_dim)
        
        # For multi-layer decoder, repeat the hidden state
        if self.n_layers > 1:
            decoder_hidden = decoder_hidden.repeat(self.n_layers, 1, 1)
        
        # Decode using same embeddings (teacher forcing)
        decoder_outputs, _ = self.decoder(embedded, decoder_hidden)
        # decoder_outputs: (batch, seq_len, hidden_dim)
        
        # Slot classification
        slot_logits = self.slot_classifier(decoder_outputs)  # (batch, seq_len, num_slots)
        
        return intent_logits, slot_logits
    
    def encode(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input sequence.
        
        Returns:
            encoder_outputs: (batch, seq_len, hidden_dim * 2)
            encoder_final: (batch, hidden_dim * 2)
        """
        embedded = self.dropout(self.embedding(input_ids))
        encoder_outputs, encoder_hidden = self.encoder(embedded)
        
        forward_hidden = encoder_hidden[-2, :, :]
        backward_hidden = encoder_hidden[-1, :, :]
        encoder_final = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        return encoder_outputs, encoder_final
    
    def decode(
        self, 
        embedded: torch.Tensor, 
        decoder_hidden: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode with given embeddings and initial hidden state.
        
        Returns:
            slot_logits: (batch, seq_len, num_slots)
        """
        decoder_outputs, _ = self.decoder(embedded, decoder_hidden)
        slot_logits = self.slot_classifier(decoder_outputs)
        return slot_logits
