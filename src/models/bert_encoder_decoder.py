"""
Model 3: Encoder-Decoder with Frozen BERT Embeddings

Architecture:
- Frozen BERT for contextualized embeddings
- Bidirectional GRU Encoder
- Unidirectional GRU Decoder
- Intent classification head (from encoder final state)
- Slot filling head (from decoder outputs)

Handles subword alignment using first-subword strategy.
"""

import torch
import torch.nn as nn
from transformers import BertModel
from typing import Tuple, Optional


class BertEncoderDecoderNLU(nn.Module):
    """
    Joint Intent Classification and Slot Filling model using
    Frozen BERT embeddings with Encoder-Decoder architecture.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_intents: int = 7,
        num_slots: int = 73,
        n_layers: int = 1,
        dropout: float = 0.3,
        bert_model_name: str = 'bert-base-cased'
    ):
        """
        Args:
            hidden_dim: Hidden dimension of GRU layers
            num_intents: Number of intent classes
            num_slots: Number of slot labels
            n_layers: Number of GRU layers
            dropout: Dropout probability
            bert_model_name: Name of the BERT model to use
        """
        super(BertEncoderDecoderNLU, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Load and freeze BERT
        print(f"Loading BERT model '{bert_model_name}'... This may take a moment.")
        self.bert = BertModel.from_pretrained(bert_model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
        print("BERT model loaded and frozen successfully!")
        
        # BERT embedding dimension
        self.bert_dim = self.bert.config.hidden_size  # 768 for base
        
        # Encoder: Bidirectional GRU on top of BERT embeddings
        self.encoder = nn.GRU(
            input_size=self.bert_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # Linear layer to project encoder hidden state for decoder initialization
        self.encoder_to_decoder = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Decoder: Unidirectional GRU
        self.decoder = nn.GRU(
            input_size=self.bert_dim,
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
        
        # Get BERT embeddings (frozen)
        with torch.no_grad():
            bert_output = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        bert_embeddings = bert_output.last_hidden_state  # (batch, seq_len, 768)
        bert_embeddings = self.dropout(bert_embeddings)
        
        # Encode
        encoder_outputs, encoder_hidden = self.encoder(bert_embeddings)
        # encoder_outputs: (batch, seq_len, hidden_dim * 2)
        # encoder_hidden: (n_layers * 2, batch, hidden_dim)
        
        # Get final encoder hidden state for intent classification
        forward_hidden = encoder_hidden[-2, :, :]  # (batch, hidden_dim)
        backward_hidden = encoder_hidden[-1, :, :]  # (batch, hidden_dim)
        encoder_final = torch.cat([forward_hidden, backward_hidden], dim=1)  # (batch, hidden_dim * 2)
        
        # Intent classification
        intent_logits = self.intent_classifier(encoder_final)  # (batch, num_intents)
        
        # Prepare decoder initial hidden state
        decoder_hidden = self.encoder_to_decoder(encoder_final)  # (batch, hidden_dim)
        decoder_hidden = decoder_hidden.unsqueeze(0)  # (1, batch, hidden_dim)
        
        if self.n_layers > 1:
            decoder_hidden = decoder_hidden.repeat(self.n_layers, 1, 1)
        
        # Decode using BERT embeddings (teacher forcing)
        decoder_outputs, _ = self.decoder(bert_embeddings, decoder_hidden)
        # decoder_outputs: (batch, seq_len, hidden_dim)
        
        # Slot classification
        slot_logits = self.slot_classifier(decoder_outputs)  # (batch, seq_len, num_slots)
        
        return intent_logits, slot_logits
