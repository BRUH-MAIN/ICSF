"""
Training utilities for T5 pipeline.

NOTE: The actual training loop should be in a notebook or script,
not imported from here. This module provides helper functions.
"""

import torch
from torch.optim import AdamW
from typing import Dict, List, Tuple, Optional
from pathlib import Path


def get_optimizer(model, lr: float = 5e-5, weight_decay: float = 0.01):
    """
    Get AdamW optimizer for the model.
    
    Args:
        model: The T5JointModel
        lr: Learning rate
        weight_decay: Weight decay coefficient
    
    Returns:
        AdamW optimizer
    """
    # Don't apply weight decay to bias and LayerNorm
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    return AdamW(optimizer_grouped_parameters, lr=lr)


def save_model(model, save_path: Path, save_tokenizer: bool = True):
    """
    Save model weights and optionally tokenizer.
    
    Args:
        model: The T5JointModel to save
        save_path: Directory to save to
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save the underlying T5 model
    model.model.save_pretrained(save_path)
    
    if save_tokenizer:
        model.tokenizer.save_pretrained(save_path)
    
    print(f"Model saved to {save_path}")


def load_model_weights(model, load_path: Path):
    """
    Load model weights from a saved checkpoint.
    
    Args:
        model: The T5JointModel to load weights into
        load_path: Directory containing saved model
    """
    from transformers import T5ForConditionalGeneration
    
    load_path = Path(load_path)
    model.model = T5ForConditionalGeneration.from_pretrained(load_path)
    print(f"Model loaded from {load_path}")
