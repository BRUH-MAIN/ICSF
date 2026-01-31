"""
Joint Intent Classification and Slot Filling Models

Model 1: EncoderDecoderNLU - Bidirectional GRU encoder + Unidirectional GRU decoder
Model 2: TransformerNLU - Transformer encoder with multi-head attention
Model 3: BertEncoderDecoderNLU - Frozen BERT embeddings + GRU encoder-decoder
"""

from .BiLSTMIntentClassifier import BiLSTMIntentClassifier
from .encoder_decoder import EncoderDecoderNLU
from .transformer_model import TransformerNLU, TransformerNLUWithCLS
from .bert_encoder_decoder import BertEncoderDecoderNLU, BertEncoderDecoderNLUWithAttention

__all__ = [
    'BiLSTMIntentClassifier',
    'EncoderDecoderNLU',
    'TransformerNLU',
    'TransformerNLUWithCLS',
    'BertEncoderDecoderNLU',
    'BertEncoderDecoderNLUWithAttention'
]
