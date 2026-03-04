"""
Data loading utilities for Joint Intent Classification and Slot Filling.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler, SequentialSampler

from src.utils import (
    build_vocab, 
    load_slot_vocab, 
    load_intent_vocab, 
    encode_slots, 
    encode_words,
    align_bert_tokens_to_words
)


def parse_line(line: str) -> Dict:
    """
    Parse a single line from the SNIPS dataset.
    
    Args:
        line: Raw line from dataset file
        
    Returns:
        Dictionary with intent_label, words (list), slot_labels (list), length

    Example:
        # Input:
        #   line = 'add:O the:O tune:B-music_item by:O misato:B-artist watanabe:I-artist to:O the:O Trapeo:B-playlist playlist:O <=> AddToPlaylist'
        #
        # Output:
        #   {
        #       'intent_label': 'AddToPlaylist',
        #       'words':        ['add', 'the', 'tune', 'by', 'misato', 'watanabe', 'to', 'the', 'Trapeo', 'playlist'],
        #       'slot_labels':  ['O', 'O', 'B-music_item', 'O', 'B-artist', 'I-artist', 'O', 'O', 'B-playlist', 'O'],
        #       'length': 10
        #   }
    """
    utterance_data, intent_label = line.split(" <=> ")
    items = utterance_data.split()
    words = [item.rsplit(':', 1)[0] for item in items]
    slot_labels = [item.rsplit(':', 1)[1] for item in items]
    
    return {
        'intent_label': intent_label,
        'words': words,
        'slot_labels': slot_labels,
        'length': len(words)
    }


def load_data(data_path: str) -> List[Dict]:
    """
    Load and parse dataset file.
    
    Args:
        data_path: Path to dataset file (train, valid, or test)
        
    Returns:
        List of parsed data dictionaries

    Example:
        # Input:
        #   data_path = 'dataset/train'
        #
        # Output (list of 13084 dicts; first element shown):
        #   [{
        #       'intent_label': 'AddToPlaylist',
        #       'words':       ['Add', 'Don', 'and', 'Sherri', 'to', 'my', 'Meditate', ...],
        #       'slot_labels': ['O', 'B-entity_name', 'I-entity_name', 'I-entity_name', 'O', ...],
        #       'length': 14
        #   }, ...]
        #
        #   len(result) == 13084  (train) / 700 (valid) / 700 (test)
    """
    lines = Path(data_path).read_text('utf-8').strip().splitlines()
    return [parse_line(line) for line in lines]


class JointNLUDataset(Dataset):
    """
    PyTorch Dataset for Joint Intent Classification and Slot Filling.
    For use with Models 1 & 2 (learned embeddings).
    """
    
    def __init__(
        self,
        data: List[Dict],
        word_vocab: Dict[str, int],
        slot_vocab: Dict[str, int],
        intent_vocab: Dict[str, int],
        max_len: int = 50
    ):
        """
        Args:
            data: List of parsed data dictionaries
            word_vocab: Word to index mapping
            slot_vocab: Slot label to index mapping
            intent_vocab: Intent label to index mapping
            max_len: Maximum sequence length
        """
        self.data = data
        self.word_vocab = word_vocab
        self.slot_vocab = slot_vocab
        self.intent_vocab = intent_vocab
        self.max_len = max_len
        self.pad_idx = word_vocab['<PAD>']
        self.slot_pad_idx = slot_vocab['<PAD>']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Sample output for idx=0 (intent='AddToPlaylist', length=14, max_len=50):
        #   {
        #       'input_ids':      tensor([42, 71, 18, 92, 5, 23, ...  0, 0, 0]),  # shape (50,)
        #       'attention_mask': tensor([1, 1, 1, 1, 1, 1, ...  0, 0, 0]),       # shape (50,)
        #       'slot_labels':    tensor([1, 6, 7, 7, 1, 12, 34, ...  0, 0, 0]),  # shape (50,)
        #       'intent_label':   tensor(0),    # 0 = AddToPlaylist
        #       'length':         tensor(14)
        #   }
        item = self.data[idx]
        
        # Encode words
        word_ids = encode_words(
            item['words'], 
            self.word_vocab, 
            self.max_len, 
            pad_idx=self.pad_idx
        )
        
        # Encode slot labels
        slot_ids = encode_slots(
            item['slot_labels'], 
            self.slot_vocab, 
            self.max_len, 
            pad_idx=self.slot_pad_idx
        )
        
        # Encode intent
        intent_id = self.intent_vocab[item['intent_label']]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        length = min(item['length'], self.max_len)
        attention_mask = [1] * length + [0] * (self.max_len - length)
        
        return {
            'input_ids': torch.tensor(word_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'slot_labels': torch.tensor(slot_ids, dtype=torch.long),
            'intent_label': torch.tensor(intent_id, dtype=torch.long),
            'length': torch.tensor(length, dtype=torch.long)
        }


class BertJointNLUDataset(Dataset):
    """
    PyTorch Dataset for Joint NLU with BERT tokenization.
    For use with Model 3 (frozen BERT embeddings).
    Handles subword alignment using first-subword strategy.
    """
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        slot_vocab: Dict[str, int],
        intent_vocab: Dict[str, int],
        max_len: int = 50
    ):
        """
        Args:
            data: List of parsed data dictionaries
            tokenizer: BERT tokenizer
            slot_vocab: Slot label to index mapping
            intent_vocab: Intent label to index mapping
            max_len: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.slot_vocab = slot_vocab
        self.intent_vocab = intent_vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Sample output for idx=0 (intent='AddToPlaylist', length=14, max_len=50):
        #   {
        #       'input_ids':      tensor([101, 5765, 1103, 6894, ...  0]),   # shape (50,)  101=CLS
        #       'attention_mask': tensor([1, 1, 1, 1, 1, ...  0, 0, 0]),     # shape (50,)
        #       'slot_labels':    tensor([-100, 1, 6, -100, 7, ...  -100]),   # shape (50,)  -100=ignored
        #       'intent_label':   tensor(0),    # 0 = AddToPlaylist
        #       'word_ids':       tensor([-1, 0, 1, 1, 2, ...  -1])          # shape (50,)  -1=special token
        #   }
        item = self.data[idx]
        input_ids, attention_mask, slot_label_ids, word_ids = align_bert_tokens_to_words(
            self.tokenizer,
            item['words'],
            item['slot_labels'],
            self.max_len,
            self.slot_vocab
        )
        
        # Encode intent
        intent_id = self.intent_vocab[item['intent_label']]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'slot_labels': torch.tensor(slot_label_ids, dtype=torch.long),
            'intent_label': torch.tensor(intent_id, dtype=torch.long),
            'word_ids': torch.tensor(word_ids, dtype=torch.long)
        }


def create_dataloaders(
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: List[Dict],
    word_vocab: Dict[str, int],
    slot_vocab: Dict[str, int],
    intent_vocab: Dict[str, int],
    batch_size: int = 32,
    max_len: int = 50
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for training, validation, and testing.
    For Models 1 & 2 (learned embeddings).
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)

    Example:
        # Input:
        #   train_data = load_data('dataset/train')   # 13084 samples
        #   val_data   = load_data('dataset/valid')   # 700 samples
        #   test_data  = load_data('dataset/test')    # 700 samples
        #   batch_size = 32, max_len = 50
        #
        # Output:
        #   (DataLoader with 409 batches,   # ceil(13084 / 32), randomly shuffled
        #    DataLoader with  22 batches,   # ceil(700 / 32),   sequential
        #    DataLoader with  22 batches)   # ceil(700 / 32),   sequential
        #
        # Each batch dict has keys:
        #   'input_ids'      -> tensor of shape (32, 50)
        #   'attention_mask' -> tensor of shape (32, 50)
        #   'slot_labels'    -> tensor of shape (32, 50)
        #   'intent_label'   -> tensor of shape (32,)
        #   'length'         -> tensor of shape (32,)
    """
    train_dataset = JointNLUDataset(train_data, word_vocab, slot_vocab, intent_vocab, max_len)
    val_dataset = JointNLUDataset(val_data, word_vocab, slot_vocab, intent_vocab, max_len)
    test_dataset = JointNLUDataset(test_data, word_vocab, slot_vocab, intent_vocab, max_len)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=RandomSampler(train_dataset)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        sampler=SequentialSampler(val_dataset)
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        sampler=SequentialSampler(test_dataset)
    )
    
    return train_loader, val_loader, test_loader


def create_bert_dataloaders(
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: List[Dict],
    tokenizer,
    slot_vocab: Dict[str, int],
    intent_vocab: Dict[str, int],
    batch_size: int = 32,
    max_len: int = 50
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for BERT-based models.
    For Model 3 (frozen BERT embeddings).
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)

    Example:
        # Input:
        #   tokenizer  = BertTokenizer.from_pretrained('bert-base-cased')
        #   train_data = load_data('dataset/train')   # 13084 samples
        #   val_data   = load_data('dataset/valid')   # 700 samples
        #   test_data  = load_data('dataset/test')    # 700 samples
        #   batch_size = 32, max_len = 50
        #
        # Output:
        #   (DataLoader with 409 batches,   # ceil(13084 / 32), randomly shuffled
        #    DataLoader with  22 batches,   # ceil(700 / 32),   sequential
        #    DataLoader with  22 batches)   # ceil(700 / 32),   sequential
        #
        # Each batch dict has keys:
        #   'input_ids'      -> tensor of shape (32, 50)  # BERT token ids (CLS...SEP...PAD)
        #   'attention_mask' -> tensor of shape (32, 50)  # 1 for real tokens, 0 for padding
        #   'slot_labels'    -> tensor of shape (32, 50)  # -100 for special/subword tokens
        #   'intent_label'   -> tensor of shape (32,)
        #   'word_ids'       -> tensor of shape (32, 50)  # -1 for special tokens
    """
    train_dataset = BertJointNLUDataset(train_data, tokenizer, slot_vocab, intent_vocab, max_len)
    val_dataset = BertJointNLUDataset(val_data, tokenizer, slot_vocab, intent_vocab, max_len)
    test_dataset = BertJointNLUDataset(test_data, tokenizer, slot_vocab, intent_vocab, max_len)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=RandomSampler(train_dataset)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        sampler=SequentialSampler(val_dataset)
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        sampler=SequentialSampler(test_dataset)
    )
    
    return train_loader, val_loader, test_loader
