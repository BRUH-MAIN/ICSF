"""
Dataset module for T5 pipeline.

Provides PyTorch Dataset class for loading preprocessed JSONL data.
Returns raw strings - tokenization happens in the training loop.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from torch.utils.data import Dataset


class T5JointDataset(Dataset):
    """
    PyTorch Dataset for T5 joint intent classification and slot filling.
    
    Loads preprocessed JSONL files and returns raw text pairs.
    Tokenization is NOT done here - it's handled in the training loop.
    """
    
    def __init__(self, jsonl_path: Path):
        """
        Initialize the dataset.
        
        Args:
            jsonl_path: Path to the preprocessed JSONL file
        """
        self.data: List[Dict[str, str]] = []
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    example = json.loads(line)
                    self.data.append(example)
        
        print(f"Loaded {len(self.data)} examples from {jsonl_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        """
        Get a single example.
        
        Returns:
            Dictionary with 'input_text' and 'target_text' keys
        """
        return {
            'input_text': self.data[idx]['input_text'],
            'target_text': self.data[idx]['target_text']
        }


def collate_fn(batch: List[Dict[str, str]]) -> Dict[str, List[str]]:
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of examples from the dataset
    
    Returns:
        Dictionary with lists of input_text and target_text
    """
    return {
        'input_text': [example['input_text'] for example in batch],
        'target_text': [example['target_text'] for example in batch]
    }


def load_datasets(preprocessed_dir: Path) -> Dict[str, Optional[T5JointDataset]]:
    """
    Load all dataset splits.
    
    Args:
        preprocessed_dir: Path to directory containing preprocessed JSONL files
    
    Returns:
        Dictionary mapping split names to Dataset objects
    """
    datasets = {}
    
    for split in ['train', 'valid', 'test']:
        jsonl_path = preprocessed_dir / f'{split}.jsonl'
        if jsonl_path.exists():
            datasets[split] = T5JointDataset(jsonl_path)
        else:
            print(f"Warning: {jsonl_path} not found")
            datasets[split] = None
    
    return datasets
