"""
Utility functions for Joint Intent Classification and Slot Filling.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter


def build_vocab(sentences: List[str], min_freq: int = 1) -> Dict[str, int]:
    """
    Build a vocabulary from a list of sentences.
    
    Args:
        sentences: List of space-separated word strings
        min_freq: Minimum frequency for a word to be included
        
    Returns:
        Dictionary mapping words to indices
    """
    word_counts = Counter()
    for sentence in sentences:
        words = sentence.split()
        word_counts.update(words)
    
    # Special tokens
    vocab = {'<PAD>': 0, '<UNK>': 1}
    
    # Add words that meet minimum frequency
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab


def load_slot_vocab(vocab_path: str = 'dataset/vocab.slot') -> Dict[str, int]:
    """
    Load slot vocabulary from file.
    
    Args:
        vocab_path: Path to the vocab.slot file
        
    Returns:
        Dictionary mapping slot labels to indices
    """
    slot_labels = Path(vocab_path).read_text('utf-8').strip().split('\n')
    # Add PAD token for masking
    slot_map = {'<PAD>': 0}
    for idx, label in enumerate(slot_labels):
        slot_map[label] = idx + 1
    return slot_map


def load_intent_vocab(vocab_path: str = 'dataset/vocab.intent') -> Dict[str, int]:
    """
    Load intent vocabulary from file.
    
    Args:
        vocab_path: Path to the vocab.intent file
        
    Returns:
        Dictionary mapping intent labels to indices
    """
    intent_labels = Path(vocab_path).read_text('utf-8').strip().split('\n')
    return {label: idx for idx, label in enumerate(intent_labels)}


def encode_slots(slot_labels: List[str], slot_map: Dict[str, int], 
                 max_len: int, pad_idx: int = 0) -> List[int]:
    """
    Encode slot labels to indices with padding.
    
    Args:
        slot_labels: List of slot label strings
        slot_map: Dictionary mapping slot labels to indices
        max_len: Maximum sequence length for padding
        pad_idx: Index to use for padding
        
    Returns:
        List of encoded slot indices
    """
    encoded = [slot_map.get(label, slot_map.get('O', 1)) for label in slot_labels]
    
    # Pad or truncate
    if len(encoded) < max_len:
        encoded = encoded + [pad_idx] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]
    
    return encoded


def encode_words(words: List[str], vocab: Dict[str, int], 
                 max_len: int, pad_idx: int = 0, unk_idx: int = 1) -> List[int]:
    """
    Encode words to indices with padding.
    
    Args:
        words: List of word strings
        vocab: Dictionary mapping words to indices
        max_len: Maximum sequence length for padding
        pad_idx: Index to use for padding
        unk_idx: Index to use for unknown words
        
    Returns:
        List of encoded word indices
    """
    encoded = [vocab.get(word, unk_idx) for word in words]
    
    # Pad or truncate
    if len(encoded) < max_len:
        encoded = encoded + [pad_idx] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]
    
    return encoded


def align_bert_tokens_to_words(
    tokenizer,
    words: List[str],
    slot_labels: List[str],
    max_len: int = 50,
    slot_map: Optional[Dict[str, int]] = None
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Align BERT subword tokens back to original word-level slot labels.
    Uses the first subword token strategy.
    
    Args:
        tokenizer: BERT tokenizer
        words: List of original words
        slot_labels: List of slot labels (same length as words)
        max_len: Maximum sequence length
        slot_map: Optional slot label to index mapping
        
    Returns:
        Tuple of (input_ids, attention_mask, slot_label_ids, word_ids)
        - word_ids maps each token position to original word index (-1 for special tokens)
    """
    input_ids = [tokenizer.cls_token_id]
    attention_mask = [1]
    aligned_slot_labels = [-100]  # -100 for special tokens (ignored in loss)
    word_ids = [-1]  # -1 for CLS token
    
    for word_idx, (word, slot_label) in enumerate(zip(words, slot_labels)):
        word_tokens = tokenizer.tokenize(word)
        word_token_ids = tokenizer.convert_tokens_to_ids(word_tokens)
        
        for i, token_id in enumerate(word_token_ids):
            if len(input_ids) >= max_len - 1:  # Reserve space for SEP
                break
            input_ids.append(token_id)
            attention_mask.append(1)
            word_ids.append(word_idx)
            
            # Only use first subword for slot label
            if i == 0:
                if slot_map is not None:
                    aligned_slot_labels.append(slot_map.get(slot_label, slot_map.get('O', 1)))
                else:
                    aligned_slot_labels.append(slot_label)
            else:
                aligned_slot_labels.append(-100)  # Ignore subsequent subwords
        
        if len(input_ids) >= max_len - 1:
            break
    
    # Add SEP token
    input_ids.append(tokenizer.sep_token_id)
    attention_mask.append(1)
    aligned_slot_labels.append(-100)
    word_ids.append(-1)
    
    # Pad to max_len
    pad_length = max_len - len(input_ids)
    input_ids.extend([tokenizer.pad_token_id] * pad_length)
    attention_mask.extend([0] * pad_length)
    aligned_slot_labels.extend([-100] * pad_length)
    word_ids.extend([-1] * pad_length)
    
    return input_ids, attention_mask, aligned_slot_labels, word_ids


def compute_slot_f1(predictions: List[List[int]], 
                    labels: List[List[int]], 
                    slot_map: Dict[str, int],
                    ignore_index: int = -100) -> Tuple[float, float, float]:
    """
    Compute token-level precision, recall, and F1 for slot filling.
    
    Args:
        predictions: List of predicted slot label indices per sequence
        labels: List of true slot label indices per sequence
        slot_map: Slot label to index mapping
        ignore_index: Index to ignore in computation
        
    Returns:
        Tuple of (precision, recall, f1)
    """
    # Get O label index
    o_idx = slot_map.get('O', 1)
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for pred_seq, label_seq in zip(predictions, labels):
        for pred, label in zip(pred_seq, label_seq):
            if label == ignore_index:
                continue
            
            # Only count non-O labels
            if label != o_idx and pred == label:
                true_positives += 1
            elif label != o_idx and pred != label:
                false_negatives += 1
                if pred != o_idx:
                    false_positives += 1
            elif label == o_idx and pred != o_idx:
                false_positives += 1
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1
