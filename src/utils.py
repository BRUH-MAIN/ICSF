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

    Example:
        # Input:
        #   sentences = ['add Don and to my playlist', 'play some music']
        #   min_freq = 1
        #
        # Output:
        #   {'<PAD>': 0, '<UNK>': 1, 'add': 2, 'Don': 3, 'and': 4,
        #    'to': 5, 'my': 6, 'playlist': 7, 'play': 8, 'some': 9, 'music': 10}
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
        vocab_path: Path to the vocab.slot file  (contains 72 slot labels, one per line)
        
    Returns:
        Dictionary mapping slot labels to indices  (73 entries including <PAD>)

    Example:
        # Input:
        #   vocab_path = 'dataset/vocab.slot'
        #   (file lines: 'B-album', 'B-artist', 'B-best_rating', ..., 'O', ...)
        #
        # Output:
        #   {'<PAD>': 0, 'B-album': 1, 'B-artist': 2, 'B-best_rating': 3,
        #    'B-city': 4, 'B-condition_description': 5, ..., 'O': 40, ...}
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
        vocab_path: Path to the vocab.intent file  (contains 7 intent labels, one per line)
        
    Returns:
        Dictionary mapping intent labels to indices  (7 entries)

    Example:
        # Input:
        #   vocab_path = 'dataset/vocab.intent'
        #   (file lines: 'AddToPlaylist', 'BookRestaurant', 'GetWeather', ...)
        #
        # Output:
        #   {'AddToPlaylist': 0, 'BookRestaurant': 1, 'GetWeather': 2,
        #    'PlayMusic': 3, 'RateBook': 4, 'SearchCreativeWork': 5,
        #    'SearchScreeningEvent': 6}

        #why not add <PAD> for intent? Because intent is a single label per sequence, we don't need a PAD token for intent classification. The model will predict one of the 7 intent labels for each input sequence, and there is no need to mask or pad intent labels like we do for slot labels.
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

    Example:
        # Input:
        #   slot_labels = ['O', 'B-entity_name', 'I-entity_name', 'O']
        #   slot_map    = {'<PAD>': 0, 'O': 1, 'B-entity_name': 2, 'I-entity_name': 3}
        #   max_len = 6, pad_idx = 0
        #
        # Output:
        #   [1, 2, 3, 1, 0, 0]   # last two are padding
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

    Example:
        # Input:
        #   words   = ['add', 'Don', 'to']
        #   vocab   = {'<PAD>': 0, '<UNK>': 1, 'add': 2, 'Don': 3, 'to': 4}
        #   max_len = 5, pad_idx = 0, unk_idx = 1
        #
        # Output:
        #   [2, 3, 4, 0, 0]   # 'add'->2, 'Don'->3, 'to'->4, then two padding zeros
        #
        # With unknown word:
        #   words = ['add', 'xyz', 'to'],  'xyz' not in vocab
        #   Output: [2, 1, 4, 0, 0]        # 'xyz' maps to unk_idx=1
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

    Example:
        # Input:
        #   words       = ['add', 'misato', 'watanabe']
        #   slot_labels = ['O',   'B-artist', 'I-artist']
        #   max_len     = 8
        #   slot_map    = {'<PAD>': 0, 'O': 1, 'B-artist': 5, 'I-artist': 6}
        #
        #   BERT tokenizes: 'add' -> ['add'],
        #                   'misato' -> ['mis', '##ato'],
        #                   'watanabe' -> ['wat', '##ana', '##be']
        #
        # Output (tuple of 4 lists, each of length max_len=8):
        #   input_ids      = [101, add_id, mis_id, ato_id, wat_id, ana_id, be_id, 102]
        #                      ^CLS                                                ^SEP
        #   attention_mask = [1, 1, 1, 1, 1, 1, 1, 1]
        #   slot_label_ids = [-100, 1, 5, -100, 6, -100, -100, -100]
        #                     ^CLS  ^O ^B-artist ^subword ignored  ^SEP
        #   word_ids       = [-1, 0, 1, 1, 2, 2, 2, -1]
        #                     ^CLS ^add ^misato   ^watanabe  ^SEP
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

    Example:
        # Input:
        #   slot_map    = {'<PAD>': 0, 'O': 1, 'B-entity_name': 2, 'I-entity_name': 3}
        #   predictions = [[1, 2, 3, 1], [1, 2, 1]]   # 0=PAD(ignored), 1=O, 2=B-entity, 3=I-entity
        #   labels      = [[1, 2, 3, 1], [1, 1, 1]]   # second seq: model predicted B-entity wrongly
        #   ignore_index = -100
        #
        # Output:
        #   (0.6667, 1.0, 0.8)   # (precision, recall, f1)
        #   precision = 2 TP / (2 TP + 1 FP) = 0.6667
        #   recall    = 2 TP / (2 TP + 0 FN) = 1.0
        #   f1        = 2 * 0.6667 * 1.0 / (0.6667 + 1.0) = 0.8
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
