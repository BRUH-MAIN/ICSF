"""
Preprocessing module for T5 pipeline.

Converts the classic NLU format (seq.in, seq.out, label) into
the text-to-text format required by T5.

Input format:
    word1:tag1 word2:tag2 ... <=> intent

Output format (JSONL):
    {"input_text": "intent and slots: <utterance>", 
     "target_text": "intent: <intent> slots: <slot>=<value> ..."}
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def extract_slots_from_bio(words: List[str], tags: List[str]) -> Dict[str, str]:
    """
    Extract slot-value pairs from BIO tagged sequence.
    
    Args:
        words: List of words in the utterance
        tags: List of BIO tags (O, B-slot_name, I-slot_name)
    
    Returns:
        Dictionary mapping slot names to their values
    """
    slots = {}
    current_slot = None
    current_value = []
    
    for word, tag in zip(words, tags):
        if tag.startswith('B-'):
            # Save previous slot if exists
            if current_slot is not None:
                slots[current_slot] = ' '.join(current_value)
            # Start new slot
            current_slot = tag[2:].lower()  # Remove 'B-' prefix, lowercase
            current_value = [word]
        elif tag.startswith('I-'):
            slot_name = tag[2:].lower()
            if current_slot == slot_name:
                # Continue current slot
                current_value.append(word)
            else:
                # Malformed: I- tag without matching B- tag
                # Save current and start new
                if current_slot is not None:
                    slots[current_slot] = ' '.join(current_value)
                current_slot = slot_name
                current_value = [word]
        else:  # 'O' tag
            # Save current slot if exists
            if current_slot is not None:
                slots[current_slot] = ' '.join(current_value)
                current_slot = None
                current_value = []
    
    # Don't forget the last slot
    if current_slot is not None:
        slots[current_slot] = ' '.join(current_value)
    
    return slots


def parse_line(line: str) -> Optional[Dict]:
    """
    Parse a single line from the dataset file.
    
    Args:
        line: A line in format "word1:tag1 word2:tag2 ... <=> intent"
    
    Returns:
        Dictionary with 'intent', 'words', and 'slots' keys, or None if malformed
    """
    try:
        utterance_data, intent_label = line.strip().split(' <=> ')
        items = utterance_data.split()
        
        words = []
        tags = []
        for item in items:
            word, tag = item.rsplit(':', 1)
            words.append(word)
            tags.append(tag)
        
        slots = extract_slots_from_bio(words, tags)
        
        return {
            'intent': intent_label,
            'words': words,
            'slots': slots
        }
    except Exception:
        return None


def format_for_t5(parsed: Dict) -> Dict[str, str]:
    """
    Format a parsed example for T5 text-to-text training.
    
    Args:
        parsed: Dictionary with 'intent', 'words', and 'slots' keys
    
    Returns:
        Dictionary with 'input_text' and 'target_text' keys
    """
    utterance = ' '.join(parsed['words'])
    input_text = f"intent and slots: {utterance}"
    
    # Build target text with sorted slots for deterministic output
    intent = parsed['intent']
    slots_str = ' '.join(
        f"{k}={v}" for k, v in sorted(parsed['slots'].items())
    )
    
    if slots_str:
        target_text = f"intent: {intent} slots: {slots_str}"
    else:
        target_text = f"intent: {intent} slots:"
    
    return {
        'input_text': input_text,
        'target_text': target_text
    }


def preprocess_file(input_path: Path, output_path: Path) -> Tuple[int, int]:
    """
    Preprocess a dataset file and save as JSONL.
    
    Args:
        input_path: Path to the input file (train, valid, or test)
        output_path: Path to save the JSONL output
    
    Returns:
        Tuple of (successful_examples, dropped_examples)
    """
    successful = 0
    dropped = 0
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            parsed = parse_line(line)
            if parsed is None:
                dropped += 1
                continue
            
            formatted = format_for_t5(parsed)
            f_out.write(json.dumps(formatted) + '\n')
            successful += 1
    
    return successful, dropped


def preprocess_dataset(dataset_dir: Path, output_dir: Path) -> Dict[str, Tuple[int, int]]:
    """
    Preprocess the entire dataset (train, valid, test).
    
    Args:
        dataset_dir: Path to the dataset directory
        output_dir: Path to save preprocessed files
    
    Returns:
        Dictionary mapping split names to (successful, dropped) counts
    """
    results = {}
    
    for split in ['train', 'valid', 'test']:
        input_path = dataset_dir / split
        output_path = output_dir / f'{split}.jsonl'
        
        if input_path.exists():
            successful, dropped = preprocess_file(input_path, output_path)
            results[split] = (successful, dropped)
            print(f"Preprocessed {split}: {successful} examples, {dropped} dropped")
        else:
            print(f"Warning: {input_path} not found, skipping {split}")
    
    return results


if __name__ == '__main__':
    # Run preprocessing
    dataset_dir = Path('dataset')
    output_dir = Path('dataset/preprocessed_t5')
    
    print("Starting preprocessing...")
    results = preprocess_dataset(dataset_dir, output_dir)
    print("\nPreprocessing complete!")
    print(f"Output saved to: {output_dir}")
