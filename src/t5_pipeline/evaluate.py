"""
Evaluation module for T5 pipeline.

String-based evaluation metrics:
- Intent Accuracy
- Slot Precision / Recall / F1
- Exact Match (intent + all slots correct)
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def parse_prediction(text: str) -> Optional[Dict]:
    """
    Parse a model prediction into structured format.
    
    Args:
        text: Generated text in format "intent: X slots: a=b c=d"
    
    Returns:
        Dictionary with 'intent' and 'slots', or None if parsing fails
    """
    result = {'intent': None, 'slots': {}}
    
    try:
        text = text.strip()
        
        # Handle both "slots:" and no slots
        if 'slots:' in text:
            intent_part, slots_part = text.split('slots:', 1)
        else:
            intent_part = text
            slots_part = ''
        
        # Parse intent
        if 'intent:' in intent_part:
            result['intent'] = intent_part.split('intent:', 1)[1].strip()
        
        # Parse slots
        slots_part = slots_part.strip()
        if slots_part:
            for pair in slots_part.split():
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    result['slots'][key.strip()] = value.strip()
        
        return result
    except Exception:
        return None


def compute_slot_metrics(pred_slots: Dict[str, str], 
                         gold_slots: Dict[str, str]) -> Tuple[int, int, int]:
    """
    Compute slot-level TP, FP, FN counts.
    
    Args:
        pred_slots: Predicted slot-value pairs
        gold_slots: Ground truth slot-value pairs
    
    Returns:
        Tuple of (true_positives, false_positives, false_negatives)
    """
    tp = 0
    fp = 0
    fn = 0
    
    # Check predictions against gold
    for key, value in pred_slots.items():
        if key in gold_slots and gold_slots[key] == value:
            tp += 1
        else:
            fp += 1
    
    # Check for missed gold slots
    for key, value in gold_slots.items():
        if key not in pred_slots or pred_slots[key] != value:
            fn += 1
    
    return tp, fp, fn


def evaluate(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Evaluate model predictions against references.
    
    Args:
        predictions: List of generated prediction strings
        references: List of ground truth target strings
    
    Returns:
        Dictionary with evaluation metrics
    """
    assert len(predictions) == len(references), "Predictions and references must have same length"
    
    total = len(predictions)
    intent_correct = 0
    exact_match = 0
    parse_failures = 0
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for pred_text, ref_text in zip(predictions, references):
        pred = parse_prediction(pred_text)
        ref = parse_prediction(ref_text)
        
        if pred is None:
            parse_failures += 1
            # Count all gold slots as FN
            if ref is not None:
                total_fn += len(ref['slots'])
            continue
        
        if ref is None:
            # This shouldn't happen with proper preprocessing
            continue
        
        # Intent accuracy
        if pred['intent'] == ref['intent']:
            intent_correct += 1
        
        # Slot metrics
        tp, fp, fn = compute_slot_metrics(pred['slots'], ref['slots'])
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Exact match (intent + all slots)
        if pred['intent'] == ref['intent'] and pred['slots'] == ref['slots']:
            exact_match += 1
    
    # Calculate final metrics
    intent_accuracy = intent_correct / total if total > 0 else 0.0
    exact_match_accuracy = exact_match / total if total > 0 else 0.0
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'intent_accuracy': intent_accuracy,
        'slot_precision': precision,
        'slot_recall': recall,
        'slot_f1': f1,
        'exact_match': exact_match_accuracy,
        'parse_failure_rate': parse_failures / total if total > 0 else 0.0,
        'total_examples': total
    }


def print_metrics(metrics: Dict[str, float]):
    """Print evaluation metrics in a readable format."""
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Intent Accuracy:     {metrics['intent_accuracy']:.4f}")
    print(f"Slot Precision:      {metrics['slot_precision']:.4f}")
    print(f"Slot Recall:         {metrics['slot_recall']:.4f}")
    print(f"Slot F1:             {metrics['slot_f1']:.4f}")
    print(f"Exact Match:         {metrics['exact_match']:.4f}")
    print(f"Parse Failure Rate:  {metrics['parse_failure_rate']:.4f}")
    print(f"Total Examples:      {metrics['total_examples']}")
    print("=" * 50)
