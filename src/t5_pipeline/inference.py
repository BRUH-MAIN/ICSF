"""
Inference module for T5 pipeline.

Provides a clean interface for running inference on new utterances.
"""

import torch
from typing import Dict, List, Union
from pathlib import Path

from src.models.t5 import T5JointModel, parse_generated_output


class T5Inferencer:
    """
    Inference pipeline for T5 joint intent and slot filling model.
    """
    
    def __init__(self, model: T5JointModel = None, model_path: Path = None, 
                 model_name: str = 't5-small', device: str = None):
        """
        Initialize the inferencer.
        
        Args:
            model: Pre-loaded T5JointModel (optional)
            model_path: Path to saved model weights (optional)
            model_name: Name of T5 model variant (used if model not provided)
            device: Device to run inference on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        if model is not None:
            self.model = model
        else:
            self.model = T5JointModel(model_name=model_name)
            if model_path is not None:
                self.model.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded model weights from {model_path}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def predict(self, utterance: str, max_length: int = 128, num_beams: int = 4) -> Dict:
        """
        Run inference on a single utterance.
        
        Args:
            utterance: Raw utterance text (e.g., "book a flight to delhi")
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
        
        Returns:
            Dictionary with 'intent', 'slots', and 'raw_output' keys
        """
        # Format input
        input_text = f"intent and slots: {utterance}"
        
        # Tokenize
        encoding = self.model.tokenize_input([input_text], max_length=max_length)
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams
            )
        
        # Decode
        raw_output = self.model.decode(output_ids[0])
        
        # Parse
        parsed = parse_generated_output(raw_output)
        
        if parsed is None:
            return {
                'intent': None,
                'slots': {},
                'raw_output': raw_output,
                'parse_success': False
            }
        
        return {
            'intent': parsed['intent'],
            'slots': parsed['slots'],
            'raw_output': raw_output,
            'parse_success': True
        }
    
    def predict_batch(self, utterances: List[str], max_length: int = 128, 
                      num_beams: int = 4, batch_size: int = 32) -> List[Dict]:
        """
        Run inference on a batch of utterances.
        
        Args:
            utterances: List of raw utterance texts
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            batch_size: Batch size for inference
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(utterances), batch_size):
            batch_utterances = utterances[i:i + batch_size]
            input_texts = [f"intent and slots: {u}" for u in batch_utterances]
            
            # Tokenize
            encoding = self.model.tokenize_input(input_texts, max_length=max_length)
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    num_beams=num_beams
                )
            
            # Decode and parse each
            for j, output in enumerate(output_ids):
                raw_output = self.model.decode(output)
                parsed = parse_generated_output(raw_output)
                
                if parsed is None:
                    results.append({
                        'utterance': batch_utterances[j],
                        'intent': None,
                        'slots': {},
                        'raw_output': raw_output,
                        'parse_success': False
                    })
                else:
                    results.append({
                        'utterance': batch_utterances[j],
                        'intent': parsed['intent'],
                        'slots': parsed['slots'],
                        'raw_output': raw_output,
                        'parse_success': True
                    })
        
        return results
