import torch
from torch import nn
from transformers import T5ForConditionalGeneration, T5Tokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class T5JointModel(nn.Module):
    """
    T5-based Joint Intent Classification and Slot Filling Model.
    
    This is a text-to-text model that takes an utterance and generates
    a structured output in the format:
        "intent: X slots: a=b c=d"
    
    No custom heads. No BIO tagging. Single forward pass, single loss.
    """
    
    def __init__(self, model_name: str = 't5-small'):
        """
        Initialize the T5 model wrapper.
        
        Args:
            model_name: The pretrained T5 model to use (e.g., 't5-small', 't5-base')
        """
        super(T5JointModel, self).__init__()
        print(f"Loading T5 model '{model_name}'... This may take a moment on first run.")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        self.model_name = model_name
        print(f"T5 model '{model_name}' loaded successfully!")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor = None):
        """
        Forward pass for training with teacher forcing.
        
        Args:
            input_ids: Tokenized input utterances [batch_size, seq_len]
            attention_mask: Attention mask for inputs [batch_size, seq_len]
            labels: Tokenized target sequences [batch_size, target_seq_len]
                    Padding tokens should be replaced with -100 for loss masking.
        
        Returns:
            If labels provided: Returns loss and logits
            If no labels: Returns logits only
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask, #padding mask for input
            labels=labels #target structured string
        )
        return outputs
    
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                 max_length: int = 128, num_beams: int = 4, **kwargs):
        """
        Generate predictions for inference.
        
        Args:
            input_ids: Tokenized input utterances [batch_size, seq_len]
            attention_mask: Attention mask for inputs [batch_size, seq_len]
            max_length: Maximum length of generated sequence
            num_beams: Number of beams for beam search
            **kwargs: Additional generation parameters
        
        Returns:
            Generated token IDs [batch_size, generated_seq_len]
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            **kwargs
        )
    
    def tokenize_input(self, texts: list, max_length: int = 128):
        """
        Tokenize input utterances.
        
        Args:
            texts: List of input strings (with prefix "intent and slots: ")
            max_length: Maximum sequence length
        
        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        encoding = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        return encoding
    
    def tokenize_target(self, texts: list, max_length: int = 128):
        """
        Tokenize target sequences for training.
        
        Args:
            texts: List of target strings (e.g., "intent: X slots: a=b c=d")
            max_length: Maximum sequence length
        
        Returns:
            Tensor of label IDs with padding tokens replaced by -100
        """
        encoding = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        # Replace padding token id with -100 so it's ignored in loss
        labels = encoding['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return labels
    
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True):
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs to decode [batch_size, seq_len] or [seq_len]
            skip_special_tokens: Whether to skip special tokens
        
        Returns:
            List of decoded strings or single string
        """
        if token_ids.dim() == 1:
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)


def parse_generated_output(text: str) -> dict:
    """
    Parse the generated T5 output into structured format.
    
    Args:
        text: Generated string in format "intent: X slots: a=b c=d"
    
    Returns:
        Dictionary with 'intent' and 'slots' keys
        Returns None if parsing fails
    """
    result = {
        'intent': None,
        'slots': {}
    }
    
    try:
        text = text.strip()
        
        # Split by "slots:"
        if 'slots:' in text:
            intent_part, slots_part = text.split('slots:', 1)
        else:
            intent_part = text
            slots_part = ''
        
        # Parse intent
        if 'intent:' in intent_part:
            intent_str = intent_part.split('intent:', 1)[1].strip()
            result['intent'] = intent_str
        
        # Parse slots
        slots_part = slots_part.strip()
        if slots_part:
            slot_pairs = slots_part.split()
            for pair in slot_pairs:
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    result['slots'][key.strip()] = value.strip()
        
        return result
    
    except Exception:
        return None
