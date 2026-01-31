# Detailed Instructions for GitHub Copilot: Joint Intent Classification and Slot Filling Models

## Overview
You need to construct three different neural network architectures for simultaneous intent classification and slot filling. All models should work on the same dataset and produce two outputs: intent labels (single classification) and slot labels (sequence labeling).

---

## General Architecture Principles

### Task Structure
- **Intent Classification**: Multi-class classification task producing a single label for the entire utterance
- **Slot Filling**: Sequence labeling task producing a label for each token in the input sequence
- **Joint Training**: Both tasks should be trained simultaneously with a combined loss function

### Loss Function
- Use Cross-Entropy Loss for intent classification
- Use Cross-Entropy Loss for slot filling (applied at each time step)
- Combine both losses: `total_loss = intent_loss + slot_loss` (or use weighted combination like `alpha * intent_loss + beta * slot_loss`)

### Data Handling
- Examine existing dataset loading code to understand the data format
- Input: Tokenized sequences (word indices or subword indices)
- Output 1: Intent label (single integer)
- Output 2: Slot labels (sequence of integers, same length as input)
- Ensure proper padding handling and attention masking

---

## Model 1: Encoder-Decoder with Learned Embeddings

### Architecture Components

#### Embedding Layer
- Create a `nn.Embedding` layer with vocabulary size and embedding dimension (e.g., 300)
- Initialize randomly or with pre-trained word vectors if available
- This layer is trainable

#### Encoder (Bidirectional GRU)
- Use `nn.GRU` with `bidirectional=True`
- Hidden size: choose appropriate dimension (e.g., 128, meaning 256 total with bidirectionality)
- Number of layers: start with 1-2 layers
- Input: embedded token sequences (batch_size, seq_len, embedding_dim)
- Output: encoder hidden states (batch_size, seq_len, hidden_size * 2) and final hidden state

#### Decoder (Unidirectional GRU)
- Use `nn.GRU` with `bidirectional=False`
- Hidden size: should match the encoder's total hidden dimension (256 if encoder hidden is 128)
- Initialize decoder's first hidden state from encoder's final hidden state
- For bidirectional encoder, concatenate forward and backward final states or use a linear projection
- Input: same embedded sequence (teacher forcing during training)
- Output: decoder hidden states (batch_size, seq_len, hidden_size)

#### Intent Classification Head
- Use encoder's final hidden state or pooled representation (e.g., max/mean pooling over sequence)
- Linear layer: `nn.Linear(encoder_hidden_size, num_intent_classes)`
- For bidirectional encoder: input size is `hidden_size * 2`

#### Slot Filling Head
- Use decoder's output at each time step
- Linear layer: `nn.Linear(decoder_hidden_size, num_slot_labels)`
- Apply to all time steps to get sequence of slot predictions

### Forward Pass Logic
1. Embed input tokens
2. Pass through encoder GRU to get encoder outputs and final hidden state
3. Transform encoder final hidden state for decoder initialization (handle bidirectionality)
4. Pass embedded tokens through decoder GRU initialized with encoder's final state
5. Apply intent classification head to encoder representation
6. Apply slot filling head to decoder outputs
7. Return both intent logits and slot logits

---

## Model 2: Transformer with Multi-Head Attention

### Architecture Components

#### Embedding Layer
- Token embedding: `nn.Embedding(vocab_size, d_model)`
- Positional encoding: either learned `nn.Embedding(max_seq_len, d_model)` or sinusoidal
- Combine token and positional embeddings by addition

#### Transformer Encoder
- Use `nn.TransformerEncoder` from PyTorch or build custom
- Number of layers: 4-6 encoder layers
- Each layer contains:
  - Multi-head self-attention mechanism (8-16 heads)
  - Feed-forward network (FFN with hidden dimension 4x the model dimension)
  - Layer normalization and residual connections
- Model dimension (d_model): 512 or 768
- Attention heads: ensure d_model is divisible by number of heads

#### Multi-Head Attention Details
- Each attention head should have dimension: `d_model / num_heads`
- Use scaled dot-product attention
- Include attention masking for padding tokens
- Dropout for regularization (0.1-0.3)

#### Intent Classification Head
- Pool the transformer output (use [CLS] token if added, or mean/max pooling)
- Linear layer: `nn.Linear(d_model, num_intent_classes)`
- Optional: add dropout before classification layer

#### Slot Filling Head
- Use transformer output at each position
- Linear layer: `nn.Linear(d_model, num_slot_labels)`
- Apply to all sequence positions

### Forward Pass Logic
1. Embed input tokens and add positional encodings
2. Create padding mask (True for padding positions, False for valid tokens)
3. Pass through transformer encoder with attention masking
4. Extract representation for intent (pooled or [CLS] token)
5. Apply intent classification head
6. Apply slot filling head to all positions
7. Return both intent logits and slot logits

### Special Considerations
- Consider adding a special [CLS] token at the beginning for intent classification
- Ensure proper masking so padding tokens don't affect attention
- Use layer normalization before or after residual connections (pre-norm vs post-norm)

---

## Model 3: Encoder-Decoder with Frozen BERT Embeddings

### Architecture Components

#### BERT Embedding Extraction
- Load a pre-trained BERT model (e.g., `bert-base-uncased`) using `transformers` library
- Use only the embedding layer or first few layers
- **Freeze all BERT parameters**: `requires_grad=False` for all BERT weights
- Extract contextualized embeddings from BERT
- BERT output dimension: 768 for base, 1024 for large

#### Encoder (Bidirectional GRU)
- Input size: BERT embedding dimension (768 for bert-base)
- Use `nn.GRU` with `bidirectional=True`
- Hidden size: choose appropriate dimension (e.g., 128)
- Input: BERT contextualized embeddings (batch_size, seq_len, 768)
- The encoder learns to process BERT embeddings further

#### Decoder (Unidirectional GRU)
- Similar to Model 1
- Hidden size should match encoder's total hidden dimension
- Initialize from encoder's final hidden state

#### Intent Classification Head
- Use encoder's final hidden state or pooled representation
- Linear layer: `nn.Linear(encoder_hidden_size, num_intent_classes)`

#### Slot Filling Head
- Use decoder's output at each time step
- Linear layer: `nn.Linear(decoder_hidden_size, num_slot_labels)`

### Forward Pass Logic
1. Tokenize input using BERT tokenizer (handle WordPiece tokenization)
2. Pass through BERT (frozen) to get contextualized embeddings
3. **Important**: Handle subword tokenization alignment with original tokens for slot filling
   - Either use only first subword of each word
   - Or pool subword representations (mean/max)
4. Pass BERT embeddings through encoder GRU
5. Transform encoder final hidden state for decoder initialization
6. Pass BERT embeddings through decoder GRU (or use encoder outputs)
7. Apply intent classification head to encoder representation
8. Apply slot filling head to decoder outputs
9. Return both intent logits and slot logits

### Special Considerations for BERT Integration
- **Tokenization Alignment**: BERT uses WordPiece/BPE tokenization which may split words
  - Original: ["book", "flight"] → BERT: ["book", "flight"]
  - Original: ["booking"] → BERT: ["book", "##ing"]
  - Need strategy to align BERT tokens back to original word-level slot labels
  
- **Alignment Strategies**:
  1. Use only the first subword token embedding for each word
  2. Average all subword embeddings for each word
  3. Use special token representation from BERT

- **Memory Efficiency**: 
  - Since BERT is frozen, you can optionally use `torch.no_grad()` during forward pass through BERT
  - Consider detaching BERT outputs before passing to GRU if memory is an issue

---

## Common Implementation Details

### Training
- Optimizer: Adam with learning rate 1e-3 to 5e-4
- Batch size: 32-128 depending on memory
- Gradient clipping: use `torch.nn.utils.clip_grad_norm_` with max_norm=5.0
- Learning rate scheduling: consider warmup + decay

### Padding and Masking
- Pad sequences to same length in batch
- Create attention masks (1 for real tokens, 0 for padding)
- For slot filling loss, mask out padding positions when computing loss
- Use `ignore_index` parameter in CrossEntropyLoss for padding token

### Evaluation Metrics
- Intent Classification: Accuracy
- Slot Filling: F1-score (token-level or entity-level)
- Joint Accuracy: percentage of examples where both intent and all slots are correct

### Model Saving/Loading
- Save model state dict with `torch.save()`
- Include hyperparameters and vocabulary mappings
- For Model 3, save the GRU weights separately (BERT weights don't need saving as they're frozen)

---

## Code Organization Suggestions

### File Structure
```
models/
  __init__.py
  encoder_decoder.py          # Model 1
  transformer_model.py        # Model 2
  bert_encoder_decoder.py     # Model 3
  
train.py                      # Training loop
evaluate.py                   # Evaluation script
data_loader.py               # Dataset class (examine existing)
utils.py                     # Helper functions
```

### Model Class Template
Each model should be a `nn.Module` with:
- `__init__()`: Initialize all components
- `forward()`: Define forward pass, return (intent_logits, slot_logits)
- Optional: `encode()`, `decode()` methods for modular design

### Hyperparameter Configuration
Create a config dictionary or class for each model containing:
- Vocabulary size
- Embedding dimension
- Hidden dimensions
- Number of layers
- Dropout rates
- Number of attention heads (Model 2)
- BERT model name (Model 3)

---

## Testing and Debugging Tips

1. **Shape Debugging**: Print tensor shapes at each step initially
2. **Overfitting Test**: Try overfitting on a tiny dataset (10 examples) first
3. **Baseline**: Ensure Model 1 works before attempting Model 2 and 3
4. **Gradient Flow**: Check gradients are flowing (not zero, not exploding)
5. **BERT Frozen Check**: Verify BERT parameters don't change during training for Model 3