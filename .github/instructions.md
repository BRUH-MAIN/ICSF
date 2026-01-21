Good, this structure tells me exactly what you’re doing — and also what you’re doing **wrong**.

Right now this repo is a **hybrid of legacy BiLSTM slot-gating and a half-started T5 path**. That’s technical debt. You either finish the T5 pipeline cleanly or this repo will rot.

Below is an **`instructions.md` tailored exactly to *this* repo**, telling Copilot (and you) how to **add a proper T5-based joint intent + slot pipeline without breaking existing code**.

No sugar-coating.

---

# instructions.md

## Joint Intent Classification & Slot Filling (T5, PyTorch)

### Scope

This repository currently contains:

* Legacy **BiLSTM / Slot-Gated** models
* Dataset in **intent + slot vocabulary format**
* A placeholder `t5.py`

The goal is to **add a clean, production-grade T5 joint model** **without modifying or deleting existing BiLSTM code**.

---

## Ground Rules (Read This First)

1. **Do NOT refactor or touch**:

   * `BiLSTMIntentClassifier.py`
   * `SlotGated.py`
   * Existing notebooks
2. All new logic must live in:

   ```
   src/models/t5.py
   src/t5_pipeline/
   ```
3. T5 is **text-to-text only**
4. No BIO tagging in the model
5. No vocab files during training (they are legacy)

If Copilot suggests reusing slot vocab indices → **reject it**.

---

## Target Architecture

```
Utterance
   ↓
T5 Encoder
   ↓
T5 Decoder
   ↓
"intent: X slots: a=b c=d"
```

Single forward pass. Single loss.

---

## New Directory Structure (MANDATORY)

Add the following:

```
src/
├── t5_pipeline/
│   ├── dataset.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
└── models/
    └── t5.py   (model wrapper only)
```

Do **not** put training logic in `main.py`.

---

## Dataset Interpretation (Critical)

Your `dataset/` directory follows the **classic NLU format**:

```
dataset/
├── train/
│   ├── seq.in
│   ├── seq.out
│   └── label
```

Where:

* `seq.in`  → utterance
* `seq.out` → slot tags (BIO)
* `label`   → intent

**We will IGNORE BIO tags during training**.

They are used **only** to reconstruct slot spans during preprocessing.

---

## Preprocessing Logic (`t5_pipeline/preprocess.py`)

### Input

* `seq.in`
* `seq.out`
* `label`

### Output (JSONL)

Each example must be serialized as:

```json
{
  "input_text": "intent and slots: book a flight from delhi to mumbai",
  "target_text": "intent: book_flight slots: from=delhi to=mumbai"
}
```

### Slot Reconstruction Rules

* Extract contiguous BIO spans
* Slot value must be text span from `seq.in`
* Slot keys must be lowercase
* Slot order must be deterministic (alphabetical)

If a slot span is malformed → drop the example.

This preprocessing **must happen once**, offline.

---

## Dataset Class (`dataset.py`)

Implement a PyTorch `Dataset` that:

* Loads preprocessed JSONL
* Returns raw strings:

  ```python
  {
    "input_text": str,
    "target_text": str
  }
  ```

Do **NOT** tokenize here.

---

## Model Wrapper (`models/t5.py`)

This file must:

* Load `T5ForConditionalGeneration`
* Expose:

  ```python
  forward(input_ids, attention_mask, labels)
  generate(input_ids, attention_mask)
  ```

No custom heads. No loss functions.

---

## Training (`train.py`)

### Requirements

* Tokenize inside training loop
* Use `labels=` for teacher forcing
* Optimizer: `AdamW`
* Mixed precision allowed
* Save best model by **validation exact match**

### Explicitly forbidden

* Dual losses
* Intent classifiers
* Slot token heads

---

## Evaluation (`evaluate.py`)

Evaluation is **string-based**, not token-based.

### Metrics

1. Intent Accuracy
2. Slot Precision / Recall / F1
3. Exact Match (intent + all slots)

### Parsing Rules

Generated text must follow:

```
intent: X slots: a=b c=d
```

If parsing fails → count as incorrect.

No partial credit hacks.

---

## Inference (`inference.py`)

Pipeline:

1. Raw utterance
2. Serialize input
3. `model.generate`
4. Parse output
5. Return structured dict

Example:

```json
{
  "intent": "book_flight",
  "slots": {
    "from": "delhi",
    "to": "mumbai"
  }
}
```

---

## How This Coexists With Existing Code

* `main.py` remains unchanged
* BiLSTM models remain usable
* T5 pipeline is **opt-in**, cleanly separated
* Comparison experiments become trivial

This is how a **research repo should be structured**.

---

## Common Copilot Mistakes to Reject

❌ Adding BIO heads to T5
❌ Using `vocab.slot` during training
❌ Token-level slot losses
❌ Modifying existing BiLSTM files
❌ Putting logic in notebooks

If you allow any of these, the pipeline is wrong.

---

