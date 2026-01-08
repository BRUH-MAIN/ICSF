import torch
import torch.nn as nn
import torch.nn.functional as F

class SlotGated(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, slot_dim, intent_dim, dropout=0.3):
        super(SlotGated, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Hidden dim is doubled because of bidirectional LSTM
        self.hidden_dim = hidden_dim * 2
        
        # --- Attention Layers ---
        # Intent Attention
        self.attention_w_intent = nn.Linear(self.hidden_dim, 1)
        
        # Slot Attention
        # Transform lstm output to key for dot-product attention
        self.slot_att_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # --- Slot Gate ---
        # Gate weights to transform intent context before combination
        self.gate_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        # Gate vector/projection to produce the gate value g
        self.gate_v = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # --- Classification Heads ---
        self.intent_out = nn.Linear(self.hidden_dim, intent_dim)
        self.slot_out = nn.Linear(self.hidden_dim, slot_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, x_lengths):
        """
        x: [batch_size, seq_len] - Input word indices
        x_lengths: [batch_size] - Length of each sentence
        """
        # 1. Embedding
        embeds = self.embedding(x) # [batch, seq, embed_dim]
        embeds = self.dropout(embeds)
        
        # 2. Shared Encoder (BiLSTM)
        # Pack padded sequence for correct RNN processing
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(embeds, x_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_input)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # lstm_out: [batch, seq_len, hidden_dim]
        
        batch_size, seq_len, _ = lstm_out.size()
        
        # Create mask for padding (1 for data, 0 for pad)
        # Assuming padding index is 0
        mask = (x != 0).unsqueeze(2) # [batch, seq_len, 1]
        
        # 3. Intent Attention & Prediction
        # Compute attention scores: alpha = softmax(H * W)
        attn_scores = self.attention_w_intent(lstm_out) # [batch, seq_len, 1]
        
        # Mask padded elements (set score to very small value)
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=1) # [batch, seq_len, 1]
        
        # c_intent = sum(alpha * H)
        c_intent = torch.sum(lstm_out * attn_weights, dim=1) # [batch, hidden_dim]
        
        # Intent Prediction
        intent_logits = self.intent_out(c_intent)
        
        # 4. Slot Attention
        # Calculate context c_slot for EACH word step.
        # We use a dot-product attention where Query = h_t, Key = Linear(H).
        # key: [batch, seq_len, hidden_dim]
        key = self.slot_att_linear(lstm_out)
        
        # scores: [batch, seq_len (query), seq_len (key)]
        # Q * K^T
        slot_scores = torch.bmm(lstm_out, key.transpose(1, 2))
        
        # Mask padding in the Keys (dim 2)
        # mask.transpose: [batch, 1, seq_len]
        slot_scores = slot_scores.masked_fill(mask.transpose(1, 2) == 0, -1e9)
        
        slot_weights = F.softmax(slot_scores, dim=2) # [batch, seq_len, seq_len]
        
        # c_slot: [batch, seq_len, hidden_dim]
        c_slot = torch.bmm(slot_weights, lstm_out)
        
        # 5. Slot Gate Mechanism
        # g = tanh(c_slot + W * c_intent)
        # Expand c_intent to match sequence length: [batch, seq_len, hidden_dim]
        c_intent_expanded = c_intent.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Calculate Gate activation
        gate_input = c_slot + self.gate_linear(c_intent_expanded)
        gate_activ = torch.tanh(gate_input)
        
        # Calculate Gate Value g (element-wise or scalar)
        # Here we produce a vector gate element-wise
        g = self.gate_v(gate_activ) # [batch, seq_len, hidden_dim]
        g = torch.sigmoid(g)
        
        # 6. Final Slot Prediction
        # "Mask" or "Amplify" slot features: h_t + c_slot * g
        combined_features = lstm_out + (c_slot * g)
        
        slot_logits = self.slot_out(combined_features)
        
        return intent_logits, slot_logits
