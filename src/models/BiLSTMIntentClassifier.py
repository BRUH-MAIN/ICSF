import torch
from torch import nn
from transformers import BertModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# BiLSTM with BERT embeddings for better accuracy
class BiLSTMIntentClassifier(nn.Module):
    def __init__(self, hidden_dim, output_dim, n_layers=1, bidirectional=True, dropout=0.3):
        super(BiLSTMIntentClassifier, self).__init__()
        print("Loading BERT model... This may take a few minutes on first run.")
        self.bert = BertModel.from_pretrained('bert-base-cased')
        # Freeze BERT parameters to speed up training
        for param in self.bert.parameters():
            param.requires_grad = False
        print("BERT model loaded successfully!")
        
        embedding_dim = 768  # BERT hidden size
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                            bidirectional=bidirectional, batch_first=True, 
                            dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask):
        # Get BERT embeddings
        with torch.no_grad():  # Don't compute gradients for BERT since it's frozen
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = bert_output.last_hidden_state  # (batch_size, seq_len, 768)
        
        # Pass through LSTM
        lstm_out, (hidden, cell) = self.lstm(embeddings)
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        out = self.fc(hidden)
        return out