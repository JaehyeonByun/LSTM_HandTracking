import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        weights = torch.softmax(self.attention(lstm_output), dim=1)  
        context = torch.sum(weights * lstm_output, dim=1)
        return context

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(
            input_size,
            hidden_layer_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        self.attention = Attention(hidden_layer_size * 2)

        self.fc1 = nn.Linear(hidden_layer_size * 2, hidden_layer_size)
        self.layer_norm1 = nn.LayerNorm(hidden_layer_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size // 2)
        self.layer_norm2 = nn.LayerNorm(hidden_layer_size // 2)

        self.fc3 = nn.Linear(hidden_layer_size // 2, output_size)

    def forward(self, x):
        lstm_output, _ = self.lstm(x) 

        attention_output = self.attention(lstm_output)

        # Pass through fully connected layers
        x = self.relu(self.layer_norm1(self.fc1(attention_output)))
        x = self.dropout(x)
        x = self.relu(self.layer_norm2(self.fc2(x)))
        x = self.fc3(x)

        return x
