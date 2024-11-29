import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, 
                            num_layers=2, batch_first=True, bidirectional=False)  # Note the batch_first=True
        self.fc1 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size//2)
        self.fc3 = nn.Linear(hidden_layer_size//2, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc1(x)
        x= self.fc2(x)
        x = self.fc3(x)
        x= x[:,-1,:]
        return x