import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.relu(self.hidden(x))
        out = self.linear(x)
        return out