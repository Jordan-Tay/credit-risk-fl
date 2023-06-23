import torch
import torch.nn as nn

class MLP(nn.Module):
  def __init__(self, input_dim, output_dim, dropout=False):
    super().__init__()
    self.layer_size = 64
    if dropout:
      self.seq = nn.Sequential(
          nn.Linear(input_dim, self.layer_size),
          nn.ReLU(),
          nn.Dropout(),
          nn.Linear(self.layer_size, self.layer_size),
          nn.ReLU(),
          nn.Dropout(),
          nn.Linear(self.layer_size, self.layer_size),
          nn.ReLU(),
          nn.Dropout(),
          nn.Linear(self.layer_size, self.layer_size),
          nn.ReLU(),
          nn.Dropout(),
          nn.Linear(self.layer_size, self.layer_size),
          nn.ReLU(),
          nn.Dropout(),
          nn.Linear(self.layer_size, self.layer_size),
          nn.ReLU(),
          nn.Dropout(),
          nn.Linear(self.layer_size, self.layer_size),
          nn.ReLU(),
          nn.Dropout(),
          nn.Linear(self.layer_size, self.layer_size),
          nn.ReLU(),
          nn.Dropout(),
          nn.Linear(self.layer_size, self.layer_size),
          nn.ReLU(),
          nn.Dropout(),
          nn.Linear(self.layer_size, self.layer_size),
          nn.ReLU(),
          nn.Dropout(),
          nn.Linear(self.layer_size, self.layer_size),
          nn.ReLU(),
          nn.Dropout(),
          nn.Linear(self.layer_size, output_dim),
          nn.Sigmoid(),
      )
    else:
      self.seq = nn.Sequential(
          nn.Linear(input_dim, self.layer_size),
          nn.ReLU(),
          nn.Linear(self.layer_size, self.layer_size),
          nn.ReLU(),
          nn.Linear(self.layer_size, self.layer_size),
          nn.ReLU(),
          nn.Linear(self.layer_size, self.layer_size),
          nn.ReLU(),
          nn.Linear(self.layer_size, self.layer_size),
          nn.ReLU(),
          nn.Linear(self.layer_size, self.layer_size),
          nn.ReLU(),
          nn.Linear(self.layer_size, self.layer_size),
          nn.ReLU(),
          nn.Linear(self.layer_size, self.layer_size),
          nn.ReLU(),
          nn.Linear(self.layer_size, self.layer_size),
          nn.ReLU(),
          nn.Linear(self.layer_size, self.layer_size),
          nn.ReLU(),
          nn.Linear(self.layer_size, self.layer_size),
          nn.ReLU(),
          nn.Linear(self.layer_size, output_dim),
          nn.Sigmoid(),
      )

  def forward(self, x):
    return self.seq(x)

input_dim = 28
hidden_dim = 256
n_layers = 4
output_dim = 21

class LSTMModel(nn.Module):
  def __init__(self, input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, n_layers=n_layers):
    super().__init__()
    self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
    self.linear = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
    x, _ = self.lstm(x)
    x = x[:, -1, :]
    x = self.linear(x)
    return nn.Sigmoid()(x)

class MLP2(nn.Module):
  def __init__(self, input_dim, output_dim, dropout=True, dropout_rate=0.5):
    super().__init__()
    self.layer_size = 512
    if dropout:
      self.seq = nn.Sequential(
          nn.Linear(input_dim, self.layer_size),
          nn.ReLU(),
          nn.Dropout(dropout_rate),
          nn.Linear(self.layer_size, self.layer_size),
          nn.ReLU(),
          nn.Dropout(dropout_rate),
          nn.Linear(self.layer_size, self.layer_size),
          nn.ReLU(),
          nn.Dropout(dropout_rate),
          nn.Linear(self.layer_size, self.layer_size),
          nn.ReLU(),
          nn.Dropout(dropout_rate),
          nn.Linear(self.layer_size, output_dim),
          nn.Sigmoid(),
      )
    else:
      self.seq = nn.Sequential(
          nn.Linear(input_dim, self.layer_size),
          nn.ReLU(),
          nn.Linear(self.layer_size, self.layer_size),
          nn.ReLU(),
          nn.Linear(self.layer_size, self.layer_size),
          nn.ReLU(),
          nn.Linear(self.layer_size, self.layer_size),
          nn.ReLU(),
          nn.Linear(self.layer_size, output_dim),
          nn.Sigmoid(),
      )
  def forward(self, x):
    return self.seq(x)


class HierarchicalLSTM(nn.Module):
  def __init__(self, num_time_series, num_time_steps, num_features, hidden_size):
    super(HierarchicalLSTM, self).__init__()

    self.num_time_series = num_time_series
    self.num_time_steps = num_time_steps
    self.num_features = num_features
    self.hidden_size = hidden_size

    self.lower_lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_size, num_layers=2, batch_first=True)

    self.pooling = nn.AdaptiveAvgPool1d(1)

  def forward(self, input):
    num_samples, num_time_series, num_time_steps, num_features = input.shape

    input = input.view(num_samples * num_time_series, num_time_steps, num_features)
    output, _ = self.lower_lstm(input)
    output = output[:, -1, :].view(num_samples, num_time_series, -1).permute(0, 2, 1)
    output = self.pooling(output).squeeze()

    return output

class LSTMModel2(nn.Module):
  def __init__(self, dropout=False, dropout_rate=0.5):
    super().__init__()
    self.lstm1 = HierarchicalLSTM(4, 12, 1, 4)
    self.lstm2 = HierarchicalLSTM(4, 4, 8, 16)
    self.lstm3 = HierarchicalLSTM(4, 4, 16, 32)
    self.lstm4 = HierarchicalLSTM(4, 4, 29, 32)

    self.mlp = MLP2(184, 1, dropout=dropout, dropout_rate=dropout_rate)

  def forward(self, x):
    nts, ts1, ts2, ts3, ts4 = x
    ts1 = self.lstm1(ts1)
    ts2 = self.lstm2(ts2)
    ts3 = self.lstm3(ts3)
    ts4 = self.lstm4(ts4)
    x = torch.cat((nts, ts1, ts2, ts3, ts4), 1)
    x = self.mlp(x)
    return x