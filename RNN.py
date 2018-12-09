import torch
import torch.nn as nn
import pandas as pd


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size  # rnn OUTPUT SIZE
        # number of recurrent layers: 2 layers mean that the second layer will take the hidden_output from the first layer
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size,
                          num_layers, batch_first=True, dropout=0, nonlinearity='relu', bidirectional=False)  # batch_first=True means we expect input to be batch, seq_len, input_size
        # nonlinearity='relu' WHEN USING RNN
        self.fc = nn.Linear(hidden_size, output_size)  # 25 locations
        self.relu = nn.ReLU()

    def forward(self, input_data):
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers,
                         input_data.size(0), self.hidden_size, requires_grad=True)  # the size should be (number of layers*direction, batch, hidden_sizeß\)
        h0 = h0.cuda()  # move to gpu. if using cpu,comment
        # input: batch, seq_len, input_size
        # output: batch, seq_len, hidden_size
        output, h0 = self.rnn(input_data, h0)

        # Decode the hidden state of the last time ste

        output = self.fc(output[:, -1, :])
        #output = self.fc(h0)

        return output  # size: (sequence_size, output_size)
