'''
adopted from pytorch.org (Classifying names with a character-level RNN-Sean Robertson)
'''

import torch
import torch.nn as nn
# from libs.layers import CudaVariable, myEmbedding, myLinear, myLSTM, biLSTM,PRU, GaussianNoise


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(RNN, self).__init__()
        '''
        Linear
        '''
        
        middle1_size = 8
        middle2_size = 4

        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=False)
        self.fc1 = nn.Linear(hidden_size, middle1_size) # linear 1
        self.fc2 = nn.Linear(middle1_size, middle2_size) # linear 2
        self.fc3 = nn.Linear(middle2_size, output_size) # linear 3
        self.fc = nn.Linear(hidden_size, output_size) # linear 3

        self.softmax = nn.LogSoftmax(dim=2)
        

        '''
        GaussianNoise
        
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=False)
        self.fc = nn.Linear(hidden_size, output_size)
        add_noise = 0.1
        self.add_noise = GaussianNoise(mean=0.0, sigma=0.1)
        self.softmax = nn.LogSoftmax(dim=2)
        #self.dropout = nn.Dropout(p=0.5)
        '''

        '''
        Layer normalization
        
        self.layer_norm1 = nn.LayerNorm(input_size,eps=1e-6) # layer 1
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=False)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
        '''


        self.batch_size = batch_size
        self.hidden_size = hidden_size

    def forward(self, input):
        '''
        input = input.unsqueeze(0)
        '''
        #import pdb; pdb.set_trace()
        hidden = self.initHidden()
        hidden = (hidden[0], hidden[1])
        output, hidden = self.rnn(input, hidden)
        output = self.fc(output)
        

        # output = self.fc1(output)
        # output = self.fc2(output)
        # output = self.fc3(output)
        

        '''
        GaussianNoise
        
        hidden = self.initHidden()
        hidden = (hidden[0], hidden[1])
        output, hidden = self.rnn(input, hidden)

        output = self.add_noise(output)
        output = self.fc(output)
        #output = self.dropout(output)
        '''

        '''
        Layer normalization
        
        hidden = self.initHidden()
        hidden = (hidden[0], hidden[1])
        input = self.layer_norm1(input)
        output, hidden = self.rnn(input, hidden)

        output = self.fc(output)
        '''


        output = self.softmax(output)

        return output

    def initHidden(self):
        return torch.zeros(2, 1, self.batch_size, self.hidden_size).to('cuda')
