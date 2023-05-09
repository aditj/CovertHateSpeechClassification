import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F
from keras.preprocessing import text, sequence

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x
    
class NeuralNet(nn.Module):
    def __init__(self, embedding_matrix,max_features):
        super(NeuralNet, self).__init__()
        self.max_features = max_features
        
        self.LSTM_UNITS = 128
        self.DENSE_HIDDEN_UNITS = 4 * self.LSTM_UNITS

        embed_size = embedding_matrix.shape[1]
        
        self.embedding = nn.Embedding(self.max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)
        
        self.lstm1 = nn.LSTM(embed_size, self.LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(self.LSTM_UNITS * 2, self.LSTM_UNITS, bidirectional=True, batch_first=True)
    
        self.linear1 = nn.Linear(self.DENSE_HIDDEN_UNITS, self.DENSE_HIDDEN_UNITS)
        self.linear2 = nn.Linear(self.DENSE_HIDDEN_UNITS, self.DENSE_HIDDEN_UNITS)
        
        self.linear_out = nn.Linear(self.DENSE_HIDDEN_UNITS, 1)
        
    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)
        
        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)
        
        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)
        
        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1  = F.relu(self.linear1(h_conc))
        h_conc_linear2  = F.relu(self.linear2(h_conc))
        
        hidden = h_conc + h_conc_linear1 + h_conc_linear2
        
        result = self.linear_out(hidden)
        
        return result

