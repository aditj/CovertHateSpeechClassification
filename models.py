
import torch
from transformers import AlbertModel, BertModel
from transformers import logging
from torch.nn import functional as F
import torch.nn as nn
logging.set_verbosity_error()

class BERTClass(torch.nn.Module):
    def __init__(self,n_classes = 6):
        super(BERTClass, self).__init__()
        self.l1 = BertModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2") # AlbertModel.from_pretrained('albert-base-v2')
        self.l2 = torch.nn.Linear(128, 128)
        self.l3 = torch.nn.ReLU()
        self.l4 = torch.nn.Dropout(0.1)
        self.l5 = torch.nn.Linear(128, n_classes)
    
    def forward(self, ids, mask, token_type_ids):
        _,output_1 = self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids,return_dict=False)
        output_2 = self.l2(output_1)
        output_3 = self.l3(output_2)
        output_4 = self.l4(output_3)
        output = self.l5(output_4)
        return output


class CNNBERTClass(torch.nn.Module):
    def __init__(self,n_classes = 6):
        super(CNNBERTClass, self).__init__()
        self.l1 = BertModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2") # AlbertModel.from_pretrained('albert-base-v2')
        self.l2 = torch.nn.Conv1d(1, 32, 2)
        self.l3 = torch.nn.ReLU()
        self.l4 = torch.nn.Conv1d(32, 64, 2)
        self.l5 = torch.nn.ReLU()
        self.l6 = torch.nn.MaxPool1d(64)
        self.l7 = torch.nn.Linear(64, 64)
        self.l8 = torch.nn.ReLU()
        self.l9 = torch.nn.Dropout(0.1)
        self.l10 = torch.nn.Linear(64, n_classes)
    def forward(self, ids, mask, token_type_ids):
        _,output_1 = self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids,return_dict=False)
        output_2 = self.l2(output_1.unsqueeze(1))
        output_3 = self.l3(output_2)
        output_4 = self.l4(output_3)
        output_5 = self.l5(output_4)
        output_6 = self.l6(output_5)
        output_7 = self.l7(output_6.squeeze(2))
        output_8 = self.l8(output_7)
        output_9 = self.l9(output_8)
        output = self.l10(output_9)
        return output

class SpatialDropout(torch.nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x
    
class LSTM(torch.nn.Module):
    def __init__(self, embedding_matrix,max_features):
        super(LSTM, self).__init__()
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






