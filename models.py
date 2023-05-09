
import torch
from transformers import AlbertModel, BertModel
from transformers import logging
logging.set_verbosity_error()

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = BertModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2") # AlbertModel.from_pretrained('albert-base-v2')
        self.l2 = torch.nn.Linear(128, 128)
        self.l3 = torch.nn.ReLU()
        self.l4 = torch.nn.Dropout(0.1)
        self.l5 = torch.nn.Linear(128, 6)
    
    def forward(self, ids, mask, token_type_ids):
        _,output_1 = self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids,return_dict=False)
        output_2 = self.l2(output_1)
        output_3 = self.l3(output_2)
        output_4 = self.l4(output_3)
        output = self.l5(output_4)
        return output


class CNNBERTClass(torch.nn.Module):
    def __init__(self):
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
        self.l10 = torch.nn.Linear(64, 6)
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






