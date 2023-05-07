
import torch
from transformers import AlbertModel, BertModel
from transformers import logging
logging.set_verbosity_error()

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = BertModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2") # AlbertModel.from_pretrained('albert-base-v2')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(128, 6)
    
    def forward(self, ids, mask, token_type_ids):
        _,output_1 = self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids,return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output