# create an eavesdropper client

from client import Client,CustomDataset
from models import BERTClass
from sklearn import metrics
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2', do_lower_case=True)
from torch.utils.data import Dataset, DataLoader
class Eavesdropper():
    def __init__(self,batch_size,n_batches_per_client,max_len=20,epochs = 1,learning_rate = 1e-04,device = "cuda",n_classes = 6):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device
        self.max_len = max_len
        self.batch_size = batch_size
        self.n_batches_per_client = n_batches_per_client
        self.max_len = max_len
        self.model = BERTClass(n_classes=n_classes)
        self.load_data()
        self.model.to(self.device)
    def load_data(self):
        train_df = pd.read_csv('./data/client_datasets/client_eav_train.csv')
        train_df['list'] = train_df['list'].apply(lambda x: x.strip('][').split(','))
        train_df['list'] = train_df['list'].apply(lambda x: [float(i) for i in x])
        self.train_dataset = CustomDataset(train_df, tokenizer, self.max_len)
        self.train_loader = DataLoader(self.train_dataset, shuffle = True, batch_size=self.batch_size, num_workers=0)
        valid_df = pd.read_csv('./data/client_datasets/client_eav_valid.csv')
        valid_df['list'] = valid_df['list'].apply(lambda x: x.strip('][').split(','))
        valid_df['list'] = valid_df['list'].apply(lambda x: [float(i) for i in x])
        self.valid_dataset = CustomDataset(valid_df, tokenizer, self.max_len)
        self.valid_loader = DataLoader(self.valid_dataset, shuffle = True, batch_size=self.batch_size, num_workers=0)
        print(f'Eavesdropper initialized with {len(self.train_dataset)} training samples and {len(self.valid_dataset)} validation samples')
    def train(self,parameters):
        if parameters is not None:
            self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.Adam(params =  self.model.parameters(), lr=self.learning_rate)
        for epoch in range(self.epochs):
            ## compute number of batches
            ## take subset of train loader from n_batches_start to n_batches_end                    
            for _,data in enumerate(self.train_loader,0):
                ids = data['ids'].to(self.device,torch.long)
                mask = data['mask'].to(self.device,torch.long)
                token_type_ids = data['token_type_ids'].to(self.device,torch.long)
                targets = data['targets'].to(self.device,torch.float)
                outputs = self.model(ids, mask, token_type_ids)
                optimizer.zero_grad()
                loss = torch.nn.BCEWithLogitsLoss()(outputs, targets)
                
                ## Get gradients and make them negative
                loss.backward() 
                
                ## Update parameters 
                optimizer.step()
                
    def evaluate(self):
        self.model.eval()
        fin_targets=[]
        fin_outputs=[]
        with torch.no_grad():
            for _, data in enumerate(self.valid_loader):
                if _ > 10:
                    break
                ids = data['ids'].to(self.device,torch.long)
                mask = data['mask'].to(self.device,torch.long)
                token_type_ids = data['token_type_ids'].to(self.device,torch.long)
                targets = data['targets'].to(self.device,torch.float)
                outputs = self.model(ids, mask, token_type_ids)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
                
        outputs = np.array(fin_outputs) >= 0.5
        ## create weights for samples with all 0 classes to avoid bias
        accuracy = metrics.accuracy_score(fin_targets, outputs)
        f1_score = metrics.f1_score(fin_targets, outputs, average='micro')
        balanced_accuracy = metrics.balanced_accuracy_score(fin_targets, outputs)
        
        print("Eavesdropper Accuracy: ",accuracy)
        return accuracy,f1_score,balanced_accuracy
    def get_parameters(self):
        return self.model.state_dict()
    def set_parameters(self,parameters_state_dict):
        self.model.load_state_dict(parameters_state_dict, strict=True)
    def randomize_parameters(self):
        for layer in self.model.state_dict().keys():
            self.model.state_dict()[layer] = torch.randn(self.model.state_dict()[layer].shape)

    def check_parameters_change(self,parameters_before,parameters_after):
        for layer in parameters_before:
            if torch.equal(parameters_before[layer],parameters_after[layer]):
                continue
            else:
                print("Parameters changed")
                return True
        print("Parameters did not change")
        return False

