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
    def __init__(self,batch_size,n_batches_per_client,max_len=20,epochs = 1,learning_rate = 1e-05,device = "cuda"):
        
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device
        self.max_len = max_len
        self.batch_size = batch_size
        self.n_batches_per_client = n_batches_per_client
        self.max_len = max_len
        self.model = BERTClass()
        self.create_dataset()
        self.load_data()
        self.model.to(self.device)
    def create_dataset(self):
        df = pd.read_csv('./data/train.csv')
        df = df[['id','comment_text','toxic','severe_toxic','obscene','threat','insult','identity_hate']]
        ## filter away rows with toxic
        df_tobedropped = df[(df['obscene'] == 1)]
        train_df = df.drop(df_tobedropped.index)
        train_df['list'] = train_df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].values.tolist()
        train_df = train_df.drop(['toxic','severe_toxic','obscene','threat','insult','identity_hate'],axis=1)
        train_df = train_df.sample(n = self.batch_size*self.n_batches_per_client, random_state=200)
        train_df = train_df.reset_index(drop=True)
        train_df.to_csv('./data/client_datasets/eavesdropper_train.csv',index=False)
        n_train = len(train_df)
        n_valid = len(df) - n_train
        valid_df = df.sample(n = int(n_valid*0.05), random_state=200)
        valid_df['list'] = valid_df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].values.tolist()
        valid_df = valid_df.drop(['toxic','severe_toxic','obscene','threat','insult','identity_hate'],axis=1)
        valid_df = valid_df.reset_index(drop=True)
        valid_df.to_csv('./data/client_datasets/eavesdropper_test.csv',index=False)
    def load_data(self):
        train_df = pd.read_csv('./data/client_datasets/eavesdropper_train.csv')
        train_df['list'] = train_df['list'].apply(lambda x: x.strip('][').split(','))
        train_df['list'] = train_df['list'].apply(lambda x: [float(i) for i in x])
        
        self.train_dataset = CustomDataset(train_df, tokenizer, self.max_len)
        self.train_loader = DataLoader(self.train_dataset, shuffle = True, batch_size=self.batch_size, num_workers=0)
        
        valid_df = pd.read_csv('./data/client_datasets/eavesdropper_test.csv')
        valid_df['list'] = valid_df['list'].apply(lambda x: x.strip('][').split(','))
        valid_df['list'] = valid_df['list'].apply(lambda x: [float(i) for i in x])
        self.valid_dataset = CustomDataset(valid_df, tokenizer, self.max_len)
        self.valid_loader = DataLoader(self.valid_dataset, shuffle = True, batch_size=self.batch_size, num_workers=0)
        print(f'Eavesdropper initialized with {len(self.train_dataset)} training samples and {len(self.valid_dataset)} validation samples')
    def train(self,parameters):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.Adam(params =  self.model.parameters(), lr=self.learning_rate)
        for epoch in range(self.epochs):
            ## compute number of batches
            # take subset of train  loader from n_batches_start to n_batches_end                    
            for _,data in enumerate(self.train_loader,0):
                ids = data['ids'].to(self.device,torch.long)
                mask = data['mask'].to(self.device,torch.long)
                token_type_ids = data['token_type_ids'].to(self.device,torch.long)
                targets = data['targets'].to(self.device,torch.float)
                outputs = self.model(ids, mask, token_type_ids)
                optimizer.zero_grad()
                loss = torch.nn.BCEWithLogitsLoss()(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            #print(f'Client: {self.cid}, Batch: {_}, Loss:  {loss.item()}')
    
    def evaluate(self,parameters):
        self.set_parameters(parameters)
        self.model.eval()
        fin_targets=[]
        fin_outputs=[]
        with torch.no_grad():
            for _, data in enumerate(self.valid_loader):
                ids = data['ids'].to(self.device,torch.long)
                mask = data['mask'].to(self.device,torch.long)
                token_type_ids = data['token_type_ids'].to(self.device,torch.long)
                targets = data['targets'].to(self.device,torch.float)
                outputs = self.model(ids, mask, token_type_ids)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        outputs = np.array(fin_outputs) >= 0.5
        ## create weights for samples with all 0 classes to avoid bias
        all_zeros = np.zeros([6])
        weights = np.where(np.array(fin_targets) == all_zeros,1.0,10.0).sum(axis=1)
        print("No of samples will all zero classes: ",np.where(weights == 6.0).sum())
        accuracy = metrics.accuracy_score(fin_targets, outputs, sample_weight = weights)
        accuracy = metrics.accuracy_score(fin_targets, outputs)
        print("Eavesdropper Accuracy: ",accuracy)
        return accuracy
    def get_parameters(self):
        return self.model.state_dict()
    
    def set_parameters(self,parameters_state_dict):
       
        self.model.load_state_dict(parameters_state_dict, strict=True)
    


    

