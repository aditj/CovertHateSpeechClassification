import numpy as np
from sklearn import metrics
import torch
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from transformers import AlbertTokenizer, BertTokenizer
from transformers import logging
logging.set_verbosity_error()
import pandas as pd
#tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2', do_lower_case=True)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

class CustomDataset(Dataset):
    '''
    Custom dataset class for the client
    '''
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.comment_text
        self.targets = self.data['list']
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        # make string of a list as a list
        targets = self.targets[index]
        targets = [float(i) for i in targets]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(targets, dtype=torch.float)
        }

## Class for FL client 
class Client():
    '''
    A Federated Learning client class
    '''
    def __init__(self,cid,network,train_batch_size = 40,valid_batch_size = 32,max_len = 300,epochs = 1,learning_rate = 1e-03,device = "mps",n_classes = 6):
        '''
        Initialize the client with the following parameters:
        cid: Client ID
        network: Model to be used
        train_batch_size: Batch size for training
        valid_batch_size: Batch size for validation
        max_len: Maximum length of the input
        epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to be used mps or cuda or cpu
        n_classes: Number of classes
        '''
        self.cid = cid
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device
        self.max_len = max_len
        self.n_classes = n_classes 
        self.load_data()
        self.model = network
        self.model.to(self.device)
       
        print("Client", self.cid, "initialized with ",len(self.train_dataset),"train samples and ",len(self.valid_dataset),"valid samples")

    def load_data(self):
        '''
        Load the data for the client
        '''
        df = pd.read_csv(f'data/client_datasets/client_{self.cid}.csv')
        df['list'] = df['list'].apply(lambda x: x.strip('][').split(','))
        df['list'] = df['list'].apply(lambda x: [float(i) for i in x])
        train_df = df.sample(frac=0.8, random_state=200)
        valid_df = df.drop(train_df.index).reset_index(drop=True)
        train_df = train_df.reset_index(drop=True)
        self.train_dataset = CustomDataset(train_df, tokenizer, self.max_len)
        self.valid_dataset = CustomDataset(valid_df, tokenizer, self.max_len)
        self.train_loader = DataLoader(self.train_dataset, shuffle = True, batch_size=self.train_batch_size, num_workers=0)
        self.n_batch_per_client = len(self.train_loader)//self.train_batch_size
        self.valid_loader = DataLoader(self.valid_dataset, shuffle = True, batch_size=self.valid_batch_size, num_workers=0)
    def train(self,parameters,n_batches_max,i):
        '''
        Train the client with the given parameters (NN model parameters)
        parameters: Parameters of the model
        n_batches_max: Number of batches per client
        i: Client number
        '''
        self.set_parameters(parameters)
        self.model.train()
        ## Adam with weight decay AdamW
        optimizer = torch.optim.AdamW(params =  self.model.parameters(), lr=self.learning_rate)
        for epoch in range(self.epochs):
            ## compute number of batches
            n_batches = len(self.train_loader)
            n_batches_start = (n_batches_max*i) % n_batches
            n_batches_end = (n_batches_max*(i+1)) % n_batches
            if n_batches_end < n_batches_start:
                n_batches_end = n_batches
            # take subset of train  loader from n_batches_start to n_batches_end                    

            for _,data in enumerate(self.train_loader,0):
                if _ < n_batches_start:
                    continue
                if _ > n_batches_end:
                    break
                ids = data['ids'].to(self.device,torch.long)
                mask = data['mask'].to(self.device,torch.long)
                token_type_ids = data['token_type_ids'].to(self.device,torch.long)
                targets = data['targets'].to(self.device,torch.float)
                outputs = self.model(ids, mask, token_type_ids)
                optimizer.zero_grad()
                ### Add weights for samples with all 0 classes to avoid bias
                #pos_weights = torch.ones([self.n_classes])*10
                #pos_weights = pos_weights.to(self.device)
                loss = torch.nn.BCEWithLogitsLoss()(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            #print(f'Client: {self.cid}, Batch: {_}, Loss:  {loss.item()}')
    
    def evaluate(self,parameters,batch_size):
        '''
        Evaluate the client
        parameters: Parameters of the model
        batch_size: Batch size
        '''
        self.set_parameters(parameters)
        self.model.eval()
        fin_targets=[]
        fin_outputs=[]
        loss = []
        with torch.no_grad():
            for _, data in enumerate(self.valid_loader):
                ids = data['ids'].to(self.device,torch.long)
                mask = data['mask'].to(self.device,torch.long)
                token_type_ids = data['token_type_ids'].to(self.device,torch.long)
                targets = data['targets'].to(self.device,torch.float)
                outputs = self.model(ids, mask, token_type_ids)
               # pos_weights = torch.ones([self.n_classes])*10
               # pos_weights = pos_weights.to(self.device)
                #loss+=[torch.nn.BCEWithLogitsLoss(pos_weight = pos_weights)(outputs, targets)]
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        outputs = np.array(fin_outputs) >= 0.5
        fin_targets = np.array(fin_targets)
        ## create weights for samples with all 0 classes to avoid bias
        f1_score = metrics.f1_score(fin_targets, outputs, average='weighted',zero_division=1)
        accuracy = metrics.accuracy_score(fin_targets, outputs)
        balanced_accuracy = metrics.balanced_accuracy_score(fin_targets, outputs, adjusted=True)       
        print("Client: ",self.cid," Accuracy: ",accuracy)
        return accuracy,f1_score,balanced_accuracy
    def get_parameters(self):
        return self.model.state_dict()
    def set_parameters(self,parameters_state_dict):
        self.model.load_state_dict(parameters_state_dict, strict=True)
    def are_parameters_equal(self,parameters_state_dict):
        for layer in self.model.state_dict():
            if not torch.equal(self.model.state_dict()[layer],parameters_state_dict[layer]):
                print("Parameters are not equal")
                return False
        return True



