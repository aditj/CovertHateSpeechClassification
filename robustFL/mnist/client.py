import numpy as np
from sklearn import metrics
import torch
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from transformers import logging
logging.set_verbosity_error()
import pandas as pd
#tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')


import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

class CustomDataset(Dataset):

    def __init__(self, dataframe):
        
        self.pixels = dataframe.pixels
        self.targets = dataframe.label

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, index):
        # get list of pixels from string of list
        pixels = self.pixels[index]
        targets = self.targets[index]
        

        return {
            'pixels': torch.tensor(pixels, dtype=torch.int64),
            'targets': torch.tensor(targets, dtype=torch.float)
        }

## Class for FL client 
class Client():
    def __init__(self,cid,network,train_batch_size = 40,valid_batch_size = 32,max_len = 300,epochs = 1,learning_rate = 1e-03,device = "cuda",n_classes = 6,client_dataset_path = "./data/client_datasets/"):
        self.cid = cid
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device
        self.max_len = max_len
        self.n_classes = n_classes 
        self.client_dataset_path = client_dataset_path
        self.load_data()
        self.model = network
        self.model.to(self.device)
       
        print("Client", self.cid, "initialized with ",len(self.train_dataset),"train samples and ",len(self.valid_dataset),"valid samples")

    def load_data(self):
        df = pd.read_csv(f'{self.client_dataset_path}client_{self.cid}.csv')
        df['pixels'] = df['pixels'].apply(lambda x: x.strip('][').split(','))
        df['pixels'] = df['pixels'].apply(lambda x: [int(i) for i in x])
        train_df = df.sample(frac=0.8, random_state=200)
        valid_df = df.drop(train_df.index).reset_index(drop=True)
        train_df = train_df.reset_index(drop=True)
        self.train_dataset = CustomDataset(train_df)
        self.valid_dataset = CustomDataset(valid_df)
        self.train_loader = DataLoader(self.train_dataset, shuffle = True, batch_size=self.train_batch_size, num_workers=0)
        self.n_batch_per_client = len(self.train_loader)//self.train_batch_size
        self.valid_loader = DataLoader(self.valid_dataset, shuffle = True, batch_size=self.valid_batch_size, num_workers=0)
    def train(self,parameters,n_batches_max,i):
        self.set_parameters(parameters)
        self.model.train()
        ## Adam with weight decay AdamW
        optimizer = torch.optim.AdamW(params =  self.model.parameters(), lr=self.learning_rate)
        ## SGD
        # optimizer = torch.optim.SGD(params =  self.model.parameters(), lr=self.learning_rate)
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
                pixels = data['pixels'].to(self.device,torch.float)
                targets = data['targets'].to(self.device,torch.float)
                # reshape pixels to (batch_size,28,28,1)
                pixels = pixels.reshape(-1,1,28,28)
                outputs = self.model(pixels)
                optimizer.zero_grad()
                ### Add weights for samples with all 0 classes to avoid bias
                loss = torch.nn.CrossEntropyLoss()(outputs, targets.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            #print(f'Client: {self.cid}, Batch: {_}, Loss:  {loss.item()}')
    
    def evaluate(self,parameters,batch_size):
        self.set_parameters(parameters)
        self.model.eval()
        fin_targets=[]
        fin_outputs=[]
        loss = []
        with torch.no_grad():
            for _, data in enumerate(self.valid_loader):
                pixels = data['pixels'].to(self.device,torch.float)
                pixels = pixels.reshape(-1,1,28,28)
                targets = data['targets'].to(self.device,torch.float)
                outputs = self.model(pixels)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        outputs = np.array(fin_outputs) 
        outputs = np.argmax(outputs, axis=1)
        fin_targets = np.array(fin_targets)
        ## create weights for samples with all 0 classes to avoid bias
        accuracy = metrics.accuracy_score(fin_targets, outputs)
    
        # print("Client: ",self.cid," Accuracy: ",accuracy)
        return accuracy
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



