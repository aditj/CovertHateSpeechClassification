# create an eavesdropper client

from client import Client,CustomDataset
from models import CNNImage
from sklearn import metrics
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
class Eavesdropper():
    def __init__(self,batch_size,n_batches_per_client,max_len=20,epochs = 1,learning_rate = 5e-04,device = "cuda",n_classes = 10,client_dataset_path = "./data/client_datasets/"):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device
        self.batch_size = batch_size
        self.n_batches_per_client = n_batches_per_client
        self.model = CNNImage()
        self.client_dataset_path = client_dataset_path
        self.load_data()
        self.model.to(self.device)
    def load_data(self):
        train_df = pd.read_csv(f'{self.client_dataset_path}client_eav_train.csv')

        train_df['pixels'] = train_df['pixels'].apply(lambda x: x.strip('][').split(','))
        train_df['pixels'] = train_df['pixels'].apply(lambda x: [int(i) for i in x])
        self.train_dataset = CustomDataset(train_df)
        self.train_loader = DataLoader(self.train_dataset, shuffle = True, batch_size=self.batch_size, num_workers=0)
        valid_df = pd.read_csv(f'{self.client_dataset_path}client_eav_valid.csv')
        valid_df['pixels'] = valid_df['pixels'].apply(lambda x: x.strip('][').split(','))
        valid_df['pixels'] = valid_df['pixels'].apply(lambda x: [int(i) for i in x])
        self.valid_dataset = CustomDataset(valid_df)
        self.valid_loader = DataLoader(self.valid_dataset, shuffle = True, batch_size=self.batch_size, num_workers=0)
        print(f'Eavesdropper initialized with {len(self.train_dataset)} training samples and {len(self.valid_dataset)} validation samples')
    def train(self,parameters):
        if parameters is not None:
            self.set_parameters(parameters)
        self.model.train()
        # optimizer = torch.optim.AdamW(params =  self.model.parameters(), lr=self.learning_rate)
        optimizer = torch.optim.SGD(params =  self.model.parameters(), lr=self.learning_rate)
        for epoch in range(self.epochs):
            ## compute number of batches
            ## take subset of train loader from n_batches_start to n_batches_end                    
            for _,data in enumerate(self.train_loader,0):
                pixels = data['pixels'].to(self.device,torch.float)
                targets = data['targets'].to(self.device,torch.float)
                # reshape pixels to (batch_size,1,28,28)
                pixels = pixels.reshape(-1,1,28,28)
                outputs = self.model(pixels)
                optimizer.zero_grad()
                loss = torch.nn.CrossEntropyLoss()(outputs,targets.long())
                
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
                pixels = data['pixels'].to(self.device,torch.float)
                pixels = pixels.reshape(-1,1,28,28)
                targets = data['targets'].to(self.device,torch.float)
                outputs = self.model(pixels)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        ## create weights for samples with all 0 classes to avoid bias
        fin_outputs = np.array(fin_outputs)
        fin_outputs = np.argmax(fin_outputs,axis=1)    

        accuracy = metrics.accuracy_score(fin_targets, fin_outputs)

        print("Eavesdropper Accuracy: ",accuracy)
        return accuracy
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

