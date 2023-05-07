import torch
from client import Client
from model import BERTClass
import numpy as np
from tqdm import tqdm
LOG = "cmd.log"
import logging                                                     
logging.basicConfig(filename=LOG, filemode="w", level=logging.INFO)  


class Server():
    def __init__(self,n_clients,n_communications,parameters,device_data_matrix):
        self.n_clients = n_clients
        self.global_parameters = parameters.copy()
        self.aggregated_parameters = parameters.copy()
        self.zero_aggregated_parameters()
        self.n_communications = n_communications
        self.device_data_matrix = device_data_matrix
        self.clients = []
        self.initialize_clients()
        self.aggregated_loss = 0
        self.n_batch_per_client = self.clients[0].train_batch_size
        self.aggregated_accuracies = np.zeros(self.n_communications)
        print("Server initialized")
    def train(self):
        # for each communication round
        for i in tqdm(range(self.n_communications)):

            self.zero_aggregated_parameters() # zero the aggregated parameters
            self.aggregated_loss = 0 # zero the aggregated loss
           # clients_participating = self.device_data_matrix[i] # get the clients participating in this communication round
            clients_participating = np.ones(self.n_clients)*self.n_batch_per_client # 
            # randomly select clients
            self.percent_clients = 0.6
            self.percent_clients = np.random.uniform(0.4,0.9)
            clients_participating = np.random.choice(self.n_clients,size=int(self.percent_clients*self.n_clients),replace=False)
            for j in tqdm(clients_participating): # for each client
                # if client_batch_size == 0:  # if the client is not participating in this communication round,
                #     continue
                client_batch_size = 40
                self.clients[j].train(self.global_parameters,client_batch_size,i) # train the client
                self.add_parameters(self.clients[j].get_parameters()) # add the parameters to the aggregated parameters
                self.aggregated_loss += self.clients[j].evaluate(self.clients[j].get_parameters(),client_batch_size) # add the loss to the aggregated loss
            self.divide_parameters(len(clients_participating)) # divide the aggregated parameters by the number of clients
            self.aggregated_loss/=len(clients_participating) 
            self.aggregated_accuracies[i] = self.aggregated_loss
            self.assign_global_parameters(self.aggregated_parameters) # assign the global parameters to the aggregated parameters
            logging.info(f'Communication round: {i}, Aggregated Accuracies: {self.aggregated_loss} with {len(clients_participating)} clients') # print the loss
            print(f'Communication round: {i}, Aggregated Accuracies: {self.aggregated_loss}') # print the loss
        
    def initialize_clients(self):
        for i in range(self.n_clients): # for each client
            self.clients.append(Client(i,BERTClass())) # initialize a client
                
    def get_parameters(self):
        return self.global_parameters
    def add_parameters(self,parameters):
        for layer in self.aggregated_parameters:
            self.aggregated_parameters[layer] += parameters[layer]  # add the parameters to the aggregated parameters
    def divide_parameters(self,divisor):
        for layer in self.aggregated_parameters:
            self.aggregated_parameters[layer] = self.aggregated_parameters[layer]/divisor # divide the aggregated parameters by the divisor
    def assign_global_parameters(self,parameters):
        for layer in self.global_parameters:
            self.global_parameters[layer] = parameters[layer] # assign the global parameters to the aggregated parameters
    def zero_aggregated_parameters(self):
        for layer in self.aggregated_parameters:
            self.aggregated_parameters[layer] = torch.zeros_like(self.aggregated_parameters[layer])
    def is_equal_parameters(self):
        for layer in self.aggregated_parameters:
            if not torch.equal(self.global_parameters[layer],self.aggregated_parameters[layer]):
                return False
        return True