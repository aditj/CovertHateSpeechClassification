import torch
from client import Client
from models import BERTClass,CNNBERTClass
import numpy as np
from markovchain import MarkovChain
from tqdm import tqdm
from solvemdp import solvelp
## add date and time to log file
import datetime
now = datetime.datetime.now()
LOG = f"./data/logs/server_{now.strftime('%Y-%m-%d_%H:%M:%S')}.log"
import logging                                                     
logging.basicConfig(filename=LOG, filemode="w", level=logging.INFO)  
from eavesdropper import Eavesdropper
class Server():
    def __init__(self,n_clients,n_communications,parameters,n_classes,client_parameters):
        self.n_clients = n_clients
        self.global_parameters = parameters.copy()
        self.aggregated_parameters = parameters.copy()
        self.n_communications = n_communications
        self.n_classes = n_classes
        self.clients = []
        self.client_parameters = client_parameters
        self.model = BERTClass
        self.initialize_clients()
        self.aggregated_loss = 0
        self.aggregated_accuracies = np.zeros((self.n_communications,2))
        self.markovchain = MarkovChain(N_device=self.n_clients)
        self.successful_round = self.markovchain.successful_round
        self.device_data_matrix = self.markovchain.device_data_matrix
        self.n_batch_per_client = self.clients[0].n_batch_per_client
        self.train_batch_size = self.clients[0].train_batch_size
        self.eavesdropper_random = Eavesdropper(self.train_batch_size,self.n_batch_per_client,self.clients[0].max_len,n_classes = self.n_classes)
        self.eavesdropper_smart = Eavesdropper(self.train_batch_size,self.n_batch_per_client,self.clients[0].max_len,n_classes = self.n_classes)
        self.obfuscating_parameters = self.eavesdropper_random.get_parameters()
        self.smart_obfuscating_parameters = self.eavesdropper_smart.get_parameters()
        self.zero_aggregated_parameters()
        self.markovchain.generate_device_data_matrix()
        self.state_learning_queries = 10
        self.state_oracle = 0
        self.policy = np.zeros((self.markovchain.P.shape))
        print("Server initialized")
    def train(self):
        # for each communication round
        for i in tqdm(range(self.n_communications)):
            if self.successful_round[i] == 1 :
                self.zero_aggregated_parameters() # zero the aggregated parameters
                self.aggregated_loss = 0 # zero the aggregated loss
                self.aggregated_f1 = 0
                self.aggregated_balanced_accuracy = 0
                clients_participating = np.ones(self.n_clients)*self.n_batch_per_client # 
                # Markov chain based client selection
                clients_participating = self.device_data_matrix[i] # get the clients participating in this communication round
                # randomly select clients
                # self.percent_clients = 0.6
                # self.percent_clients = np.random.uniform(0.4,0.9)
                # clients_participating = np.random.choice(self.n_clients,size=int(self.percent_clients*self.n_clients),replace=False)
                # All Clients
                # clients_participating = np.ones(self.n_clients)*self.n_batch_per_client # 
                for j,client_batch_size in tqdm(enumerate(clients_participating,0)): # for each client
                    
                    self.clients[j].train(self.global_parameters,client_batch_size,i) # train the client
                    self.add_parameters(self.clients[j].get_parameters()) # add the parameters to the aggregated parameters
                    evaluations = self.clients[j].evaluate(self.clients[j].get_parameters(),client_batch_size) 
                    self.aggregated_loss += evaluations[0]
                    self.aggregated_f1 += evaluations[1]
                    self.aggregated_balanced_accuracy += evaluations[2]
                self.divide_parameters(len(clients_participating)) # divide the aggregated parameters by the number of clients
                self.aggregated_loss/=len(clients_participating) 
                self.aggregated_f1/=len(clients_participating)
                self.aggregated_balanced_accuracy/=len(clients_participating)
                self.assign_global_parameters(self.aggregated_parameters) # assign the global parameters to the aggregated parameters
                logging.info(f'Communication round: {i}, Aggregated Accuracies: {self.aggregated_loss}, {self.aggregated_f1}, {self.aggregated_balanced_accuracy}') # print the loss
                print(f'Communication round: {i}, Aggregated Accuracies: {self.aggregated_loss}, {self.aggregated_f1}, {self.aggregated_balanced_accuracy}') # print the loss
                
            else:

                self.randomize_eavesdropper_parameters(i)
                self.eavesdropper_random.train(self.obfuscating_parameters)
                self.smart_obfuscating_parameters = self.eavesdropper_smart.get_parameters()
                self.eavesdropper_smart.train(self.smart_obfuscating_parameters)
                evaluations = self.eavesdropper_random.evaluate(self.obfuscating_parameters)
                evaluations_smart = self.eavesdropper_smart.evaluate(self.smart_obfuscating_parameters)
                logging.info(f"Communication round {i} Eavesdropper accuracy: {evaluations[0]} F1 {evaluations[1]} Balanced Accuracy {evaluations[2]}")
                logging.info(f"Communication round {i} SmartEavesdropper accuracy: {evaluations_smart[0]} F1 {evaluations_smart[1]} Balanced Accuracy {evaluations_smart[2]}")
                print("Communication round {} failed".format(i))
                
                
    def initialize_clients(self):
        for i in range(self.n_clients): # for each client
            self.clients.append(Client(i,self.model(self.n_classes),n_classes=self.n_classes,learning_rate=self.client_parameters["learning_rate"])) # initialize a client
    def randomize_eavesdropper_parameters(self,i):
        # randomize the eavesdropper parameters
        # random seed
        np.random.seed(i)
        for layer in self.obfuscating_parameters:
            self.obfuscating_parameters[layer] = torch.as_tensor(np.random.uniform(-1,1,self.obfuscating_parameters[layer].shape),dtype=torch.float32)
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
    def is_equal_parameters(self,parameters):
        for layer in self.obfuscating_parameters:
            if not torch.equal(self.obfuscating_parameters[layer],parameters[layer]):
                return False
        return True
    def generate_policy(self):
        self.policy = solvelp(self.markovchain.C_A,self.markovchain.C_L,self.markovchain.P,self.markovchain.P.shape[0],self.markovchain.U,self.markovchain.D )
        