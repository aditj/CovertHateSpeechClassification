from models import BERTClass

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime 
now = datetime.datetime.now()

from mdp import MDP,MarkovChain
from client import Client

class Server():
    def __init__(self,n_clients,n_communications,parameters,n_classes,
                 client_parameters,generate_policy = False,greedy_policy = False,
                 experiment_condition = "test",N_successful = 30,cumm_exp_res_file="./data.log"
                 ,P_O = np.array([[0.8,0.2,0],
       [0.15,0.7,0.15],
       [0.0,0.15,0.85]], 
           ),
       N_choices = None,
       client_dataset_path = "./data/client_datasets/", 
       C_A = [[0,1.8],
                   [0,0.8],
                   [0,0.3]],
        thres_factor = 4
                   ):
        ### FL Server Related ###
        self.n_clients = n_clients
        self.global_parameters = parameters.copy()
        self.aggregated_parameters = parameters.copy()
        self.n_communications = n_communications
        self.n_classes = n_classes
        
        
        ### MDP Related ###
        self.n_total = 60000 
        self.thres_factor = thres_factor
        self.markovchain = MarkovChain(N_device=self.n_clients,N_total=self.n_total,P = P_O,N_choices = N_choices,thresfactor=self.thres_factor)
        self.markovchain.generate_device_data_matrix()
        self.successful_round = self.markovchain.successful_round
        self.device_data_matrix = self.markovchain.device_data_matrix

        ## Create file to write log
        self.experiment_condition = experiment_condition + "_"+ str(greedy_policy) + "_"
        self.file = f"./data/logs/experiment1/{now.strftime('%Y-%m-%d_%H:%M:%S')}" + self.experiment_condition + ".log"
        self.state_learning_queries = N_successful
        self.C_A = C_A
        self.get_policy(generate_policy,greedy_policy)
        self.state_oracle = 0

        ### FL Client Related ###
        self.clients = []
        self.client_parameters = client_parameters
        self.model = BERTClass
        self.initialize_clients(client_dataset_path)
        self.n_batch_per_client = self.clients[0].n_batch_per_client
        self.train_batch_size = self.clients[0].train_batch_size
        self.count_learning_queries = 0

        self.cumm_exp_res_file = cumm_exp_res_file

      
        print("Server initialized")
    def train(self):
        self.count_learning_queries = 0
        with open(self.file, "a") as f:
            f.write('-'*100)
            f.write('\n')
            f.write(self.experiment_condition)
            f.write('\n')
            f.write('-'*100)
            f.write('\n')
        for i in tqdm(range(self.n_communications)):
            self.state_oracle = self.markovchain.oracle_states[i]
            action_prob = self.policy[int(self.state_oracle*self.L + self.E*self.state_learning_queries)]
            action = np.random.choice([0,1],p=[action_prob,1-action_prob])
            tqdm.write(f"Action: {action}, State: {self.state_oracle}, Queries: {self.state_learning_queries}, Prob: {action_prob}")
            if self.state_learning_queries == 0:
                action = 0
            if action == 1:
                self.count_learning_queries += 1
                if self.successful_round[i] == 1 :
                    self.state_learning_queries -= 1
                    ### Zero the aggregated parameters and loss
                    self.zero_aggregated_parameters() # zero the aggregated parameters
                    self.aggregated_loss = 0 # zero the aggregated loss

                    clients_participating = self.select_clients(i)
                    ### Train the clients and aggregate the parameters
                    for j,client_batch_size in tqdm(enumerate(clients_participating,0)): # for each client
                        self.clients[j].train(self.global_parameters,client_batch_size,i) # train the client
                        self.add_parameters(self.clients[j].get_parameters()) # add the parameters to the aggregated parameters
                        evaluations = self.clients[j].evaluate(self.clients[j].get_parameters(),client_batch_size) 
                        self.aggregated_loss += evaluations
                        
                    
                    self.divide_parameters(len(clients_participating)) # divide the aggregated parameters by the number of clients
                    self.aggregated_loss/=len(clients_participating) 
                    
                    self.assign_global_parameters(self.aggregated_parameters) # assign the global parameters to the aggregated parameters
                    ### write in self.file
                    with open(self.file, "a") as f:
                        f.write(f'Communication round: {i}, Aggregated Accuracies: {self.aggregated_loss} {self.count_learning_queries}')
                        f.write('\n')
                    tqdm.write(f'Communication round: {i}, Aggregated Accuracies: {self.aggregated_loss}') # print the loss
                else:
                    tqdm.write("Communication round {} failed and no learning".format(i))
            else:
                tqdm.write("Communication round {} Skipped".format(i))
            
            
        with open(self.file, "a") as f:
            f.write('-'*100)
            f.write('\n')
        with open(self.cumm_exp_res_file,"a") as f:
            f.write(datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')+": "+ self.experiment_condition + " : " + str(self.aggregated_loss)+ ", " + str(self.count_learning_queries))
            f.write('\n')
        
    def initialize_clients(self,client_dataset_path):
        for i in range(self.n_clients): # for each client
            self.clients.append(Client(i,self.model(n_classes=self.n_classes),n_classes=self.n_classes,learning_rate=self.client_parameters["learning_rate"],client_dataset_path=client_dataset_path)) # initialize a client
    def select_clients(self,i):
        # All Clients
        #clients_participating = np.ones(self.n_clients)*self.n_batch_per_client # 
        # Markov chain based client selection
        clients_participating = self.device_data_matrix[i] # get the clients participating in this communication round
        # randomly select clients
        # self.percent_clients = 0.6
        # self.percent_clients = np.random.uniform(0.4,0.9)
        # clients_participating = np.random.choice(self.n_clients,size=int(self.percent_clients*self.n_clients),replace=False)
        # All Clients
        # clients_participating = np.ones(self.n_clients)*self.n_batch_per_client # 
        return clients_participating
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
    def get_policy(self,generate_policy,greedy_policy):

        
        self.L = self.state_learning_queries + 1
        self.O = self.markovchain.P.shape[0]
        U = 2
        D = 0.15
        self.E = 2
        P_O = self.markovchain.P
        fs = self.markovchain.success_prob
        ## Advesarial cost
        
        C_A = np.tile(self.C_A,self.L*self.E).reshape(self.O*self.L*self.E,U) # tiling adversarial cost 
        ## Learner Cost
        C_L = np.tile(np.concatenate([np.repeat(np.linspace(0.6,20,self.L),self.E).reshape(-1,1),np.zeros((self.L*self.E,1))],axis=1),self.O).reshape(self.O*self.L*self.E,U)

        C_L[0::self.L*self.E,:] = [0,0]
        C_L[1::self.L*self.E,:] = [0,0]
        # C_L[self.L*self.E-2::self.L*self.E,:] = [1e10,0]
        # C_L[self.L*self.E-1::self.L*self.E,:] = [1e10,0]
        if generate_policy:  
            mdp = MDP(self.L,P_O,fs,C_A,C_L,D,"lplagrange")
            

        if greedy_policy:
            self.policy = np.load('./data/input/greedy_policy.npy') 
            return
        self.policy = np.load("./data/input/policy.npy")
        # Plot the policy
        plt.figure(figsize=(10,10))
        plt.plot(np.arange(self.policy.shape[0]),self.policy[:],label='Obfuscate')
        plt.savefig('./data/plots/policy.png')
        plt.close()
    def train_randomly(self):
        self.count_learning_queries = 0
    
        with open(self.file, "a") as f:
            f.write('-'*100)
            f.write('\n')
            f.write(self.experiment_condition + ' RANDOM POLICY ')
            f.write('\n')
            f.write('-'*100)
            f.write('\n')
        action_sequence_indices = np.random.choice(np.arange(self.n_communications),size=self.state_learning_queries,replace=False)
        action_sequence = np.zeros(self.n_communications)
        action_sequence[action_sequence_indices] = 1
        for i in tqdm(range(self.n_communications)):

            action = action_sequence[i]
            tqdm.write(f"Action: {action}, State: {self.state_oracle}, Queries: {self.state_learning_queries}, Prob: 1")
            if self.state_learning_queries == 0:
                action = 0
            if action == 1:
                self.count_learning_queries += 1
                
                self.state_learning_queries -= 1
                ### Zero the aggregated parameters and loss
                self.zero_aggregated_parameters() # zero the aggregated parameters
                self.aggregated_loss = 0 # zero the aggregated loss

                clients_participating = self.select_clients(i)
                ### Train the clients and aggregate the parameters
                for j,client_batch_size in tqdm(enumerate(clients_participating,0)): # for each client
                    self.clients[j].train(self.global_parameters,client_batch_size,i) # train the client
                    self.add_parameters(self.clients[j].get_parameters()) # add the parameters to the aggregated parameters
                    evaluations = self.clients[j].evaluate(self.clients[j].get_parameters(),client_batch_size) 
                    self.aggregated_loss += evaluations
                
                self.divide_parameters(len(clients_participating)) # divide the aggregated parameters by the number of clients
                self.aggregated_loss/=len(clients_participating) 
                
                self.assign_global_parameters(self.aggregated_parameters) # assign the global parameters to the aggregated parameters
                ### write in self.file
                with open(self.file, "a") as f:
                    f.write(f'Communication round: {i}, Aggregated Accuracies: {self.aggregated_loss} {self.count_learning_queries}')
                    f.write('\n')
                tqdm.write(f'Communication round: {i}, Aggregated Accuracies: {self.aggregated_loss}') # print the loss
            else:
                tqdm.write("Communication round {} Skipped".format(i))
            
            
            
        with open(self.file, "a") as f:
            f.write('-'*100)
            f.write('\n')
        with open(self.cumm_exp_res_file,"a") as f:
            f.write(datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')+": "+ self.experiment_condition + " , " + str(self.aggregated_loss))
            f.write('\n')