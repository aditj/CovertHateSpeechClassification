### Successfully created dataset 
### Now create the server and clients
from server import Server
from client import Client 
from models import CNNImage
import os
import numpy as np
import pandas as pd
from create_datasets import create_datasets_clients
import matplotlib.pyplot as plt
from utils.randomseed import seed_everything
from tqdm import tqdm
import logging
def main():
    cumm_exp_res_file = "./data/logs/cummulative_experiment_results_MDP.txt"
    Ps = [[
       [0.7, 0.2, 0.05, 0.03,  0.01, 0.01, 0],
       [0.6, 0.2, 0.04, 0.02,  0.01, 0.01, 0.13],
       [0.5, 0.2, 0.04, 0.02,  0.01, 0.01, 0.23],
       [0.4, 0.2, 0.04, 0.02,  0.01, 0.01, 0.33],
       [0.3, 0.2, 0.03, 0.015, 0.01, 0.01, 0.43],
       [0.2, 0.2, 0.03, 0.015, 0.01, 0.01, 0.53],
       [0.1, 0.2, 0.03, 0.015, 0.01, 0.01, 0.63]
    ],]
    # Get experiment conditions which have been done
    experiment_conditions_done = []
    with open(cumm_exp_res_file,"r") as f:
        # iterate over lines
        lines = f.readlines()
        for line in lines:
            experiment_conditions_done.append(line.split(":")[3].replace("_True_acc","").replace("_False_acc","").split("_MNIST")[0].replace(" ",""))
    # get value counts of experiment_conditions_done
    experiment_conditions_done = pd.Series(experiment_conditions_done).value_counts()
    print(experiment_conditions_done)
    N_communication_rounds = 100 # Number of communication rounds
    N_successful = 30 # Number of successful communication roundss
    model = CNNImage # Model to be used
    dataset = "MNIST" # Dataset to be used

    n_classes = 10 # Number of classes
    
    fraction_of_data = 1 # Fraction of data to be used
    
    
    # Filename to store all "final" results of use with date
    client_parameters = {"learning_rate":0.0001} # Parameters for the client
    
    GENERATE_DATA = True # Generate data or not
    GENERATE_POLICY = True # Generate policy or not

    N_device = 40
    eavesdropper_training_size = 0.1 # Proportion of data available with the eavesdropper wrt another user
    eavesdropper_training_classes = 4 # Number of good classes available with the eavesdropper
    eavesdropper_training_prop = 0.9  # ratio of good classes to bad classes in the eavesdropper dataset
    for P in Ps:         
        if eavesdropper_training_prop == "normal":
            eavesdropper_training_prop = eavesdropper_training_classes/10
        
        experiment_condition = str(N_device)+"_"+str(N_successful)+"_"+str(N_communication_rounds) +  "_" + str(eavesdropper_training_size) + "_" + str(eavesdropper_training_classes) + "_" + str(eavesdropper_training_prop) # Experiment condition
        N_exp_runs = 10 # Number of experiment runs

        if experiment_condition in experiment_conditions_done.index:
            if experiment_conditions_done[experiment_condition] >= 2*N_exp_runs:
                print(f"Experiment {experiment_condition} condition already done")
                continue                        
            else: 
                N_exp_runs = N_exp_runs - (experiment_conditions_done[experiment_condition]//2)
                print(f"Experiment {experiment_condition} condition done incompletely, continuing with {N_exp_runs} more runs")
        else:
            print(f"Experiment {experiment_condition} condition not done, continuing with {N_exp_runs} runs")
        try:     
            for k in tqdm(range(N_exp_runs)):
                seed_everything(k) # Set seed
                
                if GENERATE_DATA:
                    create_datasets_clients(N_device = N_device, fraction_of_data = fraction_of_data,eavesdropper_training_size=eavesdropper_training_size,eavesdropper_training_classes=eavesdropper_training_classes,eavesdropper_training_prop=eavesdropper_training_prop)
                    tqdm.write("Datasets created")    
                

                parameters = Client(0,model(n_classes)).get_parameters() # Get parameters from client
                tqdm.write("Initial Parameters initialized")
                
                s_nongreedy = Server(N_device,N_communication_rounds,parameters,n_classes=n_classes,client_parameters=client_parameters,generate_policy = GENERATE_POLICY,experiment_condition=experiment_condition,greedy_policy= False,N_successful=N_successful,cumm_exp_res_file=cumm_exp_res_file)
                tqdm.write("Server initialized for non greedy policy")

                s_nongreedy.train()
                tqdm.write(f"Training complete for {k} run non greedy policy")
                
                ## Greedy Policy ##
                s = Server(N_device,N_communication_rounds,parameters,n_classes=n_classes,client_parameters=client_parameters,generate_policy = GENERATE_POLICY,greedy_policy= True,experiment_condition=experiment_condition,N_successful=N_successful,cumm_exp_res_file=cumm_exp_res_file)
                tqdm.write("Server initialized for greedy policy")
                
                s.train()
                tqdm.write(f"Training complete for {k} run greedy policy")
        except UnboundLocalError:
            continue
main()