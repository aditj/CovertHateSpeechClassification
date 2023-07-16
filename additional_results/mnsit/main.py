### Successfully created dataset 
### Now create the server and clients


from server import Server
from client import Client 
from models import CNNImage
import numpy as np
from create_datasets import create_datasets_clients
import matplotlib.pyplot as plt
from utils.randomseed import seed_everything
def main():
    N_device = 20 # Number of devices
    N_communication_rounds = 45 # Number of communication rounds
    model = CNNImage # Model to be used

    fraction_of_data = 0.5 # Fraction of data to be used
    n_classes = 1 # Number of classes
    client_parameters = {"learning_rate":0.0001} # Parameters for the client
    GENERATE_DATA = True # Generate data or not
    for k in range(1,10):
        seed_everything(k) # Set seed
        if GENERATE_DATA:
            create_datasets_clients(N_device = N_device, fraction_of_data = fraction_of_data)
            print("Datasets created")
            
        GENERATE_POLICY = True # Generate policy or not

        parameters = Client(0,model(n_classes)).get_parameters() # Get parameters from client
        print("Initial Parameters initialized")
        
        s_nongreedy = Server(N_device,N_communication_rounds,parameters,n_classes=n_classes,client_parameters=client_parameters,generate_policy = GENERATE_POLICY)
        print("Server initialized for non greedy policy")
        s_nongreedy.train()
        print(f"Training complete for {k} run non greedy policy")
        
        ## Greedy Policy ##
        s = Server(N_device,N_communication_rounds,parameters,n_classes=n_classes,client_parameters=client_parameters,generate_policy = GENERATE_POLICY,greedy_policy= True)
        print("Server initialized for greedy policy")
        s.train()
        print(f"Training complete for {k} run greedy policy")
main()