### Successfully created dataset 
### Now create the server and clients


from server import Server
from client import Client 
from models import BERTClass,CNNBERTClass
import numpy as np
from create_datasets import create_datasets_clients
import matplotlib.pyplot as plt

def main():
    N_device = 20
    N_communication_rounds = 100
    model = BERTClass

    fraction_of_data = 0.5
    n_classes = 1
    client_parameters = {"learning_rate":0.0001}
    GENERATE_DATA = True
    if GENERATE_DATA:
        create_datasets_clients(N_device = N_device, fraction_of_data = fraction_of_data)
        print("Datasets created")
        
    GENERATE_POLICY = True

    parameters = Client(0,model(n_classes)).get_parameters()
    print("Initial Parameters initialized")
    s = Server(N_device,N_communication_rounds,parameters,n_classes=n_classes,client_parameters=client_parameters,generate_policy = GENERATE_POLICY)
    print("Server initialized")
    
    s.train()
    print("Training complete")

    
    plt.plot(s.aggregated_accuracies)
    plt.savefig("./data/accuracies.png")
main()