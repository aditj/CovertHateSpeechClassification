### Successfully created dataset 
### Now create the server and clients


from server import Server
from client import Client 
from models import BERTClass,CNNBERTClass
import numpy as np
from create_datasets import create_datasets_clients
import matplotlib.pyplot as plt

def main():
    N_device = 100
    N_communication_rounds = 100
    fraction_of_data = 1
    
    GENERATE_DATA = True
    if GENERATE_DATA:
        create_datasets_clients(N_device = N_device, fraction_of_data = fraction_of_data)
        print("Datasets created")
        
    

    parameters = Client(0,CNNBERTClass()).get_parameters()
    print("Initial Parameters initialized")
    s = Server(N_device,N_communication_rounds,parameters)
    print("Server initialized")
    
    s.train()
    print("Training complete")
    np.save("./data/aggregated_accuracies.npy",s.aggregated_accuracies)
    print("Accuracies saved")

    
    plt.plot(s.aggregated_accuracies)
    plt.savefig("./data/accuracies.png")
main()