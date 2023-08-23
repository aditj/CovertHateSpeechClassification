### Create datasets
#### Partition the entire dataset into N Clients
import pandas as pd
import os
import numpy as np
## Function to create datasets
def create_datasets_clients(N_device = 10, 
                            fraction_of_data = 1,
                            batch_size = 40,
                            client_dataset_path= "./data/client_datasets/",
                            ):
    # check if client_datasets folder exists
    if not os.path.exists(client_dataset_path):
        os.makedirs(client_dataset_path)
    else: 
        # delte all files inside the folder
        for filename in os.listdir(client_dataset_path):
            os.remove(client_dataset_path+filename)
    df = pd.read_csv("./data/train.csv",index_col = False)
    # relabel label as target
    df['label'].rename('target', inplace=True)
    
    ## Take fraction of data
    df = df.sample(frac=fraction_of_data).reset_index(drop=True)
    ## Create a list of the labels
    df['pixels'] = df.drop('label', axis=1).values.tolist()

    df = df[['pixels', 'label']].copy()
    #### Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    N_total = df.shape[0]
    N_batch = int(N_total/N_device)

    #### Partion the dataset into N devices disjoint datasets and save dataset
    #### Serially partition the dataset
    for i in range(N_device):
        client_dataset = df.iloc[i*N_batch:(i+1)*N_batch,:]
        client_dataset = client_dataset.reset_index(drop=True)
        client_dataset.to_csv(f"{client_dataset_path}client_{i}.csv")
    print("Datasets created for ",N_device," clients")
  
    
