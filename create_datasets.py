### Create datasets
#### Partition the entire dataset into N Clients
import pandas as pd
from markovchain import generate_device_data_matrix

## Function to create datasets
def create_datasets_clients(N_device = 100, fraction_of_data = 1):
    df = pd.read_csv("./data/train.csv")
    df = df.sample(frac=fraction_of_data).reset_index(drop=True)
    
    df['list'] = df[df.columns[2:]].values.tolist()
    df = df[['comment_text', 'list']].copy()
    #### Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    N_total = df.shape[0]
    N_batch = int(N_total/N_device)
    generate_device_data_matrix(N_device = N_device,N_total = N_total)
    print("Device data matrix generated")

    #### Partion the dataset into N devices disjoint datasets and save dataset
    #### Serially partition the dataset
    for i in range(N_device):
        client_dataset = df.iloc[i*N_batch:(i+1)*N_batch,:]
        client_dataset = client_dataset.reset_index(drop=True)
        client_dataset.to_csv(f"./data/client_datasets/client_{i}.csv")
