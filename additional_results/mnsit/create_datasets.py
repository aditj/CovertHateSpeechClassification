### Create datasets
#### Partition the entire dataset into N Clients
import pandas as pd
import os
import numpy as np
## Function to create datasets
def create_datasets_clients(N_device = 10, fraction_of_data = 1,batch_size = 40,eavesdropper_training_size = 1,eavesdropper_training_classes = 5,eavesdropper_training_prop = 0.1):
    # check if client_datasets folder exists
    if not os.path.exists("./data/client_datasets"):
        os.makedirs("./data/client_datasets")
    else: 
        # delte all files inside the folder
        for filename in os.listdir("./data/client_datasets"):
            os.remove("./data/client_datasets/"+filename)
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
        client_dataset.to_csv(f"./data/client_datasets/client_{i}.csv")
    print("Datasets created for ",N_device," clients")
    ## Create dataset for eavesdropper
    # Filter out the df if list column has 0 
    # have 10% 0 and 90% 1
    classes = np.random.choice(10,eavesdropper_training_classes,replace=False)
    eavesdropper_training_size = int( eavesdropper_training_size*N_batch)
    good_class_size = int(eavesdropper_training_size*eavesdropper_training_prop)
    bad_class_size = eavesdropper_training_size - good_class_size
    df_eav_1 = df[df['label'].apply(lambda x: x  in classes)].sample(n = good_class_size,)
    df_eav_0 = df[df['label'].apply(lambda x: x not in classes)].sample(n = bad_class_size)
    train_df_eav = pd.concat([df_eav_1,df_eav_0])
    train_df_eav = train_df_eav.reset_index(drop=True)

    valid_df_eav = df.drop(train_df_eav.index).reset_index(drop=True)
    valid_df_eav = valid_df_eav.sample(n = int(N_batch)).reset_index(drop=True)
    print("Eavesdropper dataset created with shape: ",train_df_eav.shape,valid_df_eav.shape)
    train_df_eav.to_csv(f"./data/client_datasets/client_eav_train.csv",index=False)
    valid_df_eav.to_csv(f"./data/client_datasets/client_eav_valid.csv",index=False)

