### Python Code for Covert Optimization for Hate Speech Classification


### Structure of the code base
```
main.py: Main file to run the simulations
server.py: Contains the server class
client.py: Contains the client class
eavesdropper.py: Contains the eavesdropper class
mdp.py: Contains the MDP and the MarkovChain class 
create_datasets.py: Contains the functions to create the datasets for clients and eavesdropper
models.py: Contains the Neural Network models
utils/: Folder containing utility functions
```

Create these directories in the root folder:
```
data/: Directory to store the dataset and results
data/input: Directory to store intermediate variables
data/input/spsa: Directory to store the SPSA parameters
data/client_datasets: Directory to store client datasets
data/logs: Directory to store logs
data/logs/experiment1: Directory to store logs for experiment 1
data/plots: Directory to store plots

```

Main prerequisites are:
```
 pytorch, ortools, seaborn, pandas, numpy, matplotlib, tqdm
```
but all prerequisites can be installed using the following command:
```
pip install -r requirements.txt
```


Python Version: *3.9.6*



Change the following variables in main.py:
```
- N_device: Number of client devices used in the federated learning setup
- N_communication_rounds: Number of rounds of federated learning
model: Model to be used for federated learning (Any model can be used we have implemented BERT+Linear) 
- fraction_of_data: Fraction of data to be used from the entire dataset
- n_classes: Number of classes in the dataset (toxic or not toxic for hatespeech classification dataset hence 1 class)
- client_parameters: Parameters for client neural network training (right now only learning rate can be set)
```


## Running the code (detailed instructions)

### Steps to create dataset
1. Download the dataset from https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data (train.csv.zip)
2. Unzip the file and place it in the data folder with the name train.csv
3. Run the following code to preprocess the data
``` python utils/preprocess.py```

### Steps to run the code:
1. Change the variables (mentioned above) in main.py
2. Create the required directories (mentioned above)
3. Create the dataset using the steps mentioned above
4. Run the code using ```python main.py``` for eavesdropper mentioned under different regimes (10% and no data) these regimes are run automatically
5. Generate the plots using ```python utils/generateplotforspsa.py``` (Plot 2 in Paper) and ```python utils/generateplotfromlog.py``` (Plot 1 in Paper)
6. The results are stored in the data/plots folder 



