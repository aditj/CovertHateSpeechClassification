### Covert Optimization for Hate Speech Classification


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
data/df_treated_comments_comments.csv: Preprocessed training data
data/input: Directory to store intermediate variables
data/client_datasets: Directory to store client datasets
data/logs: Directory to store logs
```

Main prerequisites are:
```
 pytorch, ortools, seaborn, pandas, numpy, matplotlib, tqdm
```

Change the following variables in main.py:
```
num_clients: Number of clients
```
and run using ```python main.py```. 

Figures can be generated using ```python utils/generateplotforspsa.py``` and ```python utils/generateplotfromlog.py```.

####
