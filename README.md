### Covert Optimization for Hate Speech Classification
### Outline of simulations
Read best practices for simulating deep nn and FL

### What do you want to simulate? 


#### A bit of terminology 
Number of communication round $n_comm$
Number of epochs trained at each round: $n_epoch$
Number of 
#### 
General Markovian Federated Learning Simulation: 

An oracle which has datasets where different number of clients participate at each communication round different number of devices partcipate in performing learning
the partipating devised perform gradient descent (or modified version) at each communication round for some epochs and return the weights. The learner aggregates the weights anc returns the aggregated weights to the clients and this goes on. 
The number of clients participating is markovian in nature. 

Learner private markovian fl simulation 
The learner can now have a threshold on the noise it can tolerate in the response. 
The number of clients or the state of the oracle is known to the learner, but the response can still be very noisy and that is *assumed* to be a function of the total datapoints used in training. We *assume* the other factors remain constant and the threshold on the noise is a threshold on the total sample size (Is this conclusion sound?). Therefore at each communication round the number of devices is markovian and the number of samples per indiviual devices is probabilistic leading to a probability of error which can be emperically calculated. 

Now the learner can choose to skip based on the number of clients and the total number of successful epochs left. 

#### How do you simulate the markov chain based oracle
Assume that each device has equal number of samples so N devices have N_s samples  or N_s/BS = N_b batches, each making total number of $NN_s = N_train$ samples. 

Now at each communication step i,  N_i devices participate and each select a subset 

We assume 3 state markov chain where either of $N_A < N_B < N_C$ devices can participate. Each time each paticipating takes out of either N_s or N_s/2 samples iid. The markov chain has transition matrix $P$. The training is done for some epochs on the batches then. 


iid vs non-iid
#### We then write the code for the server and client
There are N_clients initialized each of which have a test and training set loader
The server runs for N_comm rounds
in each round it samples the said number of clients
Updates the weights
If the total>threshold:
Let's the client train the NN on the number of batches in the matrix 
(or the complete batch or some batch size)
and the clients sends the weights back
the weights are then averaged out 
the loss is also computed for the training data a and then aggregated 
and this process is repeated again


####
