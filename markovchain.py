import numpy as np

# function for generating device data matrix
def generate_device_data_matrix(T = 1000,N_device = 100,N_total = 150000,thresfactor = 2):
    Batch_size = 40
    N_batches = N_total//Batch_size

    N_batch_per_device = N_batches//N_device
    thres = N_batches//thresfactor
    P = np.array([[0.8,0.2,0],
       [0.3,0.2,0.5],
       [0,0.2,0.8]])
    N_choices = np.array([ N_device//2.5, N_device//1.75, N_device//1.25],dtype=int)
    N_window = N_device//5 -1

    n_success = np.zeros(N_choices.shape[0])
    n_visits = np.zeros(N_choices.shape[0])
    X = 0
    device_data_matrix = np.zeros((T, N_device),dtype=np.int16)
    for t in range(T):
        no_of_devices_selected = int(np.random.uniform(N_choices[X] - N_window, N_choices[X] + N_window))
        
        devices_selected = np.random.choice(range(N_device),size=no_of_devices_selected,replace=False)
        # device_data_matrix[t,devices_selected] = np.random.choice(,size = N_choices[X],replace=True)
        device_data_matrix[t,devices_selected] = np.random.randint(0,N_batch_per_device,size = no_of_devices_selected)
        n_success[X] += device_data_matrix[t,:].sum()>thres
        n_visits[X] += 1
        X = np.random.choice(range(3),p = P[X])

    success_prob = n_success/n_visits
    print("Success probability for each oracle state: ",success_prob)
    np.save("./data/device_data_matrix.npy",device_data_matrix)


sucess_probs = []
for thres in np.linspace(1,10,100):
    sucess_probs.append(generate_device_data_matrix(thresfactor=thres))
import matplotlib.pyplot as plt
plt.plot(np.linspace(1,10,100),sucess_probs)
plt.xlabel("Threshold factor")
plt.ylabel("Success probability")
plt.savefig("./data/success_prob_vs_threshold_factor.png")

