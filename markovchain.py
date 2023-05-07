import numpy as np

# function for generating device data matrix
def generate_device_data_matrix(T = 1000,N_device = 100,N_total = 10000):
    N_batch = N_total//N_device
    P = [
        [0.5, 0.3, 0.2],
        [0.3,0.4,0.3],
        [0.2,0.3,0.5]
    ]
    N_choices = np.array([ N_device//3, N_device//2, N_device//1.25],dtype=int)


    batch_choice = np.array([N_batch//4,N_batch//2,N_batch] ,dtype=int)
    n_success = np.zeros(N_choices.shape[0])
    n_visits = np.zeros(N_choices.shape[0])
    X = 0
    thres = 5000
    device_data_matrix = np.zeros((T, N_device),dtype=np.int16)
    for t in range(T):
        devices_selected = np.random.choice(range(N_device),size=N_choices[X],replace=False)
        # device_data_matrix[t,devices_selected] = np.random.choice(,size = N_choices[X],replace=True)
        device_data_matrix[t,devices_selected] = np.random.randint(0,N_batch,size = N_choices[X])
        n_success[X] += device_data_matrix[t,:].sum()>thres
        n_visits[X] += 1
        X = np.random.choice(range(3),p = P[X])

    success_prob = n_success/n_visits
    print("Success probability for each oracle state: ",success_prob)
    np.save("./data/device_data_matrix.npy",device_data_matrix)