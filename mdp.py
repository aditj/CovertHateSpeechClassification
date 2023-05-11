import numpy as np
class MarkovChain():
    def __init__(self,T = 1000,N_device = 100,N_total = 150000,thresfactor = 4,P = np.array([[0.8,0.2,0],
        [0.3,0.2,0.5],
        [0,0.2,0.8]])):
        self.T = T
        self.N_device = N_device
        self.N_total = N_total
        self.thresfactor = thresfactor
        self.Batch_size = 40
        self.N_batches = N_total//self.Batch_size
        self.N_batch_per_device = self.N_batches//self.N_device
        self.thres = self.N_batches//self.thresfactor
        self.P = P
        self.N_choices = np.array([ N_device//2.5, N_device//1.75, N_device//1.25],dtype=int)
        self.N_window = N_device//5 -1
        self.n_success = np.zeros(self.N_choices.shape[0])
        self.n_visits = np.zeros(self.N_choices.shape[0])
        self.X = 0
        self.device_data_matrix = np.zeros((T, N_device),dtype=np.int16)
        self.successful_round = np.zeros(T,dtype=np.int16)
        ## Parameters of MDP
        
    def generate_device_data_matrix(self):
        for t in range(self.T):
            no_of_devices_selected = int(np.random.uniform(self.N_choices[self.X] - self.N_window, self.N_choices[self.X] + self.N_window))
            devices_selected = np.random.choice(range(self.N_device),size=no_of_devices_selected,replace=False)
            self.device_data_matrix[t,devices_selected] = np.random.randint(0,self.N_batch_per_device,size = no_of_devices_selected)
            self.n_success[self.X] += self.device_data_matrix[t,:].sum()>self.thres
            self.n_visits[self.X] += 1
            self.X = np.random.choice(range(3),p = self.P[self.X])
            if self.device_data_matrix[t,:].sum()>self.thres:
                self.successful_round[t] = 1
        self.success_prob = self.n_success/self.n_visits
        print("Success probability for each oracle state: ",self.success_prob)
def generate_success_prob():
    sucess_probs = []
    m = MarkovChain()
    for thres in np.linspace(1,10,100):
        sucess_probs.append(m.generate_device_data_matrix())
    import matplotlib.pyplot as plt
    plt.plot(np.linspace(1,10,100),sucess_probs)
    plt.xlabel("Threshold factor")
    plt.ylabel("Success probability")
    plt.savefig("./data/success_prob_vs_threshold_factor.png")


# function for generating device data matrix
class MDP():
    def __init__(self,M,P_O,fs,C_A,C_L):
        self.O = 3
        self.L = M
        self.X = self.O*self.L
        self.U = 2
        self.D = 0.6
        self.P_O = P_O
        self.fs = fs
        self.P  = self.generate_P()
        self.C_A = C_A
        self.C_L = C_L
    def generate_P():
        