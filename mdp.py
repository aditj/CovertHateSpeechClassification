import numpy as np
from utils.solvemdp import solvelp,spsa
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
        self.oracle_states = np.zeros(T)
    def generate_device_data_matrix(self):
        for t in range(self.T):
            no_of_devices_selected = int(np.random.uniform(self.N_choices[self.X] - self.N_window, self.N_choices[self.X] + self.N_window))
            devices_selected = np.random.choice(range(self.N_device),size=no_of_devices_selected,replace=False)
            self.device_data_matrix[t,devices_selected] = np.random.randint(0,self.N_batch_per_device,size = no_of_devices_selected)
            self.n_success[self.X] += self.device_data_matrix[t,:].sum()>self.thres
            self.n_visits[self.X] += 1
            self.oracle_states[t] = self.X
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
    # import matplotlib.pyplot as plt
    # plt.plot(np.linspace(1,10,100),sucess_probs)
    # plt.xlabel("Threshold factor")
    # plt.ylabel("Success probability")
    # plt.savefig("./data/success_prob_vs_threshold_factor.png")


# function for generating device data matrix
class MDP():
    def __init__(self,M,P_O,fs,C_A,C_L,solvemethod = "lp"):
        self.O = P_O.shape[0]
        self.L = M
        self.X = self.O*self.L
        self.U = 2
        self.D = 0.5
        self.delta = 0.1
        self.P_O = P_O
        self.fs = fs
        self.generate_P()
        self.C_A = C_A
        self.C_L = C_L
        self.solvemdp(solvemethod)
    def generate_P(self):
        self.P = np.zeros((self.U, self.O*self.L,  self.O*self.L))
        for u in range(self.U):
            for o in range(self.O):
                f = (self.fs[o])*u # Probability of success for learner state
                delta = self.delta # Probability of new arrival
                # Probability transition for learner state
                P_L = (1-delta)*(f)*np.roll(np.identity(self.L),-1,axis=1) +  ((delta)*(f)+(1-delta)*(1-f))*np.roll(np.identity(self.L),0,axis=1) + delta*(1-f)*np.roll(np.identity(self.L),1,axis=1)# Probability transition for learner state               
                P_L[0,-1] = 0
                P_L[-1,0] = 0
                if ((P_L>0).sum(axis = 1)>3).sum()>0:
                    print("more learner transitions than required")
                P_L = P_L/P_L.sum(axis = 1).reshape(self.L,1)
                self.P[u,o*self.L:(o+1)*self.L,:] = np.tile(P_L,(1,self.O))*np.repeat(self.P_O[o],self.L,axis = 0)
            self.P[:,self.L-1::self.L,0::self.L] = 0
            self.P[u,:,:] = self.P[u,:,:]/self.P[u,:,:].sum(axis = 1).reshape(self.O*self.L,1)
            self.P[u,:,:] = np.around(self.P[u,:,:],decimals = 4)
        if self.fs.shape != (self.O):
            print("please correct dimension of fs")
        if self.P_O.shape != (self.O,self.O):
            print("please correct dimension of P_O")
        ### Stochastic Check 
        if (np.around(self.P.sum(axis = 2),7) != 1).sum() > 0:
            print("P is not stochastic")
    def policyfrom(self,probpolicy,X):
        probpolicy += 1e-10
        policy = (probpolicy/probpolicy.sum(axis = 1).reshape(X,1))[:,0]
        # replace na with 0
        return policy
    def solvemdp(self,method):
        if method == "lp":
            probpolicy = solvelp(self.C_A,self.C_L,self.P,self.X,self.U,self.D)
            self.policy = self.policyfrom(probpolicy,self.X)
            np.save("./data/input/policy.npy",self.policy)
        elif method == "vi":
            print("vi not defined")
          #  self.V = solvevi(self.P,self.C_A,self.C_L)
        elif method == "spsa":
            self.policy = spsa(self.P,self.C_A,self.C_L)
        else:
            print("Please enter a valid method")