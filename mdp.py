import numpy as np
from utils.solvemdp import solvelp,spsa,solvelp_generalcost

class MarkovChain():
    def __init__(self,T = 1000,N_device = 100,N_total = 150000,thresfactor = 4,P = np.array([[0.7,0.3,0],
       [0.15,0.7,0.15],
       [0.0,0.15,0.85]])):
        self.T = T
        self.N_device = N_device
        self.N_total = N_total
        self.thresfactor = thresfactor
        self.Batch_size = 40
        self.N_batches = N_total//self.Batch_size
        self.N_batch_per_device = self.N_batches//self.N_device
        self.thres = self.N_batches//self.thresfactor
        self.P = P
        self.N_choices = np.array([ N_device//2.25, N_device//1.75, N_device//1.1],dtype=int)
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
        self.success_prob = np.zeros((self.P.shape[0],2))
        for i in range(self.P.shape[0]):
            self.success_prob[i,0] = 1 - self.n_success[i]/self.n_visits[i]
            self.success_prob[i,1] = self.n_success[i]/self.n_visits[i]
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
    def __init__(self,M,P_O,fs,C_A,C_L,D,solvemethod = "lp"):
        self.O = P_O.shape[0]
        self.L = M
        self.E = 2
        self.X = self.O*self.L*self.E
        self.U = 2
        self.D = D
        self.delta = 0.2
        self.P_O = P_O
        self.fs = fs
        self.generate_P()
        self.C_A = C_A
        self.C_L = C_L
        self.solvemdp(solvemethod)
        self.generate_greedy_policy()
    def generate_P(self):
        self.P = np.zeros((self.U,self.O*self.L*self.E,self.O*self.L*self.E))
        for a in range(self.U):
            for o in range(self.O):
                for l in range(self.L):
                    for e in range(self.E):
                        if l == 0:
                            p_success = 0
                        else:
                            p_success = self.fs[o,a]
                        for o_prime in range(self.O):
                            p_o_o_prime = self.P_O[o,o_prime]
                            for l_prime in range(self.L):
                                l_transition_success = l_prime == l + e - 1
                                l_transition_failure = l_prime == l + e 
                                if l==self.L-1: 
                                    l_transition_success = l_prime == l - 1
                                    l_transition_failure = l_prime == l
                                for e_prime in range(self.E):
                                    p_e_prime = e_prime*self.delta + (1-e_prime)*(1-self.delta)
                                    if l == self.L-1 or l==self.L-2:
                                        p_e_prime = 1 - e_prime
                                    self.P[a,o*self.L*self.E+l*self.E+e,o_prime*self.L*self.E+l_prime*self.E+e_prime] = p_o_o_prime*(l_transition_success*p_success + l_transition_failure*(1-p_success))*p_e_prime 
                                
        # check if P is stochastic
        if (np.around(self.P.sum(axis = 2),4) != 1).sum() > 0:
            print(np.where(self.P.sum(axis = 2)<0.99)[1])
            print(self.P.sum(axis = 2))
            print("P is not stochastic")
    def policyfrom(self,probpolicy,X):
        probpolicy += 1e-10
        policy = (probpolicy/probpolicy.sum(axis = 1).reshape(X,1))[:,0]
        # replace na with 0
        return policy
    def solvemdp(self,method):
        if method == "lp":
            probpolicy = solvelp(self.C_A,self.C_L,self.P,self.X,self.U,self.D,alpha=1)
            self.policy = self.policyfrom(probpolicy,self.X)
            np.save("./data/input/policy.npy",self.policy)
        elif method == "vi":
            print("vi not defined")
          #  self.V = solvevi(self.P,self.C_A,self.C_L)
        elif method=="lplagrange":
            C_lamb = lambda lambd: self.C_A + (lambd)*self.C_L
            
            running_c = 0
            lambds = np.linspace(0.2,0.4,10)
            avgcost = np.zeros(len(lambds))

            for i,lamb in enumerate(lambds):
                probpolicy = solvelp_generalcost(C_lamb(lamb),self.P,self.O*self.E*self.L,self.U,)
                avgcost[i] = self.occupation_measure(probpolicy,self.C_L,self.O,self.L,self.E,self.U)
                print(lamb,avgcost[i])

                probpolicy = probpolicy/(1e-10+ np.sum(probpolicy,axis = 1).reshape(self.O*self.L*self.E,1))
                
                policy2 = probpolicy
                if avgcost[i] < self.D:
                    break
                policy1 = probpolicy
            lamb1 = lambds[i-1]
            # policy1 = solvelp(C_A + lamb1*C_L,P,O*E*L,U)
            occupation1 = avgcost[i-1]
            # policy1 = policy1/(np.sum(policy1,axis = 1).reshape(O*L*E,1))
            policy1[self.L*self.E-1::self.L*self.E,:] = [0,1]
            lamb2 = lambds[i]
            # policy2 = solvelp(C_A + lamb2*C_L,P,O*E*L,U)
            policy2[self.L*self.E-1::self.L*self.E,:] = [0,1]
            occupation2 = avgcost[i]
            # policy2 = policy2/(np.sum(policy2,axis = 1).reshape(O*L*E,1))
            alpha = (self.D - occupation2)/(occupation1 - occupation2)
            print(alpha,occupation1,occupation2)
            probpolicy = alpha*policy1 + (1-alpha)*policy2
            self.policy = self.policyfrom(probpolicy,self.X)
            np.save("./data/input/policy.npy",self.policy)
        elif method == "spsa":
            self.policy = spsa(self.P,self.C_A,self.C_L)
        else:
            print("Please enter a valid method")
    def occupation_measure(self,probpolicy,C,O,L,E,U):
        c = 0
        for i in range(O*L*E):
            for u in range(U):
                c += C[i,u]*probpolicy[i,u]
        return c
    def generate_greedy_policy(self):
        self.greedy_policy = np.zeros_like(self.policy)
        np.save("./data/input/greedy_policy.npy",self.greedy_policy)