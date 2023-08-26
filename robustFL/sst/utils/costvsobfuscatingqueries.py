# import mdp from parent directory
from mdp import MDP,MarkovChain
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils.randomseed import seed_everything
import seaborn as sns


def average_action(probpolicy,C,O,L,E,P,T):
        cost = 0
        x = 2*L*E + (L-4)*E
        for t in range(T):
                a = np.random.choice([0,1],p = [probpolicy[x],1-probpolicy[x]])
                cost += a
                x = np.random.choice(np.arange(O*L*E),p = P[a,x,:])
        return cost/T

markovchain = MarkovChain(N_device=20)
markovchain.generate_device_data_matrix()

state_learning_queries = 30
L = state_learning_queries + 1
O = 3
U = 2
E = 2
D = 0.3
P_O = markovchain.P
fs = markovchain.success_prob
## Advesarial cost
C_A = [[0,1.8],
    [0,0.8],
    [0,0.3]]
C_A = np.tile(C_A,L*E).reshape(O*L*E,U) # tiling adversarial cost 
C_Ls = np.linspace(0,4.5,1000)
n_iter = 1000
average_actions = np.zeros((len(C_Ls),n_iter))
j = 0 
calculate = False
if calculate:
    for c_l in C_Ls:
        ## Learner Cost
        C_L = np.tile(np.concatenate([np.repeat(np.linspace(c_l,10,L),E).reshape(-1,1),np.zeros((L*E,1))],axis=1),O).reshape(O*L*E,U)
        C_L[0::L*E,:] = [0,0]
        C_L[1::L*E,:] = [0,0]
        # C_L[L*E-2::L*E,:] = [1e10,0]
        # C_L[L*E-1::L*E,:] = [1e10,0]

        mdp = MDP(L,P_O,fs,C_A,C_L,D,"lplagrange")  
        n_communications = 120

        for i in range(n_iter):
            average_actions[j,i] = average_action(mdp.policy,C_A,O,L,E,mdp.P,n_communications)
        j+=1
    np.save("./data/average_actions_vs_cost.npy",average_actions)
average_actions = np.load("./data/average_actions_vs_cost.npy")
sns.set_style("whitegrid")
plt.plot(C_Ls,average_actions.mean(axis=1))
plt.xlabel("Cost of learning",fontsize=20)
plt.ylabel("Average proportion of $u = 1$",fontsize=20)
plt.savefig("./data/average_actions_vs_cost.pdf")
