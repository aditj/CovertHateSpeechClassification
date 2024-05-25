# import mdp from parent directory

from mdp import MDP,MarkovChain
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
num_private_end = 0
def average_cost(probpolicy,C,P,T):
        cost = 0
        x = 0
        for t in range(T):
                a = np.random.choice([0,1],p = probpolicy[x,:])
                cost += C[x,a]*probpolicy[x,a]
                x = np.random.choice(np.arange(O*L*E),p = P[a,x,:])
        return cost/T
for k in range(1000):
        markovchain = MarkovChain(N_device=20)
        markovchain.generate_device_data_matrix()
        successful_round = markovchain.successful_round
        device_data_matrix = markovchain.device_data_matrix
        generate_policy = True
        
        if generate_policy:
                state_learning_queries = 16
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
        [0,0.4]]
                C_A = np.tile(C_A,L*E).reshape(O*L*E,U) # tiling adversarial cost 
                ## Learner Cost
                C_L = np.tile(np.concatenate([np.repeat(np.linspace(0.6,5,L),E).reshape(-1,1),np.zeros((L*E,1))],axis=1),O).reshape(O*L*E,U)
                C_L[0::L*E,:] = [0,0]
                C_L[1::L*E,:] = [0,0]
                C_L[L*E-2::L*E,:] = [1e10,0]
                C_L[L*E-1::L*E,:] = [1e10,0]

                mdp = MDP(L,P_O,fs,C_A,C_L,D,"lplagrange")
                

                # Plot the policy
        # count number of successful rounds and simulate
        n_communications = 45 
        count_learning_queries = 0
        print(mdp.policy)
        policy = mdp.policy
        for i in tqdm(range(n_communications)):
                oracle_state = markovchain.oracle_states[i]
                
                action_prob = mdp.policy[int(oracle_state*L + E*state_learning_queries)]
                
                action = np.random.choice([0,1],p=[action_prob,1-action_prob])

                if state_learning_queries == 0:
                        action = 0
                if action == 1:
                        count_learning_queries += 1
                        if successful_round[i] == 1:
                                state_learning_queries -= 1
                if count_learning_queries/(i+1) > 0.5:
                        print(f"Not Private {i}",count_learning_queries,action,state_learning_queries)
                else:
                        print(f"Private {i}",count_learning_queries,action,state_learning_queries)
                        if i == n_communications - 1:
                                num_private_end += 1
        plt.figure(figsize=(10,10))
        plt.plot(np.arange(policy.shape[0]//2),policy[::2],label='Obfuscate')
        plt.savefig('./data/plots/policy.png')
        plt.close()
print(f"Number of private rounds at the end {num_private_end}")