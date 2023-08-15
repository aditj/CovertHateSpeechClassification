# import mdp from parent directory

from mdp import MDP,MarkovChain
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
num_private_end = 0
end_query_count = np.zeros(10)
def average_cost(probpolicy,C,P,T):
        cost = 0
        x = 0
        for t in range(T):
                a = np.random.choice([0,1],p = probpolicy[x,:])
                cost += C[x,a]*probpolicy[x,a]
                x = np.random.choice(np.arange(O*L*E),p = P[a,x,:])
        return cost/T
from utils.randomseed import seed_everything
for k in range(10):
        seed_everything(k)
        N_device = 100
        P = np.array([
       [0.7, 0.2, 0.05, 0.03,  0.02, 0.0, 0],
       [0.4, 0.42, 0.05, 0.03,  0.05, 0.05, 0],
       [0.2, 0.3, 0.37, 0.02,  0.04, 0.04, 0.03],
       [0.025, 0.1, 0.2, 0.35,  0.2, 0.1, 0.025],
       [0.03, 0.04, 0.04, 0.02, 0.37, 0.3, 0.2],
       [0, 0.02, 0.03, 0.03, 0.1, 0.42, 0.4],
       [0, 0.0, 0.02, 0.03, 0.05, 0.2, 0.7]
    ]) 
        print(P.sum(axis=1))
        # length 7 n_choice
        N_choices = np.array([N_device//2.75,N_device//2.4,N_device//2.2,N_device//2,N_device//1.8,N_device//1.7,N_device//1],dtype=int)
        markovchain = MarkovChain(N_device=N_device,P = P, N_choices = N_choices,thresfactor = 8,N_total = 15000,T=10000)
        markovchain.generate_device_data_matrix()
        generate_policy = True
        if generate_policy:
                        state_learning_queries = 30
                        L = state_learning_queries + 1
                        O = markovchain.P.shape[0]
                        U = 2
                        E = 2
                        D = 0.2
                        P_O = markovchain.P
                        fs = markovchain.success_prob
                        ## Advesarial cost
                        C_A = [[
                                [0,1.8],
                                [0,1.4],
                                [0,1.1],
                                [0,0.8],
                                [0,0.5],
                                [0,0.3],
                                [0,0.1],
                        ]]
                        C_A = np.tile(C_A,L*E).reshape(O*L*E,U) # tiling adversarial cost 
                        ## Learner Cost
                        C_L = np.tile(np.concatenate([np.repeat(np.linspace(0.6,20,L),E).reshape(-1,1),np.zeros((L*E,1))],axis=1),O).reshape(O*L*E,U)
                        C_L[0::L*E,:] = [0,0]
                        C_L[1::L*E,:] = [0,0]
                        # C_L[L*E-2::L*E,:] = [1e10,0]
                        # C_L[L*E-1::L*E,:] = [1e10,0]

                        mdp = MDP(L,P_O,fs,C_A,C_L,D,"lplagrange")      
        successful_round = markovchain.successful_round
        device_data_matrix = markovchain.device_data_matrix
                # Plot the policy
        # count number of successful rounds and simulate
        n_communications = 120 
        count_learning_queries = 0
        print(mdp.policy)
        policy = mdp.policy
        num_good_oracle_states = 0
        for i in tqdm(range(n_communications)):
                oracle_state = markovchain.oracle_states[i]
                

                action_prob = mdp.policy[int(oracle_state*L + E*state_learning_queries)]
                
                action = np.random.choice([0,1],p=[action_prob,1-action_prob])

                if state_learning_queries == 0:
                        action = 0
                if action == 1:
                        count_learning_queries += 1
                        if successful_round[i] == 1:
                                if oracle_state >= 3:
                                        num_good_oracle_states += 1
                                state_learning_queries -= 1
                if count_learning_queries/(i+1) > 0.5:
                        print(f"Not Private {i}",count_learning_queries,action,state_learning_queries)
                else:
                        print(f"Private {i}",count_learning_queries,action,state_learning_queries)
                        if i == n_communications - 1:
                                num_private_end += 1
                                end_query_count[k] = state_learning_queries
        plt.figure(figsize=(10,10))
        plt.plot(np.arange(policy.shape[0]//2),policy[::2],label='Obfuscate')
        plt.savefig('./data/plots/policy.png')
        plt.close()
        print(k,num_private_end,end_query_count[:k].mean(),num_good_oracle_states)
