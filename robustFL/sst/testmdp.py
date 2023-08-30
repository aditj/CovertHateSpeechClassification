# import mdp from parent directory

from mdp import MDP,MarkovChain
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
def average_cost(probpolicy,C,P,T):
        cost = 0
        x = 0
        for t in range(T):
                a = np.random.choice([0,1],p = probpolicy[x,:])
                cost += C[x,a]*probpolicy[x,a]
                x = np.random.choice(np.arange(O*L*E),p = P[a,x,:])
        return cost/T
from utils.randomseed import seed_everything

FILE_PATH = "./data/logs/MDP_TEST.txt"

N_device = 100
P = np.array([
[0.8, 0.1, 0.05, 0.03,  0.02],
[0.6, 0.2, 0.1, 0.03,  0.07],
[0.5, 0.2, 0.2, 0,  0.1],
[0.4,0.2,0.2,0.1,0.1],
[0.02,0.02,0.1,0.18,0.68 ]
]) 
# length 7 n_choice
N_choices = np.array([N_device//2.75,N_device//2.4,N_device//2.2,N_device//2,N_device//1.8,N_device//1.7,N_device//1],dtype=int)
N_choices = np.array([N_device//50,N_device//20,N_device//4,N_device//2,N_device//1],dtype=int)

thresfactor = 10
markovchain = MarkovChain(N_device=N_device,P = P, N_choices = N_choices,thresfactor = thresfactor,N_total = 60000,T=10000)

num_succ_updates = 7
state_learning_queries = num_succ_updates
L = state_learning_queries + 1
O = markovchain.P.shape[0]
U = 2
E = 2
D = 0.15
P_O = markovchain.P
## Advesarial cost
C_A = [[
        [0,1.8],
        [0,1.6],
        [0,0.7],
        [0,0.3],
        [0,0.1],
]]
C_A = np.tile(C_A,L*E).reshape(O*L*E,U) # tiling adversarial cost 
## Learner Cost
C_L = np.tile(np.concatenate([np.repeat(np.linspace(0.6,20,L),E).reshape(-1,1),np.zeros((L*E,1))],axis=1),O).reshape(O*L*E,U)
C_L[0::L*E,:] = [0,0]
C_L[1::L*E,:] = [0,0]
markovchain.generate_device_data_matrix()
successful_round = markovchain.successful_round
device_data_matrix = markovchain.device_data_matrix
fs = markovchain.success_prob
n_communications = 100
testMDP = True
if testMDP:
        for k in range(10):
                seed_everything(k)
                # C_L[L*E-2::L*E,:] = [1e10,0]
                # C_L[L*E-1::L*E,:] = [1e10,0]
                markovchain = MarkovChain(N_device=N_device,P = P, N_choices = N_choices,thresfactor = thresfactor,N_total = 60000,T=10000)
                markovchain.generate_device_data_matrix()
                successful_round = markovchain.successful_round
                device_data_matrix = markovchain.device_data_matrix
                fs = markovchain.success_prob
                mdp = MDP(L,P_O,fs,C_A,C_L,D,"lplagrange")      
                
                        # Plot the policy
                # count number of successful rounds and simulate
                state_learning_queries = num_succ_updates

                count_learning_queries = 0
                policy = mdp.policy
                num_good_oracle_states = 0
                total_learning_queries = 0 
                batch_size = np.zeros(state_learning_queries)
                j = 0
                for i in tqdm(range(n_communications)):
                        oracle_state = markovchain.oracle_states[i]
                        action_prob = mdp.policy[int(oracle_state*L + E*state_learning_queries)]
                        action = np.random.choice([0,1],p=[action_prob,1-action_prob])
                        if state_learning_queries == 0:
                                action = 0
                        total_learning_queries += action

                        if action == 1:
                                if successful_round[i] == 1:
                                        batch_size[j] = device_data_matrix[i].sum()
                                        j += 1
                                        if oracle_state >= 3:
                                                num_good_oracle_states += 1
                                        state_learning_queries -= 1
                # Plot the policy
                plt.figure(figsize=(10, 5))
                plt.plot(np.arange(L*O),1 - policy[::2])
                for i in range(O):
                        plt.vlines(i*L,0,1,linestyles='dashed')
                plt.xlabel("State")
                plt.ylabel("Probability of taking action 1")
                plt.savefig("./data/plots/MDP_policy.png")

                with open(FILE_PATH,"a") as f:
                        f.write(f"Optimal Policy {k} {num_good_oracle_states} {batch_size.mean()} {total_learning_queries} {state_learning_queries}\n")

# greedy policy
if testMDP:
        for k in range(10):
                seed_everything(k)
                # C_L[L*E-2::L*E,:] = [1e10,0]
                # C_L[L*E-1::L*E,:] = [1e10,0]
                markovchain = MarkovChain(N_device=N_device,P = P, N_choices = N_choices,thresfactor = thresfactor,N_total = 60000,T=10000)
                markovchain.generate_device_data_matrix()
                successful_round = markovchain.successful_round
                device_data_matrix = markovchain.device_data_matrix
                
                        # Plot the policy
                # count number of successful rounds and simulate
                state_learning_queries = num_succ_updates

                count_learning_queries = 0
                num_good_oracle_states = 0
                total_learning_queries = 0 
                batch_size = np.zeros(state_learning_queries)
                j = 0
                for i in tqdm(range(n_communications)):
                        oracle_state = markovchain.oracle_states[i]
                        action = 1
                        if state_learning_queries == 0:
                                action = 0
                        total_learning_queries += action
                        
                        if action == 1:
                                if successful_round[i] == 1:
                                        batch_size[j] = device_data_matrix[i].sum()
                                        j += 1
                                        if oracle_state >= 3:
                                                num_good_oracle_states += 1
                                        state_learning_queries -= 1
                # Plot the policy
                
                with open(FILE_PATH,"a") as f:
                        f.write(f"Greedy Policy {k} {num_good_oracle_states} {batch_size.mean()} {total_learning_queries} {state_learning_queries}\n")


# random policy
batch_size = np.zeros((100,num_succ_updates))

for k in range(10):
        seed_everything(k)
        markovchain = MarkovChain(N_device=N_device,P = P, N_choices = N_choices,thresfactor = thresfactor,N_total = 60000,T=10000)
        markovchain.generate_device_data_matrix()
        successful_round = markovchain.successful_round
        device_data_matrix = markovchain.device_data_matrix
        fs = markovchain.success_prob
        
        state_learning_queries = num_succ_updates
        count_learning_queries = 0
        action_sequence_indices = np.random.choice(np.arange(n_communications),size=state_learning_queries,replace=False)
        action_sequence = np.zeros(n_communications)
        action_sequence[action_sequence_indices] = 1        
        
        
        j = 0
        for i in range(n_communications):
                action = action_sequence[i]
                if state_learning_queries == 0:
                        action = 0
                if action == 1:
                        batch_size[k,j] = device_data_matrix[i].sum() 
                        j += 1
                        count_learning_queries += 1
                        state_learning_queries -= 1
        with open(FILE_PATH,"a") as f:
                f.write(f"Random Policy {k} {count_learning_queries} {batch_size[k,:].mean()}\n")
