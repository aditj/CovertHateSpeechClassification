import numpy as np
import matplotlib.pyplot as plt
from ortools.linear_solver import pywraplp
from solvemdp import spsa
import seaborn as sns
import tqdm as tqdm
import sys
# get these from arguments
args = sys.argv
do_lagrange_method, do_spsa_method, plot_spsa = int(args[1]), int(args[2]), int(args[3])
print(do_lagrange_method, do_spsa_method, plot_spsa)
PLOT_DIR = './data/plots/'

def createP(O,L,E,P_O,fs,M,delta):
    P = np.zeros((A,O*L*E,O*L*E))
    for a in range(A):
        for o in range(O):
            for l in range(L):
                for e in range(E):
                    if l == 0: # if the learner state is 0 then the learner can only go to 1
                        p_success = 0
                    else:
                        p_success = fs[o,a]
                    for o_prime in range(O):
                        p_o_o_prime = P_O[o,o_prime] # probability of transition to o_prime from o
                        for l_prime in range(L):
                            l_transition_success = l_prime == l + e*M - 1
                            l_transition_failure = l_prime == l + e*M 
                            if l>=L-M: 
                                l_transition_success = l_prime == min(l+M-1,L-2)
                                l_transition_failure = l_prime == min(l+M,L-1)
                            for e_prime in range(E):
                                p_e_prime = e_prime*delta + (1-e_prime)*(1-delta)
                                if l >= L-M:
                                    p_e_prime = 1 - e_prime
                                P[a,o*L*E+l*E+e,o_prime*L*E+l_prime*E+e_prime] = p_o_o_prime*(l_transition_success*p_success + l_transition_failure*(1-p_success))*p_e_prime 
    return P

def solvelp(C,P,X,U,alpha = 1):
    solver = pywraplp.Solver.CreateSolver('GLOP')
    infinity = solver.infinity()
    pi = {}
    for i in range(X):
        for u in range(U): 
            pi[i,u] =  solver.NumVar(0,infinity, 'pi[%i][%i]'%(i,u))
    objective = solver.Objective()
    for i in range(X): 
        for u in range(U):  
            objective.SetCoefficient(pi[i,u], C[i][u])
    if alpha == 1:
        prob_constraint = [pi[i,u] for i in range(X) for u in range(U)]
        solver.Add(sum(prob_constraint) == 1)
    O = 3
    E = 2
    L = X//O//E
    for j in range(X):
        transition_constraint_left = [P[u][i][j]*pi[i,u] for u in range(U) for i in range(X)]
        
        transition_constraint_right = [pi[j,u] for u in range(U) ] #(1-alpha)*1/X# [P[u][i][j]*pi[i,u] for i in range(X) for u in range(U)]
        solver.Add(sum(transition_constraint_left) == sum(transition_constraint_right))

    objective.SetMinimization()

    status = solver.Solve()
    
    probpolicy = np.zeros((X,U))
    if status == pywraplp.Solver.OPTIMAL:
        for i in range(X):
            for u in range(U):
                probpolicy[i,u] =  pi[i,u].solution_value()
    else:
        print('The problem does not have an optimal solution.')

    try:
        index = np.where(probpolicy[:,0]==1)[0][0]
        print(index)
        O = 3
        E = 2
        L = X//O//E
        print(solver.ComputeConstraintActivities())
    except:
        pass
    
    return probpolicy
def average_cost(probpolicy,C,P,T):
    cost = 0
    x = 0
    for t in range(T):
        a = np.random.choice([0,1],p = probpolicy[x,:])
        cost += C[x,a]*probpolicy[x,a]
        x = np.random.choice(np.arange(O*L*E),p = P[a,x,:])
    return cost/T


O = 3
L = 60
U = 2
A = U
X = O*L
E = 2
P_O = np.array([[0.7,0.2,0.1],
       [0.1,0.7,0.2],
       [0.1,0.2,0.7]])

### Complete Probability Transition Kernel
P = np.zeros((A, O*L, O*L))
### Probability of Success for different states and action pairs
fs = np.array([[0,0.1],[0,0.6],[0,0.9]])
fs_change = np.array([[0,0.1],[0,0.6],[0,0.9]])
### MAKE P have transitions from either -1 or M-1 or M


M= 10
delta = 0.03 # Arrival rate 
M_change = 4
delta_change = 0.1

#### Cost
## Advesarial cost
C_A = [[0,1.8],
     [0,0.8],
     [0,0.3]]
C_A = np.tile(C_A,L*E).reshape(O*L*E,A) # tiling adversarial cost 
## Learner Cost
C_L = np.tile(np.concatenate([np.repeat(np.linspace(0.6,10,L),E).reshape(-1,1),np.zeros((L*E,1))],axis=1),[O,1])
C_L[0::L*E,:] = [0,0]
C_L[1::L*E,:] = [0,0]
# C_L = [[0.6,0],
#     [0.6,0],
#     [0.6,0]
# ]
# C_L = np.tile(C_L,L*E).reshape(O*L*E,A) # tiling learner cost
# Each oracle section is L*E length, hence the last two states of each oracle section are absorbing states with high cost for not learning 
# the index of the last two states of each oracle section is L*E-2 and L*E-1 ( corresponding to l = L-1)
C_L[L*E-2::L*E,:] = [1e2,0]
C_L[L*E-1::L*E,:] = [1e2,0]
## Learning constraint
D = 0.4

# Create threedimensional probability transition matrix for each action with proper tests
P = createP(O,L,E,P_O,fs,M,delta)
P_change = createP(O,L,E,P_O,fs_change,M_change,delta_change)
# check if P is stochastic
if (np.around(P.sum(axis = 2),4) != 1).sum() > 0:
  print(np.where(P.sum(axis = 2)<0.99)[1])
  print(P.sum(axis = 2))
  print("P is not stochastic")

for j in range(P_O.shape[1]):
    for i in range(j+1,P_O.shape[1]):
        for l in range(P_O.shape[1]):
            assert P_O[i,l:].sum() >= P_O[j,l:].sum()
        
for a in range(A):
    plt.figure()
    plt.imshow(P[a,:,:])
    plt.colorbar()
    plt.title("P for action {}".format(a))
    plt.savefig(PLOT_DIR + "P_{}.png".format(a))
    plt.close()



## Check constraints
print(delta*M/fs[0,1]<(1 - (D/C_L[2,0])))

lambds = np.linspace(.1,.4,10)
if do_lagrange_method:
    C_lamb = lambda lambd: C_A + (lambd)*C_L
    def occupation_measure(probpolicy,C,O,L,E,U):
        c = 0
        for i in range(O*L*E):
            for u in range(U):
                c += C[i,u]*probpolicy[i,u]
        return c
    running_c = 0
    avgcost = np.zeros(len(lambds))

    for i,lamb in enumerate(lambds):
        probpolicy =solvelp(C_lamb(lamb),P,O*E*L,U)
        avgcost[i] = occupation_measure(probpolicy,C_L,O,L,E,U)
        print(lamb,avgcost[i])
        probpolicy = probpolicy/(np.sum(probpolicy,axis = 1).reshape(O*L*E,1))
        policy2 = probpolicy
        if avgcost[i] < D:
            break
        policy1 = probpolicy
    lamb1 = lambds[i-1]
    # policy1 = solvelp(C_A + lamb1*C_L,P,O*E*L,U)
    occupation1 = avgcost[i-1]
    # policy1 = policy1/(np.sum(policy1,axis = 1).reshape(O*L*E,1))

    policy1[L*E-1::L*E,:] = [0,1]
    lamb2 = lambds[i]
    # policy2 = solvelp(C_A + lamb2*C_L,P,O*E*L,U)
    policy2[L*E-1::L*E,:] = [0,1]
    occupation2 = avgcost[i]
    # policy2 = policy2/(np.sum(policy2,axis = 1).reshape(O*L*E,1))
    alpha = (D - occupation2)/(occupation1 - occupation2)
    print(alpha,occupation1,occupation2)
    policy = alpha*policy1 + (1-alpha)*policy2
    # make three subplots
    fig, axs = plt.subplots(3, 1,figsize=(20,10))
    axs[0].plot(np.arange(O*L),policy1[0::2,1])
    axs[1].plot(np.arange(O*L),policy2[0::2,1])
    axs[2].plot(np.arange(O*L),policy[0::2,1])
    axs[0].set_title('Policy 1')
    axs[1].set_title('Policy 2')
    axs[2].set_title('Policy')
    # add vlines on each plot at L*E intervals
    for o in range(O+1):
        axs[0].axvline(x=L*o,color='r')
        axs[1].axvline(x=L*o,color='r')
        axs[2].axvline(x=L*o,color='r')
    axs[0].set_xticks(np.tile(np.arange(L),O))
    # rotate the tick labels and set their alignment.
    plt.setp(axs[0].get_xticklabels(), rotation=90)
    plt.savefig(PLOT_DIR + "policy.png")
    plt.close()
n_iter = 4000
change_iter = 2000
if do_spsa_method:
    parameter_initial_values = [10,5]
    n_samples = 100
    parameters_spsa = np.zeros((n_samples,n_iter,2*O*E+1))
    delt = np.linspace(3,3,n_iter)#*np.pi/8
    delt[n_iter//3:2*n_iter//3]*=1
    delt[2*n_iter//3:]*=1
    # delt[8*n_iter//9:]*=0.9
    epsilon = np.linspace(1.5,1.5,n_iter)
    T = 200
    lamb = 10
    rho = 2e1
    parameters_initial = np.append(np.tile(parameter_initial_values,O*E),np.pi/4)
    for i in tqdm.tqdm(range(n_samples)):    
        np.random.seed(i)
        parameters_spsa[i] = spsa(parameters_initial,delt,n_iter,T,P,D,lamb,epsilon,rho,L,O,E,A,C_A,C_L,tau=0.5,P_change = P_change, change_iter = change_iter)
        print("Sample: ",i)
    np.save("./data/input/spsa/parameters_spsa",parameters_spsa)


if plot_spsa:
    n_iter_plot = n_iter
    parameters_spsa = np.load("./data/input/spsa/parameters_spsa.npy")
    # negative is set to zero
    parameters_spsa[parameters_spsa<0] = 0
    parameters_spsa[parameters_spsa>50 ] = 50
    parameters_spsa_o_1 = np.sort(parameters_spsa[:,:n_iter_plot,0:2],axis = 2)[:,:,1].mean(axis = 0)
    parameters_spsa_o_2 = np.sort(parameters_spsa[:,:n_iter_plot,4:6],axis = 2)[:,:,1].mean(axis = 0)
    parameters_spsa_o_3 = np.sort(parameters_spsa[:,:n_iter_plot,8:10],axis = 2)[:,:,1].mean(axis = 0)
    # Standard deviation
    parameters_spsa_o_1_std = np.sort(parameters_spsa[:,:n_iter_plot,0:2],axis = 2)[:,:,1].std(axis = 0)
    parameters_spsa_o_2_std = np.sort(parameters_spsa[:,:n_iter_plot,4:6],axis = 2)[:,:,1].std(axis = 0)
    parameters_spsa_o_3_std = np.sort(parameters_spsa[:,:n_iter_plot,8:10],axis = 2)[:,:,1].std(axis = 0)
    # plot the mean
    plt.figure(figsize=(7.5,5))
    sns.set_style("darkgrid")
    plt.plot(np.arange(n_iter_plot),parameters_spsa_o_1,label = r"$\theta_2 \ O = 1$",color="red")
    # plot shaded standard deviation area around the mean
    plt.fill_between(np.arange(n_iter_plot),parameters_spsa_o_1-parameters_spsa_o_1_std,parameters_spsa_o_1+parameters_spsa_o_1_std,alpha=0.3,color="red")

    plt.plot(np.arange(n_iter_plot),parameters_spsa_o_2,label = r"$\theta_2 \ O = 2$",color="black")
    plt.fill_between(np.arange(n_iter_plot),parameters_spsa_o_2-parameters_spsa_o_2_std,parameters_spsa_o_2+parameters_spsa_o_2_std,alpha=0.3,color="black")
    plt.plot(np.arange(n_iter_plot),parameters_spsa_o_3,label = r"$\theta_2 \ O = 3$",color="blue")
    plt.fill_between(np.arange(n_iter_plot),parameters_spsa_o_3-parameters_spsa_o_3_std,parameters_spsa_o_3+parameters_spsa_o_3_std,alpha=0.3,color="blue")
    plt.scatter(n_iter_plot,16,label="$\phi_2 \ O = 1$",color="red")
    plt.scatter(n_iter_plot//2,11,color="red")

    plt.scatter(n_iter_plot//2,3,label="$\phi_2 \ O = 2$",color="black")
    plt.scatter(n_iter_plot,2,color="black")
    plt.scatter(n_iter_plot//2,0,label="$\phi_2 \ O = 3$",color="blue")
    plt.scatter(n_iter_plot,0,color="blue")
    plt.vlines(x=n_iter_plot//2,color="black",ymin=0,ymax = 20,linestyle="--",alpha=0.5)
    plt.legend(fontsize=14,loc="upper left")
    
    plt.xlabel("Iterations",fontsize = 16)
    plt.ylabel("Parameters",fontsize = 16)
    # tight layout
    plt.tight_layout()
    plt.savefig("./data/plots/figspsa.pdf")

