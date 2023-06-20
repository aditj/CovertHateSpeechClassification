import numpy as np
import matplotlib.pyplot as plt
from ortools.linear_solver import pywraplp
from solvemdp import spsa
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
L = 20
U = 2
A = U
X = O*L
E = 2
P_O = np.array([[0.85,0.15,0],
       [0.15,0.7,0.15],
       [0.0,0.15,0.85]])

### Complete Probability Transition Kernel
P = np.zeros((A, O*L, O*L))
### Probability of Failure for different states and action pairs
fs = np.array([[0,0.3],[0,0.75],[0,0.95]])

### MAKE P have transitions from either -1 or M-1 or M
M= 10
delta = 0.18# Arrival rate 

# Create threedimensional probability transition matrix for each action with proper tests
P = np.zeros((A,O*L*E,O*L*E))
for a in range(A):
  for o in range(O):
      for l in range(L):
        for e in range(E):
          if l == 0:
            p_success = 0
          else:
            p_success = fs[o,a]
          for o_prime in range(O):
              p_o_o_prime = P_O[o,o_prime]
              for l_prime in range(L):
                l_transition_success = l_prime == l + e - 1
                l_transition_failure = l_prime == l + e 
                if l==L-1: 
                    l_transition_success = l_prime == l - 1
                    l_transition_failure = l_prime == l
                for e_prime in range(E):
                    p_e_prime = e_prime*delta + (1-e_prime)*(1-delta)
                    if l == L-1 or l==L-2:
                        p_e_prime = 1 - e_prime
                    P[a,o*L*E+l*E+e,o_prime*L*E+l_prime*E+e_prime] = p_o_o_prime*(l_transition_success*p_success + l_transition_failure*(1-p_success))*p_e_prime 
                  
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
    plt.savefig("P_{}.png".format(a))

#### Cost
## Advesarial cost
C_A = [[0,1.8],
     [0,0.8],
     [0,0.3]]
C_A = np.tile(C_A,L*E).reshape(O*L*E,A) # tiling adversarial cost 
## Learner Cost
C_L = np.tile(np.concatenate([np.repeat(np.linspace(0.6,10,L),E).reshape(-1,1),np.zeros((L*E,1))],axis=1),[O,1])
# C_L = [[0.8,0],
#     [0.7,0],
#     [0.6,0]
# ]
# C_L = np.tile(C_L,L*E).reshape(O*L*E,A) # tiling learner cost
# C_L = np.tile(C_L,L).reshape(O*L,A) # tiling learner cost
C_L[0::L*E,:] = [0,0]
C_L[1::L*E,:] = [0,0]
# C_L[L*E-2::L*E,:] = [1,0]
# C_L[L*E-1::L*E,:] = [1,0]

D = 0.4
print(delta*M/fs[0,1]<(1 - (D/C_L[2,0])))

print(delta*M/fs[0,1],(1 - (D/C_L[2,0])))
C_lamb = lambda lambd: C_A + (lambd)*C_L
def occupation_measure(probpolicy,C,O,L,E,U):
    c = 0
    for i in range(O*L*E):
        for u in range(U):
            c += C[i,u]*probpolicy[i,u]
    return c
running_c = 0
lambds = np.linspace(0.2,0.4,10)
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
fig, axs = plt.subplots(3, 1)
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
plt.savefig("policy.png")

parameter_initial_values = np.linspace(7,10,5)
n_iter = 4000
if True:
    for p_1 in parameter_initial_values:
    
        parameters_initial = np.append(np.tile([p_1,12],O*E),np.pi/2)
        delt = np.linspace(0.5,0.4,n_iter)
        T = 1000
        lamb = 1
        epsilon = np.linspace(0.4,0.2,n_iter)
        rho = 2
        parameters_spsa = spsa(parameters_initial,delt,n_iter,T,P,D,lamb,epsilon,rho,L,O,E,A,C_A,C_L)
        np.save("./data/input/spsa/parameters_spsa_"+str(p_1),parameters_spsa)

## Plot these parameters  for different initial values and oracle states
if True:
  n_iter_sample = n_iter
  sns.set_style("darkgrid")
  sns.set_context("paper")
  for p_1 in parameter_initial_values:
      parameters_spsa = np.load("./data/input/spsa/parameters_spsa_"+str(p_1)+".npy")
      plt.plot(np.arange(n_iter_sample),parameters_spsa[:n_iter_sample,0],color = 'red')
  # plt.plot(np.arange(n_iter),parameters_spsa[:,])
      plt.plot(np.arange(n_iter_sample),parameters_spsa[:n_iter_sample,2],color = 'blue')
      plt.plot(np.arange(n_iter_sample),parameters_spsa[:n_iter_sample,4],color = 'green')
  # plt.legend(fontsize=12)
  #plt.legend(fontsize=12)
  plt.xlabel("Iterations",size = 16)
  plt.ylabel("Threshold Parameter of $y^L$", size = 16)
  plt.xticks(size=16)
  plt.yticks(size=16)
  plt.savefig("fig2.pdf",bbox_inches='tight',pad_inches=0)