## Generate plot for spsa convergence
## Import libraries
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from solvemdp import spsa,solvelp
# check if P_O is 1st order stochastic dominant
O = 3
L = 40
U = 2
A = U
X = O*L
E = 2

#### Probability transition kernel 
### Oracle
P_O = np.array([[0.85,0.15,0],
       [0.15,0.7,0.15],
       [0.0,0.15,0.85]])
for i in range(P_O.shape[0]):
    for j in range(i):
        if i!=j:
            assert P_O[i,i]>=P_O[j,i]
### Complete Probability Transition Kernel
P = np.zeros((A, O*L, O*L))
### Probability of Failure for different states and action pairs
fs = np.array([[0,0.3],[0,0.5],[0,0.9]])

### MAKE P have transitions from either -1 or M-1 or M
M = 5
delta = 0.2# Arrival rate 

# Create threedimensional probability transition matrix for each action with proper tests
P = np.zeros((A,O*L*E,O*L*E))
for a in range(A):
  for o in range(O):
      for l in range(L):
        for e in range(E):
          # last L 
          # if l == L-1 and e == E-1:
          #    continue
          
          if l == 0:
            p_success = 0
          else:
            p_success = fs[o,a]
          for o_prime in range(O):
              p_o_o_prime = P_O[o,o_prime]
              for l_prime in range(L):
                l_transition_success = l_prime == l + e - 1
                l_transition_failure = l_prime == l + e 
                for e_prime in range(E):
                    p_e_prime = e_prime*delta + (1-e_prime)*(1-delta)
                    if l == L-1:
                        p_e_prime = 1 - e_prime
                    P[a,o*L*E+l*E+e,o_prime*L*E+l_prime*E+e_prime] = p_o_o_prime*(l_transition_success*p_success + l_transition_failure*(1-p_success))*p_e_prime 
                    if l == L-1 and e == 1 and l_prime == L-1 and e_prime == 0:
                       P[a,o*L*E+l*E+e,o_prime*L*E+l_prime*E+e_prime] = p_o_o_prime

                    # if o*L*E+l*E+e == 7 and e_prime == 0 and e == 1 and a == 0 and o==0 and o_prime==0 and (l_transition_failure or l_transition_success):
                    #     print(l,e,l_prime,p_success,p_o_o_prime,l_transition_success,l_transition_failure,p_e_prime)
                    #     print(p_o_o_prime,(l_transition_success*p_success + l_transition_failure*(1-p_success)),p_e_prime)
                    #     print(P[a,o*L*E+l*E+e,o_prime*L*E+l_prime*E+e_prime])

# check if P is stochastic
if (np.around(P.sum(axis = 2),4) != 1).sum() > 0:
  print(np.where(P.sum(axis = 2)<0.99)[1])
  print(P.sum(axis = 2))
  print("P is not stochastic")

#### Cost
## Advesarial cost
C_A = [[0,1.0],
     [0,1.0],
     [0,1.0]]
C_A = np.tile(C_A,L*E).reshape(O*L*E,A) # tiling adversarial cost 
## Learner Cost

C_L = np.tile(np.concatenate([np.repeat(np.linspace(0.6,0.6,L),E).reshape(-1,1),np.zeros((L*E,1))],axis=1),O).reshape(O*L*E,A)
print
# C_L = [[0.8,0],
#     [0.7,0],
#     [0.6,0]
# ]
# C_L = np.tile(C_L,L*E).reshape(O*L*E,A) # tiling learner cost
# C_L = np.tile(C_L,L).reshape(O*L,A) # tiling learner cost
C_L[0::L*E,:] = [0,0]
C_L[1::L*E,:] = [0,0]
C_L[L*E-2::L*E,:] = [1e10,0]
C_L[L*E-1::L*E,:] = [1e10,0]

D = 0.3
constraint = (delta*M/fs[0,1])< (1 - (D/C_L[2,0]))
print(delta*M/fs[0,1],(1 - (D/C_L[2,0])),constraint)
probpolicy = solvelp(C_A,C_L,P,X = O*L*E,U = 2,D = D,alpha = 1)

def average_cost(probpolicy,C,P,T):
    cost = 0
    x = 0
    for t in range(T):
        cost += np.dot(C[x,:],probpolicy[x,:])
        a = np.random.choice([0,1],p = probpolicy[x,:])
        x = np.random.choice(np.arange(O*L*E),p = P[a,x,:])
    return cost/T

# when sum of row is non ze  ro, normalize

probpolicy = probpolicy/(np.sum(probpolicy,axis = 1).reshape(O*L*E,1))
random_policy = np.ones((O*L*E,A))/A

print(average_cost(probpolicy,C_L,P,1000))
print(average_cost(probpolicy,C_A,P,1000))

print(average_cost(random_policy,C_L,P,1000))
print(average_cost(random_policy,C_A,P,1000))

C_L = np.tile(np.concatenate([np.repeat(np.linspace(0,L,L),E).reshape(-1,1),np.zeros((L*E,1))],axis=1),O).reshape(O*L*E,A)
print(average_cost(probpolicy,C_L,P,1000))
print(probpolicy[L*E+10:L*E+40:2,1])
print(average_cost(random_policy,C_L,P,1000))


#print(probpolicy[:,1])
#probpolicy = np.round(probpolicy,1)
# plt.plot(np.arange(O*L),probpolicy[1::2,1],color = 'red')
print(probpolicy)
plt.clf()

plt.plot(np.arange(O*L),probpolicy[0::2,1],color = 'blue',label="Optimal Policy using LP")
plt.legend()
## Write text for oracle states 
for i in range(O):
    plt.text(i*L+L/2,1.1,"Oracle State"+str(i),fontsize = 15, horizontalalignment='center',verticalalignment='center')
plt.xticks(ticks = np.arange(O*L,step=10,), labels = np.tile(np.arange(L,step=10),O))
# Plot vertical lines for oracle states
for i in range(O+1):
  plt.axvline(x = i*L,color = 'black',linestyle = '--')
plt.xlabel("Learner State")
plt.ylabel("Probability of choosing action 1")
# white solid background
# plt.savefig("./data/plots/figure_policy_white.png",bbox_inches='tight')
# plt.savefig("./data/plots/figure_policy_white.png",bbox_inches='tight', transparent=True)
print("Here")
plt.savefig("./data/plots/figure_policy.png")

# clear plot

parameter_initial_values = np.linspace(7,7,1)
n_iter = 4000
if True:
    for p_1 in parameter_initial_values:
        parameters_initial = np.append(np.tile([p_1,12],O*E),np.pi/2)
        #delt = np.append(np.repeat(np.pi/2,O*(A)),np.pi/4) #np.power(0.99,(np.arange(1,n_iter+1)))+0.001
        delt = np.linspace(0.5,0.4,n_iter)
        T = 1000
        lamb = 1
        epsilon = np.linspace(0.4,0.2,n_iter)
        rho = 2
        parameters_spsa = spsa(parameters_initial,delt,n_iter,T,P,D,lamb,epsilon,rho,L,O,E,A,C_A,C_L,tau=0.9)
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
