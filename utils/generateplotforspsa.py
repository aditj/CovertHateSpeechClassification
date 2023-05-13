## Generate plot for spsa convergence
## Import libraries
import numpy as np
import matplotlib.pyplot as plt
from solvemdp import spsa
P_O = np.array([[0.8,0.2,0],
       [0.3,0.2,0.5],
       [0,0.2,0.8]])
O = 3
L = 20
U = 2
A = U
X = O*L


D = 0.5

#### Probability transition kernel 
### Oracle
P_O = np.array([[0.8,0.2,0],
       [0.3,0.2,0.5],
       [0,0.2,0.8]])
### Complete Probability Transition Kernel
P = np.zeros((A, O*L, O*L))
### Probability of Failure for different states and action pairs
fs = np.array([[0,0.2],[0,0.75],[0,0.95]])

### 
for a in range(A):
  for o in range(O):
    f = fs[o,a]
    delta = 0.25 # Arrival rate 
    P_L = (1-delta)*(f)*np.roll(np.identity(L),-1,axis=1) +  ((delta)*(f)+(1-delta)*(1-f))*np.roll(np.identity(L),0,axis=1) + delta*(1-f)*np.roll(np.identity(L),1,axis=1)# Probability transition for learner state
    P_L[0,-1] = 0
    P_L[-1,0] = 0
    # Check for learner state
    if ((P_L>0).sum(axis = 1)>3).sum()>0:
      print("more learner transitions than required")
    P_L = P_L/P_L.sum(axis=1).reshape(L,1) # normalization

    P[a,L*o:(L)*(o+1),:] = np.tile(P_L,(1,O))*np.repeat(P_O[o,:],L,axis=0) # setting the learner transition matrix for o row and a action
    # Check for learner state combined with oracle transition
    if ((P[a,L*o:(L)*(o+1),:]>0).sum(axis =1 )>O*3).sum()>0:
      print("more learner transitions than required")

  P[:,L-1::L,0::L] = 0 # setting the transition from L-1 to 0 to 0 
  P[a,:,:] = P[a,:,:]/P[a,:,:].sum(axis = 1).reshape(O*L,1) ## Normalization 
  P[a,:,:] = np.around(P[a,:,:],7) # Rounding

### Dimension Check
if fs.shape != (O,A):
  print("please correct dimension of fs")
if P_O.shape != (O,O):
  print("please correct dimension of P_O")
### Stochastic Check 
if (np.around(P.sum(axis = 2),5) != 1).sum() > 0:
  print("P is not stochastic")

#### Cost
## Advesarial cost
C_A = [[0,1.6],
     [0,0.7],
     [0,0.2]]
C_A = np.tile(C_A,L).reshape(O*L,A) # tiling adversarial cost 
## Learner Cost
C_L =  np.tile([1.0,0],O*L).reshape(O*L,A)
#    [0,1.55,1.75,3],
#     [0,1.75,2.25,3],
#     [0,2,3,3.25]
# ]
C_L[0::L,:] = [0.5,0]

parameter_initial_values = np.arange(4,12)


for p_1 in parameter_initial_values:
  
    D = 0.6
    parameters_initial = np.append(np.tile([p_1,12],O),np.pi/2)
    n_iter = 6000
    #delt = np.append(np.repeat(np.pi/2,O*(A)),np.pi/4) #np.power(0.99,(np.arange(1,n_iter+1)))+0.001
    delt = np.linspace(0.5,0.4,n_iter)
    T = 1000
    lamb = 1
    epsilon = np.linspace(0.4,0,n_iter)
    rho = 2
    parameters_spsa = spsa(parameters_initial,delt,n_iter,T,P,D,lamb,epsilon,rho,L,O,A,C_A,C_L)
    np.save("parameters_spsa_"+str(p_1),parameters_spsa)

## Plot these parameters  for different initial values and oracle states
n_iter_sample = 6000
for p_1 in parameter_initial_values:
    parameters_spsa = np.load("parameters_spsa_"+str(p_1)+".npy")
    plt.plot(np.arange(n_iter_sample),parameters_spsa[:n_iter_sample,0],label="$y^O = O_1$")
# plt.plot(np.arange(n_iter),parameters_spsa[:,])
    plt.plot(np.arange(n_iter_sample),parameters_spsa[:n_iter_sample,2],label="$y^O = O_2$")
    plt.plot(np.arange(n_iter_sample),parameters_spsa[:n_iter_sample,4],label="$y^O = O_3$")
plt.legend(fontsize=12)
plt.legend(fontsize=12)
plt.xlabel("Iterations",size = 16)
plt.ylabel("Threshold Parameter of $y^L$", size = 16)
plt.xticks(size=16)
plt.yticks(size=16)
plt.savefig("fig2.pdf",bbox_inches='tight',pad_inches=0)
plt.show()