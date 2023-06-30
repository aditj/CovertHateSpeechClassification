### This file contains functions for solving MDPs and SPSA algorithm
### import packages     
from ortools.linear_solver import pywraplp
import numpy as np
#### Functions for solving MDPs using Linear Programming
def solvelp(C_A,C_L,P,X,U,D,alpha = 1):
    solver = pywraplp.Solver.CreateSolver('GLOP')
    infinity = solver.infinity()
    pi = {}
    for i in range(X):
        for u in range(U): 
            pi[i,u] =  solver.NumVar(0,infinity, 'pi[%i][%i]'%(i,u))
    constraint = solver.RowConstraint(0, D, '')
    for i in range(X): 
        for u in range(U): 
            constraint.SetCoefficient(pi[i,u],C_L[i][u])
    objective = solver.Objective()

    for i in range(X): 
        for u in range(U):  
            objective.SetCoefficient(pi[i,u], C_A[i][u])
    if alpha == 1:
        prob_constraint = [pi[i,u] for i in range(X) for u in range(U)]
        solver.Add(sum(prob_constraint) == 1)
    O = 3
    E = 2
    L = X//O//E
    for j in range(X):
        transition_constraint_left = [P[u][i][j]*pi[i,u] for u in range(U) for i in range(X)]
        # for i in range(X):

            # if i%(L*E)==0 or (i-1)%(L*E)==0:
            #     transition_constraint_left += [pi[i,0]*(int(j==i)-alpha*P[0][i][j]) ]
            # else:
            #     for u in range(U):
            #         transition_constraint_left += [pi[i,u]*(int(j==i)-alpha*P[u][i][j]) ]

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
    print(solver.Objective().Value())
    print(solver.ComputeConstraintActivities()[0])
    return probpolicy
def solvelp_generalcost(C,P,X,U,alpha = 1):
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

def sigmoid(x,thres=0,scale=1,tau=1):
    return scale/(1 + np.exp(-(x-thres)/tau))

def policy_from_sigmoid2d(parameters,L,O,E,A,tau):
    policy = np.zeros((O*L*E,1),dtype = float)
    n_thresholds = int(parameters.shape[0]-1)//O//E
    assert n_thresholds==2
    q = np.sin(parameters[-1])**2
    a=0
    for o in range(O):
        for e in range(E):
            paras = parameters[o*n_thresholds+e*n_thresholds:(o+1)*n_thresholds+e*n_thresholds]
            thresholds = np.array(paras).reshape(n_thresholds)
            for l in range(L):
                policyvalue = sigmoid(l,thresholds[0],q,tau) + sigmoid(l,thresholds[1],1-q,tau) 
                policy[o*L*E + l*E + e] = policyvalue
        
    return policy

def averagecost_randompolicy(cost,n_iter,P,policy,X_0 = 0):
  X = X_0
  A = P.shape[0]
  O = P.shape[2]
  averagecost = np.zeros(n_iter)
  
  for i in range(n_iter):
    u = policy[X]
    q_k = u[0]
    a_i = int(np.random.choice([0,1],p=[1-q_k,q_k]))
    averagecost[i] = cost[X][a_i]
    if np.round(np.sum(P[a_i,X]),3)!=1:
       print(P[a_i,X].sum())
    P[a_i,X] = P[a_i,X]/np.sum(P[a_i,X])
    X = np.random.choice(np.arange(O),p = P[a_i,X])
  return averagecost.mean()

### SPSA algorithm for solving MDPs
def spsa(initial_parameters,delta,n_iter,T,P,D,lamb,epsilon,rho,L,O,E,A,C_A,C_L,tau=0.3):
  m = initial_parameters.shape[0]
  
  parameters = initial_parameters.copy().reshape(m)
  #policies = np.zeros((n_iter,L*O,2))
  parameters_store = np.zeros((n_iter,m))
  for i in range(n_iter):
    np.random.seed(i)
    pertub = np.random.binomial(1,0.5,(m))
    parameters_plus = parameters + pertub*delta[i]
    parameters_minus = parameters - pertub*delta[i]
    assert parameters_plus.shape == parameters.shape
    policy = policy_from_sigmoid2d(parameters,L,O,E,A,tau)
    policy_plus = policy_from_sigmoid2d(parameters_plus,L,O,E,A,tau,)
    policy_minus = policy_from_sigmoid2d(parameters_minus,L,O,E,A,tau)
    C_A_plus = averagecost_randompolicy(C_A,T,P,policy_plus)
    C_A_minus = averagecost_randompolicy(C_A,T,P,policy_minus)
    C_L_plus = averagecost_randompolicy(C_L,T,P,policy_plus)
    C_L_minus = averagecost_randompolicy(C_L,T,P,policy_minus)
    C_L_avg = averagecost_randompolicy(C_L,T,P,policy)
    C_A_avg = averagecost_randompolicy(C_A,T,P,policy)
   # print(C_L_plus,C_L_minus)
    del_C_A = np.zeros(pertub.shape[0])
    del_C_L = np.zeros(pertub.shape[0])

    for j in range(pertub.shape[0]):
      if pertub[j]!=0:
        del_C_A[j] = (C_A_plus - C_A_minus)/(pertub[j]*delta[i])
        del_C_L[j] = (C_L_plus - C_L_minus)/(pertub[j]*delta[i])
    assert del_C_A.shape == parameters.shape
    parameters = parameters - epsilon[i]*(del_C_A + del_C_L*np.max([0, rho*(C_L_avg-D) ]))
    lamb = np.max([(1-epsilon[i]/rho)*lamb, lamb + epsilon[i]*(C_L_avg - D)])
    if i%10==0:
        print(i,C_A_avg)
    parameters_store[i] = parameters
    tau = 0.9999*tau
  print(C_A_avg,C_L_avg-D,lamb,parameters)
  return parameters_store

