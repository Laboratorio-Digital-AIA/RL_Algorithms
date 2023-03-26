
# .. Santiago Bobadilla Suarez
# .. Last edit: 18/04/2021

# ...................................................................... Libraries

import numpy as np                      # .. Mathematic operations
import pandas as pd                     # .. Data Frame visualization   
from gurobipy import *                  # .. Gurobi optimizer

# ........................................................................................
# ...................................................................... Policy Evaluation

# ... 𝑣(𝑠)←∑_𝑎 𝜋(𝑎|𝑠)∑_𝑠′,𝑟 𝑝 (𝑠′,𝑟|𝑠,𝑎)[𝑟+𝛾𝑣(𝑠′)]

# ............ PARAMETERS:

# .. S: Array with the states. Has to be NUMERIC!
# .. V: Inicial zero array where we compute the policy value for each state
# .. r: Pre-define rewards matrix for each state vs decision.
# .. cubeProbs: Cube of probabilities of changing to another state
# .. gamma: Discount factor
# .. theta: Limit rate of convergence
# .. pi: Array that state for each state the decision made. (The position is equivalent to the state)

# ............ OUTPUT:  

# .. v: Array of the final value of the policy in each state

def evaluate_policy(S, V, r, cubeProbs, gamma, theta, pi):
    delta = float('inf')                                                            # .. Start with a big gap 
    while delta > theta:                                                            # .. Do continue until it converge
        delta = 0                                                                   # .. For each iteration delta is reset
        for s in S:                                                                 # .. Go for each state one by one
            V_prim = V.copy()                                                       # .. Made a copy of the present value (NEW MEMORY DIRECTION)
            V[s] = r[s, pi[s]] + (gamma*np.dot(cubeProbs[s,:, pi[s]], V_prim))      # .. Do Bellman Update
            delta = max(delta, abs(V_prim[s] - V[s]))                               # .. Select the biggest error, and update delta
            
    return V                                                                        # .. When it converge we return the value array


# .......................................................................................
# ...................................................................... Policy Iteration

# ............ PARAMETERS:                                                                                  ... Step used: 

# .. S: Array with the states. Has to be NUMERIC!                                                           ... 'evaluate_policy' and 'policyImprovement'
# .. V: Inicial zero array where we compute the policy value for each state                                 ... 'evaluate_policy' and 'policyImprovement'
# .. r: Pre-define rewards matrix for each state vs decision.                                               ... 'evaluate_policy' and 'policyImprovement'
# .. cubeProbs: Cube of probabilities of changing to another state                                          ... 'evaluate_policy' and 'policyImprovement'
# .. gamma: Discount factor                                                                                 ... 'evaluate_policy' and 'policyImprovement'
# .. theta: Limit rate of convergence                                                                       ... Only in 'evaluate_policy'
# .. pi: Array that state for each state the decision made. (The position is equivalent to the state)       ... 'evaluate_policy' and 'policyImprovement'

# ............ OUTPUT:  

# .. v: Array of the final value of the policy in each state
# .. pi: Array with the optimal decisions for each state.

def policy_iteration(S, V, r, cubeProbs, gamma, theta, pi):
    policyStable = False                                                            # .. Define that the optimal policy have not happen
    while not policyStable:                                                         # .. Continue until it reachs the optimal policy
        V = evaluate_policy(S, V, r, cubeProbs, gamma, theta, pi)                   # .. Evaluate the given policy
        pi, policyStable = policy_improvement(S, V, r, cubeProbs, gamma, pi)         # .. See if we can improve the value of the policy

    return pi, V                                                                    # .. Return the optimal values and decisions

def policy_improvement(S, V, r, cubeProbs, gamma, pi):
    policyStable = True                                                             # .. Define that the policy can´t be improved
    for s in S:                                                                     # .. For each state we check the optimality
        old = pi[s]                                                                 # .. Save the pre-define decision gave by the evaluated policy
        new = np.argmax( r[s,:] + gamma*np.dot(V, cubeProbs[s,:,:]) )               # .. Check the best decision for state 's'
        if old != new:                                                              
            policyStable = False                                                    # .. Cause there's a better decision, we state that this is not the optimal policy
            pi[s] = new                                                             # .. Update the new decision 
    
    return pi, policyStable                                                         # .. Return the new decision to evaluated and state that we are not in the optimal


# .......................................................................................
# ...................................................................... Value Iteration

# .. 𝑣(𝑠)←max𝑎∑𝑠′,𝑟𝑝(𝑠′,𝑟|𝑠,𝑎)[𝑟+𝛾𝑣(𝑠′)]

# ............ PARAMETERS:

# .. S: Array with the states. Has to be NUMERIC!
# .. V: Inicial zero array where we compute the policy value for each state
# .. r: Pre-define rewards matrix for each state vs decision.
# .. cubeProbs: Cube of probabilities of changing to another state
# .. gamma: Discount factor
# .. theta: Limit rate of convergence

# ............ OUTPUT:  

# .. v: Array of the final value of the policy in each state
# .. pi: Array with the optimal decisions for each state.

def value_iteration(S, V, r, cubeProbs, gamma, theta):
    pi = [-1 for s in S]                                                            # .. Define array to state the optimal decisions (None when we start)
    delta = float('inf')                                                            # .. Start with a big gap 
    while delta > theta:                                                            # .. Do continue until it converge
        delta = 0                                                                   # .. For each iteration delta is reset
        for s in S:                                                                 # .. Go for each state one by one
            V_prim = V.copy()                                                       # .. Made a copy of the present value (NEW MEMORY DIRECTION)
            aux = r[s, :] + (gamma*np.dot(V_prim, cubeProbs[s,:,:]))                # .. Do Bellman Update
            V[s] = np.max(aux)                                                      # .. Select biggest value as the local optimal of the state
            pi[s] = np.argmax(aux)                                                  # .. Select the decision relate to the biggest value as the local optimal 
            delta = max(delta, abs(V_prim[s] - V[s]))                               # .. Select the biggest error, and update delta
            
    return pi, V                                                                    # .. When it converge we return the value array, and the respective actions


# ..........................................................................................................
# ...................................................................... Lineal Algebra -- Policy Evaluation 

# .. V𝜋' = (I - 𝛾P𝜋')-1R

# ............ PARAMETERS:

# .. S: Array with the states. Has to be numeric!
# .. r: Pre-define rewards for each state thanks to the decision made.
# .. gamma: Discount factor
# .. probs: Probabilities of changing to another state taking in consideration the pre-define policy

# ............ OUTPUT:  

# .. Array of the final value of the policy in each state

def lineal_algebra_PE(S, r, gamma, probs):
    I = np.identity(len(S))                                             # .. Create of a Identity matrix 
    return np.matmul (np.linalg.inv(I - gamma*probs), r)                # .. Soluction of the linear ecuations


# ..............................................................................................................
# ...................................................................... Primal Optimal Problem Resolution for m

# ............ PARAMETERS:

# .. S: Array with the states. Has to be STRING!
# .. D: Array with decisions. Has to be NUMERIC!
# .. gamma: Discount factor
# .. dicReward: Dictionary with the rewards of every state acording to the possible actions
# .. dicProbsDes: Dictionary with the Probabilities of changing to another state taking in consideration the action select

# ............ OUTPUT:  

# .. A guriby model ready to optimize

def primal_MDP(S, D, dicReward, dicProbsDes, gamma):

    m = Model('MDP Santiago')                                                                  # .. We created a model
    variables = {v : m.addVar(vtype = GRB.CONTINUOUS, lb = -1e8, name = v) for v in S}         # .. Add decision variables (one 'per' state)

    for v in S:                                                                                # .. Add Lineal Constrains
        for d in D:                                                                            # .. ej: VB ≥4 + (0.05·VE+ 0.7·VB+ 0.15·VP+ 0.1·VR)𝛾
            m.addConstr(variables[v] >= dicReward[v,d] + quicksum(variables[i]*dicProbsDes[d,v,i] for i in S)*gamma, name = v + str(d))
        
    m.setObjective(quicksum(variables[v] for v in S), GRB.MINIMIZE)                            # .. Set FO and objective
    m.setParam("OutputFlag", 0)                                                                # .. Hide Model Stats
    m.update()                                                                                 # .. Update object model

    return m                                                                                   # .. Return model                                                     



