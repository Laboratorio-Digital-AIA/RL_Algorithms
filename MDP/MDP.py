
# .. Santiago Bobadilla Suarez
# .. Last edit: 18/04/2021

# .................................................
# ....................................... Libraries

import numpy as np                      # .. Mathematic operations
import pandas as pd                     # .. Data Frame visualization   
from gurobipy import *                  # .. Gurobi optimizer
import algthmSBob as algthm             # .. Personal Santiago Bobadilla Algorithms implementations

# ...........................................................
# ....................................... GENERAL MDP PROBLEM

print("------------------------------------------------")
print("-------------- GENERAL PROBLEM -----------------")
print("------------------------------------------------")

# ....................................... States

# ..... There are four type of students:
# .. Excelent (E)
# .. Good (G)
# .. Average (A)
# .. Bad (B)

states = ["E", "G", "A", "B"]           

# ....................................... Decisions

# .. 0. Take class
# .. 1. Do homework
# .. 2. Read text book 

decisions = [0, 1, 2]

# ....................................... Reward in each state for each action
# ..... NO REWARD: 
# .. Excelent-0, and Bad-2      -->      Penalized

rewards = np.array(([-10000, 3, 0], [4,2,-10], [3, -5, -10], [-3, -10, -10000]))       # .. Matrix creation
seeRewards = pd.DataFrame(rewards, index = states, columns = decisions)                # .. DataFrame creation
print("Rewards: \n", seeRewards, "\n----------")                                       # .. Visualization

# ..................................... Matrix Probabilities por each decision

# .. Decision 0
# . There is no probability for i = E to the j. 
probD0 = np.array(([0, 0, 0, 0], [0.05, 0.7, 0.15, 0.1], [0, 0.2, 0.5, 0.3], [0, 0, 0.1, 0.9]))                   # .. Matrix creation
seeProbD0 = pd.DataFrame(probD0, index = states, columns = states)                                                # .. DataFrame creation
print("Decision 0 probabilities: \n", seeProbD0, "\n----------")                                                  # .. Visualization

# .. Decision 1
probD1 = np.array(([0.6, 0.4, 0, 0], [0.25, 0.6, 0.1, 0.05], [0.1, 0.3, 0.5, 0.1], [0, 0.05, 0.25, 0.7]))         # .. Matrix creation
seeProbD1 = pd.DataFrame(probD1, index = states, columns = states)                                                # .. DataFrame creation
print("Decision 1 probabilities: \n", seeProbD1, "\n----------")                                                  # .. Visualization

# .. Decision 2
# . There is no probability for i = B to the j. 
probD2 = np.array(([0.9, 0.1, 0, 0], [0.5, 0.5, 0, 0], [0.2, 0.3, 0.5, 0], [0, 0, 0, 0]))                         # .. Matrix creation
seeProbD2 = pd.DataFrame(probD2, index = states, columns = states)                                                # .. DataFrame creation
print("Decision 2 probabilities: \n", seeProbD2, "\n")                                                            # .. Visualization

# ..................................... Cube of probabilities matrix

cubeProbs = np.zeros((len(states), len(states), len(decisions)))                        # .. Cube where the layers are the respective matrix

cubeProbs[:,:,0] = probD0                                                               # .. Matrix: Decision 0 (First Layer)
cubeProbs[:,:,1] = probD1                                                               # .. Matrix: Decision 1 (Second Layer)
cubeProbs[:,:,2] = probD2                                                               # .. Matrix: Decision 2 (Third Layer)

# .............................................................................................................
# ............................................... ALGOTHIMS ...................................................
# .............................................................................................................

print("................................................")
print("................... ALGORITHMS .................")
print("................................................\n")

# ........... GENERAL PARAMETERS


S = range(len(states))                  # .. S: Array of states (numeric)
A = decisions                           # .. A: Array of actions/decisions
V = np.zeros(len(states))               # .. V: Initial array for value compute
pi = [-1 for s in S]                    # .. Pi: Array that define the action for every state. (Starts with none decision made.)
r = rewards                             # .. r: Reward for each state
theta = 1e-8                            # .. theta: Limit rate of convergence
gamma = 0.9                             # .. gamma: Discount factor

# .................................................................. POLICY EVALUATION - RL
# .........................................................................................

print("------------ POLICY EVALUATION .. RL")
print("------------------------------------\n")

# ....................................... DEFINE POLICY

# .. If theta student is 'Excelent (E)' he will take decision '1'.
# .. If theta student is 'Good (G)' he will take decision '0'.
# .. If theta student is 'Average (A)' he will take decision '2'.
# .. If theta student is 'Bad (E)' he will take decision '2'.

# .. WARNING: Rember not all students can take make all decisions. 

# ....... Arbitrary define of decision for each State

pi[0] = 1                   # .. pi[state] = decision
pi[1] = 0
pi[2] = 2
pi[3] = 1

# ....................................... RL - POLICY EVALUATION
V = algthm.evaluate_policy(S, V, r, cubeProbs, gamma, theta, pi)

# ....................................... See results
seeV = pd.DataFrame(V, index = states, columns = ["Value"])                        # .. DataFrame creation
print("Value for the define policy: \n", seeV, "\n")                               # .. Visualization

# .................................................................. POLICY EVALUATION - LINEAL ALGEBRA
# .....................................................................................................

print("------------ LINEAR ALGEBRA ... POLICY EVALUATION")
print("-------------------------------------------------\n")

# .. WARNING: We are using the same policy.

# .. probs: Probabilities of changing to another state taking in consideration theta pre-define policy
# .. r_prime: Immediate return for each state with the respective decision made.

probs = np.zeros((len(states), len(states)))            # .. Matriz creation

probs[0,:] = probD1[0,:]                                # .. probs[row,:] = matrix_decision[state,:]
probs[1,:] = probD0[1,:]
probs[2,:] = probD2[2,:]
probs[3,:] = probD1[3,:]

r_prime = [3, 4, -10, -10]

# ....................................... LINEAL ALGEBRA - POLICY EVALUATION
la_V = algthm.lineal_algebra_PE(S, r_prime, gamma, probs)

# ....................................... See results
seela_V = pd.DataFrame(la_V, index = states, columns = ["Value"])                  # .. DataFrame creation
print("Value for the define policy: \n", seela_V, "\n")                            # .. Visualization


# .................................................................. POLICY ITERATION - RL
# .........................................................................................

print("------------ POLICY ITERATION .. RL")
print("-----------------------------------\n")

# ....................................... RL - POLICY ITERATION
pi_sub_prime, V_prime = algthm.policy_iteration(S, V, r, cubeProbs, gamma, theta, pi)

# ....................................... See results
seePI_V = pd.DataFrame(V_prime, index = states, columns = ["Value"])                  # .. DataFrame creation
print("Value for the policy iteration: \n", seePI_V, "\n")                            # .. Visualization

seeDes = pd.DataFrame(pi_sub_prime, index = states, columns = ["Decisions"])      # .. DataFrame creation
print("Decisions in the optimal policy: \n", seeDes, "\n")                            # .. Visualization

print("\nLa función objetivo es: ")
print(np.sum(V_prime))                                                                # .. Visualization

# .................................................................. VALUE EVALUATION - RL
# .........................................................................................

print("\n------------ VALUE ITERATION .. RL")
print("----------------------------------\n")

# ....................................... RL - VALUE ITERATION
pi_sub_prime_two, V_prime_two = algthm.value_iteration(S, V, r, cubeProbs, gamma, theta)

# ....................................... See results
seePI_V = pd.DataFrame(V_prime_two, index = states, columns = ["Value"])              # .. DataFrame creation
print("Value for the policy iteration: \n", seePI_V, "\n")                            # .. Visualization

seeDes_two = pd.DataFrame(pi_sub_prime_two, index = states, columns = ["Decisions"])      # .. DataFrame creation
print("Decisions in the optimal policy: \n", seeDes_two, "\n")                            # .. Visualization
                                                         # .. Visualization
print("-----------------------------------------------------------\n")


# .................................................................. GENERAL RESOLUTION - PRIMAL OPTIMAL PROBLEM
# ..............................................................................................................

# .............. DICTIONARIES

# .. Rewards
dicRewards = {(e,d) : rewards[states.index(e), decisions.index(d)] for d in decisions for e in states}

# .. Probabilities
dicProbsDes = {}

for e in states:
    for d in states:
        dicProbsDes[0,e,d] = probD0[states.index(e), states.index(d)]
        dicProbsDes[1,e,d] = probD1[states.index(e), states.index(d)]
        dicProbsDes[2,e,d] = probD2[states.index(e), states.index(d)]        

# ....................................... OPTIMIZER - PRIMAL PROBLEM
m = algthm.primal_MDP(states, decisions, dicRewards, dicProbsDes, gamma)
m.optimize()

print("\n------------ PRIMAL OPTIMAL PROBLEM .. GENERAL RESOLUTION")
print("-----------------------------------------------------------\n")

# .. Print Results if there is optimal solution
if m.status == GRB.OPTIMAL:
    
    print('Objective function:', m.objVal)                      # .. Objective Function
    print('-------------------------------------')

    # .. Value of the decision variables
    print("The respective value are: ")
    for v in m.getVars():                                       # .. Value of the variables
        print(v.varName, round(v.x, 3))

    print("\nLas respecitivas acciones son: ")
    result ={}
    for s in states:
        for a in decisions:
            if m.getConstrByName(s+str(a)).pi != 0:
                result[s] = a

    print(result)
    print('-------------------------------------')

