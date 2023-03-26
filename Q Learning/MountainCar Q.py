
# ... Santiago Bobadilla Suarez
# ... Last Modify: 26/04/2021

# ................... Q ~ LEARNING
# ................................

# ...This implementation is based on the tutorial of: https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/ 
# .....................................................................................................................................

import gym                           # ... Environment of problem's: https://gym.openai.com/docs/
import numpy as np
import matplotlib.pyplot as plt
import os

# ... The objective of this script is to explain the implementation of Q ~ Learning 
# ... algorithm in the problem of a car trying to climb a mountain.

# ... I. Observation of general statistics to understand better the problem.
# ..........................................................................

print("------------------------\n ------- MOUNTAIN CAR V0 \n ------------------------ \n")

env = gym.make("MountainCar-v0")         # ... Specific problem to aboard: https://gym.openai.com/envs/MountainCar-v0/
env.reset()                              # ... Start form cero the instance of the environment

print("..... General Statistics of the Problem:")

print("  The size of our frame (possible states) is: ", env.observation_space)     # ... Understand the possible states that the problem has.
print("  The action size is: ", env.action_space.n)                                # ... Understand the possible actions that the problem has.

print("\n ..... General Statistics of the States:")

print("  The maximun state is: ", env.observation_space.high)                      # ... Obtain the minimum of the states
print("  The minimun state is: ", env.observation_space.low)                       # ... Obtain the maximum of the states

# ... II. Creation of the Q ~ Table
# .................................

# ... 1. Define the dimensions of the Q ~ Table and their step size

print("\n ..... Descrete Representation of the 'Continues' Space of the Problem:")

DESCRETE_OS_SIZE = [40] * len(env.observation_space.high)                                                   # ... Definitions of the dimensions (hyper parameter)
DESCRETE_OS_WIN_SIZE  = (env.observation_space.high - env.observation_space.low) / DESCRETE_OS_SIZE         # ... Definition of the step base on the size of the Q ~ Table and the range of the steps

print("   Dimensions of the Q ~ Table: ", DESCRETE_OS_SIZE )
print("   Length of the step: ", DESCRETE_OS_WIN_SIZE)

# ... 2. Generate the Q ~ Table

print("\n ..... Q ~ Table:")

q_table = np.random.uniform(low = -2, high = 0, size = ( DESCRETE_OS_SIZE  + [env.action_space.n] ))     # ... Initialize the Q ~ Table with Random Uniform values for each action in each state

print("   Shape: ", q_table.shape)

# ... III. Define Parameters 
# ..........................

LEARNING_RATE = 0.1                                                                     # ... Anything between 0 to 1. Learn faster when it approaches to 1. 
DISCOUNT = 0.95                                                                         # ... It's a meassure of how important is for us furute actions over current actions.
EPISODES = 25000                                                                        # ... Amount of replicas the RL will have to learn.
SHOW_EVERY = 5000                                                                       # ... Let us know you are still alive.

epsilon = 0.5                                                                           # ... Between 0 to 1. Chance to explore. 

START_EPSILON_DECAY = 1                                                                 # ... Episode where the epsilon will start to have effect
END_EPSILON_DECAY = EPISODES // 2                                                       # ... Episode where the epsilon will stop having effect

epsilon_decay_value = epsilon / (END_EPSILON_DECAY - START_EPSILON_DECAY)               # ... Rate of decay of the epsilon after each step

eps_reward = []                                                                         # ... Array to save the historical record of the rewards of the episodes
aggr_ep_reward = { 'eps' : [], 'avg' : [], 'min' : [], 'max' : [] }                     # ... Dictionary with relevant statistic information

# ... IV. Helper function to fit values to the Q ~ Table
# ......................................................

def get_descrete_state(state):                                                          # ... Receive an state (x,y -> for this problem) 
    discrete_state = (state - env.observation_space.low) / DESCRETE_OS_WIN_SIZE         # ... Transform it to a valid place in the table
    return tuple(discrete_state.astype(np.int))                                         # ... Return the valid state as the form (x,y -> for this problem)

# ... V. Run the Q ~ Learning Algorithm  
# .....................................

print("\n ..... Q ~ Learning:")

absolute_file_path = os.path.abspath(__file__)[:-16]                                                        # ... Obtain where the file is locate as absolute path taking of the name of this script
separador = absolute_file_path[-1]                                                                          # ... Obtain the '\' to replace it with '/' in the path 

for episode in range(EPISODES):                                                                             # ... Run all episodes that we define

   # if episode % 10 == 0:                                                                                   # ... When the module with 10 is 0 we are going to save the Q ~ Table
       # path = absolute_file_path.replace(separador, '/') + "qtables/" + str(episode) + "-qtable.npy"       # ... Define where we are going to save it. Will be save as .npy
       # np.save(path, q_table)                                                                              # ... Save the plot

    episode_reward = 0                                                                                      # ... Restar to 0 a aux variable to take track of the reward of the episode

    if episode % 50 == 0:                                                                           # ... Define if in this episode we are showing how the RL is learning
        render = True                                                                                       # ... Yes, show !
    else:                                                                                                   #
        render = False                                                                                      # ... No, don't show ! 

    discrete_state = get_descrete_state(env.reset())                                                        # ... When we start each episode it start in the initial position
    done = False                                                                                            # ... Iterate until the environment is done.

    while not done:                                                                                         # ... Start the interaction with the environment.

        if np.random.random() > epsilon:                                                                    # ... Define the exploration-exploitation paradigm 
            action = np.argmax(q_table[discrete_state])                                                     # ... Select the optimal action
        else:                                                                                               #
            action = np.random.randint(0, env.action_space.n)                                               # ... Explore an aleatory action 

        
        new_state, reward, done, _ = env.step(action)                                                       # ... Execute the action in the environment and get the new information
        episode_reward += reward                                                                            # ... Update the aux variable with the new reward information
        new_discrete_state = get_descrete_state(new_state)                                                  # ... Look up the relate state of environment to our Q ~ Table

        if render:                                                                                          # ... Show the progress if TRUE
            env.render()                                                                                    # ... Show the environment !

        if not done:                                                                                        # ... If we are not done, we update the Q ~ Table
            # ... Formula in: https://wikimedia.org/api/rest_v1/media/math/render/svg/678cb558a9d59c33ef4810c9618baf34a9577686
            new_q = (1-LEARNING_RATE)* q_table[discrete_state + (action,)] + LEARNING_RATE * (reward + (np.max(q_table[new_discrete_state]) * DISCOUNT))
            q_table[discrete_state + (action,)] = new_q

        elif new_state[0] >= env.goal_position:                                                             # ... If we reach the goal, we made it
            #print(f"We made it to the flag on episode {episode}")
            q_table[discrete_state + (action,)] = 0                                                         # ... That state in the Q ~ Table is award by not being penalysed 

        discrete_state = new_discrete_state                                                                 # ... Update the action to continue the iteration

    if END_EPSILON_DECAY >= episode >= START_EPSILON_DECAY:                                                 # ... When that episode is done iterating, observe if the epsilon must be decay
        epsilon -= epsilon_decay_value                                                                      # ... Decay epsilon

    eps_reward.append(episode_reward)                                                                       # ... Save the reward of the state

    if not episode % SHOW_EVERY:                                                                            # ... Register statistic that are meaning full to the problem
        average_reward = sum(eps_reward[-SHOW_EVERY:])/len(eps_reward[-SHOW_EVERY:])                        # ... Obtain the average reward up to that point

        aggr_ep_reward['eps'].append(episode)                                                               # ... Save the episode
        aggr_ep_reward['avg'].append(average_reward)                                                        # ... Save the average
        aggr_ep_reward['min'].append(min(eps_reward[-SHOW_EVERY:]))                                         # ... Save the minimum 
        aggr_ep_reward['max'].append(max(eps_reward[-SHOW_EVERY:]))                                         # ... Save the maximum 
                                                                                                            # ... Print the stats 
        print(f"Episode: {episode} Avg: {average_reward} Min: {min(eps_reward[-SHOW_EVERY:])} Max: {max(eps_reward[-SHOW_EVERY:])}" )

env.close()                                                                                                 # ... Close the environment to avoid problems

plt.plot(aggr_ep_reward['eps'], aggr_ep_reward['avg'], label = "Avg")       # ... Plot the stats of the average
plt.plot(aggr_ep_reward['eps'], aggr_ep_reward['min'], label = "Min")       # ... Plot the stats of the average
plt.plot(aggr_ep_reward['eps'], aggr_ep_reward['max'], label = "Max")       # ... Plot the stats of the average
plt.legend(loc=4)                                                           # ... Put the legends in the buttom right position
plt.show()                                                                  # ... Show

