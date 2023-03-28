#environment_q_learning.py
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
from gurobipy import *
import pandas as pd
import Models
import itertools
import copy
import json

style.use("ggplot")
print("QL\n")
def get_bounds(HPR):
    
    copy = HPR.copy()
    copy.relax()
    
    ub, lb = {}, {}

    for var in copy.getVars():
    	if var.VarName in lead_vars:
	    	if var.vtype == 'B':
	    		lb[var.VarName] = 0
	    		ub[var.VarName] = 1
	    	elif var.vtype == 'I':
		        copy.setObjective(var, GRB.MAXIMIZE)
		        copy.update()
		        copy.optimize()
		        if copy.Status == 2:
		            ub[var.VarName] = copy.ObjVal
		        else:
		            ub[var.VarName] = min(var.ub,1e8)

		        copy.setObjective(var, GRB.MINIMIZE)
		        copy.update()
		        copy.optimize()
		        
		        if copy.Status == 2:
		            lb[var.VarName] = copy.ObjVal
		        else:
		            lb[var.VarName] = max(var.lb,-1e8)
	    	else:
		    	lb[var.VarName] = max(var.lb,-1e8)
	    		ub[var.VarName] = min(var.Ub,1e8)
	    
    return {var.VarName:[int(lb[var.VarName]),int(ub[var.VarName])] for var in copy.getVars() if var.VarName in lead_vars}

def getVarTypes(m):
    INT = [v.VarName for v in m.getVars() if v.vtype == 'I' or v.vtype == 'B']
    CON = [v.VarName for v in m.getVars() if v.vtype == 'C']
    return INT,CON

class Agent:	
	def __init__(self, variables_dict, actions):

		if not bilevel_solutions:
			modelo = model.copy()
			for v in modelo.getVars():
			    v.setAttr('vtype', 'C')
			    modelo.update()

			variables = {var.VarName:(var,np.random.uniform(-ct[var.VarName],ct[var.VarName])) if var.VarName in lead_vars else (var,np.random.uniform(dt[var.VarName],ct[var.VarName])) for var in modelo.getVars()}
			#variables = {var.VarName:(var,np.random.uniform(-2,2)) for var in modelo.getVars()}
			modelo.setObjective(quicksum(variables[var][0]*variables[var][1] for var in variables.keys()), GRB.MINIMIZE)
			modelo.update()
			modelo.optimize()
			position_aux = {var.VarName:int(var.x) if var.VarName in INT else var.x for var in modelo.getVars()}
			self.at_position = {i:position_aux[i] for i in position_aux.keys() if i in lead_vars}
		elif np.random.random() > omega:
			modelo = model.copy()
			#starting_solution_index = np.random.randint(0,len(bilevel_solutions))
			if np.random.random() > delta:
				self.at_position = {var.VarName:p_bound[var.VarName] for var in modelo.getVars() if var.VarName in lead_vars}
			else:
				bilev_sol = np.random.randint(0, len(bilevel_solutions))
				self.at_position = {var.VarName: bilevel_solutions[bilev_sol][var.VarName] for var in modelo.getVars() if
									var.VarName in lead_vars}
		else:
			modelo = model.copy()
			for v in modelo.getVars():
			    v.setAttr('vtype', 'C')
			    modelo.update()

			variables = {var.VarName:(var,np.random.uniform(-ct[var.VarName],ct[var.VarName])) if var.VarName in lead_vars else (var,dt[var.VarName]) for var in modelo.getVars()}
			modelo.setObjective(quicksum(variables[var][0]*variables[var][1] for var in variables.keys()), GRB.MINIMIZE)
			modelo.update()
			modelo.optimize()
			position_aux = {var.VarName:int(var.x) if var.VarName in INT else var.x for var in modelo.getVars()}
			self.at_position = {i:position_aux[i] for i in position_aux.keys() if i in lead_vars}
		
		self.actions = actions
		self.num_actions = len(self.actions)

	def move(self, choice):
		#choice = {0: {x:0, y: 0, z: 0}, 1:{x:-1, y: 0, z: 0}, 2: {x:1, y: 0, z: 0}...}
		for movement in self.actions[choice].keys():
			self.at_position[movement] += self.actions[choice][movement] 
			if self.at_position[movement] < 0:
				self.at_position[movement] = 0
			elif self.at_position[movement] > bds[movement][1]:
				self.at_position[movement] = bds[movement][1]


#Integer instances: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, knapsack, interdiction

model = Models.HPR_toy_9()
follower = Models.follower_toy_9
ins = "9"
print('loaded')

copy_model = model.copy()
for v in copy_model.getVars():
    v.setAttr('vtype', 'C')
    copy_model.update()
print("model relaxed")

copy_model.optimize()

if copy_model.Status != 2:
	print("Model ubounded or infeasible")
	sys.exit()

F = follower({i.VarName:0 for i in model.getVars()})
fol_vars = [i.VarName for i in F.getVars()]
lead_vars = [i.VarName for i in model.getVars() if i.VarName not in fol_vars]
dt = {i.VarName:i.obj for i in F.getVars()}
ct = {i.VarName:i.obj for i in model.getVars()}

OF_start = copy_model.ObjVal

print(f'optimized with LB: {OF_start}')

bds = get_bounds(model)
print("bounds loaded")
INT, CON = getVarTypes(model)
print("variables domain retrieved")

HM_EPISODES = 1000
epsilon = 1
min_eps = 0.1
SHOW_EVERY = 10
EPS_DECAY = 0.999 
DISCOUNT = 0.9
omega = 0.5
delta = 0.5
LEARNING_RATE = 0.1
bilevel_solutions = []

penalty_hpr_infeas = -10e10
penalty_fol_infeas = -10e10
penalty_stay = -1e10

variables_dict = {}
cont = -1
for var in model.getVars():
	if var.VarName in INT and var.VarName in lead_vars:
		cont += 1
		variables_dict[cont] = var.VarName
actions = {}
iterator = 0
actions[iterator] = {variables_dict[k]:0 for k in variables_dict.keys()}
for i in variables_dict.keys():
	for j in [-1,1]:
		iterator += 1
		actions[iterator] = {variables_dict[k]:0 for k in variables_dict.keys()}
		actions[iterator][variables_dict[i]] = j
print("actions built")
steps = min(250,len(actions))
print(steps)
time.sleep(2)

episode_rewards = []
p_bound = {v.VarName: np.nan for v in model.getVars()}
p_bound["OF"] = 1e8
p_bound["F_OF"] = 1e8
optimal_solutions = [p_bound]
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}
q_table = {}
t_inicio = time.time()

step = 0

for episode in range(HM_EPISODES):

	step+=1

	position = Agent(variables_dict, actions)
	if episode % SHOW_EVERY == 0 and episode > 0:
		print(f"on # {episode}, epsilon: {epsilon}\n")
		moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode= "valid")

		font = {'family': 'serif',
		        'color':  'darkred',
		        'weight': 'normal',
		        'size': 16,
		        }

		plt.plot([i for i in range(len(moving_avg))], moving_avg)
		plt.ylabel(f"avg reward {SHOW_EVERY}") #ma")
		plt.xlabel(f"episode #")
		#plt.title(f'Instance: {ins}', fontdict=font, loc='center')
		plt.title(f'TEST', fontdict=font, loc='center')
		plt.grid(True)
		plt.savefig('TEST_performance.png')
		plt.close()
		with open('bilevel_sols_instance_'+str(ins)+'.txt', 'w') as convert_file:
			convert_file.write(json.dumps(bilevel_solutions))
		with open('optimal_solutions_instance_'+str(ins)+'.txt', 'w') as convert_file:
			convert_file.write(json.dumps(optimal_solutions))

	episode_reward = 0
	for i in range(min(step,steps)):
		obs = tuple([position.at_position[i] for i in position.at_position.keys() if i in INT])

		if obs not in q_table.keys():
			q_table[obs] = [-OF_start for i in range(position.num_actions)]

		if np.random.random() > epsilon:
			action = np.argmax(q_table[obs])
		else:
			action = np.random.randint(0,position.num_actions)

		position.move(action)

		new_obs = tuple([position.at_position[i] for i in position.at_position.keys() if i in INT])

		if new_obs not in q_table.keys():
			q_table[new_obs] = [-OF_start for i in range(position.num_actions)]

		#REWARDSTRUCTURE
		
		h = model.copy()
		current_solution = {}
		for v in h.getVars():
			v.setAttr('vtype', 'C')
			h.update()

		for v in h.getVars():
			if v.VarName in INT and v.VarName in lead_vars:
				h.addConstr(v == position.at_position[v.VarName])
		
		h.update()
		h.optimize()
		
		if h.Status != 2:
			reward = penalty_hpr_infeas
		else:			
			f = follower(position.at_position)
			f.update()
			f.optimize()
			
			bilev_feas = True
			if f.Status != 2:
				bilev_feas = False

			if not bilev_feas:
				reward = penalty_fol_infeas
			else:
				
				y_hat = {y.VarName:y.x for y in f.getVars()}
				h_2 = model.copy()
				
				for v in h_2.getVars():
					v.setAttr('vtype', 'C')
					h_2.update()

				for v in h_2.getVars():
					if v.VarName in lead_vars:
						h_2.addConstr(v == position.at_position[v.VarName])
					else:
						h_2.addConstr(v == y_hat[v.VarName])
				
				h_2.update()
				h_2.optimize()

				bilev_feas_2 = True
				if h_2.Status != 2:
					bilev_feas_2 = False

				if not bilev_feas_2:
					reward = penalty_hpr_infeas

				else:

					if new_obs == obs and h_2.ObjVal != p_bound["OF"] and p_bound["OF"] < 1e8:
						reward = penalty_stay
	
					else:

						HPR_OF = h_2.ObjVal 

						reward = -(HPR_OF)

						current_solution = {var.VarName: var.x for var in h_2.getVars()}
						current_solution["OF"] = HPR_OF
						current_solution["F_OF"] = f.ObjVal 
					
					if current_solution and current_solution not in bilevel_solutions:
						bilevel_solutions.append(current_solution)

					if h_2.ObjVal == optimal_solutions[0]["OF"] and current_solution not in optimal_solutions:
						optimal_solutions.append(current_solution)

					if HPR_OF < p_bound["OF"]:
						for var in current_solution.keys():
							p_bound[var] = current_solution[var]
						p_bound["OF"] = HPR_OF
						p_bound["F_OF"] = f.ObjVal
						optimal_solutions = [p_bound]
						print(f"Incumbent update on episode: {episode}")
						print(f"Best values and OF: {p_bound}\n")
					
					
		max_future_q = np.max(q_table[new_obs])
		current_q = q_table[obs][action]
		new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

		q_table[obs][action] = new_q

		episode_reward += reward

	episode_rewards.append(episode_reward)
	epsilon *= EPS_DECAY
	if epsilon < min_eps:
		EPS_DECAY = 1
		epsilon = 0.3

	if not episode % SHOW_EVERY:
		average_reward = sum(episode_rewards[-SHOW_EVERY:])/len(episode_rewards[-SHOW_EVERY:])
		aggr_ep_rewards['ep'].append(episode)
		aggr_ep_rewards['avg'].append(average_reward)
		aggr_ep_rewards['min'].append(min(episode_rewards[-SHOW_EVERY:]))
		aggr_ep_rewards['max'].append(max(episode_rewards[-SHOW_EVERY:]))
	
timecpu = str(time.time() - t_inicio)

print(f"time_run: {timecpu}\n")
#print(f"Best values and OF: {p_bound}\n")
print(f'Optimal solutions: {optimal_solutions}')
print(f'number of bilevel solutions: {len(bilevel_solutions)}')
#print(bilevel_solutions)
with open('bilevel_sols_instance_'+str(ins)+'.txt', 'w') as convert_file:
			convert_file.write(json.dumps(bilevel_solutions))
with open('optimal_solutions_instance_'+str(ins)+'.txt', 'w') as convert_file:
	convert_file.write(json.dumps(optimal_solutions))

