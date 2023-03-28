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
import winsound
import PPP_Env_V2_MARL
from math import *

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
	def __init__(self, variables_dict, actions, bds, update_bounds):

		#if not bilevel_solutions:
			
		modelo = model.copy()
		
		if np.random.random() < 0.5:
			variables = {var.VarName:(var,np.random.uniform(-ct[var.VarName],ct[var.VarName])) if var.VarName in lead_vars else (var,np.random.uniform(dt[var.VarName],ct[var.VarName])) for var in modelo.getVars()}
		else:
			variables = {var.VarName:(var,np.random.uniform(-10,11)) for var in modelo.getVars()}
		
		modelo.setObjective(quicksum(variables[var][0]*variables[var][1] for var in variables.keys()), GRB.MINIMIZE)
		modelo.update()
		modelo.addConstr(quicksum(v*original_objective[v.varName] for v in modelo.getVars()) <= incumbent)
		modelo.update()
		
		if update_bounds:
			print('Bounds update:\n')
			bds_actual = bds
			bds = get_bounds(modelo)
			if bds_actual == bds:
				print('No change in bounds')
			else:
				print('New bounds')
			update_bounds = False


		if np.random.random() < 0.5:		
			for v in modelo.getVars():
			    v.setAttr('vtype', 'C')
			    modelo.update()
		
		
		modelo.optimize()
		
		solution_vertex = {var.VarName:var.x for var in modelo.getVars()}
		#print('solution_vertex')
		#print(solution_vertex)
		if solution_vertex not in vertices:
			vertices.append(solution_vertex)

		
		if np.random.random() > omega and p_bound["OF"] < GRB.INFINITY and p_bound["F_OF"] < GRB.INFINITY:
			modelo = model.copy()
			self.at_position = {var.VarName:p_bound[var.VarName] for var in modelo.getVars() if var.VarName in lead_vars}
			#else:
			#	bilev_sol = np.random.randint(0, len(bilevel_solutions))
			#	self.at_position = {var.VarName: bilevel_solutions[bilev_sol][var.VarName] for var in modelo.getVars() if
			#						var.VarName in lead_vars}
		else:
			
			modelo = model.copy()
			
			number_of_lambdas = np.random.randint(1, len(vertices)+1)

			vertices_selected = []

			while len(vertices_selected) < number_of_lambdas:

				vertex_selected = np.random.randint(0, len(vertices))
				if vertex_selected not in vertices_selected:
					vertices_selected.append(vertex_selected) 

			lambdas = {'lambda_'+str(j):np.random.uniform(0,100) for j in vertices_selected}
			lambda_total = sum(lambdas[j] for j in lambdas.keys())
			weighted_lambdas = {j:lambdas[j]/lambda_total for j in lambdas.keys()}

			feasibility_check = {var:sum(weighted_lambdas['lambda_'+str(j)]*vertices[j][var] for j in vertices_selected) for var in original_variables.keys()}

			feasibility_check = {var:int(feasibility_check[var]) if var in lead_vars else feasibility_check[var] and var in INT for var in feasibility_check.keys()}

			position_aux = feasibility_check
			#print('position_aux')
			#print(position_aux)
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

class Instance():

    def get_level(self, perf):

            if perf < .2:
                return 1
            elif perf < .4:
                return 2
            elif perf < .6:
                return 3
            elif perf < .8:
                return 4
            else:
                return 5

    def incentive(self, perf, choice='sigmoid'):

        # Samuel has discretized the function according to the performance level
        if self.W[1] == 0:
            return 0

        if choice == 'sigmoid':
            rate, offset = 10, self.threshold
            incent = 1/(1 + exp(-rate*(perf-offset)))

        elif choice == 'linear':
            slope, offset = 1, 0
            incent = offset + slope*perf

        return incent

    def __init__(self):
        self.T = range(30) # Planning horizon
        self.W = [0, 1] # [shock?, inspection?]
        self.NUM_INTERVALS = 5
        # discrete levels of performance
        self.L = range(1, self.NUM_INTERVALS + 1)
        # Performance treshold (discrete)
        self.threshold = 0.6
        '''
        Necessary parmeters to model the deterioration
        '''
        self.fail = .2 # failure threshold (continuous)
        self.ttf = 10.0 # time to failure
        self.Lambda = -log(self.fail)/self.ttf
        self.gamma = [exp(-self.Lambda*tau) for tau in range(len(self.T)+2)]
        # perf_tt = exp(-Lambda*tt) --> Lambda = -ln(perf_tt)/tt

        '''Princpipal'''
        self.g = {1: 2, 2: 47, 3: 500, 4: 953, 5: 998}
        self.g_star = 595


        '''Agent'''
        self.FC = 3 # 4.4663 Fixed maintenance cost
        self.VC = self.FC # Variable maintenance cost

        '''delta = [exp(-I.Lambda*tau) - exp(-I.Lambda*(tau-1)) for tau in range(1,I.T)]
                    
                    gamma_2 = [1]
                    for tau in range(1,I.T):
                        gamma_2.append(gamma_2[tau-1]+delta[tau-1])
                
                    print(gamma_1==gamma_2)
                    aaaaa
                    '''

        self.cf = [self.FC for _ in range(len(self.T))]
        self.cv = [self.VC for _ in range(len(self.T))]

        self.xi_L = {1:0, 2:.21, 3:.41, 4:.61, 5:.81}
        self.xi_U = {1:.2, 2:.4, 3:.6, 4:.8, 5:1}   
        self.bond = {}
        self.c_sup_i = 70
        self.a = 1
        self.epsilon = 1.4
        for level in self.L:
            average_l = 0
            count_l = 0
            for gamma_val in self.gamma:
                if self.get_level(gamma_val) == level:
                    average_l += 7*self.incentive(gamma_val) 
                    count_l += 1
            self.bond[level] = average_l/count_l

I = Instance()# instance.PPP_Ins(INC=4, INS=3)

m = Models.HPR_PPP(I)
model = Models.HPR_PPP(I)
follower = Models.follower_PPP
ins = "PPP"
save = True
print('loaded')

HM_EPISODES = int(1e4)
epsilon = 1
min_eps = 0.1
SHOW_EVERY = 1
EPS_DECAY = 0.999 
DISCOUNT = 0.9
omega = 0.5
delta = 0.5
LEARNING_RATE = 0.1


copy_model = model.copy()
'''for v in copy_model.getVars():
    v.setAttr('vtype', 'C')
    copy_model.update()
print("model relaxed")
'''
copy_model.optimize()

if copy_model.Status != 2:
	print("Model ubounded or infeasible")
	sys.exit()

F = follower(I,{i.VarName:0 for i in model.getVars()})
original_variables = {var.VarName:var for var in model.getVars()} 
original_objective = {v.VarName:v.Obj for v in model.getVars()}
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
#time.sleep(2)

episode_rewards = []
p_bound = {v.VarName: np.nan for v in model.getVars()}
p_bound["OF"] = GRB.INFINITY
p_bound["F_OF"] = GRB.INFINITY
optimal_solutions = [p_bound]
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}
q_table = {}
t_inicio = time.time()
vertices = []
step = 0
incumbent = GRB.INFINITY
update_bounds = False

for episode in range(HM_EPISODES):

	step+=1

	position = Agent(variables_dict, actions, bds, update_bounds)
	#print('position')
	#print(position)
	update_bounds = False
	if episode % SHOW_EVERY == 0 and episode > 0:
		print(f"on # {episode}, epsilon: {epsilon}\n")
		timecpu = str(time.time() - t_inicio)
		print(f"time_run: {timecpu}\n")
		print(f'Optimal solutions: {optimal_solutions}')
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
		if save:
			plt.savefig('instance_'+str(ins)+'_TEST_performance.png')
			plt.close()
			
			with open('bilevel_sols_instance_'+str(ins)+'.txt', 'w') as convert_file:
				convert_file.write(json.dumps(bilevel_solutions))
			with open('optimal_solutions_instance_'+str(ins)+'.txt', 'w') as convert_file:
				convert_file.write(json.dumps(optimal_solutions))

	episode_reward = 0
	for i in range(min(step,steps)):

		obs = tuple([position.at_position[i] for i in position.at_position.keys() if i in INT])
		#print('obs')
		#print(obs)
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
		#print('position.at_position')
		#print(position.at_position)
		for v in h.getVars():
			if v.VarName in INT and v.VarName in lead_vars:
				h.addConstr(v == position.at_position[v.VarName])
		
		h.update()
		h.optimize()
		
		if h.Status != 2:
			reward = penalty_hpr_infeas
		else:			
			f = follower(I,position.at_position)
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
					if v.VarName in lead_vars:
						v.setAttr('vtype', 'C')
						h_2.update()

				for v in h_2.getVars():
					if v.VarName in lead_vars:
						h_2.addConstr(v == position.at_position[v.VarName])
					
					h_2.addConstr(quicksum(v*dt[v.varName] for v in h_2.getVars() if v.VarName in fol_vars) <= f.objVal)
				
				h_2.update()
				h_2.optimize()

				bilev_feas_2 = True
				if h_2.Status != 2:
					bilev_feas_2 = False

				if not bilev_feas_2:
					reward = penalty_hpr_infeas

				else:

					if new_obs == obs and h_2.ObjVal != p_bound["OF"] and p_bound["OF"] < GRB.INFINITY:
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
							#duration = 100# milliseconds
							#freq = 1000  # Hz
							#winsound.Beep(freq, duration)
							timecpu = str(time.time() - t_inicio)
							
							for var in current_solution.keys():
								p_bound[var] = current_solution[var]
							p_bound["OF"] = HPR_OF
							incumbent = HPR_OF
							vertices = []
							update_bounds = True
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
if save:
	with open('bilevel_sols_instance_'+str(ins)+'.txt', 'w') as convert_file:
		convert_file.write(json.dumps(bilevel_solutions))
	with open('optimal_solutions_instance_'+str(ins)+'.txt', 'w') as convert_file:
		convert_file.write(json.dumps(optimal_solutions))

duration = 100# milliseconds
freq = 1000  # Hz
winsound.Beep(freq, duration)