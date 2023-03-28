#DSA_project_proposal.py
from gurobipy import *
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
import os
from math import *
import random
import itertools
import copy
import json
import Models
import winsound
import time
from sklearn.linear_model import LinearRegression
from statistics import mean
import PPP_Env_V2_MARL

def bilev_feas_check(y_star, dt, y_hat):
	bilev_feas = False
	if sum(dt[v]*y_star[v] for v in y_star.keys()) <= sum(dt[v]*y_hat[v] for v in y_hat.keys()):
		bilev_feas = True
	return bilev_feas

def is_mipsol(INT, x_hat):    
    tolerance = 1e-4
    for i in INT:
        if abs(floor(x_hat[i])-x_hat[i])>tolerance:
            return False
    return True

def is_int_sol(INT, x_hat):    
    tolerance = 1e-4
    for i in x_hat.keys():
        if abs(floor(x_hat[i])-x_hat[i])>tolerance:
            return False
    return True

def getVarTypes(m):
    INT =[v.VarName for v in m.getVars() if v.vtype == 'I' or v.vtype == 'B']
    CONT =[v.VarName for v in m.getVars() if v.vtype == 'C']
    return INT,CONT

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

follower = Models.follower_PPP
print("Models loaded")
ins = 'PPP'
save = True
#save = False
save_gap_plot = True
n = 7
n_lead = 5
time_to_vertex = 7
time_to_representation = 7
time_to_look = 500
integer_epsilon = 0.5
update_every = 1
#thresh = 1

INT,CONT = getVarTypes(m)

copy_model = m.copy()
for v in copy_model.getVars():
    v.setAttr('vtype', 'C')
    copy_model.update()
print("HPR Model relaxed")

copy_model.optimize()

if copy_model.Status != 2:
	print("Model ubounded or infeasible")
	sys.exit()

ceiling_solution = copy_model.objVal

f = follower(I,{i.VarName:0 for i in m.getVars()})
fol_vars = [i.VarName for i in f.getVars()]
lead_vars = [i.VarName for i in m.getVars() if i.VarName not in fol_vars]
ct = {i.VarName:i.obj for i in m.getVars()}
dt = {i.VarName:i.obj for i in f.getVars()}
dt = {i.VarName:dt[i.VarName] if i.varName in fol_vars else 0 for i in m.getVars()}



best_of_sol = copy_model.objVal
worst_of_sol = m.copy()
worst_of_sol.setObjective(quicksum(worst_of_sol.getVarByName(var)*ct[var] for var in ct.keys()), GRB.MAXIMIZE)
worst_of_sol.update()
worst_of_sol.optimize()
worst_of_sol = sum(worst_of_sol.getVarByName(var).x*ct[var] for var in ct.keys())

fol_of_sol = m.copy()
fol_of_sol.setObjective(quicksum(fol_of_sol.getVarByName(var)*dt[var] for var in dt.keys()), GRB.MINIMIZE)
fol_of_sol.optimize()
fol_of_sol = sum(fol_of_sol.getVarByName(var).x*ct[var] for var in ct.keys())

of_radar_bounds = [best_of_sol, worst_of_sol]
of_radar_steps = sorted([of_radar_bounds[0]+((of_radar_bounds[1] - of_radar_bounds[0]) / n)*i for i in range(0,n+1)], reverse = True)

INT = [i for i in INT if i in lead_vars]

#bounds = get_bounds(m)
original_variables = {var.VarName:var for var in m.getVars()} 
original_objective = {v.VarName:v.Obj for v in m.getVars()} 
SolutionPool = []
vertices = []
best_value = [fol_of_sol]
incumbent = GRB.INFINITY
copy_model = m.copy()
copy_model.optimize()
incumbent_fol = -GRB.INFINITY
incumbent_sol = {}
current_best_lp_sol = {} 
not_bilev_sols = [copy_model.objVal]
done = False
replication = 0
temporal_vertices_list = []
predictors = []
response = []
coeff_list = [[] for var in range(len(original_variables.keys()))]

UB = [fol_of_sol]
LB = [fol_of_sol]
time_start = time.time()
Time = [time.time()-time_start]

full_cycle = False

while not done:
	for objective_epsilon in of_radar_steps:

		if objective_epsilon == of_radar_steps[-1]:
			full_cycle = True

		restart = False
		leader_vars_obj = m.copy()
		for v in leader_vars_obj.getVars():
		    v.setAttr('vtype', 'C')
		    leader_vars_obj.update()
		leader_vars_obj.addConstr(quicksum(v*original_objective[v.varName] for v in leader_vars_obj.getVars()) <= objective_epsilon)
		leader_vars_obj.update()
		leader_vars_obj.addConstr(quicksum(v*original_objective[v.varName] for v in leader_vars_obj.getVars()) >= objective_epsilon-((incumbent - of_radar_bounds[0]) / n))
		leader_vars_obj.update()
		leader_vars_obj.setObjective(0, GRB.MINIMIZE)
		leader_vars_obj.update()
		leader_vars_obj.optimize()
		if leader_vars_obj.status == 2:
			leader_vars_obj.setObjective(quicksum(leader_vars_obj.getVarByName(var)*ct[var] for var in ct.keys() if var in lead_vars), GRB.MINIMIZE)
			leader_vars_obj.update()
			leader_vars_obj.optimize()
			min_val_lead_vars = leader_vars_obj.objVal
			leader_vars_obj.setObjective(quicksum(leader_vars_obj.getVarByName(var)*ct[var] for var in ct.keys() if var in lead_vars), GRB.MAXIMIZE)
			leader_vars_obj.update()
			leader_vars_obj.optimize() 
			max_val_lead_vars = leader_vars_obj.objVal
			of_radar_bounds_lead = [min_val_lead_vars, max_val_lead_vars]
			of_radar_steps_lead = sorted([of_radar_bounds_lead[0]+((of_radar_bounds_lead[-1] - of_radar_bounds_lead[0]) / n_lead)*i for i in range(0,n_lead+1)], reverse = True)

			for objective_epsilon_lead in of_radar_steps_lead:
				vertices = [] 
				window = objective_epsilon-((incumbent - of_radar_bounds[0]) / n)
				window_lead = objective_epsilon_lead-((of_radar_bounds_lead[-1] - of_radar_bounds_lead[0]) / n_lead)
				
				#Maximize leader
				leader_max = m.copy()
				for v in leader_max.getVars():
					v.setAttr('vtype', 'C')
					leader_max.update()
				leader_max.setObjective(quicksum(ct[var]*leader_max.getVarByName(var) for var in ct.keys()), GRB.MAXIMIZE)
				leader_max.addConstr(quicksum(v*original_objective[v.varName] for v in leader_max.getVars()) <= objective_epsilon)
				leader_max.addConstr(quicksum(v*original_objective[v.varName] for v in leader_max.getVars()) >= objective_epsilon-((incumbent - of_radar_bounds[0]) / n))
				leader_max.addConstr(quicksum(v*original_objective[v.varName] for v in leader_max.getVars() if v.VarName in lead_vars) <= objective_epsilon_lead)
				leader_max.addConstr(quicksum(v*original_objective[v.varName] for v in leader_max.getVars() if v.VarName in lead_vars) >= objective_epsilon_lead-((of_radar_bounds_lead[-1] - of_radar_bounds_lead[0]) / n))
				leader_max.update()
				leader_max.optimize()
				
				if leader_max.status == 2:
					leader_max_sol = {var.varName:var.x for var in leader_max.getVars()}
					if leader_max_sol not in vertices:
						vertices.append(leader_max_sol)

				#Minimize leader
				leader_min = m.copy()
				for v in leader_min.getVars():
					v.setAttr('vtype', 'C')
					leader_min.update()
				leader_min.setObjective(quicksum(ct[var]*leader_min.getVarByName(var) for var in ct.keys()), GRB.MINIMIZE)
				leader_min.addConstr(quicksum(v*original_objective[v.varName] for v in leader_min.getVars()) <= objective_epsilon)
				leader_min.addConstr(quicksum(v*original_objective[v.varName] for v in leader_min.getVars()) >= objective_epsilon-((incumbent - of_radar_bounds[0]) / n))
				leader_min.addConstr(quicksum(v*original_objective[v.varName] for v in leader_min.getVars() if v.VarName in lead_vars) <= objective_epsilon_lead)
				leader_min.addConstr(quicksum(v*original_objective[v.varName] for v in leader_min.getVars() if v.VarName in lead_vars) >= objective_epsilon_lead-((of_radar_bounds_lead[-1] - of_radar_bounds_lead[0]) / n_lead))
				leader_min.update()
				leader_min.optimize()
				
				if leader_min.status == 2:
					leader_min_sol = {var.varName:var.x for var in leader_min.getVars()}
					if leader_min_sol not in vertices:
						vertices.append(leader_min_sol)

				#Maximize leader vars
				leader_max_vars = m.copy()
				for v in leader_max_vars.getVars():
					v.setAttr('vtype', 'C')
					leader_max_vars.update()
				leader_max_vars.setObjective(quicksum(ct[var]*leader_max_vars.getVarByName(var) for var in ct.keys() if var in lead_vars), GRB.MAXIMIZE)
				leader_max_vars.addConstr(quicksum(v*original_objective[v.varName] for v in leader_max_vars.getVars()) <= objective_epsilon)
				leader_max_vars.addConstr(quicksum(v*original_objective[v.varName] for v in leader_max_vars.getVars()) >= objective_epsilon-((incumbent - of_radar_bounds[0]) / n))
				leader_max_vars.addConstr(quicksum(v*original_objective[v.varName] for v in leader_max_vars.getVars() if v.VarName in lead_vars) <= objective_epsilon_lead)
				leader_max_vars.addConstr(quicksum(v*original_objective[v.varName] for v in leader_max_vars.getVars() if v.VarName in lead_vars) >= objective_epsilon_lead-((of_radar_bounds_lead[-1] - of_radar_bounds_lead[0]) / n_lead))
				leader_max_vars.update()
				leader_max_vars.optimize()
				
				if leader_max_vars.status == 2:
					leader_max_vars_sol = {var.varName:var.x for var in leader_max_vars.getVars()}
					if leader_max_vars_sol not in vertices:
						vertices.append(leader_max_vars_sol)

				#Minimize leader vars
				leader_min_vars = m.copy()
				for v in leader_min_vars.getVars():
					v.setAttr('vtype', 'C')
					leader_min_vars.update()
				leader_min_vars.setObjective(quicksum(ct[var]*leader_min_vars.getVarByName(var) for var in ct.keys() if var in lead_vars), GRB.MINIMIZE)
				leader_min_vars.addConstr(quicksum(v*original_objective[v.varName] for v in leader_min_vars.getVars()) <= objective_epsilon)
				leader_min_vars.addConstr(quicksum(v*original_objective[v.varName] for v in leader_min_vars.getVars()) >= objective_epsilon-((incumbent - of_radar_bounds[0]) / n))
				leader_min_vars.addConstr(quicksum(v*original_objective[v.varName] for v in leader_min_vars.getVars() if v.VarName in lead_vars) <= objective_epsilon_lead)
				leader_min_vars.addConstr(quicksum(v*original_objective[v.varName] for v in leader_min_vars.getVars() if v.VarName in lead_vars) >= objective_epsilon_lead-((of_radar_bounds_lead[-1] - of_radar_bounds_lead[0]) / n_lead))
				leader_min_vars.update()
				leader_min_vars.optimize()
				
				if leader_min_vars.status == 2:
					leader_min_vars_sol = {var.varName:var.x for var in leader_min_vars.getVars()}
					if leader_min_vars_sol not in vertices:
						vertices.append(leader_min_vars_sol)

				#Maximize follower
				follower_max = m.copy()
				for v in follower_max.getVars():
					v.setAttr('vtype', 'C')
					follower_max.update()
				follower_max.setObjective(quicksum(dt[var]*follower_max.getVarByName(var) for var in dt.keys()), GRB.MAXIMIZE)
				follower_max.addConstr(quicksum(v*original_objective[v.varName] for v in follower_max.getVars()) <= objective_epsilon)
				follower_max.addConstr(quicksum(v*original_objective[v.varName] for v in follower_max.getVars()) >= objective_epsilon-((incumbent - of_radar_bounds[0]) / n))
				follower_max.addConstr(quicksum(v*original_objective[v.varName] for v in follower_max.getVars() if v.VarName in lead_vars) <= objective_epsilon_lead)
				follower_max.addConstr(quicksum(v*original_objective[v.varName] for v in follower_max.getVars() if v.VarName in lead_vars) >= objective_epsilon_lead-((of_radar_bounds_lead[-1] - of_radar_bounds_lead[0]) / n_lead))
				follower_max.update()
				follower_max.optimize()
				
				if follower_max.status == 2:
					follower_max_sol = {var.varName:var.x for var in follower_max.getVars()}
					if follower_max_sol not in vertices:
						vertices.append(follower_max_sol)

				#Minimize follower
				follower_min = m.copy()
				for v in follower_min.getVars():
					v.setAttr('vtype', 'C')
					follower_min.update()
				follower_min.setObjective(quicksum(dt[var]*follower_min.getVarByName(var) for var in dt.keys()), GRB.MINIMIZE)
				follower_min.addConstr(quicksum(v*original_objective[v.varName] for v in follower_min.getVars()) <= objective_epsilon)
				follower_min.addConstr(quicksum(v*original_objective[v.varName] for v in follower_min.getVars()) >= objective_epsilon-((incumbent - of_radar_bounds[0]) / n))
				follower_min.addConstr(quicksum(v*original_objective[v.varName] for v in follower_min.getVars() if v.VarName in lead_vars) <= objective_epsilon_lead)
				follower_min.addConstr(quicksum(v*original_objective[v.varName] for v in follower_min.getVars() if v.VarName in lead_vars) >= objective_epsilon_lead-((of_radar_bounds_lead[-1] - of_radar_bounds_lead[0]) / n_lead))
				follower_min.update()
				follower_min.optimize()


				if follower_min.status == 2:
					follower_min_sol = {var.varName:var.x for var in follower_min.getVars()}
					if follower_min_sol not in vertices:
						vertices.append(follower_min_sol)

				#Combined
				combined_1 = m.copy()
				combined_2 = m.copy()
				combined_3 = m.copy()
				combined_4 = m.copy()
				for v in combined_1.getVars():
					v.setAttr('vtype', 'C')
					combined_1.update()
				for v in combined_2.getVars():
					v.setAttr('vtype', 'C')
					combined_2.update()
				for v in combined_3.getVars():
					v.setAttr('vtype', 'C')
					combined_3.update()
				for v in combined_4.getVars():
					v.setAttr('vtype', 'C')
					combined_4.update()
				
				combined_1.setObjective(quicksum(ct[var.varName]*var for var in combined_1.getVars() if var.VarName in lead_vars)+
					quicksum(dt[var.varName]*var for var in combined_1.getVars() if var.VarName in fol_vars), GRB.MINIMIZE)
				combined_2.setObjective(quicksum(-ct[var.varName]*var for var in combined_2.getVars() if var.VarName in lead_vars)+
					quicksum(-dt[var.varName]*var for var in combined_2.getVars() if var.VarName in fol_vars), GRB.MINIMIZE)
				combined_3.setObjective(quicksum(-ct[var.varName]*var for var in combined_3.getVars() if var.VarName in lead_vars)+
					quicksum(dt[var.varName]*var for var in combined_3.getVars() if var.VarName in fol_vars), GRB.MINIMIZE)
				combined_4.setObjective(quicksum(ct[var.varName]*var for var in combined_4.getVars() if var.VarName in lead_vars)+
					quicksum(-dt[var.varName]*var for var in combined_4.getVars() if var.VarName in fol_vars), GRB.MINIMIZE)
				
				combined_1.addConstr(quicksum(v*original_objective[v.varName] for v in combined_1.getVars()) <= objective_epsilon)
				combined_1.addConstr(quicksum(v*original_objective[v.varName] for v in combined_1.getVars()) >= objective_epsilon-((incumbent - of_radar_bounds[0]) / n))
				combined_1.addConstr(quicksum(v*original_objective[v.varName] for v in combined_1.getVars() if v.VarName in lead_vars) <= objective_epsilon_lead)
				combined_1.addConstr(quicksum(v*original_objective[v.varName] for v in combined_1.getVars() if v.VarName in lead_vars) >= objective_epsilon_lead-((of_radar_bounds_lead[-1] - of_radar_bounds_lead[0]) / n_lead))

				combined_2.addConstr(quicksum(v*original_objective[v.varName] for v in combined_2.getVars()) <= objective_epsilon)
				combined_2.addConstr(quicksum(v*original_objective[v.varName] for v in combined_2.getVars()) >= objective_epsilon-((incumbent - of_radar_bounds[0]) / n))
				combined_2.addConstr(quicksum(v*original_objective[v.varName] for v in combined_2.getVars() if v.VarName in lead_vars) <= objective_epsilon_lead)
				combined_2.addConstr(quicksum(v*original_objective[v.varName] for v in combined_2.getVars() if v.VarName in lead_vars) >= objective_epsilon_lead-((of_radar_bounds_lead[-1] - of_radar_bounds_lead[0]) / n_lead))

				combined_3.addConstr(quicksum(v*original_objective[v.varName] for v in combined_3.getVars()) <= objective_epsilon)
				combined_3.addConstr(quicksum(v*original_objective[v.varName] for v in combined_3.getVars()) >= objective_epsilon-((incumbent - of_radar_bounds[0]) / n))
				combined_3.addConstr(quicksum(v*original_objective[v.varName] for v in combined_3.getVars() if v.VarName in lead_vars) <= objective_epsilon_lead)
				combined_3.addConstr(quicksum(v*original_objective[v.varName] for v in combined_3.getVars() if v.VarName in lead_vars) >= objective_epsilon_lead-((of_radar_bounds_lead[-1] - of_radar_bounds_lead[0]) / n_lead))

				combined_4.addConstr(quicksum(v*original_objective[v.varName] for v in combined_4.getVars()) <= objective_epsilon)
				combined_4.addConstr(quicksum(v*original_objective[v.varName] for v in combined_4.getVars()) >= objective_epsilon-((incumbent - of_radar_bounds[0]) / n))
				combined_4.addConstr(quicksum(v*original_objective[v.varName] for v in combined_4.getVars() if v.VarName in lead_vars) <= objective_epsilon_lead)
				combined_4.addConstr(quicksum(v*original_objective[v.varName] for v in combined_4.getVars() if v.VarName in lead_vars) >= objective_epsilon_lead-((of_radar_bounds_lead[-1] - of_radar_bounds_lead[0]) / n_lead))

				combined_1.update()
				combined_2.update()
				combined_3.update()
				combined_4.update()
				combined_1.optimize()
				combined_2.optimize()
				combined_3.optimize()
				combined_4.optimize()

				if combined_1.status == 2:
					combined_1_sol = {var.varName:var.x for var in combined_1.getVars()}
					if combined_1_sol not in vertices:
						vertices.append(combined_1_sol)

				if combined_2.status == 2:
					combined_2_sol = {var.varName:var.x for var in combined_2.getVars()}
					if combined_2_sol not in vertices:
						vertices.append(combined_2_sol)

				if combined_3.status == 2:
					combined_3_sol = {var.varName:var.x for var in combined_3.getVars()}
					if combined_3_sol not in vertices:
						vertices.append(combined_3_sol)
				
				if combined_4.status == 2:
					combined_4_sol = {var.varName:var.x for var in combined_4.getVars()}
					if combined_4_sol not in vertices:
						vertices.append(combined_4_sol)
				
				if vertices:
					for vertex in vertices:
						f2 = follower(I,vertex)
						#for v in f2.getVars():
						#	v.setAttr('vtype', 'C')
						#	f2.update()
						f2.optimize()

						if f2.Status == 2:
										
							y_hat = {y.VarName:y.x for y in f2.getVars()}
							m3 = m.copy()

							for v in m3.getVars():
								if v.VarName in lead_vars:
									v.setAttr('vtype', 'C')
									m3.update()

							for v in m3.getVars():
								if v.VarName in lead_vars:
									m3.addConstr(v == vertex[v.VarName])
									m3.update()
							
							m3.addConstr(quicksum(v*dt[v.varName] for v in m3.getVars() if v.VarName in fol_vars) <= f2.objVal)
							m3.update()


							m3.update()
							m3.optimize()
							
							if m3.Status == 2:
									
								leader_vars_solution = {var:vertex[var] for var in vertex.keys() if var in lead_vars}
								y_not_follower_solutions = {var:vertex[var] for var in vertex.keys() if var in fol_vars}
								y_follower_solutions = {var.VarName: var.x for var in m3.getVars() if var.VarName in fol_vars}
								x_sol = []
								for var in original_variables.keys():
									if var in leader_vars_solution.keys():
										x_sol.append(leader_vars_solution[var])
									else:
										x_sol.append(y_not_follower_solutions[var])
								
								HPR_sol = m3.objVal
								if x_sol not in predictors:
									predictors.append(x_sol)
									response.append(HPR_sol)
								current_solution = {}	
								current_solution["x"] = leader_vars_solution
								current_solution["y_star"] = y_not_follower_solutions
								current_solution["y_hat"] = y_follower_solutions					
								current_solution["Lead_OF"] = sum(vertex[var]*original_objective[var] for var in original_objective.keys())
								current_solution["HPR_OF"] = m3.objVal
								current_solution["FOL_OF"] = sum(m3.getVarByName(v).x*dt[v] for v in dt.keys())

					
								if m3.objVal < best_value[-1]:
									#print("Updated bilev lb on vertex generation")
									best_value.append(m3.objVal)
									current_best_lp_sol = {'leader_vars_solution': leader_vars_solution, 'y_follower_solutions': y_follower_solutions, 'HPR_OF':m3.objVal, 'FOL_OF':sum(m3.getVarByName(v).x*dt[v] for v in dt.keys())}
									
									UB.append(UB[-1])
									LB.append(best_value[-1])
									Time.append(time.time()-time_start)
									
								if is_mipsol(INT, leader_vars_solution) and m3.objVal < incumbent:
									#duration = 100# milliseconds
									#freq = 1000  # Hz
									#winsound.Beep(freq, duration)
									#print("Updated bilev incumbent on vertex generation")
									
									incumbent = m3.objVal
									incumbent_fol = sum(m3.getVarByName(v).x*dt[v] for v in dt.keys())
									time_incumbent = time.time()
									incumbent_sol = {'leader_vars_solution': leader_vars_solution, 'y_follower_solutions': y_follower_solutions, 'HPR_OF':m3.objVal, 'FOL_OF':sum(m3.getVarByName(v).x*dt[v] for v in dt.keys())}
									restart = True
									of_radar_steps = sorted([of_radar_bounds[0]+((incumbent - of_radar_bounds[0]) / n)*i for i in range(0,n+1)], reverse = True)
									
									UB.append(incumbent)
									LB.append(LB[-1])
									Time.append(time.time()-time_start)

								if current_solution not in SolutionPool:
									SolutionPool.append(current_solution)

				
				if not restart:
					time_of_vertex = time.time()
					while time.time()-time_of_vertex <= time_to_vertex:
						#Random vertex
						replication += 1
						add_vertex = True
						current_solution = {}
						m2, m3, f2 = np.nan, np.nan, np.nan
						#print("RVERT")
						m2 = m.copy()
						
						if np.random.random() < integer_epsilon:
							for v in m2.getVars():
								if v.varName in fol_vars:
									v.setAttr('vtype', 'C')
									m2.update()

						if np.random.random() < 0.5:
							variables = {var.VarName:(var,np.random.uniform(-ct[var.VarName],ct[var.VarName])) if var.VarName in lead_vars else (var,np.random.uniform(dt[var.VarName],ct[var.VarName])) for var in m2.getVars()}
						else:
							variables = {var.VarName:(var,np.random.uniform(-10,11)) for var in m2.getVars()}
						
						
						m2.setObjective(quicksum(variables[var][0]*variables[var][1] for var in variables.keys()), GRB.MAXIMIZE)
						m2.update()
							
						m2.addConstr(quicksum(v*original_objective[v.varName] for v in m2.getVars()) <= objective_epsilon)
						m2.addConstr(quicksum(v*original_objective[v.varName] for v in m2.getVars()) >= objective_epsilon-((incumbent - of_radar_bounds[0]) / n))
						m2.addConstr(quicksum(v*original_objective[v.varName] for v in m2.getVars() if v.VarName in lead_vars) <= objective_epsilon_lead)
						m2.addConstr(quicksum(v*original_objective[v.varName] for v in m2.getVars() if v.VarName in lead_vars) >= objective_epsilon_lead-((of_radar_bounds_lead[-1] - of_radar_bounds_lead[0]) / n_lead))
						m2.update() 

						#else:
						#m2.addConstr(quicksum(m2.getVarByName(v)*dt[v] for v in dt.keys()) >= incumbent_fol)
						
						m2.optimize()
						if m2.status == 2:

							OF_check = {var.varName:var.x for var in m2.getVars()}

							if add_vertex and OF_check not in vertices:
								vertices.append(OF_check)

							f2 = follower(I,OF_check)
							#for v in f2.getVars():
							#	v.setAttr('vtype', 'C')
							#	f2.update()
							f2.optimize()

							if f2.Status == 2:
											
								y_hat = {y.VarName:y.x for y in f2.getVars()}
								m3 = m.copy()

								for v in m3.getVars():
									if v.VarName in lead_vars:
										v.setAttr('vtype', 'C')
										m3.update()

								for v in m3.getVars():
									if v.VarName in lead_vars:
										m3.addConstr(v == OF_check[v.VarName])
										m3.update()
								
								m3.addConstr(quicksum(v*dt[v.varName] for v in m3.getVars() if v.VarName in fol_vars) <= f2.objVal)
								m3.update()


								m3.update()
								m3.optimize()
								
								if m3.Status == 2:
										
									leader_vars_solution = {var.VarName: var.x for var in m2.getVars() if var.VarName in lead_vars}
									y_not_follower_solutions = {var.VarName: var.x for var in m2.getVars() if var.VarName in fol_vars}
									y_follower_solutions = {var.VarName: var.x for var in m3.getVars() if var.VarName in fol_vars}
									x_sol = []
									for var in original_variables.keys():
										if var in leader_vars_solution.keys():
											x_sol.append(leader_vars_solution[var])
										else:
											x_sol.append(y_not_follower_solutions[var])
									
									HPR_sol = m3.objVal
									if x_sol not in predictors:
										predictors.append(x_sol)
										response.append(HPR_sol)

									current_solution["x"] = leader_vars_solution
									current_solution["y_star"] = y_not_follower_solutions
									current_solution["y_hat"] = y_follower_solutions					
									current_solution["Lead_OF"] = sum(m2.getVarByName(v).x*original_objective[v] for v in original_objective.keys())
									current_solution["HPR_OF"] = m3.objVal
									current_solution["FOL_OF"] = sum(m3.getVarByName(v).x*dt[v] for v in dt.keys())

									if m3.objVal < best_value[-1]:
										#print("Updated bilev lb on vertex generation")
										best_value.append(m3.objVal)
										current_best_lp_sol = {'leader_vars_solution': leader_vars_solution, 'y_follower_solutions': y_follower_solutions, 'HPR_OF':m3.objVal, 'FOL_OF':sum(m3.getVarByName(v).x*dt[v] for v in dt.keys())}
										
										UB.append(UB[-1])
										LB.append(best_value[-1])
										Time.append(time.time()-time_start)
				
									if is_mipsol(INT, leader_vars_solution) and m3.objVal < incumbent:
										#duration = 100# milliseconds
										#freq = 1000  # Hz
										#winsound.Beep(freq, duration)
										#print("Updated bilev incumbent on vertex generation")
										
										incumbent = m3.objVal
										incumbent_fol = sum(m3.getVarByName(v).x*dt[v] for v in dt.keys())
										time_incumbent = time.time()
										incumbent_sol = {'leader_vars_solution': leader_vars_solution, 'y_follower_solutions': y_follower_solutions, 'HPR_OF':m3.objVal, 'FOL_OF':sum(m3.getVarByName(v).x*dt[v] for v in dt.keys())}
										restart = True
										of_radar_steps = sorted([of_radar_bounds[0]+((incumbent - of_radar_bounds[0]) / n)*i for i in range(0,n+1)], reverse = True)
										
										UB.append(incumbent)
										LB.append(LB[-1])
										Time.append(time.time()-time_start)


									if current_solution not in SolutionPool:
										SolutionPool.append(current_solution)
				###############################################################################################ML####################################################

				lm_enter = False
				if not restart:
					if len(predictors) >= 2 and replication % update_every == 0:
						lm_enter = True
						lm = LinearRegression().fit(predictors, response)
						for var_coeff in range(len(coeff_list)):
							coeff_list[var_coeff].append(lm.coef_[var_coeff])

						artificial_model = m.copy()
						for v in artificial_model.getVars():
							if v.varName in fol_vars:
								v.setAttr('vtype', 'C')
								artificial_model.update()
						
						vars_in_art_model = [v for v in artificial_model.getVars()]
						artificial_model.setObjective(quicksum(lm.coef_[var]*vars_in_art_model[var] for var in range(len(vars_in_art_model)) if vars_in_art_model[var].varName in lead_vars), GRB.MINIMIZE)
						
						artificial_model.addConstr(quicksum(v*original_objective[v.varName] for v in artificial_model.getVars()) <= objective_epsilon)
						artificial_model.addConstr(quicksum(v*original_objective[v.varName] for v in artificial_model.getVars()) >= objective_epsilon-((incumbent - of_radar_bounds[0]) / n))
						artificial_model.addConstr(quicksum(v*original_objective[v.varName] for v in artificial_model.getVars() if v.VarName in lead_vars) <= objective_epsilon_lead)
						artificial_model.addConstr(quicksum(v*original_objective[v.varName] for v in artificial_model.getVars() if v.VarName in lead_vars) >= objective_epsilon_lead-((of_radar_bounds_lead[-1] - of_radar_bounds_lead[0]) / n_lead))
						artificial_model.update()
						
						artificial_model.optimize()
						if artificial_model.status == 2:
							x_hat_art = {v.varName:v.x for v in artificial_model.getVars()}
							f2_art = follower(I,x_hat_art)
							#for v in f2.getVars():
							#	v.setAttr('vtype', 'C')
							#	f2.update()
							f2_art.optimize()

							if f2_art.Status == 2:
											
								y_hat_art = {y.VarName:y.x for y in f2_art.getVars()}
								m3_art = m.copy()

								for v in m3_art.getVars():
									if v.VarName in lead_vars:
										m3_art.addConstr(v == x_hat_art[v.VarName])
										m3_art.update()
								
								m3_art.addConstr(quicksum(v*dt[v.varName] for v in m3_art.getVars() if v.VarName in fol_vars) <= f2_art.objVal)
								m3_art.update()
								m3_art.update()
								m3_art.optimize()
								
								if m3_art.Status == 2:
									art_vertex = {var.varName:var.x for var in m3_art.getVars()}
									leader_vars_solution = {var.VarName: var.x for var in artificial_model.getVars() if var.VarName in lead_vars}
									y_not_follower_solutions = {var.VarName: var.x for var in artificial_model.getVars() if var.VarName in fol_vars}
									y_follower_solutions = {var.VarName: var.x for var in m3_art.getVars() if var.VarName in fol_vars}
									x_sol = []
									for var in original_variables.keys():
										if var in leader_vars_solution.keys():
											x_sol.append(leader_vars_solution[var])
										else:
											x_sol.append(y_not_follower_solutions[var])
									
									HPR_sol = m3_art.objVal
									if x_sol not in predictors:
										predictors.append(x_sol)
										response.append(HPR_sol)
									
									current_solution = {}
									current_solution["x"] = leader_vars_solution
									current_solution["y_star"] = y_not_follower_solutions
									current_solution["y_hat"] = y_follower_solutions					
									current_solution["Lead_OF"] = sum(artificial_model.getVarByName(v).x*original_objective[v] for v in original_objective.keys())
									current_solution["HPR_OF"] = m3_art.objVal
									current_solution["FOL_OF"] = sum(m3_art.getVarByName(v).x*dt[v] for v in dt.keys())

									if m3_art.objVal < best_value[-1]:
										#print("Updated bilev lb on vertex generation")
										best_value.append(m3_art.objVal)
										current_best_lp_sol = {'leader_vars_solution': leader_vars_solution, 'y_follower_solutions': y_follower_solutions, 'HPR_OF':m3_art.objVal, 'FOL_OF':sum(m3_art.getVarByName(v).x*dt[v] for v in dt.keys())}
										
										UB.append(UB[-1])
										LB.append(best_value[-1])
										Time.append(time.time()-time_start)

									if is_mipsol(INT, leader_vars_solution) and m3_art.objVal < incumbent:
										#duration = 100# milliseconds
										#freq = 1000  # Hz
										#winsound.Beep(freq, duration)
										#print("Updated bilev incumbent on vertex generation")
										
										incumbent = m3_art.objVal
										incumbent_fol = sum(m3_art.getVarByName(v).x*dt[v] for v in dt.keys())
										time_incumbent = time.time()
										incumbent_sol = {'leader_vars_solution': leader_vars_solution, 'y_follower_solutions': y_follower_solutions, 'HPR_OF':m3_art.objVal, 'FOL_OF':sum(m3_art.getVarByName(v).x*dt[v] for v in dt.keys())}
										restart = True
										of_radar_steps = sorted([of_radar_bounds[0]+((incumbent - of_radar_bounds[0]) / n)*i for i in range(0,n+1)], reverse = True)
										
										UB.append(incumbent)
										LB.append(LB[-1])
										Time.append(time.time()-time_start)

									if add_vertex and art_vertex not in vertices:
										vertices.append(art_vertex)
				
				#Random interior point
				if not restart:
					if vertices:
						time_of_representation = time.time()
						while time.time()-time_of_representation <= time_to_representation:
							
							#for _ in range(0,int(max_samples)):	
							current_solution = {}
							m2, m3, f2 = np.nan, np.nan, np.nan
							#print("RINT")
							
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

							if np.random.random() < integer_epsilon:
								feasibility_check = {var:int(feasibility_check[var]) if var in lead_vars and var in INT else feasibility_check[var]  for var in feasibility_check.keys()}
							m2 = m.copy()
							
							for v in m2.getVars():
								v.setAttr('vtype', 'C')
								m2.update()

							for var in m2.getVars():
								m2.addConstr(var == feasibility_check[var.VarName])    	
								m2.update()

							m2.optimize()
							if m2.status == 2:
								f2 = follower(I,feasibility_check)
								#for v in f2.getVars():
								#	v.setAttr('vtype', 'C')
								#	f2.update()
								f2.optimize()
								
								if f2.Status == 2:
												
									y_hat = {y.VarName:y.x for y in f2.getVars()}
									
									m3 = m.copy()
									
									for v in m3.getVars():
										if v.VarName in lead_vars:
											v.setAttr('vtype', 'C')
											m3.update()

									for v in m3.getVars():
										if v.VarName in lead_vars:
											m3.addConstr(v == feasibility_check[v.VarName])
											m3.update()
									
									m3.addConstr(quicksum(v*dt[v.varName] for v in m3.getVars() if v.VarName in fol_vars) <= f2.objVal)
									
									m3.update()
									m3.optimize()

									if m3.Status == 2:					
										leader_vars_solution = {var.VarName: var.x for var in m2.getVars() if var.VarName in lead_vars}
										y_not_follower_solutions = {var.VarName: var.x for var in m2.getVars() if var.VarName in fol_vars}
										y_follower_solutions = {var.VarName: var.x for var in m3.getVars() if var.VarName in fol_vars}
										
										x_sol = []
										HPR_sol = m3.objVal
										for var in original_variables.keys():
											if var in leader_vars_solution.keys():
												x_sol.append(leader_vars_solution[var])
											else:
												x_sol.append(y_not_follower_solutions[var])
										if x_sol not in predictors:
											predictors.append(x_sol)
											response.append(HPR_sol)

										
										current_solution["x"] = leader_vars_solution
										current_solution["y_star"] = y_not_follower_solutions
										current_solution["y_hat"] = y_follower_solutions					
										current_solution["Lead_OF"] = m2.objVal
										current_solution["HPR_OF"] = m3.objVal
										current_solution["FOL_OF"] = sum(m3.getVarByName(v).x*dt[v] for v in dt.keys())
										
										if m3.objVal < best_value[-1]:
											#print("Updated bilev lb on random point generation")
											best_value.append(m3.objVal)
											current_best_lp_sol = {'leader_vars_solution': leader_vars_solution, 'y_follower_solutions': y_follower_solutions, 'HPR_OF':m3.objVal, 'FOL_OF':sum(m3.getVarByName(v).x*dt[v] for v in dt.keys())}
											
											UB.append(UB[-1])
											LB.append(best_value[-1])
											Time.append(time.time()-time_start)
										
										if is_mipsol(INT, leader_vars_solution) and m3.objVal < incumbent:
											#duration = 100# milliseconds
											#freq = 1000  # Hz
											#winsound.Beep(freq, duration)
											#print("Updated bilev incumbent on random point generation")
											incumbent = m3.objVal
											incumbent_fol = sum(m3.getVarByName(v).x*dt[v] for v in dt.keys())
											time_incumbent = time.time()
											incumbent_sol = {'leader_vars_solution': leader_vars_solution, 'y_follower_solutions': y_follower_solutions, 'HPR_OF':m3.objVal, 'FOL_OF':sum(m3.getVarByName(v).x*dt[v] for v in dt.keys())}
											of_radar_steps = sorted([of_radar_bounds[0]+((incumbent - of_radar_bounds[0]) / n)*i for i in range(0,n+1)], reverse = True)
											restart = True

											UB.append(incumbent)
											LB.append(LB[-1])
											Time.append(time.time()-time_start)

										if current_solution not in SolutionPool:
											SolutionPool.append(current_solution)
							if restart:
								print('Break time_of_representation while')
								break

				gap_not_bilevel = abs(incumbent - ceiling_solution)/abs(ceiling_solution+1e-10)
				gap_bilevel = abs(incumbent - best_value[-1])/abs(best_value[-1]+1e-10)

				print("\n")
				print(f'Restart: {restart}')
				print(f'Full cycle: {full_cycle}')
				print(f'realization: {replication}')
				#print(f'vertices: {vertices}')
				print(f'incumbent mip sol: {incumbent_sol}')
				print(f'incumbent lp sol: {current_best_lp_sol}')
				if lm_enter and len(predictors) >= 2:
					var_coeff_values = {}
					for variable in range(len(original_variables.keys())):
						var_coeff_values[list(original_variables.keys())[variable]] = mean(coeff_list[variable])
					print(f'Hidden true objective coefficients: {var_coeff_values}')
				print(f'Actual objective in for loop: {objective_epsilon}')
				print(f'Actual objective in for scope: {window}')
				print(f'Actual leader_variable_objective in for loop: {objective_epsilon_lead}')
				print(f'Actual leader_variable_objective in for scope: {window_lead}')
				print(f'incumbent MIBLP OF value: {incumbent}')
				print(f'Lower bound bilevel: {best_value[-1]}')
				print(f'Lower bound (not bilevel): {ceiling_solution}')
				print(f'incumbent follower: {incumbent_fol}')
				print(f'HPR_OF range: {best_value}')
				print(f'Leader_OF range: {[of_radar_steps[0],of_radar_steps[-1]]}')
				print(f'gap not bilevel: {gap_not_bilevel}')
				print(f'gap bilevel: {gap_bilevel}')
				print(f'Done: {done}')
				
				if incumbent_sol:
					print(f'Time since last incumbent update: {time.time()-time_incumbent}')

				if restart:
					print('Break leader_obj_epsilon for')
					break
				
				if incumbent_sol and time.time()-time_incumbent > time_to_look and full_cycle:
					print('Break objective_epsilon while case time ended 1')
					break
			
		if restart:
			print('Break objective_epsilon for case restart')
			break
		if incumbent_sol and time.time()-time_incumbent > time_to_look and full_cycle:
			print('Break objective_epsilon while case time ended 2')
			break
				
	if incumbent_sol and time.time()-time_incumbent > time_to_look and full_cycle:
		print('Break done (big) while')
		done = True
		break
	
if SolutionPool:
	#final_solutions = []
	output = {}
	for element in range(len(SolutionPool)):
		#print(SolutionPool[element]['y_star'])
		#print(SolutionPool[element]['y_hat'])
		#print(dt)
		#if bilev_feas_chech(SolutionPool[element]['y_star'], dt, SolutionPool[element]['y_hat']):
		#	final_solutions.append(SolutionPool[element])
		for item in SolutionPool[element].keys():
			if isinstance(SolutionPool[element][item], dict):
				for sub_item in SolutionPool[element][item].keys():
					if str(item)+'_'+str(sub_item) in output.keys():
						output[str(item)+'_'+str(sub_item)].append(SolutionPool[element][item][sub_item])
					else:
						output[str(item)+'_'+str(sub_item)] = [SolutionPool[element][item][sub_item]]	
			else:
				if item in output.keys():
					output[item].append(SolutionPool[element][item])
				else:
					output[item] = [SolutionPool[element][item]]

	
#print(output)
	if save:
		with open('bilevel_sols_instance_'+str(ins)+'.txt', 'w') as convert_file:
			convert_file.write(json.dumps(SolutionPool))
		with open('var_coeff_values_'+str(ins)+'.txt', 'w') as convert_file:
			convert_file.write(json.dumps(var_coeff_values))
		with open('optimal_solutions'+str(ins)+'.txt', 'w') as convert_file:
			convert_file.write(json.dumps(incumbent_sol))
		Training_data = pd.DataFrame(output, columns = output.keys())
		Training_data.to_csv('Data_project_MIBLP_idea_'+str(ins)+'.csv')

	if save_gap_plot:
		location = 'upper right'
		linewidth = 1
		size=10
		style.use("ggplot")
		
		# Time.pop(0)
		# UB.pop(0)
		# LB.pop(0)

		print(Time)
		print(UB)
		print(LB)
		
		plt.scatter(Time, UB, c='b', label = 'Upper bound', s = size)
		plt.plot(Time, UB, c='b', linewidth=linewidth)
		plt.scatter(Time, LB, c='r', label = 'Lower bound', s = size)
		plt.plot(Time, LB, c='r', linewidth=linewidth)
		
		plt.legend(loc=location)
		plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
		plt.title('Bounds behavior for instance '+str(ins)+'. Final gap: '+str(round(gap_bilevel*100,2))+'%')
		plt.xlabel('Time (s)')
		plt.ylabel('Bound value')
		plt.savefig('Instance_'+str(ins)+'_Bounds.png')
		plt.show()	

else:
	print("Something weird is happenning")

duration = 500# milliseconds
freq = 1000  # Hz
winsound.Beep(freq, duration)