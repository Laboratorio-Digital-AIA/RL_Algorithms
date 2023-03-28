from gurobipy import *
import numpy as np
import pandas as pd
import os
from math import floor, log, exp
import Models
import PPP_Env_V2_MARL
import datetime
import matplotlib.pyplot as plt
import copy


def create_MASTER_MAT_and_MASTER_b(HPR):
    
    #Nombres de las restricciones
    constr_names=[c.constrName for c in HPR.getConstrs()]
    #Vector b del RHS
    b = [c.rhs for c in HPR.getConstrs()]
    #Nombres de las variables
    var_names = [v.VarName for v in HPR.getVars()]
    #Crea la matriz A
    A = np.zeros(shape=(HPR.numConstrs,HPR.numVars))
    A = pd.DataFrame(A,index=constr_names,columns=var_names)
    for c in HPR.getConstrs():
        for v in HPR.getVars():
            A.loc[c.constrName,v.VarName] = HPR.getCoeff(c,v)

    MASTER_MAT = A    
    MASTER_b = pd.Series(b, index = [c.constrName for c in HPR.getConstrs()]) 

    return MASTER_MAT, MASTER_b
        
def node_is_mipsol(INT, x_hat):
    
    tolerance = 1e-4
    for i in INT:
        if abs(floor(x_hat[i])-x_hat[i])>tolerance:
            return False
    return True

def getVarTypes(m):
    INT =[v.VarName for v in m.getVars() if v.vtype == 'I' or v.vtype == 'B']
    CON =[v.VarName for v in m.getVars() if v.vtype == 'C']
    return INT,CON
    
def solvebnc(this_node):
            
    c = HPR_STD.copy()
    for v in c.getVars():
        v.setAttr('vtype', 'C')
        c.update()
    c.setParam(GRB.Param.OutputFlag, 0)
    c.setParam(GRB.Param.DualReductions, 0)
    #c.setParam(GRB.Param.Presolve, 0)
    #c.setParam(GRB.Param.Heuristics, 0)
    #c.setParam(GRB.Param.PreSparsify, 0)

    # Slicing of the master matrix for the current node
    MAT = MASTER_MAT.loc[CONSTR[this_node],:]
    b = MASTER_b[CONSTR[this_node]]
    MAT = MAT.loc[:, (MAT != 0).any(axis=0)]
    
    
    #MAT = MAT.values
    if not MAT.empty:

        or_var_names = [j.VarName for j in c.getVars()]
        aux = [i for i in list(MAT.columns) if i not in or_var_names]
        for i in range(len(aux)):
            c.addVar(vtype = GRB.CONTINUOUS, name = aux[i])
        c.update()
        for i in range(MAT.shape[0]):
            c.addConstr(quicksum(MAT.iloc[i,j]*c.getVarByName(MAT.columns[j]) for j in range(MAT.shape[1])) == b[i], name = MAT.index[i])
        c.update()
        
    c.optimize()
    
    

    stat = c.Status

    if stat == 2:

        of = c.ObjVal
        x_hat = {i.VarName:round(i.x, 6) for i in c.getVars()}
        #Nombres de las variables básicas:
        bv_names = [v.VarName for v in c.getVars() if v.vbasis == 0]
        bc_names = [con.constrName for con in c.getConstrs() if con.cbasis == -1]
        

        return x_hat, of, stat, bv_names, bc_names

        
    else:

        return {}, np.nan, stat, [],[]


    '''

    Solve normal

    '''


def solve(m, constrs,  Phi_data):
    copia = m.copy()
    copia.setParam(GRB.Param.OutputFlag, 0)
    copia.setParam(GRB.Param.DualReductions, 0)
    #copia.setParam(GRB.Param.Presolve, 0)
    #copia.setParam(GRB.Param.Heuristics, 0)
    for c in constrs:
        x = copia.getVarByName(c[0])
        
        if c[1] == '>':
            copia.addConstr(x>=c[2])
        elif c[1] == '<':
            copia.addConstr(x<=c[2])
        elif c[1] == '=':
            copia.addConstr(x==c[2])
        else:
            print("Error: Algo raro pasa")
    
    if Phi_data:
        suma = quicksum(copia.getVarByName(i)*Phi_data[1][i] for i in Phi_data[1])
        copia.addConstr(suma <= Phi_data[0])
        
    copia.update()
    
    modelo = copia.copy()
        
    modelo.optimize()
    stat = modelo.Status
    
    if stat == 2:
        x_hat = {v.VarName:v.X for v in modelo.getVars()}
        OF = modelo.ObjVal
    else:
        x_hat, OF = {}, np.nan

    return x_hat, OF, stat

def INT_CUTS(this_node, Bvars, Bcons, xhat):

    A = MASTER_MAT.loc[Bcons,:]
    B = MASTER_MAT.loc[Bcons,Bvars]
    b = MASTER_b[Bcons]
    Binv = np.linalg.inv(B.values)
    xnames = xhat.keys()
    
    g0 = []
    g = []
    gb =[]
    
    #HC+ Only
    CLEAD_aux = {i.VarName:i.obj for i in HPR.getVars() if i.VarName not in CFOL}

    for var in CLEAD_aux:
        g0.append(xhat[var] - 1)
        g0.append(xhat[var] + 1)
        g.append([1 if i == var else 0 for i in A.columns])
        g.append([1 if i == var else 0 for i in A.columns])
        gb.append([1 if i == var else 0 for i in B.columns])
        gb.append([1 if i == var else 0 for i in B.columns])
    


    G = np.array(g)
    Gb = np.array(gb)

    u = [np.dot(row, Binv) for row in Gb]
    
    Ans_g = []
    Ans_g0 = []
    g_bar = []
    g0_bar = []
    
    for i in range(len(u)):
        
        Ans_g.append(np.dot(u[i], A.values))
        Ans_g0.append(np.dot(u[i], b.values))
        
        g_bar.append(G[i,:] - Ans_g[i])
        g0_bar.append(g0[i] - Ans_g0[i])
    
    
    for i in range(len(g0_bar)):
        if g0_bar[i] == 0:
            g0_bar[i] = 1e-8

    gamma = []
    for j in range(len(A.columns)):
        gammita = []
        for i in range(len(g0_bar)):
            gammita.append(g_bar[i][j]/g0_bar[i])
        gamma.append(max(gammita))
    
    gamma = {A.columns[i]:gamma[i] for i in range(len(A.columns))}
    
    if all(value >= 0 for value in gamma.values()):
        
        for i in A.columns:
            if i in INT:
                gamma[i] = min(1, gamma[i])
            else:
                gamma[i] = gamma[i]
    
    return gamma

def choose_branch_var(x_hat, inherited_constrs, follower_vars, inte, mipnode, leader_vars, branching_priority):
        
           
    aux = {i:x_hat[i] - floor(x_hat[i]) for i in x_hat}

    x = {i:branching_priority[i] for i in x_hat if i in inte and aux[i] >= 1e-3}

    if x:
        key = max(x.items(), key=operator.itemgetter(1))[0]
        
        return key, floor(x_hat[key])
    else:
        return np.nan, np.nan

#################################################################
def BC_node(constrs, parent_id):
    
    # Global variables to keep track of Branch & Bound
    global TREE, OF, FOLOF, SOL, CONSTR, LB, UB, BILEV_FEAS, NODE_TYPE, INCUMBENT, LV_BOUNDS
    # Global variables with the problem's mathematical models and parameters information
    global HPR, HPR_STD, FOL, INT, CON, CFOL, CLEAD, BRANCH_PRIORITY
    # Global variables aggregating all models contstraints and variables
    global MASTER_MAT, MASTER_b, MASTER_MAT_OR, THRESH, or_bounds, I, MAX_CUTS
    
   
    gap = abs(UB-LB)/abs(LB)
    print(UB)
    print("\n")
    print(gap)
    print("\n")

    if  gap > THRESH and len(TREE) < int(1e4):
        
        print("\n___________________________________________________\n")
        
        
        TREE.append(parent_id)
        
        CONSTR.append(constrs)
        
        current_node = len(TREE)-1
        print("This is node " + str(current_node) + " (son of " + str(parent_id) + ")")
        

        x_hat, of, stat, Bvars, Bcons = solvebnc(current_node)
        print("___________________________________________________\n")

        if stat == 2:
            print("FEASIBLE\n")

            if node_is_mipsol(INT, x_hat):
                print("MIPSOL_ENTER\n")
                SOL.append(x_hat)
                OF.append(of)
                FOLOF.append("-")
                BILEV_FEAS.append(False)
                
                #Evalua Factibilidad Binivel
                y_star = {i:x_hat[i] for i in x_hat if i in CFOL}
                
                condition = True

                F = FOL(I,x_hat)
                y_hat, Phi, stat_f = solve(F, [], ()) 
                
                for var in CFOL.keys():
                    if y_star[var] != y_hat[var]:
                        condition = False
                        break

                if condition:
                    print("MISPOL_BILEVEL_FEAS\n")
                    BILEV_FEAS[current_node] = True
                    FOLOF[current_node] = Phi
                    
                    if INCUMBENT:
                        
                        if of < UB:
                            print("INCUMBENT_UPDATE\n")
                            UB = of
                            INCUMBENT = [current_node]
                        
                else:

                    print("MIPSOL_NOTBILEV_FEAS\n") 

                    mipsol = True
                    k = 0
                    need_to_branch = False
                    while mipsol:
                        
                        F = FOL(I,x_hat)
                        y_hat, Phi, stat_f = solve(F, [], ())

                        #Evalua Factibilidad Binivel
                        y_star = {i:x_hat[i] for i in x_hat if i in CFOL}
                        
                        condition = True

                        for var in CFOL.keys():
                            if y_star[var] != y_hat[var]:
                                condition = False
                                break
                        
                        if condition: 
                            #Toca mirar esta condicion !!!! (mipsol no es falso pero lo usamos para el while)
                            mipsol = False
                            print("MIPSOL_BILEV_FEAS_AFTER_CUT\n")
                            if of < UB:
                                print("MIPSOL_BILEV_FEAS_AFTER_CUT_UB_UPDATE\n")
                                BILEV_FEAS[current_node] = True
                                SOL[current_node] = x_hat
                                OF[current_node] = of
                                FOLOF[current_node] = Phi
                                UB = of
                                INCUMBENT = [current_node]
                        else:

                            x_constr = [(i, '=', x_hat[i]) for i in x_hat if i not in CFOL and i in [v.VarName for v in HPR.getVars()]]
                
                            phi_constr = (Phi, CFOL)
                            
                            C = HPR.copy()
                            x_hat_new, of_new, stat_new  = solve(C, x_constr, phi_constr)
                            
                            if stat_new == 2:
                                print("BEST_RESPONSE_IN_INTCUT\n")    
                                if of_new < UB:
                                    print("UB_UPDATE\n")
                                    SOL[current_node] = x_hat_new
                                    OF[current_node] = of_new
                                    FOLOF[current_node] = Phi
                                    BILEV_FEAS[current_node] = True
                                    UB = of_new
                                    INCUMBENT = [current_node]
                                
                            
                        k += 1
                                                      
                        
                        gamma = INT_CUTS(current_node, Bvars, Bcons, x_hat)
                        
                        gamma = {i:round(gamma[i], 6) for i in gamma if gamma[i] > 1e-3}

                        gamma_aux = pd.DataFrame(gamma, index = ["IC_" + str(current_node) + "_" + str(k)])
                        MASTER_MAT = MASTER_MAT.append(gamma_aux)
                        MASTER_MAT = MASTER_MAT.fillna(0)
                        aux_IC_slack = [-1 if i == "IC_" + str(current_node) + "_" + str(k) else 0 for i in MASTER_MAT.index]
                        MASTER_MAT["s_IC_" + str(current_node) + "_" + str(k)] = aux_IC_slack
                        aux_b_ic = pd.Series(1, index = ["IC_" + str(current_node) + "_" + str(k)])
                        MASTER_b = MASTER_b.append(aux_b_ic)
                        print("\nThis is cut " + str(k))
                        
                        CONSTR[current_node] += ["IC_" + str(current_node) + "_" + str(k)]

                        x_hat, of, stat, Bvars, Bcons = solvebnc(current_node)
                        

                        if stat == 2:
                            print("AFTER_CUT_NOT_DAMAGED\n")
                            if not node_is_mipsol(INT, x_hat):
                                mipsol = False
                                need_to_branch = True
                                print("NOT_MIPSOL_AFTER_CUT\n")
                                
                                
                            if node_is_mipsol(INT, x_hat) and k > MAX_CUTS:
                                print("MAX_CUTS_REACHED\n")
                                mipsol = False
                            
                                
                        elif stat != 2:
                            print("AFTER_CUT_INFEASIBLE\n")
                            mipsol = False

                    if need_to_branch:
                        branch_var, val = choose_branch_var(x_hat, CONSTR[current_node], CFOL.keys(), INT, True, CLEAD.keys(), BRANCH_PRIORITY)
                        if not np.isnan(val):
                            
                            #MASTER_MAT.append([0 for i in MASTER_MAT.index], axis = 1)
                            AuxUP = pd.DataFrame({branch_var:1, "s_BB_" + str(current_node) + "_U": - 1}, index = ["BB_" + str(current_node) + "_U"])
                            MASTER_MAT = MASTER_MAT.append(AuxUP)
                            AuxDOWN = pd.DataFrame({branch_var:1, "s_BB_" + str(current_node) + "_L":  1}, index = ["BB_" + str(current_node) + "_L"])
                            MASTER_MAT = MASTER_MAT.append(AuxDOWN)
                            MASTER_MAT.fillna(0, inplace = True)
                            AuxbUP = pd.Series(int(val)+1, index = ["BB_" + str(current_node) + "_U"])
                            MASTER_b = MASTER_b.append(AuxbUP)
                            AuxbDOWN = pd.Series(int(val), index = ["BB_" + str(current_node) + "_L"])
                            MASTER_b =MASTER_b.append(AuxbDOWN)

                            branches = {"Up":CONSTR[current_node] + ["BB_" + str(current_node) + "_U"],
                                        "Dn":CONSTR[current_node] + ["BB_" + str(current_node) + "_L"]}

                            #if current_time - init_time >= timelimit:
                            for b in branches:
                                BC_node(branches[b], current_node)
                    
    
            else:
                print("MIPNODE\n")
            

                SOL.append(x_hat)
                OF.append(of)
                FOLOF.append("-")
                BILEV_FEAS.append(False)
            
                branch_var, val = choose_branch_var(x_hat, CONSTR[current_node], CFOL.keys(), INT, True, CLEAD.keys(), BRANCH_PRIORITY)
                
                if not np.isnan(val):
                    
                    
                    #MASTER_MAT.append([0 for i in MASTER_MAT.index], axis = 1)
                    AuxUP = pd.DataFrame({branch_var:1, "s_BB_" + str(current_node) + "_U": - 1}, index = ["BB_" + str(current_node) + "_U"])
                    MASTER_MAT = MASTER_MAT.append(AuxUP)
                    AuxDOWN = pd.DataFrame({branch_var:1, "s_BB_" + str(current_node) + "_L":  1}, index = ["BB_" + str(current_node) + "_L"])
                    MASTER_MAT = MASTER_MAT.append(AuxDOWN)
                    MASTER_MAT.fillna(0, inplace = True)
                    AuxbUP = pd.Series(int(val)+1, index = ["BB_" + str(current_node) + "_U"])
                    MASTER_b = MASTER_b.append(AuxbUP)
                    AuxbDOWN = pd.Series(int(val), index = ["BB_" + str(current_node) + "_L"])
                    MASTER_b =MASTER_b.append(AuxbDOWN)

                    branches = {"Up":CONSTR[current_node] + ["BB_" + str(current_node) + "_U"],
                                "Dn":CONSTR[current_node] + ["BB_" + str(current_node) + "_L"]}
                    
                    #if current_time - init_time >= timelimit:
                    for b in branches:
                        BC_node(branches[b], current_node)

        else:
            print("INFEASIBLE_HPR\n")
            BILEV_FEAS.append(False)
            SOL.append("-")
            OF.append("-")
            FOLOF.append("-")
   
                
'''
    Initialization of recursion (Root node)
'''


def run_BnC(my_instance):
    global TREE, OF, SOL, FOLOF, CONSTR, LB, UB, BILEV_FEAS, NODE_TYPE, INCUMBENT
    # Global variables with the problem's mathematical models and parameters information
    global HPR, HPR_STD, FOL, FOL_FUB, INT, CON, CFOL, CLEAD, BRANCH_PRIORITY
    # Global variables aggregating all models contstraints and variables
    global MASTER_MAT, MASTER_b, MASTER_MAT_OR, MASTER_b_OR, LV_BOUNDS, THRESH, or_bounds, I, MAX_CUTS
    
    I = my_instance

   
    begin_time = datetime.datetime.now()

    
    HPR_STD = Models.HPR_PPP_std(I)
    HPR = Models.HPR_PPP(I)
    FOL = Models.follower_PPP
    INT, CON = getVarTypes(HPR_STD)
    CFOL = []
    CLEAD = []
    BILEV_FEAS = []
    NODE_TYPE = []
    BRANCH_PRIORITY = []
    

    MASTER_MAT, MASTER_b = create_MASTER_MAT_and_MASTER_b(HPR_STD)

    MASTER_MAT_OR = MASTER_MAT.copy()

    MASTER_b_OR = MASTER_b.copy()

    F = FOL(I,{i.VarName:0 for i in HPR_STD.getVars()})

    CFOL = {i.VarName:i.obj for i in F.getVars()}
    
    CLEAD = {i.VarName:i.obj for i in HPR.getVars() if i.VarName not in CFOL}
        
    BRANCH_PRIORITY = {i:3 for i in CLEAD}
    
    for i in CFOL:
        BRANCH_PRIORITY[i] = 2
            
    TREE = [] #Master list of nodes
    OF = [] #List of Values
    FOLOF = []
    SOL = [] #List of Dictionary
    CONSTR = [] #List of Lists of Tuples 
    HPR_LP = HPR.copy()
    
    for var in HPR_LP.getVars():
        var.setAttr('vtype','C')
        HPR_LP.update()
    HPR_LP.optimize()
    LB = HPR_LP.objVal#-24278.75 #26184
    del HPR_LP
    UB = 1e8
    INCUMBENT = []
    
    THRESH = 0.05
    print("This is threshold: "+str(THRESH))
    MAX_CUTS = 10
    BC_node([], '0')

    if INCUMBENT:
        name = "Instance_"+str(10_000)#str(57)   

        bilev_feas_nodes = []
        for node in range(len(TREE)):
            if BILEV_FEAS[node] and node != INCUMBENT[0]:
                bilev_feas_nodes.append(node)

                inspection = [SOL[node][i] for i in SOL[node].keys() if i[0] == "q"]
                maintenance = [SOL[node][i] for i in SOL[node].keys() if i[0] == "x"]
                performance = [SOL[node][i] for i in SOL[node].keys() if i[0] == "v"]
                no_periods_last_maintenance = [SOL[node][i] for i in SOL[node].keys() if i[0] == "y"]
                pplus = [SOL[node][i] for i in SOL[node].keys() if i[0] == "e"]
                pminus = [SOL[node][i] for i in SOL[node].keys() if i[0] == "s"]
                pdot = [SOL[node][i] for i in SOL[node].keys() if i[0] == "c"]
                social_ben_per_period_leader = [sum(I.g[l]*SOL[node]['z_'+str((t,l))] for l in I.L) for t in range(len(I.T))]


                PPP_metrics = {"Inspection": inspection,
                            "Maintenance": maintenance,
                            "Performance": performance,
                            "# of periods since last maintenance": no_periods_last_maintenance,
                            "Earnings": pplus,
                            "Expenditures": pminus,
                            "Available cash": pdot,
                            "Social_ben":social_ben_per_period_leader}
                
                system_performance_output = pd.DataFrame(PPP_metrics, columns = PPP_metrics.keys())

                
                BnC_metrics = {"Dual Bound": LB,
                                "Primal Bound": UB,
                                "Leader's Objective": OF,
                                "Bilevel feasibility": BILEV_FEAS,
                                "Follower's Objective": FOLOF}
                
                BnC_output = pd.DataFrame(BnC_metrics, columns = BnC_metrics.keys())
                system_performance_output.to_csv("Mantenimientos_Via_"+str(name)+"_node_"+str(node)+".csv")
                BnC_output.to_csv("Bilev_BnC_summary_Via_"+str(name)+"_node_"+str(node)+".csv")

        print("This is incumbent node")
        print(INCUMBENT[0])
        print("This is solution")
        print(SOL[INCUMBENT[0]])
        print("This is objective")
        print(OF[INCUMBENT[0]])

        bilev_feas_nodes.append(INCUMBENT[0])
        inspection = [SOL[INCUMBENT[0]][i] for i in SOL[INCUMBENT[0]].keys() if i[0] == "q"]
        maintenance = [SOL[INCUMBENT[0]][i] for i in SOL[INCUMBENT[0]].keys() if i[0] == "x"]
        performance = [SOL[INCUMBENT[0]][i] for i in SOL[INCUMBENT[0]].keys() if i[0] == "v"]
        no_periods_last_maintenance = [SOL[INCUMBENT[0]][i] for i in SOL[INCUMBENT[0]].keys() if i[0] == "y"]
        pplus = [SOL[INCUMBENT[0]][i] for i in SOL[INCUMBENT[0]].keys() if i[0] == "e"]
        pminus = [SOL[INCUMBENT[0]][i] for i in SOL[INCUMBENT[0]].keys() if i[0] == "s"]
        pdot = [SOL[INCUMBENT[0]][i] for i in SOL[INCUMBENT[0]].keys() if i[0] == "c"]
        social_ben_per_period_leader = [sum(I.g[l]*SOL[INCUMBENT[0]]['z_'+str((t,l))] for l in I.L) for t in range(len(I.T))]


        PPP_metrics = {"Inspection": inspection,
                    "Maintenance": maintenance,
                    "Performance": performance,
                    "# of periods since last maintenance": no_periods_last_maintenance,
                    "Earnings": pplus,
                    "Expenditures": pminus,
                    "Available cash": pdot,
                    "Social_ben":social_ben_per_period_leader}
        
        system_performance_output = pd.DataFrame(PPP_metrics, columns = PPP_metrics.keys())

        
        BnC_metrics = {"Dual Bound": LB,
                        "Primal Bound": UB,
                        "Leader's Objective": OF,
                        "Bilevel feasibility": BILEV_FEAS,
                        "Follower's Objective": FOLOF}
        
        BnC_output = pd.DataFrame(BnC_metrics, columns = BnC_metrics.keys())
        
        #system_performance_output.to_csv("Mantenimientos.csv")
        #BnC_output.to_csv("Bilev_BnC_summary.csv")
        system_performance_output.to_csv("Mantenimientos_Via_"+str(name)+"_incumbent_node_"+str(INCUMBENT[0])+".csv")
        BnC_output.to_csv("Bilev_BnC_summary_Via_"+str(name)+"_incumbent_node_"+str(INCUMBENT[0])+".csv")
        #return system_performance_output, BnC_output, "Solution found"
        print(bilev_feas_nodes)

    else:
        print('No solution found')
        #return pd.DataFrame({1:[0,0,0]}), pd.DataFrame({1:[0,0,0]}), "No solution found"

# for i in range(1,5):
#     my_instance = instance.PPP_Ins(INC=i, INS=3)
#     DF_ppp, DF_bnc, status = run_BnC(my_instance)

#     name = "Instance_"+str(i)   
#     DF_ppp.to_csv("Mantenimientos_Via_"+name+".csv")
#     DF_bnc.to_csv("Bilev_BnC_summary_"+name+".csv")


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

my_instance = Instance()# instance.PPP_Ins(INC=4, INS=3)
# DF_ppp, DF_bnc, status = 
run_BnC(my_instance)

# DF_ppp.to_csv("Mantenimientos_Via_"+name+'_'+str(INCUMBENT[0])+".csv")
# DF_bnc.to_csv("Bilev_BnC_summary_"+name+'_'+str(INCUMBENT[0])+".csv")
