from gurobipy import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from math import *
import PPP_Env_V2_MARL
from matplotlib import style
style.use("ggplot")

def follower_PPP(I,x_param):
    

    Follower = Model('Follower_PPP')
    #Follower.setParam('DualReductions', 0)
    Follower.setParam('OutputFlag', 0)
    '''
    FOLLOWER VARIABLES
    '''
   
    x = {t:Follower.addVar(vtype=GRB.BINARY, name="x_"+str(t)) for t in I.T}                             # Whether a maintenance action is applied at t
    y = {t:Follower.addVar(vtype=GRB.INTEGER, name="y_"+str(t)) for t in I.T}                                # Number of periods after last restoration
    b = {(t,tau):Follower.addVar(vtype=GRB.BINARY, name="b_"+str((t,tau))) for t in I.T for tau in I.T}    # Whether yt=tau
    z = {(t,l):Follower.addVar(vtype=GRB.BINARY, name="z_"+str((t,l))) for t in I.T for l in I.L}             # Whether system is at service level l at t
    v = {t:Follower.addVar(vtype=GRB.CONTINUOUS, name="v_"+str(t)) for t in I.T}                            # Performance at t
    pplus = {t:Follower.addVar(vtype=GRB.CONTINUOUS, name="earn_"+str(t)) for t in I.T}             # Earnings at t
    pminus = {t:Follower.addVar(vtype=GRB.CONTINUOUS, name="spend_"+str(t)) for t in I.T}               # Expenditures at t
    pdot = {t:Follower.addVar(vtype=GRB.CONTINUOUS, name="cash_"+str(t)) for t in I.T}              # Money at t
    w = {t:Follower.addVar(vtype=GRB.INTEGER, name="w_"+str(t)) for t in I.T}                           # Linearization of y*x
    u = {t:Follower.addVar(vtype=GRB.CONTINUOUS, name="u_"+str(t)) for t in I.T}                        # Lineartization for v*x
    aux = {(t,l):Follower.addVar(vtype=GRB.BINARY, name="aux_"+str((t,l))) for t in I.T for l in I.L}              # variable for linearization ztl*qt
    '''
    OBJECTIVE
    '''
    #Follower Objective
    Follower.setObjective(-quicksum(pplus[t]-pminus[t] for t in I.T), GRB.MINIMIZE)
    '''
    FOLLOWER CONSTRAINTS
    '''
    #Initialization
    Follower.addConstr(y[0] == 0, "iniY") 
    Follower.addConstr(w[0] == 0, "iniW")   
    Follower.addConstr(u[0] == 0, "iniU") 
    Follower.addConstr(pdot[0] == pplus[0] - pminus[0], "cash_"+str(0)) 
    
    for t in I.T:
        if t>0:   
            # Restoration inventory
            Follower.addConstr(y[t] == y[t-1] + 1 - w[t] - x[t], "inv_"+str(t))
            
            # Linearization of w (for inventory)
            Follower.addConstr(w[t] <= y[t-1], "linW1_"+str(t))
            Follower.addConstr(w[t] >= y[t-1] - len(I.T)*(1-x[t]), "linW2_"+str(t))
            Follower.addConstr(w[t] <= len(I.T)*x[t], "linW3_"+str(t))
            
            # Linearization for v (to get ObjFcn right)
            Follower.addConstr(u[t] <= v[t], "linU1_"+str(t))
            Follower.addConstr(u[t] >= v[t] - (1-x[t]), "linU2_"+str(t))
            Follower.addConstr(u[t] <= x[t], "linU3_"+str(t))
            
            # Update available cash
            Follower.addConstr(pdot[t] == pdot[t-1] + pplus[t] - pminus[t], "cash_"+str(t))
            
        # Mandatory to improve the performance if it is below the minimum
        #HPR.addConstr(v[t] >= I.minP, "minPerf_"+str(t))
        
        # Binarization of y (to retrieve performance)
        Follower.addConstr(y[t] == quicksum(tau*b[t,tau] for tau in I.T), "binY1_"+str(t))
        Follower.addConstr(quicksum(b[t,tau] for tau in I.T) == 1, "binY2_"+str(t))
        
        # Quantification of v (get performance)
        Follower.addConstr(v[t] == quicksum(I.gamma[tau]*b[t,tau] for tau in I.T), "quantV_"+str(t))
        
        # Linearization for service level
        Follower.addConstr(v[t] <= quicksum(I.xi_U[l]*z[t,l] for l in I.L), "rangeU_"+str(t))
        Follower.addConstr(v[t] >= quicksum(I.xi_L[l]*z[t,l] for l in I.L), "rangeL_"+str(t))
        
        # Specification of service-level (ranges)
        Follower.addConstr(quicksum(z[t,l] for l in I.L) == 1, "1_serv_"+str(t))
        
        
        # Profit (budget balance)
        #HPR.addConstr(pplus[t] == I.alpha + I.f[t] + quicksum((I.d[l,t+1]+I.k[t])*z[t,l] for l in I.L), "earn_"+str(t))
        Follower.addConstr(pminus[t] == (I.cf[t]+I.cv[t])*x[t]-I.cv[t]*u[t], "spend_"+str(t))
        Follower.addConstr(pminus[t] <= pdot[t] , "bud_"+str(t))
        
    # Return
    Follower.addConstr(quicksum(pplus[t] for t in I.T) >= (1+I.epsilon)*quicksum(pminus[t] for t in I.T), "return")
    
    #Earnings quicksum(q[t]*z[t,l]*k[l] for l in I.L) linealization
    for t in I.T:
        for l in I.L:
            Follower.addConstr(aux[t,l] <= x_param["q_"+str(t)], name = "binaux1_"+str((t,l)))
            Follower.addConstr(aux[t,l] <= z[t,l], name = "binaux2_"+str((t,l)))
            Follower.addConstr(aux[t,l] >= x_param["q_"+str(t)] + z[t,l] - 1, name = "binaux3_"+str((t,l)))
    
    for t in I.T:
        Follower.addConstr(pplus[t] == I.a + quicksum(aux[t,l]*I.bond[l] for l in I.L), name = "Agents_earnings_"+str(t))
    '''
    for t in I.T:
        for l in I.L:
            Follower.addConstr(x_param["aux_"+str((t,l))] <= z[t,l], name = "binaux2_"+str((t,l)))
            Follower.addConstr(x_param["aux_"+str((t,l))] >= x_param["q_"+str(t)] + z[t,l] - 1, name = "binaux3_"+str((t,l)))
    
    for t in I.T:
        Follower.addConstr(pplus[t] == I.a + quicksum(x_param["q_"+str(t)]*z[t,l]*I.bond[l] for l in I.L), name = "Agents_earnings_"+str(t))
    '''
    Follower.update()
    
    return Follower


'''
private = follower_PPP(I,x_param)

private.optimize()

if private.status == 2:

    print(f'Objective: {private.objVal}')
    
    # for v in private.getVars():
    #     print(f'name: {v.VarName}, value: {v.x}')

    inspection = [x_param[i] for i in x_param.keys()]
    maintenance = [i.x for i in private.getVars() if i.VarName[0] == "x"]
    performance = [i.x for i in private.getVars() if i.VarName[0] == "v"]

    PPP_metrics = {"Inspection": inspection,
                        "Maintenance": maintenance,
                        "Performance": performance}
    df = pd.DataFrame(PPP_metrics, columns = PPP_metrics.keys())

    fig = plt.figure(figsize =(20, 10))
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    corr = 0.4
    Inspection = {t:(t,df["Inspection"][t]) for t in range(df.shape[0]) if df["Inspection"][t] == 1}
    Maintenance = {t:(t,df["Maintenance"][t]) for t in range(df.shape[0]) if df["Maintenance"][t] == 1}
    ax.plot(range(df.shape[0]), df["Performance"], 'k--', linewidth = 1.5, label = "Performance")
    ax.plot([Inspection[t][0] for t in Inspection], [Inspection[t][1] for t in Inspection], 'rs', label = "Inspection actions")
    ax.plot([Maintenance[t][0] for t in Maintenance], [Maintenance[t][1] for t in Maintenance], 'b^', label = "Maintenance actions")
    ax.set_xlabel("Period", size=15)
    ax.set_ylabel("Road's Performance", size=15)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='best', borderaxespad=0.)
    plt.suptitle("Private's perspective" , fontsize=15)
    plt.grid(True)
    #plt.savefig('Leader perspective.png')
    plt.show()
    # plt.close(fig)

aaaaaaaaaaaa
'''

def HPR_PPP(I):
    
    HPR = Model('HPR_PPP')
    
    HPR.setParam('OutputFlag', 0)#setParam("NumericFocus",True)
    '''
    LEADER VARIABLES
    '''
    q = {t:HPR.addVar(vtype=GRB.BINARY, name="q_"+str(t)) for t in I.T}                             # Whether a maintenance action is applied at t
    #r = {(k,l):HPR.addVar(vtype=GRB.BINARY, name="r_"+str((k,l))) for k in I.K for l in I.L}                             # Whether reward k in K is given at service level l in L
    '''
    FOLLOWER VARIABLES
    '''
    x = {t:HPR.addVar(vtype=GRB.BINARY, name="x_"+str(t)) for t in I.T}                             # Whether a maintenance action is applied at t
    y = {t:HPR.addVar(vtype=GRB.INTEGER, name="y_"+str(t)) for t in I.T}					             # Number of periods after last restoration
    b = {(t,tau):HPR.addVar(vtype=GRB.BINARY, name="b_"+str((t,tau))) for t in I.T for tau in I.T}    # Whether yt=tau
    z = {(t,l):HPR.addVar(vtype=GRB.BINARY, name="z_"+str((t,l))) for t in I.T for l in I.L}		      # Whether system is at service level l at t
    v = {t:HPR.addVar(vtype=GRB.CONTINUOUS, name="v_"+str(t)) for t in I.T}							# Performance at t
    pplus = {t:HPR.addVar(vtype=GRB.CONTINUOUS, name="earn_"+str(t)) for t in I.T}				# Earnings at t
    pminus = {t:HPR.addVar(vtype=GRB.CONTINUOUS, name="spend_"+str(t)) for t in I.T}				# Expenditures at t
    pdot = {t:HPR.addVar(vtype=GRB.CONTINUOUS, name="cash_"+str(t)) for t in I.T}				# Money at t
    w = {t:HPR.addVar(vtype=GRB.INTEGER, name="w_"+str(t)) for t in I.T}							# Linearization of y*x
    u = {t:HPR.addVar(vtype=GRB.CONTINUOUS, name="u_"+str(t)) for t in I.T}						# Lineartization for v*x
    #m = {(k,l,t):HPR.addVar(vtype=GRB.BINARY, name="m_"+str((k,l,t))) for k in I.K for l in I.L for t in I.T}                             # Linearization of z*r
    aux = {(t,l):HPR.addVar(vtype=GRB.BINARY, name="aux_"+str((t,l))) for t in I.T for l in I.L}		      # variable for linearization ztl*qt
    
    '''
    OBJECTIVE
    '''
    #Leader objective
    HPR.setObjective(-quicksum(I.g[l]*z[t,l] for l in I.L for t in I.T) + quicksum(q[t]*I.c_sup_i for t in I.T) + quicksum(aux[t,l]*I.bond[l] for t in I.T for l in I.L) + I.a*len(I.T), GRB.MINIMIZE)
    '''
    FOLLOWER CONSTRAINTS
    '''
    #Initialization
    HPR.addConstr(y[0] == 0, "iniY") 
    HPR.addConstr(w[0] == 0, "iniW") 	
    HPR.addConstr(u[0] == 0, "iniU") 
    HPR.addConstr(pdot[0] == pplus[0] - pminus[0], "cash_"+str(0)) 
    
    for t in I.T:
        if t>0:   
            # Restoration inventory
            HPR.addConstr(y[t] == y[t-1] + 1 - w[t] - x[t], "inv_"+str(t))
            
            # Linearization of w (for inventory)
            HPR.addConstr(w[t] <= y[t-1], "linW1_"+str(t))
            HPR.addConstr(w[t] >= y[t-1] - len(I.T)*(1-x[t]), "linW2_"+str(t))
            HPR.addConstr(w[t] <= len(I.T)*x[t], "linW3_"+str(t))
            
            # Linearization for v (to get ObjFcn right)
            HPR.addConstr(u[t] <= v[t], "linU1_"+str(t))
            HPR.addConstr(u[t] >= v[t] - (1-x[t]), "linU2_"+str(t))
            HPR.addConstr(u[t] <= x[t], "linU3_"+str(t))
            
            # Update available cash
            HPR.addConstr(pdot[t] == pdot[t-1] + pplus[t] - pminus[t], "cash_"+str(t))
            
        # Mandatory to improve the performance if it is below the minimum
        #HPR.addConstr(v[t] >= I.minP, "minPerf_"+str(t))
        
        # Binarization of y (to retrieve performance)
        HPR.addConstr(y[t] == quicksum(tau*b[t,tau] for tau in I.T), "binY1_"+str(t))
        HPR.addConstr(quicksum(b[t,tau] for tau in I.T) == 1, "binY2_"+str(t))
        
        # Quantification of v (get performance)
        HPR.addConstr(v[t] == quicksum(I.gamma[tau]*b[t,tau] for tau in I.T), "quantV_"+str(t))
        
        # Linearization for service level
        HPR.addConstr(v[t] <= quicksum(I.xi_U[l]*z[t,l] for l in I.L), "rangeU_"+str(t))
        HPR.addConstr(v[t] >= quicksum(I.xi_L[l]*z[t,l] for l in I.L), "rangeL_"+str(t))
        
        # Specification of service-level (ranges)
        HPR.addConstr(quicksum(z[t,l] for l in I.L) == 1, "1_serv_"+str(t))
        
        
        # Profit (budget balance)
        #HPR.addConstr(pplus[t] == I.alpha + I.f[t] + quicksum((I.d[l,t+1]+I.k[t])*z[t,l] for l in I.L), "earn_"+str(t))
        HPR.addConstr(pminus[t] == (I.cf[t]+I.cv[t])*x[t]-I.cv[t]*u[t], "spend_"+str(t))
        HPR.addConstr(pminus[t] <= pdot[t] , "bud_"+str(t))
        
    # Return
    HPR.addConstr(quicksum(pplus[t] for t in I.T) >= (1+I.epsilon)*quicksum(pminus[t] for t in I.T), "return")
    
    #Earnings quicksum(q[t]*z[t,l]*k[l] for l in I.L) linealization
    for t in I.T:
        for l in I.L:
            HPR.addConstr(aux[t,l] <= q[t], name = "binaux1_"+str((t,l)))
            HPR.addConstr(aux[t,l] <= z[t,l], name = "binaux2_"+str((t,l)))
            HPR.addConstr(aux[t,l] >= q[t] + z[t,l] - 1, name = "binaux3_"+str((t,l)))
    
    for t in I.T:
        HPR.addConstr(pplus[t] == I.a + quicksum(aux[t,l]*I.bond[l] for l in I.L), name = "Agents_earnings_"+str(t))
    
    '''
    LEADER CONSTRAINTS
    '''
    
    #Leader budget
    #HPR.addConstr(quicksum(q[t]*I.c_sup_i for t in I.T) <= I.Beta, "Leader_budget")
    
    #Minimum social profit
    for t in I.T:
        HPR.addConstr(quicksum(I.g[l]*z[t,l] for l in I.L) >= I.g_star, name = "social_profit_" + str(t))
    
    
    HPR.update()

    return HPR

def HPR_PPP_std(I):
    
    HPR_std = Model('HPR_PPP_std')
    
    HPR_std.setParam('OutputFlag', 0)#setParam("NumericFocus",True)
    '''
    LEADER VARIABLES
    '''
    q = {t:HPR_std.addVar(vtype=GRB.BINARY, name="q_"+str(t)) for t in I.T}                             # Whether a maintenance action is applied at t
    #r = {(k,l):HPR.addVar(vtype=GRB.BINARY, name="r_"+str((k,l))) for k in I.K for l in I.L}                             # Whether reward k in K is given at service level l in L
    '''
    FOLLOWER VARIABLES
    '''
    x = {t:HPR_std.addVar(vtype=GRB.BINARY, name="x_"+str(t)) for t in I.T}                             # Whether a maintenance action is applied at t
    y = {t:HPR_std.addVar(vtype=GRB.INTEGER, name="y_"+str(t)) for t in I.T}					             # Number of periods after last restoration
    b = {(t,tau):HPR_std.addVar(vtype=GRB.BINARY, name="b_"+str((t,tau))) for t in I.T for tau in I.T}    # Whether yt=tau
    z = {(t,l):HPR_std.addVar(vtype=GRB.BINARY, name="z_"+str((t,l))) for t in I.T for l in I.L}		      # Whether system is at service level l at t
    v = {t:HPR_std.addVar(vtype=GRB.CONTINUOUS, name="v_"+str(t)) for t in I.T}							# Performance at t
    pplus = {t:HPR_std.addVar(vtype=GRB.CONTINUOUS, name="earn_"+str(t)) for t in I.T}				# Earnings at t
    pminus = {t:HPR_std.addVar(vtype=GRB.CONTINUOUS, name="spend_"+str(t)) for t in I.T}				# Expenditures at t
    pdot = {t:HPR_std.addVar(vtype=GRB.CONTINUOUS, name="cash_"+str(t)) for t in I.T}				# Money at t
    w = {t:HPR_std.addVar(vtype=GRB.INTEGER, name="w_"+str(t)) for t in I.T}							# Linearization of y*x
    u = {t:HPR_std.addVar(vtype=GRB.CONTINUOUS, name="u_"+str(t)) for t in I.T}						# Lineartization for v*x
    #m = {(k,l,t):HPR.addVar(vtype=GRB.BINARY, name="m_"+str((k,l,t))) for k in I.K for l in I.L for t in I.T}                             # Linearization of z*r
    aux = {(t,l):HPR_std.addVar(vtype=GRB.BINARY, name="aux_"+str((t,l))) for t in I.T for l in I.L}		      # variable for linealization zlt*qt
    
    '''
    OBJECTIVE
    '''
    #Leader objective
    HPR_std.setObjective(-quicksum(I.g[l]*z[t,l] for l in I.L for t in I.T) + quicksum(q[t]*I.c_sup_i for t in I.T) + quicksum(aux[t,l]*I.bond[l] for t in I.T for l in I.L) + I.a*len(I.T), GRB.MINIMIZE)
    '''
    FOLLOWER CONSTRAINTS
    '''
    #Initialization
    HPR_std.addConstr(y[0] == 0, "iniY") 
    HPR_std.addConstr(w[0] == 0, "iniW") 	
    HPR_std.addConstr(u[0] == 0, "iniU") 
    HPR_std.addConstr(pdot[0] == pplus[0] - pminus[0], "cash_"+str(0)) 
    
    for t in I.T:
        if t>0:   
            # Restoration inventory
            HPR_std.addConstr(y[t] == y[t-1] + 1 - w[t] - x[t], "inv_"+str(t))
            
            # Linearization of w (for inventory)
            {t:HPR_std.addVar(vtype=GRB.CONTINUOUS, name="s_linW1_"+str(t))}
            {t:HPR_std.addVar(vtype=GRB.CONTINUOUS, name="s_linW2_"+str(t))}
            {t:HPR_std.addVar(vtype=GRB.CONTINUOUS, name="s_linW3_"+str(t))}
            HPR_std.update()
            HPR_std.addConstr(w[t] + HPR_std.getVarByName("s_linW1_"+str(t)) == y[t-1], "linW1_"+str(t))
            HPR_std.addConstr(w[t] - HPR_std.getVarByName("s_linW2_"+str(t)) == y[t-1] - len(I.T)*(1-x[t]), "linW2_"+str(t))
            HPR_std.addConstr(w[t] + HPR_std.getVarByName("s_linW3_"+str(t)) == len(I.T)*x[t], "linW3_"+str(t))
            
            # Linearization for v (to get ObjFcn right)
            {t:HPR_std.addVar(vtype=GRB.CONTINUOUS, name="s_linU1_"+str(t))}
            {t:HPR_std.addVar(vtype=GRB.CONTINUOUS, name="s_linU2_"+str(t))}
            {t:HPR_std.addVar(vtype=GRB.CONTINUOUS, name="s_linU3_"+str(t))}
            HPR_std.update()
            HPR_std.addConstr(u[t] + HPR_std.getVarByName("s_linU1_"+str(t)) == v[t], "linU1_"+str(t))
            HPR_std.addConstr(u[t] - HPR_std.getVarByName("s_linU2_"+str(t)) == v[t] - (1-x[t]), "linU2_"+str(t))
            HPR_std.addConstr(u[t] + HPR_std.getVarByName("s_linU3_"+str(t)) == x[t], "linU3_"+str(t))
            
            # Update available cash
            HPR_std.addConstr(pdot[t] == pdot[t-1] + pplus[t] - pminus[t], "cash_"+str(t))
            
        # Mandatory to improve the performance if it is below the minimum
        #HPR.addConstr(v[t] >= I.minP, "minPerf_"+str(t))
        
        # Binarization of y (to retrieve performance)
        HPR_std.addConstr(y[t] == quicksum(tau*b[t,tau] for tau in I.T), "binY1_"+str(t))
        HPR_std.addConstr(quicksum(b[t,tau] for tau in I.T) == 1, "binY2_"+str(t))
        
        # Quantification of v (get performance)
        HPR_std.addConstr(v[t] == quicksum(I.gamma[tau]*b[t,tau] for tau in I.T), "quantV_"+str(t))
        
        # Linearization for service level
        HPR_std.addVar(vtype=GRB.CONTINUOUS, name="s_rangeU_"+str(t))
        HPR_std.addVar(vtype=GRB.CONTINUOUS, name="s_rangeL_"+str(t))
        HPR_std.update()
        HPR_std.addConstr(v[t] + HPR_std.getVarByName("s_rangeU_"+str(t)) == quicksum(I.xi_U[l]*z[t,l] for l in I.L), "rangeU_"+str(t))
        HPR_std.addConstr(v[t] - HPR_std.getVarByName("s_rangeL_"+str(t)) == quicksum(I.xi_L[l]*z[t,l] for l in I.L), "rangeL_"+str(t))
        
        # Specification of service-level (ranges)
        HPR_std.addConstr(quicksum(z[t,l] for l in I.L) == 1, "1_serv_"+str(t))
        
        
        # Profit (budget balance)
        #HPR.addConstr(pplus[t] == I.alpha + I.f[t] + quicksum((I.d[l,t+1]+I.k[t])*z[t,l] for l in I.L), "earn_"+str(t))
        HPR_std.addConstr(pminus[t] == (I.cf[t]+I.cv[t])*x[t]-I.cv[t]*u[t], "spend_"+str(t))
        HPR_std.addConstr(pminus[t] <= pdot[t] , "bud_"+str(t))
        
    # Return
    HPR_std.addVar(vtype=GRB.CONTINUOUS, name="s_return")
    HPR_std.update()
    HPR_std.addConstr(quicksum(pplus[t] for t in I.T) - HPR_std.getVarByName("s_return") == (1+I.epsilon)*quicksum(pminus[t] for t in I.T), "return")
    
    #Earnings quicksum(q[t]*z[t,l]*k[l] for l in I.L) linealization
    for t in I.T:
        for l in I.L:
            
            {(t,l):HPR_std.addVar(vtype=GRB.CONTINUOUS, name="s_binaux1_"+str((t,l)))}
            {(t,l):HPR_std.addVar(vtype=GRB.CONTINUOUS, name="s_binaux2_"+str((t,l)))}
            {(t,l):HPR_std.addVar(vtype=GRB.CONTINUOUS, name="s_binaux3_"+str((t,l)))}
            HPR_std.update()
            HPR_std.addConstr(aux[t,l] + HPR_std.getVarByName("s_binaux1_"+str((t,l))) == q[t], name = "binaux1_"+str((t,l)))
            HPR_std.addConstr(aux[t,l] + HPR_std.getVarByName("s_binaux2_"+str((t,l))) == z[t,l], name = "binaux2_"+str((t,l)))
            HPR_std.addConstr(aux[t,l] - HPR_std.getVarByName("s_binaux3_"+str((t,l))) == q[t] + z[t,l] - 1, name = "binaux3_"+str((t,l)))
    
    for t in I.T:
        HPR_std.addConstr(pplus[t] == I.a + quicksum(aux[t,l]*I.bond[l] for l in I.L), name = "Agents_earnings_"+str(t))
    
    '''
    LEADER CONSTRAINTS
    '''
    
    #Leader budget
    #HPR.addConstr(quicksum(q[t]*I.c_sup_i for t in I.T) <= I.Beta, "Leader_budget")
    
    #Minimum social profit
    for t in I.T:
        {t:HPR_std.addVar(vtype=GRB.CONTINUOUS, name="s_social_profit_"+str(t))}
        HPR_std.update()
        HPR_std.addConstr(quicksum(I.g[l]*z[t,l] for l in I.L) - HPR_std.getVarByName("s_social_profit_"+str(t)) == I.g_star, name = "social_profit_" + str(t))
    
    
    HPR_std.update()
    
    return HPR_std

# class Instance():

#     def get_level(self, perf):

#             if perf < .2:
#                 return 1
#             elif perf < .4:
#                 return 2
#             elif perf < .6:
#                 return 3
#             elif perf < .8:
#                 return 4
#             else:
#                 return 5

#     def incentive(self, perf, choice='sigmoid'):

#         # Samuel has discretized the function according to the performance level
#         if self.W[1] == 0:
#             return 0

#         if choice == 'sigmoid':
#             rate, offset = 10, self.threshold
#             incent = 1/(1 + exp(-rate*(perf-offset)))

#         elif choice == 'linear':
#             slope, offset = 1, 0
#             incent = offset + slope*perf

#         return incent

#     def __init__(self):
#         self.T = range(30) # Planning horizon
#         self.W = [0, 1] # [shock?, inspection?]
#         self.NUM_INTERVALS = 5
#         # discrete levels of performance
#         self.L = range(1, self.NUM_INTERVALS + 1)
#         # Performance treshold (discrete)
#         self.threshold = 0.6
#         '''
#         Necessary parmeters to model the deterioration
#         '''
#         self.fail = .2 # failure threshold (continuous)
#         self.ttf = 10.0 # time to failure
#         self.Lambda = -log(self.fail)/self.ttf
#         self.gamma = [exp(-self.Lambda*tau) for tau in range(len(self.T)+2)]
#         # perf_tt = exp(-Lambda*tt) --> Lambda = -ln(perf_tt)/tt

#         '''Princpipal'''
#         self.g = {1: 2, 2: 47, 3: 500, 4: 953, 5: 998}
#         self.g_star = 595


#         '''Agent'''
#         self.FC = 3 # 4.4663 Fixed maintenance cost
#         self.VC = self.FC # Variable maintenance cost

#         '''delta = [exp(-I.Lambda*tau) - exp(-I.Lambda*(tau-1)) for tau in range(1,I.T)]
                    
#                     gamma_2 = [1]
#                     for tau in range(1,I.T):
#                         gamma_2.append(gamma_2[tau-1]+delta[tau-1])
                
#                     print(gamma_1==gamma_2)
#                     aaaaa
#                     '''

#         self.cf = [self.FC for _ in range(len(self.T))]
#         self.cv = [self.VC for _ in range(len(self.T))]

#         self.xi_L = {1:0, 2:.21, 3:.41, 4:.61, 5:.81}
#         self.xi_U = {1:.2, 2:.4, 3:.6, 4:.8, 5:1}   
#         self.bond = {}
#         self.c_sup_i = 70
#         self.a = 1
#         self.epsilon = 1.4
#         for level in self.L:
#             average_l = 0
#             count_l = 0
#             for gamma_val in self.gamma:
#                 if self.get_level(gamma_val) == level:
#                     average_l += 7*self.incentive(gamma_val) 
#                     count_l += 1
#             self.bond[level] = average_l/count_l

# I = Instance()
# min_social_ben = [I.g_star for i in range(len(I.T))]
# leader = HPR_PPP(I)
# leader.optimize()
# print(f'Leader status: {leader.status}, objVal: {leader.objVal}')
# inspection = [v.x for v in leader.getVars() if v.varName[0] == "q"]
# maintenance = [v.x for v in leader.getVars() if v.varName[0] == "x"]
# performance = [v.x for v in leader.getVars() if v.varName[0] == "v"]
# #social_ben = {v.varName:v.x for v in leader.getVars() if v.varName[0] == "z"}
# social_ben_per_period_leader = [sum(I.g[l]*leader.getVarByName('z_'+str((t,l))).x for l in I.L) for t in range(len(I.T))]
# # no_periods_last_maintenance = [SOL[INCUMBENT[0]][i] for i in SOL[INCUMBENT[0]].keys() if i[0] == "y"]
# # pplus = [SOL[INCUMBENT[0]][i] for i in SOL[INCUMBENT[0]].keys() if i[0] == "e"]
# # pminus = [SOL[INCUMBENT[0]][i] for i in SOL[INCUMBENT[0]].keys() if i[0] == "s"]
# # pdot = [SOL[INCUMBENT[0]][i] for i in SOL[INCUMBENT[0]].keys() if i[0] == "c"]


# PPP_metrics = {"Inspection": inspection,
#             "Maintenance": maintenance,
#             "Performance": performance,
#             "Social_ben":social_ben_per_period_leader}#,
#             # "# of periods since last maintenance": no_periods_last_maintenance,
#             # "Earnings": pplus,
#             # "Expenditures": pminus,
#             # "Available cash": pdot}
# DF_ppp = pd.DataFrame(PPP_metrics, columns = PPP_metrics.keys())
# DF_ppp.to_csv("Mantenimientos_Via_leader.csv")


# x_param = {v.varName:v.x for v in leader.getVars()}
# follower = follower_PPP(I,x_param)
# dt = {i.VarName:i.obj for i in follower.getVars()}
# follower.optimize()
# maintenance = [v.x for v in follower.getVars() if v.varName[0] == "x"]
# performance = [v.x for v in follower.getVars() if v.varName[0] == "v"]
# # no_periods_last_maintenance = [SOL[INCUMBENT[0]][i] for i in SOL[INCUMBENT[0]].keys() if i[0] == "y"]
# # pplus = [SOL[INCUMBENT[0]][i] for i in SOL[INCUMBENT[0]].keys() if i[0] == "e"]
# # pminus = [SOL[INCUMBENT[0]][i] for i in SOL[INCUMBENT[0]].keys() if i[0] == "s"]
# # pdot = [SOL[INCUMBENT[0]][i] for i in SOL[INCUMBENT[0]].keys() if i[0] == "c"]
# #social_ben = {v.varName:v.x for v in follower.getVars() if v.varName[0] == "z"}
# social_ben_per_period_follower = [sum(I.g[l]*follower.getVarByName('z_'+str((t,l))).x for l in I.L) for t in range(len(I.T))]

# PPP_metrics = {"Inspection": inspection,
#             "Maintenance": maintenance,
#             "Performance": performance,
#             "Social_ben":social_ben_per_period_follower}#,
#             # "# of periods since last maintenance": no_periods_last_maintenance,
#             # "Earnings": pplus,
#             # "Expenditures": pminus,
#             # "Available cash": pdot}
# DF_ppp = pd.DataFrame(PPP_metrics, columns = PPP_metrics.keys())
# DF_ppp.to_csv("Mantenimientos_Via_follower.csv")

# print(f'follower status: {follower.status}')
# ##############################################################################################PRINT ZL FOR SHOWING THE INFEASIBILITY IN THE PAPEEEEEEER

# if follower.status == 2:
    
#     y_hat = {v.varName:v.x for v in follower.getVars()}
#     phi_x = follower.objVal
#     leader_check = leader.copy()
#     lead_vars = [i.VarName for i in leader_check.getVars() if i.VarName not in dt.keys()]
    
#     for var in lead_vars:
#         leader_check.addConstr(leader_check.getVarByName(var) == x_param[var])
#         leader_check.update()
    
#     leader_check.addConstr(quicksum(leader_check.getVarByName(var)*dt[var] for var in dt.keys()) <= phi_x)
#     leader_check.update()

#     leader_check.optimize()

#     print(f'leader check status: {leader_check.status}')
