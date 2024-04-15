# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 10:49:25 2023

@author: basti
"""
import sys

from itertools import zip_longest
import pandas as pd
import urllib.request
import pandas_compat as pdc
from io import StringIO
import requests
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum, Model
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
# import winsound
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.model_selection import KFold  # import KFold
import math
import os
import glob
import argparse



# =============================================================================
# utils
# =============================================================================

def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

def contrary(selected):
    route = []
    a = 0
    i = 0
    while len(selected)-1 > i:
        for ruta in selected:
            if ruta[0] == a:
                route.append(ruta[1])
                a = ruta[1]
                i = i+1
                break

    nonselected = []
    for num in range(11):  # Iterate over numbers 0 to 10
        if num not in route:
            nonselected.append(num)
    return nonselected

# =============================================================================
# Master Problem
# =============================================================================

def MP_IO_PSI(X, Xtong, y, Lambda, structuredloss):
    #X,Xtong,y,Lambda,structuredloss= X,Xt,Y,Lambda,structuredloss
   
    c0 = X[0][12]  # [8, 19, 51, 22, 60]
    D = len(y)
    L = len(c0)

    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()

        with gp.Model("MP", env=env) as m:

            # m = gp.Model("MP")
            m.setParam('TimeLimit', 5*60)

            # variables
            c = m.addVars(L, name="c", lb=-GRB.INFINITY)
            cd = m.addVars(D, L, name="cd", lb=-GRB.INFINITY)
            Xi = m.addVars(D, name="Xi")

            z = m.addMVar(L, name="z", lb=-GRB.INFINITY)
            zf = m.addVars(D, L, name="zf", lb=-GRB.INFINITY)

            absdif = m.addVars(L, name="absdif")
            absdiff = m.addVars(D, L, name="absdiff")

            for j in range(0, L):
                m.addConstr(c0[j]-c[j] == z[j])
                m.addConstr(z[j] <= absdif[j])
                m.addConstr(-absdif[j] <= z[j])

            for i in range(0, D):
                for j in range(0, L):
                    m.addConstr(zf[i, j] == c[j]-cd[i, j])
                    m.addConstr(zf[i, j] <= absdiff[i, j])
                    m.addConstr(-absdiff[i, j] <= zf[i, j])

            m.addConstr(quicksum(c) == quicksum(c0))

            ODMatrix = X[0][11]

            for d in range(0, D):
                if len(Xtong[d]) != 0:
                    for minixhat in Xtong[d]:
                        m.addConstr(quicksum(cd[d, ODM]*ODMatrix[ODM].loc[O].iloc[De] for O, De in flatten_list(y[d]) for ODM in range(0, len(ODMatrix))) + cd[d, 4]*quicksum(X[0][7][ii] for ii in contrary(y[d])) - quicksum(
                            cd[d, ODM]*ODMatrix[ODM].loc[O].iloc[De] for O, De in flatten_list(minixhat) for ODM in range(0, len(ODMatrix))) - cd[d, 4]*quicksum(X[0][7][ii] for ii in contrary(minixhat)) - Xi[d] <= 0)

            m.setObjective(absdif.sum()+Lambda[0]/D*absdiff.sum() + Lambda[1]*Xi.sum()/D, GRB.MINIMIZE)
            m.optimize()

            ctong = [c[t].X for t in range(0, L)]
            cdtong = [[cd[i, j].X for j in range(0, L)] for i in range(0, D)]

            return ctong, cdtong, [Xi[i].X for i in range(0, D)]


# =============================================================================
# Forward Problem
# =============================================================================

def solvePCTSP(X, c, structuredloss, Y):
    
    #X,c,structuredloss,Y = X,True_Weight_Matrix,0,0
    #X,c,structuredloss,Y = X[d],cdtong[d],structuredloss,Y[d]
    N, V, K, Q, M, q, Prize_collected, Cost_Penalty, True_Weight_Matrix, Minimum_Prize, Ma, SubObj, num, c0 = X

    cW = c[0:4]  # cW,pW = [35,33,15,17], 23
    pW = c[4]

    c3 = [[0 for j in range(0, len(V))] for i in range(0, len(V))]
    for i in range(0, len(cW)):
        for j in range(0, len(V)):
            for k in range(0, len(V)):
                c3[j][k] = c3[j][k] + cW[i]*SubObj[i].loc[j].iloc[k]
    c3 = list(np.array(c3).reshape((len(V)*len(V))))
    cdict = dict(zip([(i, j) for i in range(0, int(len(V)))
                 for j in range(0, int(len(V)))], c3))

    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()

        with gp.Model("VRP", env=env) as mdl:

            # mdl = Model("VRP")

            '''Decision Variable'''
            # Arcs
            x = mdl.addVars([(i, j) for i in V for j in V], vtype=GRB.BINARY, name="x")

            '''Constraints'''
            # Flow
            mdl.addConstrs(quicksum(x[i, j] for i in V) -
                           quicksum(x[j, i] for i in V) == 0 for j in V)

            for i in V:
                mdl.addConstr(quicksum(x[i, j] for j in V) <= 1)
                mdl.addConstr(quicksum(x[j, i] for j in V) <= 1)

            # start from warehouse
            mdl.addConstr(quicksum(x[0, j] for j in V) == 1)

            # Prize collected
            mdl.addConstr(quicksum(x[i, j]*Prize_collected[j]
                          for i in V for j in V) >= Minimum_Prize)

            # you dont go to yourself"
            mdl.addConstrs(x[i, i] == 0 for i in V)

            u = mdl.addVars([i for i in V], name="u")
            mdl.addConstrs(u[j]-u[i] >= q[i] - Q[1]*(1-x[i, j]) for i in N for j in N)
            mdl.addConstrs(q[i] <= u[i] for i in N)
            mdl.addConstrs(u[i] <= Q[1] for i in N)

            ''''Objective Function'''
            mdl.setObjective(quicksum(x[i, j]*cdict[(i, j)] for i in V for j in V) + quicksum(
                pW*(1-x[i, j])*Cost_Penalty[j] for i in N for j in N), GRB.MINIMIZE)

            '''Solve'''
            mdl.Params.MIPGap = 0
            mdl.Params.TimeLimit = 30  # seconds

            mdl.optimize()

            # retrieve solution
            vals = mdl.getAttr('x', x)
            selected = [gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)[i] for i in range(round(sum(vals.values())))]

            return selected


# =============================================================================
# Code 
# =============================================================================

def train_wCVRP(X,Y,Vcount,Lambda,structuredloss,epsilon,lossfun,breakIO):
    #1: Initialize  Vcount=100 
    TimeStart = time.time() 
    D = len(Y)
    counte=0
    countv=0
    Xt=[[] for i in range(0,D)]
    rep = [0 for i in range(0,D)]
    ctong, cdtong = X[0][12], [X[0][12] for d in range(D)]
    Xi = [0 for i in range(0,D)]
    EvolLoss = []
    cutcounter = []
    TimeMP = []
    TimeFP = []
    TimeEval = [] 
    FP_call = 0
    MP_call = 0
    At = []
    AAt = []
    timelimit = 60
    #2/3/4
    init_c = list(ctong)
    print("first ctong: ", init_c)
    iter = 0
    while counte < D and timelimit > time.time()-TimeStart:
        print()
        print()
        print("restart checking the new weights against the training instances for the ", iter+1, "th time")
        counte = 0
        countv = 0 
        count2 = 0      

        #7 for each instance in the dataset
        for d in range(0,D):
            print()
            #8: Solve subroutine (Solving the forward problem and delivering another potential route with lower cost for that c)
            timeFP_d = time.time()
            FP_call = FP_call+1
            print('training instance: ', d+1, 'of: ', D)
            yhat = fwd_PCTSP(X[d],ctong,cdtong[d],structuredloss,Y[d])
            print("data_name: ", X[d][-1], "yhat: ", yhat)
            aftertimeFP_d = time.time()
            TimeFP.append((aftertimeFP_d-timeFP_d)/60)
            loss_val = lossfun(X,Y[d],yhat,ctong,cdtong[d],structuredloss,Xi[d])
            print("found yhat: ", yhat)
            print("real y: ", Y[d])
            print("loss_val: ", loss_val)

            #12: If we have found we add it as a plane
            if loss_val>epsilon+0.000001:
                countv = countv + 1
                rep[d] = rep[d] +1
                Xt[d].append(yhat)
                if yhat not in At:
                    At.append(yhat)
            #9: If we havent found a route with a better objective for this cost, skip
            else:
                counte = counte + 1
                count2 = count2 + 1

            print("countv: ", countv, "over: ", Vcount, "count2+countv: ", count2+countv, "over: ", D)
            #when we identify Vcount possible cuts
            if countv == Vcount or count2+countv == D:
                #5: Solve MP and make ctong and cdtong its optimal solution
                timeMP_d = time.time()
                MP_call = MP_call + 1
                print("Solving the Master Problem to find new weights")
                print("old ctong: ", ctong)
                ctong, cdtong, Xi = MP_IO_PSI(X,Xt,Y,Lambda ,structuredloss)
                print("new ctong: ", ctong)
                aftertimeMP_d = time.time()
                TimeMP.append((aftertimeMP_d-timeMP_d)/60)
                if breakIO:  # if we want to break after each IO
                    break
                else:
                    count2 = count2+countv
                    countv = 0

            if time.time()-TimeStart > timelimit:
                break

        iter += 1
        print("counte: ", counte)

        
    if time.time()-TimeStart > timelimit:
         print("Over Time Limit")
         # winsound.Beep(622, 200)
         # winsound.Beep(523, 200)
    # else:
    #      winsound.Beep(523, 200)
    #      winsound.Beep(659, 200)
    
    TotalTime = (time.time() - TimeStart)/60

    print("final weights: ", ctong)
    # print("final cdtong: ", cdtong)
    print("initial weights: ", init_c)
    
    return ctong, EvolLoss, TotalTime, sum(TimeMP), sum(TimeFP), TimeEval,cutcounter,FP_call,MP_call, Xt ,AAt


# =============================================================================
# Create data from artificial data in GitHub
# =============================================================================

def noise_weight_matrix(Matrix, noise):
    c0 = [0 for i in Matrix]
    for i in range(len(Matrix)):
        c0[i] = Matrix[i] * (1+np.random.uniform(-noise, noise))
    return c0


def createPCTSPdata(text, n):

    content = text

    L = len(content.split('\n')[0].split())
    L2 = len(content.split('\n')[6].split())

    Prize_collected = [0]+[int(content.split('\n')[0].split()[i])
                           for i in range(L)]

    Weight_Penalty = int(content.split('\n')[2].split()[0])

    Cost_Penalty = [0] + [int(content.split('\n')[4].split()[i])
                          for i in range(L)]

    True_Weight_Matrix = [int(content.split('\n')[6].split()[i])
                          for i in range(L2)]

    True_Weight_Matrix.append(Weight_Penalty)

    Minimum_Prize = int(content.split('\n')[8].split()[0])

    def FormatMatrix(MatrixA):
        MatrixA = MatrixA.rename(index=dict(
            enumerate(range(1, 11))), columns=dict(enumerate(range(1, 11))))
        MatrixA.loc['0'] = 0
        MatrixA['0'] = 0
        MatrixA = MatrixA[list(MatrixA.columns[-1:]) +
                          list(MatrixA.columns[:-1])]
        MatrixA = MatrixA.reindex(range(11))
        MatrixA.loc[0] = 0
        return MatrixA

    MatrixA = pd.DataFrame(pd.DataFrame([int(content.split('\n')[10:20][j].split(
        '\t')[i]) for j in range(10) for i in range(10)]).values.reshape(10, 10))
    MatrixA = FormatMatrix(MatrixA)

    MatrixB = pd.DataFrame(pd.DataFrame([int(content.split('\n')[21:31][j].split(
        '\t')[i]) for j in range(10) for i in range(10)]).values.reshape(10, 10))
    MatrixB = FormatMatrix(MatrixB)

    MatrixC = pd.DataFrame(pd.DataFrame([int(content.split('\n')[32:42][j].split(
        '\t')[i]) for j in range(10) for i in range(10)]).values.reshape(10, 10))
    MatrixC = FormatMatrix(MatrixC)

    MatrixD = pd.DataFrame(pd.DataFrame([int(content.split('\n')[43:53][j].split(
        '\t')[i]) for j in range(10) for i in range(10)]).values.reshape(10, 10))
    MatrixD = FormatMatrix(MatrixD)

    SubObj = [MatrixA, MatrixB, MatrixC, MatrixD]

    N = [i for i in range(0, n)]
    #Clients and warehouse
    V = N + [n]
    # vehicle type
    K = [1]
    # capacities
    Q = {1: 1000}
    # number of vehicles
    M = {1: 1}
    # demand
    q = {i: np.random.randint(1, 2) for i in range(0, n+1)}

    X = [N, V, K, Q, M, q, Prize_collected, Cost_Penalty,
         True_Weight_Matrix, Minimum_Prize, MatrixA, SubObj, 0, 0]

    return X, solvePCTSP(X, True_Weight_Matrix, 0, 0)

# =============================================================================
# Loss Functions
# =============================================================================

def lossfun_IO_PCTSP(X, Y1, Y2, c, cd, structloss, Xi):
    #X,Y1,Y2,ctong,cd,structloss,Xi =X,Y[d],yhat,cdtong[d],cdtong[d],structuredloss,Xi[d]
    return lossfun_PCTSP(X, Y1, Y2, cd, structloss, Xi)


def lossfun_PCTSP(X, Y1, Y2, c, structloss, Xi):
    #c=cdtong[d]
    #X,Y1,Y2,c,structloss,Xi =  X,Y[d],yhat,cdtong[d],structuredloss,Xi[d]
    #X,flatten_list(Y),flatten_list(XXt),ctong,structuredloss,Xi
    
    C_P = X[0][7]
    cW = c[0:4]  # cW,pW = [35,33,15,17], 23
    pW = c[4]

    V, SubObj = X[0][1], X[0][11]

    c3 = [[0 for j in range(0, len(V))] for i in range(0, len(V))]
    for i in range(0, len(cW)):
        for j in range(0, len(V)):
            for k in range(0, len(V)):
                c3[j][k] = c3[j][k] + cW[i]*SubObj[i].loc[j].iloc[k]
    c3 = list(np.array(c3).reshape((len(V)*len(V))))
    cdict = dict(zip([(i, j) for i in range(0, int(len(V)))
                 for j in range(0, int(len(V)))], c3))

    sumahat = 0
    for ruta in flatten_list(Y1):
        sumahat = sumahat + (cdict[ruta])

    for ruta in contrary(Y1):
        sumahat = sumahat + c[4] * C_P[ruta]*pW

    sumat = 0
    for ruta in flatten_list(Y2):
        sumat = sumat + (cdict[ruta]) 

    for ruta in contrary(Y2):
        sumat = sumat + c[4] * C_P[ruta]*pW

    if type(structloss) == int:
        loss = sumahat - sumat - Xi
    else:
        loss = sumahat - sumat + structloss(Y1, Y2) - Xi

    return loss


def fwd_PCTSP(a, b, c, d=0, e=0):
    return solvePCTSP(a, c, d, e)



# =============================================================================
# Eval solution
# =============================================================================

def comparePCTSP(y1, y2):
    evalsol = []
    for i in range(len(y1)):
        evalsol.append(sum(c1 != c2 for c1, c2 in zip_longest(y1[i], y2[i])))
    return evalsol

def deliversol(x, c):
    YSOL = []
    for i in range(len(x)):
        Y_sol = solvePCTSP(x[i], c, 0, 0)
        YSOL.append(Y_sol)
    return YSOL


# =============================================================================
# Args
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-folder_path", "--folder_path",default='C:/Users/basti/Downloads/PreferenceDrivenOptimization-master/PreferenceDrivenOptimization-master/solvers/PCTSP_instances/size_10/')
    parser.add_argument("-n", "--n",default = 10)
    parser.add_argument("-noise", "--noise",default = 0.8)
    parser.add_argument("-Lambda", "--Lambda",default = [100,100])
    parser.add_argument("-Vcount", "--Vcount",default = 10)
    parser.add_argument("-ResetDataset", "--ResetDataset",action='store_true')
    
    args = parser.parse_args()
    
    return args

# =============================================================================
#
# =============================================================================


if __name__ == '__main__':

    args = parse_args()
    print(args)
    # sys.exit(0)
    seed = 5
    tsize = 0.8


    n = args.n    
    noise = args.noise
    bio = args.ResetDataset
    Vcount = args.Vcount
    LambdaIO = args.Lambda
    
    # Set the path of the folder containing text files
    folder_path = "../instances/pctsp/size_10/"
    
    # Use the glob module to get a list of all text files in the folder
    text_files = glob.glob(os.path.join(folder_path, "*.txt"))
    
    dataX, dataY = [], []

    # Loop through the list of text files and read them one by one
    for ind, file_path in enumerate(text_files):
        print("Reading and solving instance {} over {}".format(ind+1, len(text_files)))
        with open(file_path, "r") as f:
            text = f.read()
            dataXd, dataYd = (createPCTSPdata(text, n))
            dataX.append(
                dataXd + [(file_path[-7:][:3]).replace("_", "0").replace("a", "0")])
            dataY.append(
                [dataYd] + [(file_path[-7:][:3]).replace("_", "0").replace("a", "0")])
            print("data_name: ", dataX[-1][-1], "yhat: ", dataY[-1][0])
            print()
    
    for dataelem in dataX:
        dataelem.pop(13)
    
    if noise == "unif":
        c0 = [0.01, 0.01, 0.01, 0.01, 0.01]
    elif noise == "zero":
        c0 = [0, 0, 0, 0, 0] 
    else:
        c0 = noise_weight_matrix(dataX[0][8], noise)

    for dataelem in dataX:
        dataelem[12] = c0

    data0 = dataX, dataY

    data = [dataX[i] for i in range(len(dataY))], [
        dataY[i][0] for i in range(len(dataY))]

    dataX_train, dataX_test = train_test_split(data[0], train_size=tsize, random_state=42)
    dataY_train, dataY_test = train_test_split(data[1], train_size=tsize, random_state=42)
    # dataX_train, dataX_test = train_test_split(data[0], train_size=tsize, shuffle=False)
    # dataY_train, dataY_test = train_test_split(data[1], train_size=tsize, shuffle=False)

    c_trained, IO_eLoss, tt, t1, t2, t3, cc, fp_call, mp_call, Xt, At = train_wCVRP(X=dataX_train, Y=dataY_train, Vcount=Vcount, Lambda=LambdaIO, structuredloss=0, epsilon=0, lossfun=lossfun_IO_PCTSP, breakIO=bio)
    
    Result = {'Type': "IOn", "Lambda": LambdaIO, "D": len(dataX), "Vcount": Vcount, "tt": tt, "c_trained": c_trained, "t1": t1, "t2": t2, "c0": c0, "noise": noise,"n": n, "breakIO": bio, "fpcall": fp_call, "mpcall": mp_call, 'eval': sum(comparePCTSP(dataY_test, deliversol(dataX_test, c_trained)))/len(dataY_test)}
    # print(Result)
    
    '''

c_trained_unif = [0.0017339023683498056,0.004002428344806245, 0.006349015389584025, 0.004727245586233617, 0.03318740831102631]
c_trained_ = [7.947194168771232, 19.135752259037766,50.63767039649641, 21.785728099508113, 60.008306975651244]

sol_cTrue = deliversol(dataX_test, [8, 19, 51, 22, 60])
sol_c0 = deliversol(dataX_test, c0)

sol1 = deliversol(dataX_test, w_sol1)
sol2 = deliversol(dataX_test, w_sol2)
sol_ctrained = deliversol(dataX_test, c_trained)
sol_ctrained_unif = deliversol(dataX_test, c_trained_unif)


comparePCTSP(dataY_test, sol_cTrue)
comparePCTSP(dataY_test, sol_c0)
comparePCTSP(dataY_test, sol1)
comparePCTSP(dataY_test, sol_ctrained_unif)
sum(comparePCTSP(dataY_test, sol_ctrained_unif))/len(dataY_test)

'''