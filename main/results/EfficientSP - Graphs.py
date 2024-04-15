# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:30:13 2023

@author: basti
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def graphmarker(file2): 
    file3 = file2
    file3['bla'] = -1
    k = file3.iloc[1]['iter_updated']
    for i in range(len(file3)):
        if file3.iloc[i]['iter_updated'] == k:
            file3['bla'][i] = file2.iloc[i]['cosine'] + 0.05
    return file3


file1 = pd.read_csv('results/resultsFinal/H2_KP_SOP_OPTI_t5_inst_n_1000_id_NoCache')
file2 = pd.read_csv('results/resultsFinal/H2_KP_SOP_OPTI_t10_inst_n_1000_id_NoCache')
file3 = pd.read_csv('results/resultsFinal/H2_KP_SOP_OPTI_t30_inst_n_1000_id_NoCache')
file4 = pd.read_csv('results/resultsFinal/H2_KP_SOP_OPTI_t3600_inst_n_1000_id_NoCache')

filea = pd.read_csv('results/resultsFinal/H2_KP_SOP_OPTI_t10_inst_n_1000_id_Cache')
fileb = pd.read_csv('results/resultsFinal/H2_KP_SAT_OPTI_t10_inst_n_1000_id_NoCache')
filec = pd.read_csv('results/resultsFinal/H2_KP_SAT_OPTI_t10_inst_n_1000_id_Cache')

plt.figure(figsize=(10, 6)) 
plt.plot(file1['update_time'], file1['cosine'], label='heur,t=10', linestyle='-',color = 'forestgreen')
plt.plot(file2['update_time'], file2['cosine'], label='heur,t=30', linestyle='-',color = 'goldenrod')
plt.plot(file3['update_time'], file3['cosine'], label='heur,t=30', linestyle='-',color = 'teal')
plt.plot(file4['update_time'], file4['cosine'], label='exact', linestyle='-',color = 'purple')
plt.xlabel('Time', fontsize=32)
plt.ylabel('Cosine' ,  fontsize=36)
plt.xticks(fontsize = 17) 
plt.yticks(fontsize = 18)
plt.legend(fontsize = 22)
plt.ylim(0.9,2.1)
plt.xlim(-100,3800)
#plt.title('(a) KP-1000', fontsize=18)
plt.grid(True)
plt.savefig('test1.pdf', format='pdf')
plt.show()

plt.figure(figsize=(10,6)) 
plt.plot(file2['update_time'], file2['cosine'], label='heur,t=10', linestyle='-',color = 'teal')
plt.plot(filea['update_time'], filea['cosine'], label='cache+heur,t=10', linestyle='--',color = 'firebrick')
plt.plot(fileb['update_time'], fileb['cosine'], label='any,t=10', linestyle='-',color = 'mediumseagreen')
plt.plot(filec['update_time'], filec['cosine'], label='cache+any,t=10', linestyle='--',color = 'goldenrod')
#plt.plot(file4['update_time'], file4['cosine'], label='exact', linestyle='-',color = 'purple')

plt.plot(graphmarker(file2)['update_time'],graphmarker(file2)['bla'], marker=11, alpha=0.5, linestyle='None',color = 'teal', markersize=5)
plt.plot(graphmarker(filea)['update_time'],graphmarker(filea)['bla'], marker=11, alpha=0.5, linestyle='None',color = 'firebrick', markersize=5)
plt.plot(graphmarker(fileb)['update_time'],graphmarker(fileb)['bla'], marker=11, alpha=0.5, linestyle='None',color = 'mediumseagreen', markersize=5)
plt.plot(graphmarker(filec)['update_time'],graphmarker(filec)['bla'], marker=11, alpha=0.5, linestyle='None',color = 'goldenrod', markersize=5)
#plt.title('(a) KP-1000', fontsize=18)

plt.xlabel('Time', fontsize=32)
plt.ylabel('Cosine' ,  fontsize=36)
plt.xticks(fontsize = 17) 
plt.yticks(fontsize = 18)
plt.legend(fontsize = 22)
plt.ylim(-0.1,2.1)
plt.xlim(-100,3800)
plt.grid(True)

plt.savefig('test2.pdf', format='pdf')
plt.show()


#--------------------------------pctsp100

file1 = pd.read_csv('results/resultsFinal/H2_PCTSP_OPTI_SOP_t10_size_n01_100_NoCache')
file2 = pd.read_csv('results/resultsFinal/H2_PCTSP_OPTI_SOP_t30_size_n01_100_NoCache')
file3 = pd.read_csv('results/resultsFinal/H2_PCTSP_OPTI_SOP_t120_size_n01_100_NoCache')
file4 = pd.read_csv('results/resultsFinal/H2_PCTSP_OPTI_SOP_t3600_size_n01_100_NoCache')

filea = pd.read_csv('results/resultsFinal/H2_PCTSP_OPTI_SOP_t120_size_n01_100_Cache')
fileb = pd.read_csv('results/resultsFinal/H2_PCTSP_OPTI_SAT_t120_size_n01_100_NoCache')
filec = pd.read_csv('results/resultsFinal/H2_PCTSP_OPTI_SAT_t120_size_n01_100_Cache')
plt.figure(figsize=(10,6)) 
plt.plot(file1['update_time'], file1['cosine'], label='heur,t=10', linestyle='-',color = 'forestgreen')
plt.plot(file2['update_time'], file2['cosine'], label='heur,t=30', linestyle='-',color = 'goldenrod')
plt.plot(file3['update_time'], file3['cosine'], label='heur,t=120', linestyle='-',color = 'teal')
plt.plot(file4['update_time'], file4['cosine'], label='exact', linestyle='-',color = 'purple')
#plt.title('(b) PCTSP-100', fontsize=18)
plt.xlabel('Time', fontsize=32)
plt.ylabel('Cosine' ,  fontsize=36)
plt.xticks(fontsize = 17) 
plt.yticks(fontsize = 18)
plt.legend(fontsize = 22)
plt.ylim(0.9,2.1)
plt.xlim(-100,3800)
plt.grid(True)

plt.savefig('test3.pdf', format='pdf')
plt.show()

plt.figure(figsize=(10, 6)) 
plt.plot(file2['update_time'], file2['cosine'], label='heur,t=120', linestyle='-',color = 'teal')
plt.plot(filea['update_time'], filea['cosine'], label='cache+heur,t=120', linestyle='--',color = 'firebrick')
plt.plot(fileb['update_time'], fileb['cosine'], label='any,t=120', linestyle='-',color = 'mediumseagreen')
plt.plot(filec['update_time'], filec['cosine'], label='cache+any,t=120', linestyle='--',color = 'goldenrod')
plt.plot(graphmarker(file2)['update_time'],graphmarker(file2)['bla'], marker=11, alpha=0.5, linestyle='None',color = 'teal', markersize=5)
plt.plot(graphmarker(filea)['update_time'],graphmarker(filea)['bla'], marker=11, alpha=0.5, linestyle='None',color = 'firebrick', markersize=5)
plt.plot(graphmarker(fileb)['update_time'],graphmarker(fileb)['bla'], marker=11, alpha=0.5, linestyle='None',color = 'mediumseagreen', markersize=5)
plt.plot(graphmarker(filec)['update_time'],graphmarker(filec)['bla'], marker=11, alpha=0.5, linestyle='None',color = 'goldenrod', markersize=5)

#plt.plot(file4['update_time'], file4['cosine'], label='exact', linestyle='-',color = 'purple')
#plt.title('(b) PCTSP-100', fontsize=18)
plt.xlabel('Time', fontsize=32)
plt.ylabel('Cosine' ,  fontsize=36)
plt.xticks(fontsize = 17) 
plt.yticks(fontsize = 18)
plt.legend(fontsize = 22)
plt.ylim(-0.1,2.1)
plt.xlim(-100,3800)

plt.grid(True)

plt.savefig('test4.pdf', format='pdf')
plt.show()


#-----------200z

file1 = pd.read_csv('results/resultsFinal/H3_PCTSP_OPTI_SOP_t10_size_n01_200_NoCache')
file2 = pd.read_csv('results/resultsFinal/H3_PCTSP_OPTI_SOP_t30_size_n01_200_NoCache')
file3 = pd.read_csv('results/resultsFinal/H3_PCTSP_OPTI_SOP_t120_size_n01_200_NoCache')
file4 = pd.read_csv('results/resultsFinal/H3_PCTSP_OPTI_SOP_t3600_size_n01_200_NoCache')

filea = pd.read_csv('results/resultsFinal/H3_PCTSP_OPTI_SOP_t120_size_n01_200_Cache')
fileb = pd.read_csv('results/resultsFinal/H3_PCTSP_OPTI_SAT_t120_size_n01_200_NoCache')
filec = pd.read_csv('results/resultsFinal/H3_PCTSP_OPTI_SAT_t120_size_n01_200_Cache')

filed = pd.read_csv('results/resultsFinal/H3_PCTSP_OPTI_SOP_t30_size_n01_200_Cache')
filee = pd.read_csv('results/resultsFinal/H3_PCTSP_OPTI_SAT_t30_size_n01_200_NoCache')
filef = pd.read_csv('results/resultsFinal/H3_PCTSP_OPTI_SAT_t30_size_n01_200_Cache')
#plt.figure(figsize=(10, 6)) 
plt.figure(figsize=(10, 6))  
plt.plot(file1['update_time'], file1['cosine'], label='heur,t=10', linestyle='-',color = 'forestgreen')
plt.plot(file2['update_time'], file2['cosine'], label='heur,t=30', linestyle='-',color = 'goldenrod')
plt.plot(file3['update_time'], file3['cosine'], label='heur,t=120', linestyle='-',color = 'teal')
plt.plot(file4['update_time'], file4['cosine'], label='exact', linestyle='-',color = 'purple')
plt.xlabel('Time', fontsize=32)
plt.ylabel('Cosine' ,  fontsize=36)
plt.xticks(fontsize = 17) 
plt.yticks(fontsize = 18)
plt.legend(fontsize = 22)
#plt.title('(c) PCTSP-200', fontsize=18)
plt.ylim(0.9,2.1)
plt.xlim(-100,3800)
plt.grid(True)
plt.savefig('test5.pdf', format='pdf')
plt.show()

plt.figure(figsize=(10, 6)) 
plt.plot(file3['update_time'], file3['cosine'], label='heur,t=120', linestyle='-',color = 'teal')
plt.plot(filea['update_time'], filea['cosine'], label='cache+heur,t=120', linestyle='--',color = 'firebrick')
plt.plot(fileb['update_time'], fileb['cosine'], label='any,t=120', linestyle='-',color = 'mediumseagreen')
plt.plot(filec['update_time'], filec['cosine'], label='cache+any,t=120', linestyle='--',color = 'goldenrod')
plt.plot(graphmarker(file2)['update_time'],graphmarker(file2)['bla'], marker=11, alpha=0.5, linestyle='None',color = 'teal', markersize=5)
plt.plot(graphmarker(filea)['update_time'],graphmarker(filea)['bla'], marker=11, alpha=0.5, linestyle='None',color = 'firebrick', markersize=5)
plt.plot(graphmarker(fileb)['update_time'],graphmarker(fileb)['bla'], marker=11, alpha=0.5, linestyle='None',color = 'mediumseagreen', markersize=5)
plt.plot(graphmarker(filec)['update_time'],graphmarker(filec)['bla'], marker=11, alpha=0.5, linestyle='None',color = 'goldenrod', markersize=5)

plt.xlabel('Time', fontsize=32)
plt.ylabel('Cosine' ,  fontsize=36)
plt.xticks(fontsize = 17) 
plt.yticks(fontsize = 18)
plt.legend(fontsize = 22)
plt.grid(True)
plt.ylim(-0.1,2.1)
plt.xlim(-100,3800)
plt.savefig('test6.pdf', format='pdf')
plt.show()


plt.plot(file2['update_time'], file2['cosine'], label='heur,t=120', linestyle='-',color = 'teal')
plt.plot(filed['update_time'], filed['cosine'], label='cache+heur,t=120', linestyle='--',color = 'firebrick')
plt.plot(filee['update_time'], filee['cosine'], label='any,t=120', linestyle='-',color = 'mediumseagreen')
plt.plot(filef['update_time'], filef['cosine'], label='cache+any,t=120', linestyle='--',color = 'goldenrod')
#plt.plot(file4['update_time'], file4['cosine'], label='exact', linestyle='-',color = 'purple')

plt.xlabel('Time')
plt.ylabel('Cosine')
plt.legend()
plt.grid(True)

plt.show()


# =============================================================================
#  Optimality Gap Calc
# =============================================================================

def splitter(w):
    
    components = str(w).split("_")
    float_components = [float(value) for value in components]
    print(float_components)
    return float_components


w_PCTSPs100 = np.array([7, 13, 4, 9])
file4 = pd.read_csv('results/resultsFinal/H2_KP_SOP_OPTI_t3600_inst_n_1000_id_NoCache')
file1 = pd.read_csv('results/resultsFinal/H2_KP_SOP_OPTI_t10_inst_n_1000_id_NoCache')
filea = pd.read_csv('results/resultsFinal/H2_KP_SOP_OPTI_t10_inst_n_1000_id_Cache')
fileb = pd.read_csv('results/resultsFinal/H2_KP_SAT_OPTI_t10_inst_n_1000_id_NoCache')
filec = pd.read_csv('results/resultsFinal/H2_KP_SAT_OPTI_t10_inst_n_1000_id_Cache')
files = [file4,file1,filea,fileb,filec]
results = []
i=0
results.append('KP')

ysol = learner.predict_truew(X_test,w_PCTSPs100)
for file in files:
    i=i+1
    w_pred_t10 = splitter(file.iloc[-2]['weights'])
    ypred = learner.predict_truew(X_test,np.array(w_pred_t10))
    gap = -(eval_sol(ypred,w_PCTSPs100) - eval_sol(ysol,w_PCTSPs100))/eval_sol(ysol,w_PCTSPs100)
    results.append([i,gap,w_pred_t10])


w_PCTSPs100 = [10,75,6,9,93]
file1 = pd.read_csv('results/resultsFinal/H2_PCTSP_OPTI_SOP_t10_size_n01_100_NoCache')
file2 = pd.read_csv('results/resultsFinal/H2_PCTSP_OPTI_SOP_t30_size_n01_100_NoCache')
file3 = pd.read_csv('results/resultsFinal/H2_PCTSP_OPTI_SOP_t120_size_n01_100_NoCache')
file4 = pd.read_csv('results/resultsFinal/H2_PCTSP_OPTI_SOP_t3600_size_n01_100_NoCache')
filea = pd.read_csv('results/resultsFinal/H2_PCTSP_OPTI_SOP_t120_size_n01_100_Cache')
fileb = pd.read_csv('results/resultsFinal/H2_PCTSP_OPTI_SAT_t120_size_n01_100_NoCache')
filec = pd.read_csv('results/resultsFinal/H2_PCTSP_OPTI_SAT_t120_size_n01_100_Cache')
files = [file1,file2,file3,file4,filea,fileb,filec]
i=0
results = []
results.append('PCTSP100')

ysol = learner.predict_truew(X_test,w_PCTSPs100)
for file in files:
    i=i+1
    w_pred_t10 = splitter(file.iloc[-2]['weights'])
    ypred = learner.predict_truew(X_test,w_pred_t10)
    gap = (eval_sol(ypred,w_PCTSPs100) - eval_sol(ysol,w_PCTSPs100))/eval_sol(ysol,w_PCTSPs100)
    results.append([i,gap,w_pred_t10])
    
    
    
    
file1 = pd.read_csv('results/resultsFinal/H3_PCTSP_OPTI_SOP_t10_size_n01_200_NoCache')
file2 = pd.read_csv('results/resultsFinal/H3_PCTSP_OPTI_SOP_t30_size_n01_200_NoCache')
file3 = pd.read_csv('results/resultsFinal/H3_PCTSP_OPTI_SOP_t120_size_n01_200_NoCache')
file4 = pd.read_csv('results/resultsFinal/H3_PCTSP_OPTI_SOP_t3600_size_n01_200_NoCache')
filea = pd.read_csv('results/resultsFinal/H3_PCTSP_OPTI_SOP_t120_size_n01_200_Cache')
fileb = pd.read_csv('results/resultsFinal/H3_PCTSP_OPTI_SAT_t120_size_n01_200_NoCache')
filec = pd.read_csv('results/resultsFinal/H3_PCTSP_OPTI_SAT_t120_size_n01_200_Cache')
files = [file1,file2,file3,file4,filea,fileb,filec]
i=0

results= []
ysol = learner.predict_truew(X_test,w_PCTSPs100)
results.append('PCTSP200')
for file in files:
    i=i+1
    w_pred_t10 = splitter(file.iloc[-2]['weights'])
    ypred = learner.predict_truew(X_test,w_pred_t10)
    gap = (eval_sol(ypred,w_PCTSPs100) - eval_sol(ysol,w_PCTSPs100))/eval_sol(ysol,w_PCTSPs100)
    results.append([i,gap,w_pred_t10])
    

    
# =============================================================================
# 
# =============================================================================

filea = pd.read_csv('results/resultsFinal/F7_KP_SAT_OPTI_t10_inst_n_1000_id_NoCache')
fileb = pd.read_csv('results/resultsFinal/F7_KP_SOP_OPTI_t10_inst_n_1000_id_Cache')
filec = pd.read_csv('results/resultsFinal/F7_KP_SAT_OPTI_t10_inst_n_1000_id_Cache')

file1 = pd.read_csv('results/resultsFinal/F7_KP_SOP_OPTI_t5_inst_n_1000_id_NoCache')
file2 = pd.read_csv('results/resultsFinal/F7_KP_SOP_OPTI_t10_inst_n_1000_id_NoCache')
file3 = pd.read_csv('results/resultsFinal/F7_KP_SOP_OPTI_t30_inst_n_1000_id_NoCache')
file4 = pd.read_csv('results/resultsFinal/F7_KP_SOP_OPTI_t3600_inst_n_1000_id_NoCache')

# Plot the data from file1
plt.plot(file1['update_time'], file1['cosine'], label='timeout = 10', linestyle='-',color = 'aquamarine')
plt.plot(file2['update_time'], file2['cosine'], label='timeout = 30', linestyle='-',color = 'darkturquoise')
plt.plot(file3['update_time'], file3['cosine'], label='timeout = 120', linestyle='-',color = 'teal')
plt.plot(file4['update_time'], file4['cosine'], label='timeout = inf', linestyle='-',color = 'steelblue')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(file2['update_time'], file2['cosine'], label='SP', linestyle='-',color = 'teal')
plt.plot(filea['update_time'], filea['cosine'], label='SAT', linestyle='-',color = 'firebrick')
plt.plot(fileb['update_time'], fileb['cosine'], label='SPc', linestyle='-',color = 'mediumseagreen')
plt.plot(filec['update_time'], filec['cosine'], label='SATc', linestyle='-',color = 'goldenrod')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


filea = pd.read_csv('results/resultsFinal/F2_PCTSP_OPTI_SOP_t30_size_n01_100_Cache')
fileb = pd.read_csv('results/resultsFinal/F2_PCTSP_OPTI_SAT_t30_size_n01_100_NoCache')
filec = pd.read_csv('results/resultsFinal/F2_PCTSP_OPTI_SAT_t30_size_n01_100_Cache')

file1 = pd.read_csv('results/resultsFinal/F2_PCTSP_OPTI_SOP_t10_size_n01_100_NoCache')
file2 = pd.read_csv('results/resultsFinal/F2_PCTSP_OPTI_SOP_t30_size_n01_100_NoCache')
file3 = pd.read_csv('results/resultsFinal/F2_PCTSP_OPTI_SOP_t120_size_n01_100_NoCache')
file4 = pd.read_csv('results/resultsFinal/F2_PCTSP_OPTI_SOP_t3600_size_n01_100_NoCache')

# Plot the data from file1
plt.plot(file1['update_time'], file1['cosine'], label='timeout = 10', linestyle='-',color = 'aquamarine')
plt.plot(file2['update_time'], file2['cosine'], label='timeout = 30', linestyle='-',color = 'darkturquoise')
plt.plot(file3['update_time'], file3['cosine'], label='timeout = 120', linestyle='-',color = 'teal')
plt.plot(file4['update_time'], file4['cosine'], label='timeout = inf', linestyle='-',color = 'steelblue')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(file2['update_time'], file2['cosine'], label='SP', linestyle='-',color = 'teal')
plt.plot(filea['update_time'], filea['cosine'], label='SPc', linestyle='-',color = 'firebrick')
plt.plot(fileb['update_time'], fileb['cosine'], label='SAT', linestyle='-',color = 'mediumseagreen')
plt.plot(filec['update_time'], filec['cosine'], label='SATc', linestyle='-',color = 'goldenrod')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()




file1 = pd.read_csv('results/resultsFinal/F4_PCTSP_OPTI_SOP_t10_size_n01_200_NoCache')
file2 = pd.read_csv('results/resultsFinal/F4_PCTSP_OPTI_SOP_t30_size_n01_200_NoCache')
file4 = pd.read_csv('results/resultsFinal/F4_PCTSP_OPTI_SOP_t3600_size_n01_200_NoCache')

file3 = pd.read_csv('results/resultsFinal/F4_PCTSP_OPTI_SOP_t120_size_n01_200_NoCache')
filea = pd.read_csv('results/resultsFinal/F4_PCTSP_OPTI_SOP_t120_size_n01_200_Cache')
fileb = pd.read_csv('results/resultsFinal/F4_PCTSP_OPTI_SAT_t120_size_n01_200_NoCache')
filec = pd.read_csv('results/resultsFinal/F4_PCTSP_OPTI_SAT_t120_size_n01_200_Cache')


plt.plot(file1['update_time'], file1['cosine'], label='timeout = 10', linestyle='-',color = 'aquamarine')
plt.plot(file2['update_time'], file2['cosine'], label='timeout = 30', linestyle='-',color = 'darkturquoise')
plt.plot(file3['update_time'], file3['cosine'], label='timeout = 120', linestyle='-',color = 'teal')
plt.plot(file4['update_time'], file4['cosine'], label='timeout = inf', linestyle='-',color = 'steelblue')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot the data from file1
plt.plot(file3['update_time'], file3['cosine'], label='SP', linestyle='-',color = 'teal')
plt.plot(filea['update_time'], filea['cosine'], label='SPc', linestyle='-',color = 'firebrick')
plt.plot(fileb['update_time'], fileb['cosine'], label='SAT', linestyle='-',color = 'mediumseagreen')
plt.plot(filec['update_time'], filec['cosine'], label='SATc', linestyle='-',color = 'goldenrod')

# Customize the plot
plt.xlabel('Time')
plt.ylabel('Loss')
plt.legend()
plt.ylim(-0.1, 2.1)
# Show the plot
plt.grid(True)
plt.show()


file1 = pd.read_csv('results/resultsFinal/F4_PCTSP_OPTI_SOP_t10_size_n01_200_NoCache')
file2 = pd.read_csv('results/resultsFinal/F4_PCTSP_OPTI_SOP_t30_size_n01_200_NoCache')
file4 = pd.read_csv('results/resultsFinal/F4_PCTSP_OPTI_SOP_t3600_size_n01_200_NoCache')

file3 = pd.read_csv('results/resultsFinal/F4_PCTSP_OPTI_SOP_t120_size_n01_200_NoCache')
filea = pd.read_csv('results/resultsFinal/F4_PCTSP_OPTI_SOP_t120_size_n01_200_Cache')
fileb = pd.read_csv('results/resultsFinal/F4_PCTSP_OPTI_SAT_t120_size_n01_200_NoCache')
filec = pd.read_csv('results/resultsFinal/F4_PCTSP_OPTI_SAT_t120_size_n01_200_Cache')


plt.plot(file1['update_time'], file1['cosine'], label='timeout = 10', linestyle='-',color = 'aquamarine')
plt.plot(file2['update_time'], file2['cosine'], label='timeout = 30', linestyle='-',color = 'darkturquoise')
plt.plot(file3['update_time'], file3['cosine'], label='timeout = 120', linestyle='-',color = 'teal')
plt.plot(file4['update_time'], file4['cosine'], label='timeout = inf', linestyle='-',color = 'steelblue')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.legend()
plt.ylim(0.6,2.1)
plt.grid(True)
plt.show()

# Plot the data from file1
plt.plot(file3['update_time'], file3['cosine'], label='SP', linestyle='-',color = 'teal')
plt.plot(filea['update_time'], filea['cosine'], label='SPc', linestyle='-',color = 'firebrick')
plt.plot(fileb['update_time'], fileb['cosine'], label='SAT', linestyle='-',color = 'mediumseagreen')
plt.plot(filec['update_time'], filec['cosine'], label='SATc', linestyle='-',color = 'goldenrod')

# Customize the plot
plt.xlabel('Time')
plt.ylabel('Loss')
plt.legend()
plt.ylim(-0.1, 2.1)
# Show the plot
plt.grid(True)
plt.show()



filea = pd.read_csv('results/resultsFinal/F2_PCTSP_OPTI_SOP_t120_size_n01_200_Cache')
fileb = pd.read_csv('results/resultsFinal/F2_PCTSP_OPTI_SAT_t120_size_n01_200_NoCache')
filec = pd.read_csv('results/resultsFinal/F2_PCTSP_OPTI_SAT_t120_size_n01_200_Cache')

file1 = pd.read_csv('results/resultsFinal/F2_PCTSP_OPTI_SOP_t10_size_n01_200_NoCache')
file2 = pd.read_csv('results/resultsFinal/F2_PCTSP_OPTI_SOP_t30_size_n01_200_NoCache')
file3 = pd.read_csv('results/resultsFinal/F2_PCTSP_OPTI_SOP_t120_size_n01_200_NoCache')
file4 = pd.read_csv('results/resultsFinal/F2_PCTSP_OPTI_SOP_t3600_size_n01_200_NoCache')

# Plot the data from file1
plt.plot(file1['update_time'], file1['cosine'], label='timeout = 10', linestyle='-',color = 'aquamarine')
plt.plot(file2['update_time'], file2['cosine'], label='timeout = 30', linestyle='-',color = 'darkturquoise')
plt.plot(file3['update_time'], file3['cosine'], label='timeout = 120', linestyle='-',color = 'teal')
plt.plot(file4['update_time'], file4['cosine'], label='timeout = inf', linestyle='-',color = 'steelblue')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(file2['update_time'], file2['cosine'], label='SP', linestyle='-',color = 'teal')
plt.plot(filea['update_time'], filea['cosine'], label='SPc', linestyle='-',color = 'firebrick')
plt.plot(fileb['update_time'], fileb['cosine'], label='SAT', linestyle='-',color = 'mediumseagreen')
plt.plot(filec['update_time'], filec['cosine'], label='SATc', linestyle='-',color = 'goldenrod')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()




filea = pd.read_csv('results/PCTSP_OPTI_SOP_t120_size_n01_200_Cache__Nc_test')
fileb = pd.read_csv('results/PCTSP_OPTI_SAT_t120_size_n01_200_NoCache__Nc_test')
filec = pd.read_csv('results/PCTSP_OPTI_SAT_t120_size_n01_200_Cache__Nc_test')

file1 = pd.read_csv('results/PCTSP_OPTI_SOP_t10_size_n01_200_NoCache__Nc_test')
file2 = pd.read_csv('results/PCTSP_OPTI_SOP_t30_size_n01_200_NoCache__Nc_test')
file3 = pd.read_csv('results/PCTSP_OPTI_SOP_t120_size_n01_200_NoCache__Nc_test')
file4 = pd.read_csv('results/PCTSP_OPTI_SOP_t3600_size_n01_200_NoCache__Nc_test')

# Plot the data from file1
plt.plot(file1['update_time'], file1['cosine'], label='timeout = 10', linestyle='-',color = 'aquamarine')
plt.plot(file2['update_time'], file2['cosine'], label='timeout = 30', linestyle='-',color = 'darkturquoise')
plt.plot(file3['update_time'], file3['cosine'], label='timeout = 120', linestyle='-',color = 'teal')
plt.plot(file4['update_time'], file4['cosine'], label='timeout = inf', linestyle='-',color = 'steelblue')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(file2['update_time'], file2['cosine'], label='SP', linestyle='-',color = 'teal')
plt.plot(filea['update_time'], filea['cosine'], label='SPc', linestyle='-',color = 'firebrick')
plt.plot(fileb['update_time'], fileb['cosine'], label='SAT', linestyle='-',color = 'mediumseagreen')
plt.plot(filec['update_time'], filec['cosine'], label='SATc', linestyle='-',color = 'goldenrod')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()



# =============================================================================
# OLD
# =============================================================================



#KP

file1 = pd.read_csv('results/resultsFinal/Final_KP_SAT_OPTI_t30_inst_n_1000_id_NoCache_')
file2 = pd.read_csv('results/resultsFinal/Final_KP_SOP_OPTI_t30_inst_n_1000_id_NoCache_')
file3 = pd.read_csv('results/KP_SAT_OPTI_t120_inst_n_1000_id')
file4 = pd.read_csv('results/KP_SOP_OPTI_t120_inst_n_1000_id_NoCache_')
file5 = pd.read_csv('results/resultsFinal/Final_KP_SAT_OPTI_t3600_inst_n_1000_id_NoCache_')
file6 = pd.read_csv('results/resultsFinal/Final_KP_SOP_OPTI_t3600_inst_n_1000_id_NoCache_')

# Plot the data from file1
plt.plot(file1['update_time'], file1['cosine'], label='SAT + timeout = 30', linestyle='-',color = 'khaki')
plt.plot(file2['update_time'], file2['cosine'], label='SOP + timeout = 30', linestyle='-',color = 'green')
plt.plot(file3['update_time'], file3['cosine'], label='SAT + timeout = 120', linestyle='-',color = 'goldenrod')
plt.plot(file4['update_time'], file4['cosine'], label='SOP + timeout = 120', linestyle='-',color = 'teal')
plt.plot(file5['update_time'], file5['cosine'], label='SAT + timeout = inf', linestyle='-',color = 'firebrick')
plt.plot(file6['update_time'], file6['cosine'], label='SOP + timeout = inf', linestyle='-',color = 'steelblue')


# Customize the plot
plt.xlabel('Time')
plt.ylabel('Loss')
plt.legend()
plt.ylim(-0.1, 2.1)
# Show the plot
plt.grid(True)
plt.show()





#--------------------------------pctsp50
file1 = pd.read_csv('results/resultsFinal/Final_PCTSP_OPTI_SOP_t120_size_n01_50_NoCache_B')
file2 = pd.read_csv('results/resultsFinal/Final_PCTSP_OPTI_SAT_t120_size_n01_50_Cache_PCB')
file3 = pd.read_csv('results/resultsFinal/Final_PCTSP_OPTI_SOP_t120_size_n01_50_NoCache_B')
file4 = pd.read_csv('results/resultsFinal/Final_PCTSP_OPTI_SAT_t120_size_n01_50_Cache_PCB')

plt.plot(file1['update_time'], file1['cosine'], label='SP', linestyle='-',color ='springgreen')
plt.plot(file2['update_time'], file2['cosine'], label='SP + Cache', linestyle='-',color = 'seagreen')
plt.plot(file3['update_time'], file3['cosine'], label='SAT', linestyle='-',color='khaki')
plt.plot(file4['update_time'], file4['cosine'], label='SAT + Cache', linestyle='-',color='goldenrod')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

#50stops

file1 = pd.read_csv('results/resultsFinal/Final_PCTSP_OPTI_SAT_t30_size_n01_50_NoCache_')
file2 = pd.read_csv('results/resultsFinal/Final_PCTSP_OPTI_SOP_t30_size_n01_50_NoCache_')
file3 = pd.read_csv('results/resultsFinal/Final_PCTSP_OPTI_SAT_t120_size_n01_50_NoCache_')
file4 = pd.read_csv('results/resultsFinal/Final_PCTSP_OPTI_SOP_t120_size_n01_50_NoCache_')
file5 = pd.read_csv('results/resultsFinal/Final_PCTSP_OPTI_SAT_t3600_size_n01_50_NoCache_')
file6 = pd.read_csv('results/resultsFinal/Final_PCTSP_OPTI_SOP_t3600_size_n01_50_NoCache_cscs')

# Plot the data from file1
plt.plot(file1['update_time'], file1['cosine'], label='SAT + timeout = 30', linestyle='-',color = 'khaki')
plt.plot(file2['update_time'], file2['cosine'], label='SOP + timeout = 30', linestyle='-',color = 'darkgreen')
plt.plot(file3['update_time'], file3['cosine'], label='SAT + timeout = 120', linestyle='-',color = 'goldenrod')
plt.plot(file4['update_time'], file4['cosine'], label='SOP + timeout = 120', linestyle='-',color = 'teal')
plt.plot(file5['update_time'], file5['cosine'], label='SAT + timeout = inf', linestyle='-',color = 'firebrick')
plt.plot(file6['update_time'], file6['cosine'], label='SOP + timeout = inf', linestyle='-',color = 'steelblue')

# Customize the plot
plt.xlabel('Time')
plt.ylabel('Loss')
plt.legend()
plt.ylim(-0.1, 2.1)

plt.grid(True)
plt.show()



file1 = pd.read_csv('results/resultsFinal/Final_PCTSP_OPTI_SAT_t30_size_n01_100_NoCache_')
file2 = pd.read_csv('results/resultsFinal/Final_PCTSP_OPTI_SOP_t30_size_n01_100_NoCache_')
file3 = pd.read_csv('results/PCTSP_OPTI_SAT_t120_size_n01_100_NoCache__Nc_test')
file4 = pd.read_csv('results/PCTSP_OPTI_SOP_t120_size_n01_100_NoCache__Nc_test')
file5 = pd.read_csv('results/resultsFinal/Final_PCTSP_OPTI_SAT_t3600_size_n01_100_NoCache_')
file6 = pd.read_csv('results/resultsFinal/Final_PCTSP_OPTI_SOP_t3600_size_n01_100_NoCache_')

# Plot the data from file1
plt.plot(file1['update_time'], file1['cosine'], label='SAT + timeout = 30', linestyle='-',color = 'khaki')
plt.plot(file2['update_time'], file2['cosine'], label='SOP + timeout = 30', linestyle='-',color = 'darkgreen')
plt.plot(file3['update_time'], file3['cosine'], label='SAT + timeout = 120', linestyle='-',color = 'goldenrod')
plt.plot(file4['update_time'], file4['cosine'], label='SOP + timeout = 120', linestyle='-',color = 'teal')
plt.plot(file5['update_time'], file5['cosine'], label='SAT + timeout = inf', linestyle='-',color = 'firebrick')
plt.plot(file6['update_time'], file6['cosine'], label='SOP + timeout = inf', linestyle='-',color = 'steelblue')

# Customize the plot
plt.xlabel('Time')
plt.ylabel('Loss')
plt.legend()
plt.ylim(-0.1, 2.1)
# Show the plot
plt.grid(True)
plt.show()



file1 = pd.read_csv('results/resultsFinal/Final_PCTSP_OPTI_SAT_t30_size_n01_200_NoCache_')
file2 = pd.read_csv('results/resultsFinal/Final_PCTSP_OPTI_SOP_t30_size_n01_200_NoCache_')
file3 = pd.read_csv('results/resultsFinal/Final_PCTSP_OPTI_SAT_t120_size_n01_200_NoCache__2')
file4 = pd.read_csv('results/resultsFinal/Final_PCTSP_OPTI_SOP_t120_size_n01_200_NoCache_')
file5 = pd.read_csv('results/resultsFinal/Final_PCTSP_OPTI_SAT_t3600_size_n01_200_NoCache_')
file6 = pd.read_csv('results/resultsFinal/Final_PCTSP_OPTI_SOP_t3600_size_n01_200_NoCache_')

# Plot the data from file1
plt.plot(file1['update_time'], file1['cosine'], label='SAT + timeout = 30', linestyle='-',color = 'khaki')
plt.plot(file2['update_time'], file2['cosine'], label='SOP + timeout = 30', linestyle='-',color = 'darkgreen')
plt.plot(file3['update_time'], file3['cosine'], label='SAT + timeout = 120', linestyle='-',color = 'goldenrod')
plt.plot(file4['update_time'], file4['cosine'], label='SOP + timeout = 120', linestyle='-',color = 'teal')
plt.plot(file5['update_time'], file5['cosine'], label='SAT + timeout = inf', linestyle='-',color = 'firebrick')
plt.plot(file6['update_time'], file6['cosine'], label='SOP + timeout = inf', linestyle='-',color = 'steelblue')

# Customize the plot
plt.xlabel('Time')
plt.ylabel('Loss')
plt.legend()
plt.ylim(-0.1, 2.1)
# Show the plot
plt.grid(True)
plt.show()



#--------------------------50
file1 = pd.read_csv('results/resultsFinal/Final_PCTSP_OPTI_SOP_t10_size_n01_50_NoCache__2')
file2 = pd.read_csv('results/resultsFinal/Final_PCTSP_OPTI_SOP_t30_size_n01_50_NoCache__2')
file3 = pd.read_csv('results/resultsFinal/Final_PCTSP_OPTI_SOP_t120_size_n01_50_NoCache__2')
#file4 = pd.read_csv('results/resultsFinal/Final_PCTSP_OPTI_SOP_t600_size_n01_50_NoCache__2')
file4 = pd.read_csv('results/resultsFinal/Final_PCTSP_OPTI_SOP_t3600_size_n01_50_NoCache__2')

# Plot the data from file1
plt.plot(file1['update_time'], file1['cosine'], label='timeout = 10', linestyle='-',color = 'aquamarine')
plt.plot(file2['update_time'], file2['cosine'], label='timeout = 30', linestyle='-',color = 'darkturquoise')
plt.plot(file3['update_time'], file3['cosine'], label='timeout = 120', linestyle='-',color = 'teal')
plt.plot(file4['update_time'], file4['cosine'], label='timeout = inf', linestyle='-',color = 'steelblue')

plt.xlabel('Time')
plt.ylabel('Loss')
plt.legend()
plt.ylim(-0.1, 2.1)
# Show the plot
plt.grid(True)
plt.show()



file1 = pd.read_csv('results/resultsFinal/Final_PCTSP_OPTI_SAT_t120_size_n01_200_NoCache__2')
file2 = pd.read_csv('results/resultsFinal/Final_PCTSP_OPTI_SAT_t120_size_n01_200_NoCache_')
file3 = pd.read_csv('results/resultsFinal/Final_PCTSP_OPTI_SAT_t120_size_n01_200_Cache_PC_2')
file4 = pd.read_csv('results/resultsFinal/Final_PCTSP_OPTI_SOP_t120_size_n01_200_Cache_')


# Plot the data from file1
plt.plot(file1['update_time'], file1['cosine'], label='SAT + timeout = 30', linestyle='-',color = 'khaki')
plt.plot(file2['update_time'], file2['cosine'], label='SOP + timeout = 30', linestyle='-',color = 'darkgreen')
plt.plot(file3['update_time'], file3['cosine'], label='SAT + timeout = 120', linestyle='-',color = 'goldenrod')
plt.plot(file4['update_time'], file4['cosine'], label='SOP + timeout = 120', linestyle='-',color = 'teal')
plt.plot(file5['update_time'], file5['cosine'], label='SAT + timeout = inf', linestyle='-',color = 'firebrick')
plt.plot(file6['update_time'], file6['cosine'], label='SOP + timeout = inf', linestyle='-',color = 'steelblue')

# Customize the plot
plt.xlabel('Time')
plt.ylabel('Loss')
plt.legend()
plt.ylim(-0.1, 2.1)
# Show the plot
plt.grid(True)
plt.show()


file1 = pd.read_csv('results/resultsFinal/F6_PCTSP_OPTI_SAT_t120_size_n01_200_Cache')
file2 = pd.read_csv('results/resultsFinal/F6_PCTSP_OPTI_SAT_t120_size_n01_200_NoCache')

plt.plot(file1['update_time'], file1['cosine'], label='timeout = 10', linestyle='-',color = 'aquamarine')
plt.plot(file2['update_time'], file2['cosine'], label='timeout = 30', linestyle='-',color = 'darkturquoise')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()






file1 = pd.read_csv('results/resultsFinal/G2_PCTSP_OPTI_SOP_t10_size_n01_200_NoCache')
file2 = pd.read_csv('results/resultsFinal/G2_PCTSP_OPTI_SOP_t30_size_n01_200_NoCache')
file4 = pd.read_csv('results/resultsFinal/G2_PCTSP_OPTI_SOP_t3600_size_n01_200_NoCache')

file3 = pd.read_csv('results/resultsFinal/G2_PCTSP_OPTI_SOP_t120_size_n01_200_NoCache')
filea = pd.read_csv('results/resultsFinal/G2_PCTSP_OPTI_SOP_t120_size_n01_200_Cache')
fileb = pd.read_csv('results/resultsFinal/G2_PCTSP_OPTI_SAT_t120_size_n01_100_NoCache')
filec = pd.read_csv('results/resultsFinal/G2_PCTSP_OPTI_SAT_t120_size_n01_100_Cache')


plt.plot(file1['update_time'], file1['cosine'], label='timeout = 10', linestyle='-',color = 'aquamarine')
plt.plot(file2['update_time'], file2['cosine'], label='timeout = 30', linestyle='-',color = 'darkturquoise')
plt.plot(file3['update_time'], file3['cosine'], label='timeout = 120', linestyle='-',color = 'teal')
plt.plot(file4['update_time'], file4['cosine'], label='timeout = inf', linestyle='-',color = 'steelblue')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot the data from file1
plt.plot(file3['update_time'], file3['cosine'], label='SP', linestyle='-',color = 'teal')
plt.plot(filea['update_time'], filea['cosine'], label='SPc', linestyle='-',color = 'firebrick')
plt.plot(fileb['update_time'], fileb['cosine'], label='SAT', linestyle='-',color = 'mediumseagreen')
plt.plot(filec['update_time'], filec['cosine'], label='SATc', linestyle='-',color = 'goldenrod')

# Customize the plot
plt.xlabel('Time')
plt.ylabel('Loss')
plt.legend()
plt.ylim(-0.1, 2.1)
# Show the plot
plt.grid(True)
plt.show()


