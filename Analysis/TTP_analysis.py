import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from _utils import *
from scipy.stats import ttest_rel
parsdir = "../GLV/analysis-dual-sigmoid/model_pars/"
cs = sns.color_palette('Paired')
# resistance #
X = np.arange(0, 120*28).reshape(-1)
# clustering = pd.read_csv('./kmeans_clustering_results.csv', names=['y', 'r_HD', 'r_HI', 'Gamma', 'cluster', 'c2'], header = 0, index_col=0)
# resistance, response = clustering[clustering.cluster == 1], clustering[clustering.cluster == 0]
resistance = [11, 12, 19, 25, 36, 52, 54, 85, 88, 99, 101]
response = [1, 2, 3, 4, 6, 13, 15, 16, 17, 20, 24, 29, 30, 31, 32, 37, 40, 42, 44, 46, 50, 51,
            58, 60, 61, 62, 63, 66, 71, 75, 77, 78, 79, 84, 86, 87, 91, 92, 93, 94, 95, 96, 97,
            100, 102, 104, 105, 106, 108]
patient051 = []
for R in ['resistance', 'response']:
    A21 = []
    C2_P = []
    C2_E = []
    C2_M = []
    X_P = []
    X_E = []
    X_M = []
    if R == 'resistance':
        data = resistance
    else:
        data = response
    for i in data:
        if len(str(i)) == 1:
            patientNo = "patient00" + str(i)
        elif len(str(i)) == 2:
            patientNo = "patient0" + str(i)
        else:
            patientNo = "patient" + str(i)
        print(patientNo)
        ppo_states = pd.read_csv('../PPO_states/analysis/' + patientNo + '_evolution_states.csv', index_col=0)
        x_p = np.array(ppo_states.index).reshape(-1)
        X_P.append(x_p[-1].item())

        experts_states = pd.read_csv('../Experts_states/analysis/' + patientNo + '_experts_states.csv', index_col=0)
        x_e = np.array(experts_states.index).reshape(-1)
        X_E.append(min(x_e[-1].item(),3416))

        max_states = pd.read_csv('../MAX_states/analysis/' + patientNo + '_evolution_states.csv', index_col=0)
        x_m = np.array(max_states.index).reshape(-1)
        X_M.append(min(x_m[-1].item(), 3416))
        if patientNo == 'patient029':
            patient029= [x_p[-1], x_e[-1], x_m[-1]]
        if patientNo == 'patient036':
            patient099 = [x_p[-1], x_e[-1], x_m[-1]]


    max_x_p = max(X_P)
    max_x_e = max(X_E)
    max_x_m = max(X_M)
    X_P_arr = np.array(X_P)
    X_E_arr = np.array(X_E)
    X_M_arr = np.array(X_M)
    ttp_p = []
    ttp_e = []
    ttp_m = []
    for month in range(122):
        days = 28 * month
        end_p = np.where(X_P_arr > days)[0].shape[0]/X_P_arr.shape[0]
        ttp_p.append(end_p)
        end_e = np.where(X_E_arr > days)[0].shape[0]/X_E_arr.shape[0]
        ttp_e.append(end_e)
        end_m = np.where(X_M_arr > days)[0].shape[0]/X_M_arr.shape[0]
        ttp_m.append(end_m)
    months = np.arange(122)
    plt.style.use(['science', 'nature'])
    fig= plt.figure(figsize=(5, 4))
    plt.plot(months, ttp_p, lw=2.5, c=cs[1], label='$I^2$ADT')
    plt.plot(months, ttp_e, lw=2.5, c=cs[3], label='IADT')
    plt.plot(months, ttp_m, lw=2.5, c=cs[7], label='ADT')
    plt.xlabel('TTP (Months)', fontsize = 22)
    plt.xticks(fontsize = 20)
    plt.ylabel('PFS', fontsize = 22)
    plt.yticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1],labels = [0, 0.2, 0.4, 0.6, 0.8, 1], fontsize = 20 )
    plt.legend(fontsize = 16)
    plt.tight_layout()
    plt.savefig('./Figure/' + R + "_progression_free_survival_rate.eps", dpi=300, bbox_inches='tight')
    plt.show()
    TTP = pd.DataFrame(np.concatenate((X_P, X_E, X_M), axis = 0)/28, columns=['TTP'])
    strategy = ['I2ADT', 'IADT', 'ADT']
    TTP['Strategy'] = [i for item in [[strategy[j]] * len(data) for j in range(3)] for i in item]
    fig1 = plt.figure(figsize=(5, 4))
    p = colorAlpha_to_rgb([cs[1], cs[3], cs[7]], 0.6)
    sns.swarmplot(x = 'Strategy', y='TTP', data=TTP, size=6, palette = p)
    locs, labels = plt.xticks()
    plt.xticks(ticks=locs, labels=['$I^2$ADT', 'IADT', 'ADT'], fontsize=22)
    plt.xlabel('')
    plt.yticks(fontsize = 20)
    plt.ylabel('TTP (Months)', fontsize = 22)
    plt.hlines(np.mean(X_P)/28, -0.25, .25, colors=cs[1], lw=3)
    plt.hlines(np.mean(X_E)/28, 0.75, 1.25, colors=cs[3], lw=3)
    plt.hlines(np.mean(X_M)/28, 1.75, 2.25, colors=cs[7], lw=3)
    # if R=='resistance':
    #     plt.scatter(0, patient099[0]/28, s=120, marker='*', color='black', zorder=3)
    #     plt.scatter(1, patient099[1]/28, s=120, marker='*', color='black', zorder=3)
    #     plt.scatter(2, patient099[2]/28, s=120, marker='*', color='black', zorder=3)
    # else:
    #     plt.scatter(0, patient029[0] / 28, s=120, marker='*', color='black', zorder=3)
    #     plt.scatter(1, patient029[1] / 28, s=120, marker='*', color='black', zorder=3)
    #     plt.scatter(2, patient029[2] / 28, s=120, marker='*', color='black', zorder=3)
    plt.tight_layout()
    plt.savefig('./Figure/' + R + "_TTP_distribution.eps", dpi=300)
    plt.show()
    print(ttest_rel(X_P, X_E))




