import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from _utils import *

####################################################################
############## Analysis of competition intensity ###################
##################### Fig 5/7 a ####################################
####################################################################

parsdir = "../GLV/analysis-dual-sigmoid/model_pars/"
patientNo = 'patient036'
parslist = os.listdir(parsdir + patientNo)
PARS_LIST=[]
for arg in parslist:
    pars_df = pd.read_csv(parsdir + patientNo + '/' + arg)
    A, K, _, pars, best_pars = [np.array(pars_df.loc[i, ~np.isnan(pars_df.loc[i, :])]) for i in range(5)]
    PARS_LIST.append(best_pars)
PARS_ARR = np.stack(PARS_LIST)
pars = np.mean(PARS_ARR, axis=0)
phi36, gamma36 = pars[-3:-1]

patientNo = 'patient106'
parslist = os.listdir(parsdir + patientNo)
PARS_LIST=[]
for arg in parslist:
    pars_df = pd.read_csv(parsdir + patientNo + '/' + arg)
    A, K, _, pars, best_pars = [np.array(pars_df.loc[i, ~np.isnan(pars_df.loc[i, :])]) for i in range(5)]
    PARS_LIST.append(best_pars)
PARS_ARR = np.stack(PARS_LIST)
pars = np.mean(PARS_ARR, axis=0)
phi106, gamma106 = pars[-3:-1]
cs = sns.color_palette('Paired')
# resistance #
X = np.arange(0, 120*28).reshape(-1)
resistance = [11, 12, 19, 25, 36, 52, 54, 85, 88, 99, 101]
response = [1, 2, 3, 4, 6, 13, 15, 16, 17, 20, 24, 29, 30, 31, 32, 37, 40, 42, 44, 46, 50, 51,
            58, 60, 61, 62, 63, 66, 71, 75, 77, 78, 79, 84, 86, 87, 91, 92, 93, 94, 95, 96, 97,
            100, 102, 104, 105, 106, 108]
resistance_patients = []
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
        t = 2
    else:
        data = response
        t = 1
    for i in data:
        if len(str(i)) == 1:
            patientNo = "patient00" + str(i)
        elif len(str(i)) == 2:
            patientNo = "patient0" + str(i)
        else:
            patientNo = "patient" + str(i)
        print(patientNo)
        resistance_patients.append(patientNo)
        parslist = os.listdir(parsdir + patientNo)
        PARS_LIST = []
        for arg in parslist:
            pars_df = pd.read_csv(parsdir + patientNo + '/' + arg)
            A, K, _, pars, best_pars = [np.array(pars_df.loc[i, ~np.isnan(pars_df.loc[i, :])]) for i in range(5)]
            PARS_LIST.append(best_pars)
        PARS_ARR = np.stack(PARS_LIST)
        pars = np.mean(PARS_ARR, axis=0)
        phi, gamma = pars[-3:-1]
        a21 = t / (1 + np.exp(-gamma * X/12/28))
        A21.append(a21)
        ppo_states = pd.read_csv('../PPO_states/analysis/' + patientNo + '_evolution_states.csv', index_col=0)
        x_p = np.array(ppo_states.index).reshape(-1)
        X_P.append(x_p[-1].item())
        c2_p = (ppo_states.ad * (t / (1 + np.exp(-gamma * x_p / 12 / 28)).reshape(-1)) / K[1]) ** phi
        C2_P.append(c2_p)
        experts_states = pd.read_csv('../Experts_states/analysis/' + patientNo + '_experts_states.csv', index_col=0)
        x_e = np.array(experts_states.index).reshape(-1)
        X_E.append(x_e[-1].item())
        c2_e =  (experts_states.ad * (t / (1 + np.exp(-gamma*x_e/12/28)).reshape(-1)) / K[1]) ** phi
        C2_E.append(c2_e)
        max_states = pd.read_csv('../MAX_states/analysis/' + patientNo + '_evolution_states.csv', index_col=0)
        x_m = np.array(max_states.index).reshape(-1)
        X_M.append(x_m[-1].item())
        c2_m = (max_states.ad * (t / (1 + np.exp(-gamma * x_m / 12 / 28)).reshape(-1)) / K[1]) ** phi
        C2_M.append(c2_m)
        if patientNo=='patient036':
            xe_36 = x_e
            c2_e_36 = c2_e
            xp_36 = x_p
            c2_p_36 = c2_p
            xm_36 = x_m
            c2_m_36 = c2_m
        if patientNo=='patient106':
            xe_106=x_e
            c2_e_106 = c2_e
            xp_106 = x_p
            c2_p_106 = c2_p
            xm_106 = x_m
            c2_m_106 = c2_m

    max_x_p = max(X_P)
    max_x_e = max(X_E)
    max_x_m = max(X_M)
    for i in range(len(X_P)):
        x_p_i = X_P[i]
        C2_E[i] = C2_E[i][0:max_x_p]
        x_e_i = len(C2_E[i])
        x_m_i = X_M[i]
        C2_P[i] = np.concatenate((C2_P[i], -np.ones(int(max_x_p - x_p_i))))
        C2_E[i] = np.concatenate((C2_E[i], -np.ones(int(max_x_p - x_e_i))))
        C2_M[i] = np.concatenate((C2_M[i], -np.ones(int(max_x_m - x_m_i))))
    C2_P_ARR = np.stack(C2_P)
    C2_E_ARR = np.stack(C2_E)
    C2_M_ARR = np.stack(C2_M)
    mean_c2_p = np.array([np.mean(C2_P_ARR[C2_P_ARR[:, i] != -1, i], axis=0) for i in range(C2_P_ARR.shape[1])])
    mean_c2_e = np.array([np.mean(C2_E_ARR[C2_E_ARR[:, i] != -1, i], axis=0) for i in range(C2_E_ARR.shape[1])])
    mean_c2_m = np.array([np.mean(C2_M_ARR[C2_M_ARR[:, i] != -1, i], axis=0) for i in range(C2_M_ARR.shape[1])])
    sd_c2_p = np.array([st.sem(C2_P_ARR[C2_P_ARR[:, i] != -1, i]) for i in range(C2_P_ARR.shape[1])])
    sd_c2_e = np.array([st.sem(C2_E_ARR[C2_E_ARR[:, i] != -1, i]) for i in range(C2_E_ARR.shape[1])])
    sd_c2_m = np.array([st.sem(C2_M_ARR[C2_M_ARR[:, i] != -1, i]) for i in range(C2_M_ARR.shape[1])])
    low_c2_p_bound, high_c2_p_bound = st.t.interval(0.95, mean_c2_p.shape[0] - 1, loc=mean_c2_p, scale=sd_c2_p)
    low_c2_e_bound, high_c2_e_bound = st.t.interval(0.95, mean_c2_e.shape[0] - 1, loc=mean_c2_e, scale=sd_c2_e)
    low_c2_m_bound, high_c2_m_bound = st.t.interval(0.95, mean_c2_m.shape[0] - 1, loc=mean_c2_m, scale=sd_c2_m)
    A21 = np.stack(A21)
    mean_A21 = np.mean(A21, axis=0)
    sd_A21 = st.sem(A21)
    low_a21_bound, high_a21_bound = st.t.interval(0.95, mean_A21.shape[0] - 1, loc=mean_A21, scale=sd_A21)
    low_a21_bound[0] = 0.5
    high_a21_bound[0] = 0.5
    a21_36 = 1 / (1 + np.exp(-gamma36 * X/12/28))
    a21_106 = 1 / (1 + np.exp(-gamma106 * X/12/28))


    plt.style.use(["science", "nature"])
    fig1 = plt.figure(figsize=(5, 4))
    ax1 = fig1.add_subplot(2, 1, 1)
    # ax1.plot(X, mean_A21, lw=1.4, c=cs[1])
    # plt.fill_between(X, low_a21_bound, high_a21_bound, color=colorAlpha_to_rgb(cs[1], 0.3)[0], label='95\% CI')
    if R == 'resistance':
        ax1.plot(xp_36, c2_p_36, lw=1.4, c=cs[1], label='$I^2$ADT')
        plt.scatter(x =xp_36[-1] , y= c2_p_36[xp_36[-1]],marker=4, color='black',s=100, zorder=3)
        ax1.plot(xe_36, c2_e_36, lw=1.4, c=cs[3], label='IADT')
        plt.scatter(x=xe_36[-1], y=c2_e_36[xe_36[-1]], marker='X', color='black', s=100, zorder=3)
        ax1.plot(xm_36, c2_m_36, lw=1.4, c=cs[7], label='ADT')
        plt.scatter(x=xm_36[-1], y=c2_m_36[xm_36[-1]], marker=4, color='black', s=100, zorder=3)
        plt.yticks([0,1,2],[0, 1, 2],fontsize=20)
    else:
        ax1.plot(xp_106, c2_p_106, lw=1.4, c=cs[1], label='$I^2$ADT')
        plt.scatter(x=xp_106[-1], y=c2_p_106[xp_106[-1]], marker=4, color='black', s=100, zorder=3)
        ax1.plot(xe_106, c2_e_106, lw=1.4, c=cs[3], label='IADT')
        plt.scatter(x=xe_106[-1], y=c2_e_106[xe_106[-1]], marker='X', color='black', s=100, zorder=3)
        ax1.plot(xm_106, c2_m_106, lw=1.4, c=cs[7],  label='ADT')
        plt.scatter(x=xm_106[-1], y=c2_m_106[xm_106[-1]], marker=4, color='black', s=100, zorder=3)
        plt.yticks([0, 1, 2], [0,1, 2], fontsize=20)
    plt.xticks(ticks=[], labels=[], fontsize = 20)
    plt.ylabel('', fontsize=2)
    # plt.xlabel('Time (Days)', fontsize=45)
    plt.yticks( fontsize=20)
    #plt.legend(fontsize=20)
    ax2 = fig1.add_subplot(212)
    ax2.plot(np.arange(max_x_p+1), mean_c2_p, lw=1.4, c=cs[1], label='$I^2$ADT')
    plt.fill_between(np.arange(max_x_p+1), low_c2_p_bound, high_c2_p_bound, color=colorAlpha_to_rgb(cs[1], 0.3)[0])
    ax2.plot(np.arange(max_x_p), mean_c2_e, lw=1.4, c=cs[3], label='IADT')
    plt.fill_between(np.arange(max_x_p), low_c2_e_bound, high_c2_e_bound, color=colorAlpha_to_rgb(cs[3], 0.3)[0])
    ax2.plot(np.arange(max_x_m+1), mean_c2_m, lw=1.4, c=cs[7], label='ADT')
    plt.fill_between(np.arange(max_x_m+1), low_c2_m_bound, high_c2_m_bound, color=colorAlpha_to_rgb(cs[7], 0.3)[0])
    plt.hlines(1, -7.5, 120*28 + 7.5, lw=1.4, ls='--', color='black')
    plt.xlabel('Time (Months)', fontsize=22)
    plt.xticks(ticks = [0, 30*28, 60*28, 90*28, 120*28], labels = [0, 30, 60, 90, 120], fontsize=20)
    plt.ylabel('', fontsize=1)
    #plt.text(x=-.1, y= 0.2, s='Competition Ad. of response', fontsize=22, rotation='vertical')
    plt.yticks(ticks=[0, 1, 2],labels=[0, 1, 2],fontsize=20)
    plt.legend(ncol=3, fontsize=13.5)
    plt.tight_layout()
    plt.savefig('../Analysis/Figure/' + R +'_competition_intensity.eps', dpi=300, bbox_inches='tight')
    plt.show()

# response #
