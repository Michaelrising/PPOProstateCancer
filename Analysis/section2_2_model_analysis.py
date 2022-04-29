import scipy.stats as st
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as st
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import normalize
from _utils import *
from scipy.spatial import ConvexHull
from scipy.stats import ranksums
from scipy import interpolate
from sklearn.metrics import confusion_matrix


patientlist = [1, 2, 3, 4, 6, 11, 12, 13, 15, 16, 17, 19, 20, 24, 25, 29, 30, 31, 32, 36, 37, 40, 42, 44, 46, 50, 51,
               52, 54, 58, 60, 61, 62, 63, 66, 71, 75, 77, 78, 79, 84, 85, 86, 87, 88, 91, 92, 93, 94, 95, 96, 97,
               99, 100, 101, 102, 104, 105, 106, 108]
patient_yes = [11, 12, 19, 25, 36, 52, 54, 85, 88, 99, 101]
cs = sns.color_palette("Paired")
parsdir = "../GLV/analysis-dual-sigmoid/model_pars/"

A_list = []
K_list = []
states_list = []
ALL_F_PARS_LIST = {}
ALL_B_PARS_LIST = {}
# reading the ode parameters and the initial/terminal states
markPatient = [6, 11]
markIndex = []
finalDay_list = []
for i in patientlist:
    if len(str(i)) == 1:
        patientNo = "patient00" + str(i)
    elif len(str(i)) == 2:
        patientNo = "patient0" + str(i)
    else:
        patientNo = "patient" + str(i)
    pars_list = os.listdir(parsdir + patientNo)
    patientData = np.array(pd.read_csv('../Data/dataTanaka/Bruchovsky_et_al/' + patientNo + '.txt'))
    finalDay_list.append(patientData[-1,-1] - patientData[0,-1])
    pars_list.sort()
    f_pars_list = []
    b_pars_list = []
    for pars in pars_list:
        pars_df = pd.read_csv(parsdir + patientNo + '/' + pars)
        A, K, states, final_pars, best_pars = [np.array(pars_df.loc[i, ~np.isnan(pars_df.loc[i, :])]) for i in range(5)]
        f_pars_list.append(final_pars)
        b_pars_list.append(best_pars)
    ALL_F_PARS_LIST[patientNo] = np.stack(f_pars_list)
    ALL_B_PARS_LIST[patientNo] = np.stack(b_pars_list)
    print(i)

mean_res_index = []
mean_pars = []
for i in patientlist:
    if len(str(i)) == 1:
        patientNo = "patient00" + str(i)
    elif len(str(i)) == 2:
        patientNo = "patient0" + str(i)
    else:
        patientNo = "patient" + str(i)
    pars_arr = ALL_F_PARS_LIST[patientNo]
    mean_res_index.append(pars_arr.mean(axis=0)[-2])
    mean_pars.append(pars_arr.mean(axis=0))

# All pars distribution
mean_pars = np.stack(mean_pars)
mean_pars[:, -2] = - mean_pars[:, -2]
pars_all = pd.DataFrame(mean_pars.flatten('F'), columns = ['pars'])
items = ['r1', 'r2', 'beta1', 'beta2', 'phi', 'gamma', 'betac']
pars_all['labels'] = np.array([[items[i]] * mean_pars.shape[0] for i in range(len(items))]).flatten()
plt.style.use(['science', "nature"])
plt.figure(figsize=(5, 2))
sns.boxplot(x = 'labels', y = 'pars', data = pars_all, palette='Paired')
locs, labels = plt.xticks()
plt.xticks(ticks = locs, labels = ['$r_1$', '$r_2$', '$\\beta_1$', '$\\beta_2$', '$\\phi$', '$\\gamma$', '$\\beta_c$'], fontsize =12)
plt.xlabel('')
plt.yticks(fontsize =12)
plt.ylabel('')
plt.tight_layout()
plt.savefig('./Figure/all_pars_distribution.eps', dpi=400, bbox_inches='tight')
plt.show()

# the CI for all the paras for all the patients
ALL_F_PARS_ARR = []
for k in list(ALL_F_PARS_LIST.keys()):
    l = ALL_F_PARS_LIST[k].tolist()
    for i in range(len(l)):
        ALL_F_PARS_ARR.append(l[i])

ALL_F_PARS_ARR = np.vstack(ALL_F_PARS_ARR)
ALL_F_PARS_ARR[:, -2] = - ALL_F_PARS_ARR[:, -2]
all_pars_mean = ALL_F_PARS_ARR.mean(axis = 0)
all_pars_std = st.sem(ALL_F_PARS_ARR)
low_CI_all_pars, high_CI_all_pars = st.t.interval(0.95, 601-1, loc=all_pars_mean, scale=all_pars_std)

print("All pars Mean: {}".format(all_pars_mean))
print("CI for all pars: \n r1: ({}, {})\n r2: ({}, {}) \n beta1: ({}, {}) "
      "\n beta2: ({}, {}) \n phi:({}, {}) \n gamma: ({}, {})\n betac:({},{})".
      format(low_CI_all_pars[0], high_CI_all_pars[0],low_CI_all_pars[1], high_CI_all_pars[1],
             low_CI_all_pars[2], high_CI_all_pars[2],low_CI_all_pars[3], high_CI_all_pars[3],
             low_CI_all_pars[4], high_CI_all_pars[4],low_CI_all_pars[5], high_CI_all_pars[5],
             low_CI_all_pars[6], high_CI_all_pars[6]))
#r1 r2 distribution#
colors = [cs[2*i+1] for i in range(2)]
symbols = [ 'o', '^']
ms = [10, 9]
fig = plt.figure(figsize=(5, 4))

plt.scatter(x = mean_pars[np.isin( np.array(patientlist), np.array(patient_yes)), 0], y = mean_pars[np.isin( np.array(patientlist), np.array(patient_yes)), 1],
            s = 40,c = colorAlpha_to_rgb(colors[1], 0.6)[0], marker = '^')
plt.scatter(x = mean_pars[~np.isin( np.array(patientlist), np.array(patient_yes)), 0], y = mean_pars[~np.isin( np.array(patientlist), np.array(patient_yes)), 1],
            s=40, c = colorAlpha_to_rgb(colors[0], 0.6)[0])
y_loc,_ = plt.yticks()
plt.yticks([0.01, 0.02, 0.03, 0.04, 0.05, 0.06], [1, 2,3, 4, 5,6], fontsize = 23)
plt.text(0, 0.068, '$\\times 10^{-2}$', fontsize=12)
x_loc, _ = plt.xticks()
plt.xticks([0, .02, .04, .06, .08 , .10], [0, 2, 4, 6, 8, 10],fontsize = 23)
plt.text(0.107, 0.0025, '$\\times 10^{-2}$', fontsize=12)
# loc = [[0.017, 1.], [0.011, 0.06], [0.03, 0.65], [0.013, -0.35], [0.017, -1.45],[0.054, -.3]]
plt.xlabel('$r_1$', fontsize = 25)
plt.ylabel('$r_2$', fontsize = 25)
legends = ["response", 'resistance']
legend_elements = [Line2D([0], [0], marker = symbols[i], color='w', label=legends[i],
                          markerfacecolor=colorAlpha_to_rgb(mcolor, 0.6)[0], markersize=ms[i]) for i, mcolor in enumerate(colors)]
plt.legend(handles = legend_elements, loc='upper right', ncol=1, fontsize = 18)
plt.tight_layout()
plt.savefig("./Figure/r1_r2_distributions.eps", dpi=300, bbox_inches ='tight')
plt.show()



############## Clinician Classification ###############
patient_yes = [11, 12, 19, 25, 36, 41, 52, 54, 85, 88, 99, 101]
patient_yes = list(set(patient_yes) & set(patientlist))
patient_yes.sort()
patient_no = list(set(patientlist) - set(patient_yes))
patient_no.sort()
#### Randomization with leave-pair out CV #####
FinalDay = np.array(finalDay_list)
ResIndexlabels = pd.DataFrame(np.concatenate((np.zeros(len(patient_no)), np.ones(len(patient_yes)))),
                              index=patient_no + patient_yes)
ResIndexlabels = ResIndexlabels.sort_index()
fpr_list = []
tpr_list = []
thres_list = []
auc_list = []
PRED_TPR = []

##### Supplementary analysis of the model #######

def classify(true_y, train_res):
    sp_sen = 0
    sp = 0
    sen = 0
    clf = 0
    for res in np.unique(train_res):
        pred_y = np.zeros_like(true_y)
        pred_no = train_res < res
        pred_yes = train_res >= res
        pred_y[pred_yes] = 1
        tn, fp, fn, tp = confusion_matrix(true_y, pred_y).ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        if sp_sen <= (specificity + sensitivity) and clf < res:
            sp_sen = specificity + sensitivity
            sp = specificity
            sen = sensitivity
            clf = res
    return clf, sp_sen, sp, sen


def clf_predict(threshold, test_res, true_y):
    pred_y = []
    for res in test_res:
        if res < threshold:
            pred_y.append(0)
        else:
            pred_y.append(1)
    score = 1 - sum(abs(np.array(pred_y) - true_y))/true_y.shape[0]
    return pred_y, score


sp_matrix = np.zeros((49, 11))
sen_matrix = np.zeros((49, 11))
score_matrix = np.zeros((49, 11))
for k in patient_yes:
    position_k = np.where(np.array(patient_yes) == k)[0][0]
    for i in patient_no:
        position_i =  np.where(np.array(patient_no) == i)[0][0]
        rand_res_list = []
        for p_no in patientlist:
            if len(str(p_no)) == 1:
                patientNo = "patient00" + str(p_no)
            elif len(str(p_no)) == 2:
                patientNo = "patient0" + str(p_no)
            else:
                patientNo = "patient" + str(p_no)
            pars_arr = ALL_F_PARS_LIST[patientNo]
            rand = np.random.randint(1, pars_arr.shape[0], 1)
            res = pars_arr[rand.item(), -2]
            rand_res_list.append(res)
        rand_res_df = pd.DataFrame(-np.array(rand_res_list), index=patientlist, columns=['resIndex'])
        train_position, test_position = list(set(patientlist) - set([i, k])), [i, k]
        test_position.sort()
        train_res, test_res = rand_res_df.loc[train_position], rand_res_df.loc[test_position]
        train_labels, test_labels = ResIndexlabels.loc[train_position], ResIndexlabels.loc[test_position]
        # clf = Perceptron(fit_intercept = False, tol=1e-3, random_state=0)
        # clf.fit(train_res.to_numpy(), train_labels.to_numpy().ravel())
        clf, sp_sen, sp, sen = classify(train_labels.to_numpy().ravel(), train_res.to_numpy().ravel())
        _, test_score = clf_predict(clf, test_res.to_numpy().ravel(), test_labels.to_numpy().ravel())
        sp_matrix[position_i, position_k] = sp
        sen_matrix[position_i, position_k] = sen
        score_matrix[position_i, position_k] = test_score

mean_sp = sp_matrix.mean()
mean_sen = sen_matrix.mean()
mean_score = score_matrix.mean()
sd_sp = st.sem(sp_matrix.flatten())
sd_sen = st.sem(sen_matrix.flatten())
sd_score = st.sem(score_matrix.flatten())
low_CI_sp, high_CI_sp = st.t.interval(0.95, 49*11-1, loc=mean_sp, scale=sd_sp)
low_CI_sen, high_CI_sen = st.t.interval(0.95, 49*11-1, loc=mean_sen, scale=sd_sen)
low_CI_score, high_CI_score = st.t.interval(0.95, 49*11-1, loc=mean_score, scale=sd_score)

print("Mean Specificity:{}; CI:({},{})".format(mean_sp, low_CI_sp, high_CI_sp))
print("Mean Sensitivity:{}; CI:({},{})".format(mean_sen, low_CI_sen, high_CI_sen))
print("Mean Score:{}; CI:({},{})".format(mean_score, low_CI_score, high_CI_score))


ResIndexlabels = pd.DataFrame(np.concatenate((np.ones(len(patient_no)), np.zeros(len(patient_yes)))),
                              index=patient_no + patient_yes)
ResIndexlabels = ResIndexlabels.sort_index()

# Compare with the standard IADT classification ###
##################################
############ Fig3.c ##############
##################################
plt.style.use(['science', "nature"])
plt.figure(figsize=(5, 4))

mean_res_index = np.stack(mean_res_index)
# type_arr = np.array([2 if i in patient_yes else 1 for i in patientlist ])

CompetitionIndex = 1 / (1 + np.exp(- mean_res_index * FinalDay / 28 / 12))

fpr, tpr, thresholds = roc_curve(ResIndexlabels, - mean_res_index)
mean_auc_score = auc(fpr, tpr)

lw = 1
# x=np.mean(np.stack(fpr_list), 0)
# y=np.mean(np.stack(tpr_list), 0)
# plt.plot(x,y , lw=3, c=cs[1], ls='--',label="AUC = %0.2f" % np.mean(auc_list))
plt.plot(fpr, tpr, color=cs[7], lw=3, label="AUC = %0.2f" % mean_auc_score)
# plt.fill_between(x, low_CI_bound, high_CI_bound, alpha=0.5)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel("1-specificity", fontsize=23)
plt.ylabel("sensitivity", fontsize=23)
plt.legend(loc="lower right", fontsize=23)
plt.xticks(labels=[0,0.2,0.4,0.6,0.8, 1.0], ticks=[0, 0.2,0.4,0.6,0.8, 0.98], fontsize = 22)
plt.yticks(labels=[0.2,0.4,0.6,0.8, 1.0], ticks=[0.2,0.4,0.6,0.8, 0.98],fontsize = 22)
plt.subplots_adjust(left=0.2, right=1, top=1, bottom=0.2)
plt.tight_layout()
plt.savefig('./Figure/ROC_competition_index.eps', dpi=300, bbox_inches='tight')
plt.show()


response = patient_no
resistance = patient_yes
colors = [cs[2*i+1] for i in range(2)]
symbols = ['circle' for _ in range(2)]
all_patients_pars_mean_arr = np.stack([ALL_F_PARS_LIST[i].mean(0) for i in list(ALL_F_PARS_LIST.keys())])
all_patients_pars_mean_arr[:, -2] = -all_patients_pars_mean_arr[:, -2]
all_patients_pars_mean_df = pd.DataFrame(np.concatenate((all_patients_pars_mean_arr, ResIndexlabels), axis=1), index = patientlist,
                                         columns=['r1', 'r2', 'beta1', 'beta2', 'phi', 'gamma', 'betac', 'labels'])
all_patients_pars_mean_df['c'] = all_patients_pars_mean_df.labels.map({i: colors[i] for i in range(2)})
all_patients_pars_mean_df['s'] = all_patients_pars_mean_df.labels.map({i: symbols[i] for i in range(2)})


fig = plt.figure(figsize=(5, 4))
plt.scatter(all_patients_pars_mean_df.r2, all_patients_pars_mean_df.gamma,
            c = all_patients_pars_mean_df.c.apply(lambda x: colorAlpha_to_rgb(x, 0.6)[0]), s=40, zorder=3)

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
loc = [[0.017, 1.], [0.011, 0.06], [0.03, 0.65], [0.013, -0.35], [0.017, -1.45],[0.054, -.3]]
plt.xlabel('Growth rate of RST ($r_2$)', fontsize = 22)
plt.ylabel('RST IDX ($\gamma$)', fontsize = 22)
legends = ["response", 'resistance']
legend_elements = [Line2D([0], [0], marker = 'o', color='w', label=legends[i],
                          markerfacecolor=colorAlpha_to_rgb(mcolor, 0.6)[0], markersize=10) for i, mcolor in enumerate(colors)]
plt.legend(handles = legend_elements, loc='right', ncol=1, fontsize = 14.5)
plt.tight_layout()
plt.savefig("./Figure/r2_gamma_distributions.eps", dpi=300, bbox_inches ='tight')
plt.show()



# data_for_kmeans['cluster'] = normalized_data_for_kmeans['cluster']
# data_for_kmeans.to_csv("./kmeans_clustering_results.csv")

##################################
############ Fig3.a ##############
##################################
plt.figure(figsize=(5, 4))
sns.swarmplot(x = 'labels', y='gamma',data =all_patients_pars_mean_df ,palette=[colorAlpha_to_rgb(cs[3], 0.6)[0], colorAlpha_to_rgb(cs[1], 0.6)[0]], size=8, orient='v')
plt.hlines(all_patients_pars_mean_df.loc[response, 'gamma'].mean(),  0.75, 1.25, colors=cs[1], lw=5, ls='--', zorder=3)
plt.hlines(all_patients_pars_mean_df.loc[resistance, 'gamma'].mean(),-0.25, .25, colors=cs[3], lw=5, ls='--', zorder=3)
plt.xlabel('')
plt.xticks(ticks=[0, 1], labels=['resistance', 'response'], fontsize=23)
plt.yticks(ticks=[0, 0.2, 0.4, 0.6, 0.8], labels = [0, 2, 4, 6, 8], fontsize=23)
plt.ylabel('')
#plt.xticks(ticks=[], labels=[])
plt.tight_layout()
plt.savefig('./Figure/distribution_gamma.eps', dpi=300,  bbox_inches = 'tight')
plt.show()

plt.style.use(['science', 'nature'])
fig = plt.figure(figsize=(5, 4))
ax1 = fig.add_subplot(131)
ax1 = sns.swarmplot(x = 'labels', y='gamma',data =all_patients_pars_mean_df ,palette=[colorAlpha_to_rgb(cs[3], 0.6)[0], colorAlpha_to_rgb(cs[1], 0.6)[0]], size=8, orient='v')
plt.hlines(all_patients_pars_mean_df.loc[response, 'gamma'].mean(),  0.75, 1.25, colors=cs[1], lw=5, ls='--', zorder=3)
plt.hlines(all_patients_pars_mean_df.loc[resistance, 'gamma'].mean(),-0.25, .25, colors=cs[3], lw=5, ls='--', zorder=3)
plt.xlabel('$\gamma$', fontsize=25)
plt.yticks(ticks=[0, 0.2, 0.4, 0.6, 0.8], labels = [0, 2, 4, 6, 8], fontsize=23)
plt.ylabel('')
plt.xticks(ticks=[], labels=[])
plt.tight_layout()
ax2 = fig.add_subplot(132)
ax2 = sns.swarmplot(x='labels', y='r1', data =all_patients_pars_mean_df ,palette=[colorAlpha_to_rgb(cs[3], 0.6)[0], colorAlpha_to_rgb(cs[1], 0.6)[0]], size=8, orient='v')
loc, lab = plt.xticks()
plt.hlines(all_patients_pars_mean_df.loc[response, 'r1'].mean(), 0.75, 1.25, colors=cs[1], lw=5, ls='--', zorder=3)
plt.hlines(all_patients_pars_mean_df.loc[resistance, 'r1'].mean(), -0.25, .25, colors=cs[3], lw=5, ls='--', zorder=3)
plt.xlabel('$r_1$', fontsize=25)
ax2.set_ylabel('')
ax2.set_yticks([])
plt.ylim(0.0, 0.1)
plt.xticks(ticks=[], labels=[])
plt.tight_layout()
ax3 = fig.add_subplot(133)
ax3 = sns.swarmplot(x='labels', y='r2', data =all_patients_pars_mean_df ,palette=[colorAlpha_to_rgb(cs[3], 0.6)[0], colorAlpha_to_rgb(cs[1], 0.6)[0]], size=8, orient='v')
loc, lab = plt.xticks()
plt.hlines(all_patients_pars_mean_df.loc[response, 'r2'].mean(), 0.75, 1.25, colors=cs[1], lw=5, ls='--', zorder=3)
plt.hlines(all_patients_pars_mean_df.loc[resistance, 'r2'].mean(), -0.25, .25,  colors=cs[3], lw=5, ls='--', zorder=3)
plt.xlabel('$r_2$', fontsize=25)
ax3.yaxis.tick_right()
ax3.yaxis.set_label_position("right")
ax3.set_ylabel('', fontsize=23)
plt.xticks(ticks=[], labels=[])
loc1, lab1 = plt.yticks()
plt.yticks(ticks=[0.01,0.03,0.05,0.07, 0.09], labels = [1, 3, 5, 7, 9], fontsize=23)
# plt.text(1, 1.01, '$10^{-2}$', fontsize = 10)
plt.ylim(0.0, 0.1)
legends = ["response", 'resistance']
legend_elements = [Line2D([0], [0], marker = 'o', color='w', label=legends[i],
                          markerfacecolor=mcolor, markersize=8) for i, mcolor in enumerate(colors)]
#plt.legend(handles = legend_elements, ncol=2, fontsize = 16.5)
plt.tight_layout()
plt.savefig('./Figure/distribution_three_characteristics.eps', dpi=300,  bbox_inches = 'tight')
plt.show()

# difference of gamma between response and resistance #
mean_resistance_gamma = all_patients_pars_mean_df.loc[resistance, 'gamma'].mean()
sd_resistance_gamma = st.sem(all_patients_pars_mean_df.loc[resistance, 'gamma'])
low_CI_rst_gamma, high_CI_rst_gamma = st.t.interval(0.95, df=len(patient_yes) - 1, loc=mean_resistance_gamma, scale=sd_resistance_gamma)
print("Resistance mean gamma:{} with CI({}, {})".format(mean_resistance_gamma, low_CI_rst_gamma, high_CI_rst_gamma))

mean_response_gamma = all_patients_pars_mean_df.loc[response, 'gamma'].mean()
sd_response_gamma = st.sem(all_patients_pars_mean_df.loc[response, 'gamma'])
low_CI_rsp_gamma, high_CI_rsp_gamma = st.t.interval(0.95, df = len(patient_no), loc=mean_response_gamma, scale=sd_response_gamma)
print("Response mean gamma:{} with CI({}, {})".format(mean_response_gamma, low_CI_rsp_gamma, high_CI_rsp_gamma))

ranksums(x = all_patients_pars_mean_df.loc[resistance, 'gamma'], y = all_patients_pars_mean_df.loc[response, 'gamma'])
ranksums(x = all_patients_pars_mean_df.loc[resistance, 'r1'], y = all_patients_pars_mean_df.loc[response, 'r1'])
ranksums(x = all_patients_pars_mean_df.loc[resistance, 'r2'], y = all_patients_pars_mean_df.loc[response, 'r2'])


##################################
############ Fig3.b ##############
##################################
##  A21 trends #
X = np.arange(0, 120*28).reshape(-1)

A21_dict = {}
for R in ['resistance', 'response']:
    A21 = []
    if R == 'resistance':
        patient_group = resistance
    else:
        patient_group = response
    for i in patient_group:
        if len(str(i)) == 1:
            patientNo = "patient00" + str(i)
        elif len(str(i)) == 2:
            patientNo = "patient0" + str(i)
        else:
            patientNo = "patient" + str(i)
        print(patientNo)
        parslist = os.listdir(parsdir + patientNo)
        PARS_LIST = []
        for arg in parslist:
            pars_df = pd.read_csv(parsdir + patientNo + '/' + arg)
            A, K, _, pars, best_pars = [np.array(pars_df.loc[i, ~np.isnan(pars_df.loc[i, :])]) for i in range(5)]
            PARS_LIST.append(best_pars)
        PARS_ARR = np.stack(PARS_LIST)
        pars = np.mean(PARS_ARR, axis=0)
        phi, gamma = pars[-3:-1]
        a21 = 1 / (1 + np.exp(-gamma * X/12/28))
        A21.append(a21)
    A21 = np.stack(A21)
    mean_A21 = np.mean(A21, axis=0)
    sd_A21 = st.sem(A21)
    low_a21_bound, high_a21_bound = st.t.interval(0.95, mean_A21.shape[0] - 1, loc=mean_A21, scale=sd_A21)
    low_a21_bound[0] = 0.5
    high_a21_bound[0] = 0.5
    A21_dict[R] = (mean_A21, low_a21_bound, high_a21_bound)

plt.style.use(["science", "nature"])
fig1 = plt.figure(figsize=(5, 4))
# resistance
ax1 = fig1.add_subplot(2, 1, 1)
ax1.plot(X, A21_dict['resistance'][0], lw=4, c=cs[3])
plt.fill_between(X, A21_dict['resistance'][1], A21_dict['resistance'][2], color=colorAlpha_to_rgb(cs[3], 0.3)[0], label='resistance')
plt.xticks([], [],fontsize=23)
plt.yticks([0.1, 0.3, 0.5], [0.1, 0.3, 0.5], fontsize=23)
plt.ylabel('')
plt.xlabel('')
plt.legend(fontsize=18)
ax2 = fig1.add_subplot(2, 1, 2)
ax2.plot(X, A21_dict['response'][0], lw=4, c=cs[1])
plt.fill_between(X,  A21_dict['response'][1],  A21_dict['response'][2], color=colorAlpha_to_rgb(cs[1], 0.3)[0], label='response')
plt.xticks(ticks = [0, 30*28, 60*28, 90*28, 120*28],labels=[0, 30, 60, 90, 120],fontsize = 23)
plt.xlabel('Time (Months)', fontsize=25)
# plt.ylabel(loc = 'top', ylabel='$A_{21}$', fontsize=40)
plt.text( x=-0.2, y=1.05, s='$A_{21}$',fontsize=25, transform=ax2.transAxes)
plt.yticks(fontsize=23)
plt.ylim(0.25, 0.55)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig('./Figure/A21_changes.eps', dpi=300, bbox_inches = 'tight')
plt.show()



