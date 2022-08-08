import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, roc_curve, auc
from _utils import *
from scipy.stats import sem, t
import sys
sys.path.append(".")

patientList = [1, 2, 3, 4, 6, 11, 12, 13, 15, 16, 17, 19, 20, 24, 25, 29, 30, 31, 32, 36, 37, 40, 42, 44, 46, 50, 51,
               52, 54, 58, 60, 61, 62, 63, 66, 71, 75, 77, 78, 79, 84, 85, 86, 87, 88, 91, 92, 93, 94, 95, 96, 97,
               99, 100, 101, 102, 104, 105, 106, 108] # kick 56 83
# patientList = [1, 2, 3, 6, 11, 12, 15, 16, 17, 19, 20, 24, 25, 29, 30, 31, 32, 36, 37, 40, 42, 44, 51,
#                52, 54, 58, 60, 61, 77, 78, 79, 84, 85, 86, 87, 88, 91, 93, 94, 95, 96, 97,
#                 100, 102, 104, 105, 106, 108]
patient_no = [1, 2, 3, 4, 6, 13, 14, 15, 16, 17,
              20, 22, 24, 26, 28, 29, 30, 31, 37, 39,
              40, 42, 44, 50, 51, 55, 56, 58, 60, 61,
              62, 63, 66, 71, 75, 77, 78, 79, 81, 84,
              86, 87, 91, 93, 94, 95, 96, 97, 100, 102,
              104, 105, 106, 108, 109, 32, 46, 64, 83, 92]
patient_no = list(set(patientList) & set(patient_no))
patient_yes = [11, 12, 19, 25, 36, 41, 52, 54, 85, 88, 99, 101]
patient_yes = list(set(patientList) & set(patient_yes))
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
unmarkIndex = []
YesIndex = []
NoIndex = []
finalDay_list = []
for i in patientList:
    if len(str(i)) == 1:
        patientNo = "patient00" + str(i)
    elif len(str(i)) == 2:
        patientNo = "patient0" + str(i)
    else:
        patientNo = "patient" + str(i)
    pars_list = os.listdir(parsdir + patientNo)
    patientData = np.array(pd.read_csv('../../data/dataTanaka/Bruchovsky_et_al/' + patientNo + '.txt'))
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
mean_res_index = []
for i in patientList:
    if len(str(i)) == 1:
        patientNo = "patient00" + str(i)
    elif len(str(i)) == 2:
        patientNo = "patient0" + str(i)
    else:
        patientNo = "patient" + str(i)
    pars_arr = ALL_F_PARS_LIST[patientNo]
    mean_res_index.append(pars_arr.mean(axis=0)[-2])
    if i in markPatient:
        markIndex.append(pars_arr.mean(axis=0)[-2])
    else:
        unmarkIndex.append(pars_arr.mean(axis=0)[-2])
    if i in patient_yes:
        YesIndex.append(pars_arr.mean(axis=0)[-2])
    else:
        NoIndex.append(pars_arr.mean(axis=0)[-2])
mean_res_index = np.stack(mean_res_index)


# Draw plots
plt.style.use(['science', 'nature'])
plt.figure(figsize=(5, 4))
ax = sns.boxplot(data=-mean_res_index, color=cs[0], orient='v')
sns.swarmplot(data = -np.stack(YesIndex), color=cs[1], size=5)
sns.swarmplot(data = -np.stack(NoIndex), color=cs[3], size=5)
plt.scatter(x=-markIndex[1], y=0, color=cs[5], label='patient011', marker='*', s=150, zorder=3)
plt.scatter(x=-markIndex[0], y=0, color=cs[7], label='patient006', marker='*', s=150, zorder=3)
plt.scatter(x =-np.array(YesIndex), y =[0 for _ in range(len(YesIndex))], color='red' )
plt.scatter(x = -np.array(NoIndex), y=[0 for _ in range(len(NoIndex))], color='green')
# Add jitter with the swarmplot function
# ax = sns.swarmplot(markIndex[0], color = 'red', label = 'patient006')
# ax = sns.swarmplot(markIndex[1], color = 'yellow', label = 'patient011')
plt.xlabel("RST IDX $\gamma$", fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(labels=[], ticks=[])
plt.ylabel("", fontsize=22)
plt.legend(loc='upper right', fontsize=16.5)
plt.subplots_adjust(left=0.2, right=1, top=1, bottom=0.2)
# adding transparency to colors
for patch in ax.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor(colorAlpha_to_rgb((r, g, b), .3)[0])
plt.tight_layout()
plt.savefig('../../Analysis/Figure/ResIndex_distribution.eps', dpi=300 , bbox_inches = 'tight')
plt.show()
plt.close()

mean_res_index = []
mean_pars = []
for i in patientList:
    if len(str(i)) == 1:
        patientNo = "patient00" + str(i)
    elif len(str(i)) == 2:
        patientNo = "patient0" + str(i)
    else:
        patientNo = "patient" + str(i)
    pars_arr = ALL_F_PARS_LIST[patientNo]
    mean_res_index.append(pars_arr.mean(axis=0)[-2])
    mean_pars.append(pars_arr.mean(axis=0))
##################################
############ Fig2.c ##############
##################################
print("Fig2.d is saved as: ../../Analysis/Figure/all_pars_distribution.eps")
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
plt.savefig('../../Analysis/Figure/all_pars_distribution.eps', dpi=400, bbox_inches='tight')
plt.show()



##################################
############ Fig2.d ##############
##################################
print("Fig2.d is saved as: ../../Analysis/Figure/Validation_PSA.eps")
valdir = "../../data/model_validate"
vallist = os.listdir(valdir)
true_list = []
predict_list = []
for file in vallist:
    val_df = pd.read_csv(valdir + '/' + file)
    true_psa, pre_psa = [np.array(val_df.loc[i, :]) for i in range(2)]
    true_psa = true_psa[1:].astype(np.float)
    pre_psa = pre_psa[1:].astype(np.float)
    true_list.append(true_psa)
    predict_list.append(pre_psa)
TruePsa = np.concatenate(true_list)
PredictPsa = np.concatenate(predict_list)
r2 = r2_score(TruePsa, PredictPsa)
plt.style.use(['science', 'nature'])
plt.figure(figsize=(5, 4))
plt.scatter(TruePsa, PredictPsa, s=25)
plt.plot([0, 40], [0, 40], color=cs[5], lw=2.5)
plt.xlabel("Measured PSA ($\mu g/L$)", fontsize=22)
plt.ylabel("Simulated PSA ($\mu g/L$)", fontsize=22)
plt.xticks(labels=[0,10, 20, 30, 40], ticks=[0, 10, 20, 30 , 40], fontsize=20)
plt.yticks(labels=[0,10, 20, 30, 40], ticks=[0, 10, 20, 30 , 40],fontsize=20)
plt.text(20, 10, f"$R^2 = $ {r2:<5.2f}", fontsize=22)
plt.subplots_adjust(left=0.2, right=1, top=1, bottom=0.2)
plt.tight_layout()
plt.savefig('../../Analysis/Figure/Validation_PSA.eps', dpi=300, bbox_inches='tight')
plt.show()
plt.close()



##### The distribution and confidence interval analysis for patient-specific parameters #####
MEAN_PARS = {}
LOW_CI = {}
HIGH_CI = {}
CI = {}
for i in patientList:
    if len(str(i)) == 1:
        patientNo = "patient00" + str(i)
    elif len(str(i)) == 2:
        patientNo = "patient0" + str(i)
    else:
        patientNo = "patient" + str(i)
    pars_arr = ALL_F_PARS_LIST[patientNo]
    mean_pars_arr = np.mean(pars_arr, 0)
    sd_pars_arr = sem(pars_arr, 0)
    low_pars_bound, high_pars_bound = t.interval(0.95, 9, loc=mean_pars_arr, scale=sd_pars_arr)
    MEAN_PARS[patientNo] = mean_pars_arr
    CI[patientNo] = [(np.around(low_pars_bound[i],4),
                      np.around(high_pars_bound[i], 4)) for i in range(mean_pars_arr.shape[0])]
    LOW_CI[patientNo] = low_pars_bound
    HIGH_CI[patientNo] = high_pars_bound
pd.DataFrame.from_dict(MEAN_PARS, orient='index', columns=['r0', 'r1', 'beta1', 'beta2', 'phi', 'gamma', 'betac']).to_csv('../../Analysis/average_pars.csv')
pd.DataFrame.from_dict(LOW_CI, orient='index', columns=['r0', 'r1', 'beta1', 'beta2', 'phi', 'gamma', 'betac']).to_csv(
    '../../Analysis/csv_file/pars_low_ci.csv')
pd.DataFrame.from_dict(HIGH_CI, orient='index', columns=['r0', 'r1', 'beta1', 'beta2', 'phi', 'gamma', 'betac']).to_csv(
    '../../Analysis/csv_file/pars_high_ci.csv')
df_CI = pd.DataFrame.from_dict(CI, orient='index', columns=['r0', 'r1', 'beta1', 'beta2', 'phi', 'gamma', 'betac'])
df_CI[['r0', 'r1', 'beta1', 'beta2']].to_csv('../../Analysis/csv_file/pars_ci_14.csv', sep=';')
df_CI[['phi', 'gamma', 'betac']].to_csv('../../Analysis/csv_file/pars_ci_57.csv', sep=';')





