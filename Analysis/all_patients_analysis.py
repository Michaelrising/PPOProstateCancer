import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _utils import  *
import seaborn as sns
import scipy.stats as st
from scipy import interpolate
a1, a2, a3, a4, a5, a6 = AnyObject(), AnyObject(), AnyObject(), AnyObject(), AnyObject(), AnyObject()

##############################################################
##################### ALL Group #######################
##############################################################
cs = sns.color_palette("Paired")

all = [1, 2, 3, 4, 6, 11, 12, 13, 15, 16, 17,19, 20, 24, 25, 29, 30, 31, 32, 36, 37, 40, 42, 44, 46, 50, 51,
            52, 54, 58, 60, 61, 62, 63, 66, 71, 75, 77, 78, 79, 84, 85, 86, 87, 88, 91, 92, 93, 94, 95, 96, 97, 99,
            100, 101, 102, 104, 105, 106, 108]
all.sort()
simulation_end_cinical = pd.read_csv("../Experts_policy/analysis/s_end_list.csv",names=['Days'],index_col=0, header=0)
parsdir = "../GLV/analysis-dual-sigmoid/model_pars/"
patientLables = []
patientCPA = []
patientLEU = []
patientSurvivalTime = []
simu_stop = []
all_uniform = []
all_adaptive = []
all_all = []
pars_all_patients = []
for i in all:
    if len(str(i)) == 1:
        patientNo = "patient00" + str(i)
    elif len(str(i)) == 2:
        patientNo = "patient0" + str(i)
    else:
        patientNo = "patient" + str(i)
    print(patientNo)
    # if i in uniform:
    #     all_uniform.append(patientNo)
    # else:
    #     all_adaptive.append(patientNo)
    all_all.append(patientNo)
    doseSeq = pd.read_csv("../PPO_policy/analysis/" + patientNo + "_rl_dosages.csv", names = ["Month", "CPA", "LEU"], header=0)
    patientLables.append(patientNo)
    patientSurvivalTime.append(np.array(doseSeq).shape[0] * 28)
    doseSeq["CPA"] = doseSeq["CPA"]/200
    doseSeq["LEU"] = doseSeq["LEU"]/7.5
    patientCPA.append(np.array(doseSeq["CPA"]))
    patientLEU.append(np.array(doseSeq["LEU"]))
    if patientNo in list(simulation_end_cinical.index):
        simu_stop.append(simulation_end_cinical.loc[patientNo].Days.item())
    else:
        simu_stop.append(None)

df_ppo_CPA = pd.DataFrame(patientCPA, index = patientLables)
df_ppo_CPA = df_ppo_CPA.sort_index()
df_ppo_LEU = pd.DataFrame(patientLEU, index = patientLables)
df_ppo_LEU = df_ppo_LEU.sort_index()
df_ppo_Time = pd.DataFrame(patientSurvivalTime, index= patientLables, columns=["rl"])
df_ppo_Time = df_ppo_Time.sort_index()
simu_stop = pd.DataFrame(simu_stop, index = patientLables)
onColor = "#FF0000"
onCpa = "#FF0000"
onLeu = "#87CEEB"
offColor = "#696969"
plt.style.use(["science", 'nature'])
fig = plt.figure(figsize = (40, 0.4 * len(all_all)))
ax1 = fig.add_subplot(1,2,1)
for l, patient in enumerate(all_all):
    cpa_ppo = df_ppo_CPA.loc[patient, ~np.isnan(df_ppo_CPA.loc[patient])]
    leu_ppo = df_ppo_LEU.loc[patient, ~np.isnan(df_ppo_LEU.loc[patient])]
    for month, cpaData in enumerate(cpa_ppo):
        leuData = leu_ppo[month]
        if cpaData != 0:
            if leuData != 0:
                barcontainer = ax1.barh(patient +'-p', 28, left=month * 28, color=onColor, alpha=cpaData, hatch="///",
                        height=0.8, tick_label=None)
            else:
                barcontainer = ax1.barh(patient+'-p', 28, left=month * 28, color=onCpa,
                        alpha=cpaData, height=0.8, tick_label=None)
            # ax.barh(patientLables[patient], 28, left = month * 28, hatch = "/", label = "LEU-ON",alpha = 0, height = 0.5, tick_label = None)
        if cpaData == 0 and leuData != 0:
            barcontainer = ax1.barh(patient+'-p', 28, left=month * 28, color=onLeu, hatch="///",
                    alpha=0, height=0.8, tick_label=None)
        if ~np.isnan(df_ppo_CPA.loc[patient, month]) and cpaData == 0 and leuData == 0:
            barcontainer = ax1.barh(patient+'-p', 28, left=month * 28, color=offColor, height=0.8, tick_label=None)
    s1 = plt.scatter(x=barcontainer.patches[0].get_x() + barcontainer.patches[0].get_width(),
                y=barcontainer.patches[0].get_y() + barcontainer.patches[0].get_height() / 2, marker=4, color='black',
                s=250, label = 'EOS', zorder=3)
locs, labels = plt.yticks()
labels = all_all
plt.yticks(np.arange(0, len(all_all), 1), [], fontsize=30)
plt.ylabel("Patient No.", fontsize = 40)
plt.xticks(fontsize=40)
plt.xlabel("Time (Day)", fontsize=40)
plt.xlim(-10, 3600)

ax2 = fig.add_subplot(1,2,2)
for l, patient in enumerate(all_all):
    clinical_data = pd.read_csv("../data/dataTanaka/Bruchovsky_et_al/" + patient + ".txt", header=None)
    ONOFF = np.array(clinical_data.loc[:, 7])
    drugOnDays = 0
    drugOffDays = 0
    Days = np.array(clinical_data.loc[:, 9]) - np.array(clinical_data.loc[0, 9])
    CPA = np.array(clinical_data.loc[:, 2])
    LEU = np.array(clinical_data.loc[:, 3])
    cpa_left = 0
    leu_left = 0
    for ii in range(len(ONOFF) - 1):
        cpa = CPA[ii]
        leu = LEU[ii]
        if ~np.isnan(cpa):
            barcontainer = ax2.barh(patient + '-c', Days[ii + 1] - Days[ii], left=Days[ii], color=onColor, height=0.8, alpha = cpa/200,
                                   tick_label=None)
        if ~np.isnan(leu):
            barcontainer = ax2.barh(patient + '-c', max(28 * int(leu/7.5), Days[ii + 1] - Days[ii]), left=Days[ii], hatch="///",color =onLeu,alpha = 0,
                                   height=0.8, tick_label=None)
        if np.isnan(leu) and np.isnan(cpa):
            barcontainer = ax2.barh(patient + '-c', Days[ii + 1] - Days[ii], left=Days[ii], color=offColor, height=0.8,
                                   tick_label=None)
    if ~np.isnan(simu_stop.loc[patient].item()):
        plt.scatter(x = simu_stop.loc[patient].item(),  y = barcontainer.patches[0].get_y() + barcontainer.patches[0].get_height()/2,
                marker=4, color = 'black', s = 250, label ='EOS',zorder=3)
    else:
        CPA = [0, 50, 100, 150, 200]
        LEU = [0, 7.5]
        extraDose = pd.read_csv("../Experts_policy/analysis/prediction/" +patient+"_predicted_doseSeq.csv")
        left = Days[-1]
        extraDose = np.array(extraDose)[:, -1]
        for ii in range(extraDose.shape[0]):
            extra_cpa = CPA[int(extraDose[ii]%5)]
            extra_leu = LEU[int(extraDose[ii]//5)]
            if left > 28 * 120:
                length = 28 * 121 - left
            else:
                length = 28
            if extra_cpa:
                ax2.barh(patient+"-c", length, left=left, color=onCpa,alpha =extra_cpa/200, height=0.8, tick_label=None)
            if extra_leu:
                ax2.barh(patient + '-c', 28 , left=left, hatch="///", color=onLeu, alpha=0,
                        height=0.8, tick_label=None)
            if not extra_leu and not extra_cpa:
                ax2.barh(patient+'-c', length, left=left, color=offColor, height=0.8, alpha =1, tick_label=None)
            left += 28
            if left > 28 * 121:
                break
        plt.scatter(x=left, y=barcontainer.patches[0].get_y() + barcontainer.patches[0].get_height() / 2,
                    marker=4, color='black', s=250, label='EOS')
    s2 = plt.scatter(x=barcontainer.patches[0].get_x() + barcontainer.patches[0].get_width(),
                y=barcontainer.patches[0].get_y() + barcontainer.patches[0].get_height() / 2,
                marker="X", color='black', s=250, label='EOC', zorder=3)

locs, labels = plt.yticks()
labels = []
plt.yticks(np.arange(0, len(all_all), 2), labels, fontsize = 40)
# plt.ylabel("1 $\longleftarrow$ Patient No. $\longrightarrow$ 108", fontsize = 24)
plt.xticks(fontsize = 40)
plt.xlabel("Time (Day)", fontsize = 40)
plt.xlim(-10, 3600)
plt.legend([ a1, a2, a3, a4, s1, s2 ], ['C$\&$L-On',"Cpa-On ","Leu-On" ,'Treat-Off', 'EOS', 'EOC'],
           handler_map={a1: AnyObjectHandler(color=onColor), a2:AnyObjectHandler(color=onCpa, _hatch=None),
                        a3: AnyObjectHandler(color=colorAlpha_to_rgb(cs[0], 0)[0], alpha = 1), a4: AnyObjectHandler(color=offColor,alpha=1, _hatch=None)}
           , fontsize =30, loc = 2, bbox_to_anchor=(-0.225, 0.7))
plt.savefig("./Figure/All_patients_Strategy.pdf", dpi = 500)
plt.show()
plt.close()

patientLables = []
PSAThresholds = []
patientSurvivalTime = []
for i in all:
    if len(str(i)) == 1:
        patientNo = "patient00" + str(i)
    elif len(str(i)) == 2:
        patientNo = "patient0" + str(i)
    else:
        patientNo = "patient" + str(i)
    print(patientNo)
    statesSeq = pd.read_csv("../PPO_states/analysis/" + patientNo +'_evolution_states.csv', names = ["AD", "AI", "PSA"], header=0)
    patientLables.append(patientNo)
    psa = statesSeq['PSA']
    diff1_psa = psa.diff()
    ratio_psa = diff1_psa/psa[:-1]
    PSAThresholds.append(ratio_psa)

ppo_off= []
ppo_cpa_daily = []
ppo_leu_monthly = []
ppo_onoff_freq = []
df_ppo_drug = df_ppo_CPA + df_ppo_LEU
for patient_i in df_ppo_drug.index:
    patient_drug = np.array(df_ppo_drug.loc[patient_i, ~np.isnan(df_ppo_drug.loc[patient_i])])
    off_percentage = patient_drug[patient_drug == 0].shape[0]/patient_drug.shape[0]
    ppo_off.append(off_percentage)
    on_off = 0
    for i in range(patient_drug.shape[0] - 1):
        if patient_drug[i] !=patient_drug[i + 1]:
            on_off += 1
    ppo_onoff_freq.append(on_off)
    patient_cpa = np.array(df_ppo_CPA.loc[patient_i, ~np.isnan(df_ppo_CPA.loc[patient_i])])
    cpa_daily = sum(patient_cpa)/(patient_cpa.shape[0]) * 200
    patient_leu = np.array(df_ppo_LEU.loc[patient_i, ~np.isnan(df_ppo_LEU.loc[patient_i])])
    leu_monthly = sum(patient_leu)/(patient_leu.shape[0]) * 7.5
    ppo_cpa_daily.append(cpa_daily)
    ppo_leu_monthly.append(leu_monthly)
ppo_cpa_daily = pd.DataFrame(ppo_cpa_daily, index=patientLables)
ppo_leu_monthly = pd.DataFrame(ppo_leu_monthly, index=patientLables)
ppo_off = pd.DataFrame(ppo_off, index=patientLables)
ppo_onoff_freq = pd.DataFrame(ppo_onoff_freq, index=patientLables)

print("mean ppo fre:"+ str(ppo_onoff_freq.mean()))
off_clinical = []
cpa_clinical_daily = []
leu_clinical_monthly = []
clinical_onoff_freq = []
for patient_i in df_ppo_drug.index:
    clinical_data = pd.read_csv("../data/dataTanaka/Bruchovsky_et_al/" + patient_i + ".txt", header=None)
    onoff = np.array(clinical_data.loc[:, 7])
    Days = np.array(clinical_data.loc[:, 9].diff()[1:])
    Days = np.append(Days, 28)
    offdays = sum(Days[~onoff.astype(bool)])
    off_percentage = offdays/sum(Days)
    off_clinical.append(off_percentage)
    on_off = 0
    for i in range(onoff.shape[0] - 1):
        if onoff[i] != onoff[i + 1]:
            on_off += 1
    clinical_onoff_freq.append(on_off)
    patient_cpa = np.array(clinical_data.loc[:, 2])
    clinical_cpa_daily = np.sum(patient_cpa[~np.isnan(patient_cpa)] * Days[~np.isnan(patient_cpa)])/(clinical_data.loc[clinical_data.shape[0]-1, 9] - clinical_data.loc[0, 9])
    cpa_clinical_daily.append(clinical_cpa_daily)
    patient_leu = np.array(clinical_data.loc[:, 3])
    clinical_leu_monthly = patient_leu[~np.isnan(patient_leu)].sum()/(clinical_data.loc[clinical_data.shape[0]-1, 9] - clinical_data.loc[0, 9]) * 28
    leu_clinical_monthly.append(clinical_leu_monthly)

off_clinical = pd.DataFrame(off_clinical, index=patientLables)
cpa_clinical_daily = pd.DataFrame(cpa_clinical_daily, index=patientLables)
leu_clinical_monthly = pd.DataFrame(leu_clinical_monthly, index=patientLables)
clinical_onoff_freq = pd.DataFrame(clinical_onoff_freq, index=patientLables)

print("Mean clinical feq:" + str(clinical_onoff_freq.mean()))
from scipy.stats import ttest_rel

#
#
# ### all_uniform #####
# print('========================================')
# print('=======all Uniform T-Test========')
# print('Total therapy off:{}'.format(np.mean(ppo_off.loc[all_uniform]-off_clinical.loc[all_uniform]).item()))
# print(ttest_rel(ppo_off.loc[all_uniform], off_clinical.loc[all_uniform]))
# print('Daily CPA off: {}'.format(-np.mean(ppo_cpa_daily.loc[all_uniform]- cpa_clinical_daily.loc[all_uniform]).item()))
# print(ttest_rel(ppo_cpa_daily.loc[all_uniform], cpa_clinical_daily.loc[all_uniform]))
# print('Monthly LEU off: {}'.format(-np.mean(ppo_leu_monthly.loc[all_uniform]- leu_clinical_monthly.loc[all_uniform]).item()))
# print(ttest_rel(ppo_leu_monthly.loc[all_uniform], leu_clinical_monthly.loc[all_uniform]))
# print('========================================')
#
# ### all_adaptive #####
# print('========================================')
# print('=======all Adaptive T-Test========')
# print('Total therapy off:{}'.format(np.mean(ppo_off.loc[all_adaptive]-off_clinical.loc[all_adaptive]).item()))
# print(ttest_rel(ppo_off.loc[all_adaptive], off_clinical.loc[all_adaptive]))
# print('Daily CPA off: {}'.format(-np.mean(ppo_cpa_daily.loc[all_adaptive]- cpa_clinical_daily.loc[all_adaptive]).item()))
# print(ttest_rel(ppo_cpa_daily.loc[all_adaptive], cpa_clinical_daily.loc[all_adaptive]))
# print('Monthly LEU off: {}'.format(-np.mean(ppo_leu_monthly.loc[all_adaptive]- leu_clinical_monthly.loc[all_adaptive]).item()))
# print(ttest_rel(ppo_leu_monthly.loc[all_adaptive], leu_clinical_monthly.loc[all_adaptive]))
# print('========================================')


# all all

print('========================================')
print('=======all All T-Test========')
print('Total therapy off:{}'.format(np.mean((ppo_off-off_clinical)/off_clinical)))
print(ttest_rel(ppo_off, off_clinical))
print('Daily CPA off: {}'.format(-np.mean((ppo_cpa_daily- cpa_clinical_daily)/cpa_clinical_daily).item()))
print(ttest_rel(ppo_cpa_daily, cpa_clinical_daily))
print('Monthly LEU off: {}'.format(-np.mean((ppo_leu_monthly- leu_clinical_monthly)/leu_clinical_monthly).item()))
print(ttest_rel(ppo_leu_monthly, leu_clinical_monthly))
print('========================================')



#### determine the ascending policy ####

OFF_INTERVAL = []
ON_INTERVAL = []
OFF_ON_R = []
OFF_ON_R_PRED = []
OFF_ON_Len = []
DRUG_PRED = []
CPA_PRED = []
LEU_PRED = []
off_len = 0
on_len = 0
off_on_len = 0
for i in all:
    if len(str(i)) == 1:
        patientNo = "patient00" + str(i)
    elif len(str(i)) == 2:
        patientNo = "patient0" + str(i)
    else:
        patientNo = "patient" + str(i)
    print(patientNo)
    doseSeq = pd.read_csv("../PPO_policy/analysis/" + patientNo + "_rl_dosages.csv", names = ["Month", "CPA", "LEU"], header=0)
    doseSeq["CPA"] = doseSeq["CPA"] / 200
    doseSeq["LEU"] = doseSeq["LEU"] / 7.5
    month = np.array(doseSeq['Month']).reshape(-1)
    drug = np.array(doseSeq["CPA"] + doseSeq["LEU"]).reshape(-1)
    cpa = doseSeq["CPA"]
    leu = doseSeq["LEU"]
    off_interval = []
    on_interval = []
    off_on_ratio = []
    off = 0
    on = 0
    for ii in range(drug.shape[0]):
        if drug[ii] == 0:
            off += 1
            if ii!=0 and on != 0:
                on_interval.append(on)
            on = 0
        else:
            if ii!=0 and off != 0:
                off_interval.append(off)
            off = 0
            on += 1
    max_off = max(off_interval)
    max_on = max(on_interval)
    for ii in range(min(len(off_interval), len(on_interval))):
        ratio = off_interval[ii]/on_interval[ii]
        off_on_ratio.append(ratio)
    f = interpolate.interp1d(x = np.arange(len(off_on_ratio))/len(off_on_ratio), y = off_on_ratio, kind='linear', fill_value="extrapolate")
    pred_x = np.linspace(0, 1, 100)
    OFF_ON_R_PRED.append(f(pred_x))
    OFF_ON_R.append(off_on_ratio)
    drug_dose_change = []
    one_course_drug = 0
    one_course_cpa = 0
    one_course_leu = 0
    leu_dose_change = []
    cpa_dose_change = []

    for ii in range(len(drug)):
        if cpa[ii] != 0:
            one_course_cpa += cpa[ii]
        else:
            if ii != 0 and one_course_cpa!=0:
                cpa_dose_change.append(one_course_cpa)
            one_course_cpa = 0
        if leu[ii] != 0:
            one_course_leu += leu[ii]
        else:
            if ii != 0 and one_course_leu!=0:
                leu_dose_change.append(one_course_leu)
            one_course_leu = 0
        if drug[ii] != 0:
            one_course_drug += drug[ii]
        else:
            if ii != 0 and one_course_drug!=0:
                drug_dose_change.append(one_course_drug)
            one_course_drug = 0
        if len(cpa_dose_change) < len(drug_dose_change):
            cpa_dose_change.append(0)
        if len(leu_dose_change) < len(drug_dose_change):
            leu_dose_change.append(0)
    for ii in range(min(len(off_interval), len(on_interval))):
        drug_dose_change[ii] = drug_dose_change[ii]/(on_interval[ii] + off_interval[ii])
        leu_dose_change[ii] = leu_dose_change[ii]/(on_interval[ii] + off_interval[ii])
        cpa_dose_change[ii] = cpa_dose_change[ii]/(on_interval[ii] + off_interval[ii])
    f_drug = interpolate.interp1d(x = np.arange(len(drug_dose_change)-1)/(len(drug_dose_change)-1), y = drug_dose_change[:-1], kind='nearest',fill_value="extrapolate")
    DRUG_PRED.append(f_drug(pred_x))
    f_cpa = interpolate.interp1d(x = np.arange(len(cpa_dose_change)-1)/(len(cpa_dose_change)-1), y = cpa_dose_change[:-1], kind='nearest',fill_value="extrapolate")
    CPA_PRED.append(f_cpa(pred_x))
    f_cpa = interpolate.interp1d(x=np.arange(len(leu_dose_change)-1) / (len(leu_dose_change)-1), y=leu_dose_change[:-1],
                                 kind='nearest', fill_value="extrapolate")
    LEU_PRED.append(f_cpa(pred_x))


OFF_ON_PRED_ARR = np.stack(OFF_ON_R_PRED)
mean_off_on = OFF_ON_PRED_ARR.mean(axis = 0 )
sd_off_on = st.sem(OFF_ON_PRED_ARR, axis = 0)
low_off_on_bound, high_off_on_bound = st.t.interval(0.95, mean_off_on.shape[0] - 1, loc=mean_off_on, scale=sd_off_on)

CPA_PRED_ARR = np.stack(CPA_PRED)
mean_cpa = CPA_PRED_ARR.mean(axis = 0 )
sd_cpa = st.sem(CPA_PRED_ARR, axis = 0)
low_cpa_bound, high_cpa_bound = st.t.interval(0.95, mean_cpa.shape[0] - 1, loc=mean_cpa, scale=sd_cpa)

LEU_PRED_ARR = np.stack(LEU_PRED)
mean_leu = LEU_PRED_ARR.mean(axis = 0 )
sd_leu = st.sem(LEU_PRED_ARR, axis = 0)
low_leu_bound, high_leu_bound = st.t.interval(0.95, mean_leu.shape[0] - 1, loc=mean_leu, scale=sd_leu)

plt.style.use(["science", "nature"])

fig = plt.figure(figsize=(5, 6))
ax1 = fig.add_subplot(3,2,1)
x = np.linspace(0, 1, mean_off_on.shape[0])
face_c_pred0 = colorAlpha_to_rgb(cs[0], 1)[0]
plt.plot(x, mean_off_on, c=cs[7], lw=1.2)
plt.fill_between(x, low_off_on_bound, high_off_on_bound, color=face_c_pred0)
plt.ylabel('Treat Off/On Ratio', fontsize = 10.5)
plt.xlabel('Treatment Course', fontsize=10.5)
plt.xticks(fontsize = 9)
plt.yticks(fontsize = 9)
plt.scatter(x=1,y=mean_off_on[-1], marker=4, color='black',
                s=180, label = 'EOS', zorder=3)
plt.tight_layout()

ax2 = fig.add_subplot(3,2,3)
x = np.linspace(0, 1, mean_cpa.shape[0])
face_c_pred0 = colorAlpha_to_rgb(cs[0], 1)[0]
plt.plot(x, mean_cpa, c=cs[7], lw=1.2)
plt.fill_between(x, low_cpa_bound, high_cpa_bound, color=face_c_pred0)
plt.ylabel('Daily CPA/Cycle', fontsize = 10.5)
plt.xlabel('Treatment Course', fontsize=10.5)
plt.xticks(fontsize = 9)
plt.yticks(fontsize = 9)
plt.scatter(x=1,y=mean_cpa[-1], marker=4, color='black',
                s=180, label = 'EOS', zorder=3)
plt.tight_layout()

ax3 = fig.add_subplot(3,2,5)
x = np.linspace(0, 1, mean_leu.shape[0])
face_c_pred0 = colorAlpha_to_rgb(cs[0], 1)[0]
plt.plot(x, mean_leu, c=cs[7], lw=1.2)
plt.fill_between(x, low_leu_bound, high_leu_bound, color=face_c_pred0)
plt.ylabel('Monthly LEU/Cycle', fontsize = 10.5)
plt.xlabel('Treatment Course', fontsize=10.5)
plt.xticks(fontsize = 9)
plt.yticks(fontsize = 9)
plt.scatter(x=1,y=mean_leu[-1], marker=4, color='black',
                s=180, label = 'EOS', zorder=3)
plt.tight_layout()

# plt.savefig('./Figure/all_I2ADT_off_on_ratio_change.eps', dpi=300, bbox_inches='tight')
# plt.show()

## IADT ##
OFF_ON_R_PRED = []
CPA_R_PRED =[]
LEU_R_PRED=[]
cpa = np.array([0, 50, 100, 150, 200])/200
leu = np.array([0, 7.5])/7.5
_action_set = np.stack((np.tile(cpa, 2), np.sort(leu.repeat(5))), axis=1)
for l, patient in enumerate(all_all):
    clinical_data = pd.read_csv("../data/dataTanaka/Bruchovsky_et_al/" + patient + ".txt", header=None)
    ONOFF = np.array(clinical_data.loc[:, 7])
    Days = np.array(clinical_data.loc[:, 9]) - np.array(clinical_data.loc[0, 9])
    Days_diff = np.append(np.diff(Days), 28)
    CPA = np.array(clinical_data.loc[:, 2])/200
    LEU = np.array(clinical_data.loc[:, 3])/7.5
    off_interval = []
    on_interval = []
    cpa_interval = []
    leu_interval = []
    off_on_ratio = []
    transition = Days[0]
    drug_transition = 0

    cpa_total = np.zeros_like(CPA)
    cpa_total[np.isnan(CPA)] = 0
    cpa_total[~np.isnan(CPA)] = CPA[~np.isnan(CPA)]
    leu_total = np.zeros_like(LEU)
    leu_total[np.isnan(LEU)] = 0
    leu_total[~np.isnan(LEU)] = LEU[~np.isnan(LEU)]
    cpa_total = cpa_total * Days_diff
    for i in range(ONOFF.shape[0]-1):
        if ONOFF[i] != ONOFF[i+1]:
            if ONOFF[i] == 1:
                on_interval.append(Days[i+1] - transition)
                cpa_interval.append(cpa_total[drug_transition:i+1].sum())
                leu_interval.append(leu_total[drug_transition:i+1].sum())
                transition = Days[i+1]
            if ONOFF[i] == 0:
                off_interval.append(Days[i+1] - transition)
                transition = Days[i+1]
                drug_transition = i + 1

    # on_interval.append(Days[-1] - transition) if ONOFF[-1] == 1 else off_interval.append(Days[-1] - transition)
    # transition = Days[-1]
    if np.isnan(simu_stop.loc[patient].item()):
        extraDose = np.array(pd.read_csv("../Experts_policy/analysis/prediction/" +patient+"_predicted_doseSeq.csv", index_col=0)).reshape(-1)
        if ONOFF[-1] ==0 and extraDose[0] == 0:
            for i in range(extraDose.shape[0] - 1):
                if extraDose[i]==0 and extraDose[i + 1] != 0:
                    off_interval.append(Days[-1] + 28 * (i+1) - transition)
                    start = i + 1
                    break
        elif ONOFF[-1] == 0 and extraDose[0] != 0:
            off_interval.append(Days[-1] - transition)
            start = 0
        elif ONOFF[-1] != 0 and extraDose[0] == 0:
            on_interval.append(Days[-1] - transition)
            cpa_interval.append(cpa_total[drug_transition:].sum())
            leu_interval.append(leu_total[drug_transition:].sum())
            start = 0
        else:
            for i in range(extraDose.shape[0] - 1):
                if extraDose[i] != 0 and extraDose[i + 1] == 0:
                    on_interval.append(Days[-1] + 28 * (i+1) - transition)
                    cpa_interval.append(cpa_total[drug_transition:].sum() + 28 * _action_set[extraDose[:i + 1].astype(int), 0].sum())
                    leu_interval.append(leu_total[drug_transition:].sum() + _action_set[extraDose[:i + 1].astype(int), 1].sum())
                    start = i + 1
                    break
        transition = start
        for i in range(start, extraDose.shape[0]-1):
            if extraDose[i] * extraDose[i+1] == 0 and not (extraDose[i] == 0 and extraDose[i+1]==0):
                if extraDose[i] != 0:
                    on_interval.append((i - transition + 1) * 28)
                    cpa_interval.append(28 * _action_set[extraDose[transition:i + 1].astype(int), 0].sum())
                    leu_interval.append(_action_set[extraDose[transition:i + 1].astype(int), 0].sum())
                if extraDose[i] == 0:
                    off_interval.append((i - transition + 1) * 28)
                transition = i - start + 1
    cpa_dose_change = np.zeros(min(len(off_interval), len(on_interval)))
    leu_dose_change = np.zeros(min(len(off_interval), len(on_interval)))
    for ii in range(min(len(off_interval), len(on_interval))):
        ratio = off_interval[ii]/on_interval[ii]
        off_on_ratio.append(ratio)
        leu_dose_change[ii] = 28 * leu_interval[ii] / (on_interval[ii] + off_interval[ii])
        cpa_dose_change[ii] = cpa_interval[ii] / (on_interval[ii] + off_interval[ii])
    if len(off_on_ratio) >= 2:
        f = interpolate.interp1d(x=np.arange(len(off_on_ratio)) / max((len(off_on_ratio) - 1), 1), y=off_on_ratio, kind='nearest', fill_value="extrapolate")
        pred_x = np.linspace(0, 1, 100)
        OFF_ON_R_PRED.append(f(pred_x))
        f_cpa = interpolate.interp1d(x=np.arange(len(cpa_dose_change)) / max((len(cpa_dose_change) - 1), 1), y=cpa_dose_change, kind='nearest', fill_value="extrapolate")
        CPA_R_PRED.append(f_cpa(pred_x))
        f_leu = interpolate.interp1d(x=np.arange(len(leu_dose_change)) / max((len(leu_dose_change) - 1), 1),
                                     y=leu_dose_change, kind='nearest', fill_value="extrapolate")
        LEU_R_PRED.append(f_leu(pred_x))


OFF_ON_PRED_ARR = np.stack(OFF_ON_R_PRED)
mean_off_on = OFF_ON_PRED_ARR.mean(axis = 0 )
sd_off_on = st.sem(OFF_ON_PRED_ARR, axis = 0)
low_off_on_bound, high_off_on_bound = st.t.interval(0.95, mean_off_on.shape[0] - 1, loc=mean_off_on, scale=sd_off_on)

LEU_R_ARR = np.stack(LEU_R_PRED)
mean_leu = LEU_R_ARR.mean(axis = 0 )
sd_leu = st.sem(LEU_R_ARR, axis = 0)
low_leu_bound, high_leu_bound = st.t.interval(0.95, mean_leu.shape[0] - 1, loc=mean_leu, scale=sd_leu)

CPA_R_ARR = np.stack(CPA_R_PRED)
mean_cpa = CPA_R_ARR.mean(axis = 0 )
sd_cpa = st.sem(CPA_R_ARR, axis = 0)
low_cpa_bound, high_cpa_bound = st.t.interval(0.95, mean_cpa.shape[0] - 1, loc=mean_cpa, scale=sd_cpa)


# plt.style.use(["science", "nature"])
# fig = plt.figure(figsize=(5, 4))
ax2 = fig.add_subplot(3,2,2)
x = np.linspace(0, 1, mean_off_on.shape[0])
face_c_pred0 = colorAlpha_to_rgb(cs[0], 1)[0]
plt.plot(x, mean_off_on, c=cs[7], lw=1.2)
plt.fill_between(x, low_off_on_bound, high_off_on_bound, color=face_c_pred0)
# plt.ylabel('Treat Off/On Ratio', fontsize = 11)
plt.xlabel('Treatment Course', fontsize=10.5)
plt.xticks(fontsize = 9)
plt.yticks(fontsize = 9)
plt.scatter(x=1,y=mean_off_on[-1], marker=4, color='black',
                s=180, label = 'EOS', zorder=3)
plt.tight_layout()

ax4 = fig.add_subplot(3,2,4)
x = np.linspace(0, 1, mean_cpa.shape[0])
face_c_pred0 = colorAlpha_to_rgb(cs[0], 1)[0]
plt.plot(x, mean_cpa, c=cs[7], lw=1.2)
plt.fill_between(x, low_cpa_bound, high_cpa_bound, color=face_c_pred0)
# plt.ylabel('Treat Off/On Ratio', fontsize = 11)
plt.xlabel('Treatment Course', fontsize=10.5)
plt.xticks(fontsize = 9)
plt.yticks(fontsize = 9)
plt.scatter(x=1,y=mean_cpa[-1], marker=4, color='black',
                s=180, label = 'EOS', zorder=3)
plt.tight_layout()

ax6 = fig.add_subplot(3,2,6)
x = np.linspace(0, 1, mean_leu.shape[0])
face_c_pred0 = colorAlpha_to_rgb(cs[0], 1)[0]
plt.plot(x, mean_leu, c=cs[7], lw=1.2)
plt.fill_between(x, low_leu_bound, high_leu_bound, color=face_c_pred0)
# plt.ylabel('Treat Off/On Ratio', fontsize = 11)
plt.xlabel('Treatment Course', fontsize=10.5)
plt.xticks(fontsize = 9)
plt.yticks(fontsize = 9)
plt.scatter(x=1,y=mean_leu[-1], marker=4, color='black',
                s=180, label = 'EOS', zorder=3)
plt.tight_layout()
plt.savefig('./Figure/all_I2ADT_IADT_comparision.eps', dpi=500, bbox_inches='tight')
plt.show()
