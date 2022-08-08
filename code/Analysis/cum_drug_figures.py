import os
import glob
import time
from datetime import datetime
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

patientlist = [1, 2, 3, 4, 6, 11, 12, 13, 15, 16, 17, 19, 20, 24, 25, 29, 30, 31, 32, 36, 37, 40, 42, 44, 46, 50, 51,
               52, 54, 58, 60, 61, 62, 63, 66, 71, 75, 77, 78, 79, 84, 85, 86, 87, 88, 91, 92, 93, 94, 95, 96, 97,
               99, 100, 101, 102, 104, 105, 106, 108] # kick 56 83 out of analysis
patient_yes = [11, 12, 19, 25, 36, 52, 54, 85, 88, 99, 101]
patient_no = list(set(patientlist) - set(patient_yes))

for no in patientlist:
    if len(str(no)) == 1:
        patientNo = "patient00" + str(no)
    elif len(str(no)) == 2:
        patientNo = "patient0" + str(no)
    else:
        patientNo = "patient" + str(no)
    clinical_data = pd.read_csv("../data/dataTanaka/Bruchovsky_et_al/" + patientNo + ".txt", header=None)
    cpa = np.array(clinical_data[2])
    cpa[np.isnan(cpa)] = 0
    leu = np.array(clinical_data[3])
    leu[np.isnan(leu)] = 0
    days = np.array(clinical_data[8])
    days = days - days[0]
    days_diff = np.diff(days)
    days_diff = np.append(days_diff, 28)
    cpa_month_normalized = cpa / 200
    leu_month_normalized = leu / 7.5
    leu_month_normalized[np.where(leu_month_normalized > 1)[0]] = 1
    # prediction = np.array(pd.read_csv('Experts_policy/analysis/prediction/' + patientNo +'_predicted_doseSeq.csv',  header=0, index_col=0)).reshape(-1)
    # cpa_1 = np.array([0, 50, 100, 150, 200]) / 200
    # leu_1 = np.array([0, 7.5]) / 7.5
    # _action_set = np.stack((np.tile(cpa_1, 2), np.sort(leu_1.repeat(5))), axis=1)
    # for i in range(prediction.shape[0]):
    #     action = prediction[i]
    #     dose_cpa = _action_set[i, 0]
    #     dose_leu = _action_set[i, 1]
    #     cpa_month_normalized =np.append(cpa_month_normalized, dose_cpa)
    #     leu_month_normalized = np.append(leu_month_normalized, dose_leu)
    #     days = np.append(days, days[-1] + 28)
    #     days_diff = np.append(days_diff, 28)
    cumsum_dose1_clinical = np.cumsum(cpa_month_normalized * days_diff / 28)
    cumsum_dose2_clinical = np.cumsum(leu_month_normalized)
