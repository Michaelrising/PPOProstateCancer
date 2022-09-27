from _utils import *
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns



datadir = "../PPO_Analysis/training_analysis/"
figdir = "../PPO_Analysis/"
fileList = os.listdir(datadir)
paraSetting = ['0503', '0504', '0505', '0506', '0507']
for file in fileList:
    customSmooth(datadir, figdir, csv_name = file)
smooth_datadir = "../PPO_Analysis/training_analysis_smooth/"
#### n ########
# TEST-REWARD
N = np.arange(3, 8)
colors = sns.color_palette() #["#496B92", "#ffd9fe", "#ffa538","#024032"]
plotList = []
plt.style.use('seaborn')
plt.figure(figsize=(8, 5))
for i in range(len(paraSetting)):
    par, n, color = paraSetting[i], N[i], colors[i]
    parDataDir = smooth_datadir
    data = pd.read_csv(parDataDir + par + "-Reward_greedy_evaluate.csv")
    rewardTest = pd.DataFrame(data)
    plt.plot(rewardTest['Step'][100:], rewardTest["Value"][100:], color = color, alpha = 0.3)
    l, = plt.plot(rewardTest["Step"], rewardTest["SValue"], color = color)
    plotList.append(l)
plt.legend(handles=plotList, loc = 'lower right', labels = ["$n$ = 3","$n$ = 4","$n$ = 5", "$n$ = 6",  "$n$ = 7"], fontsize = 22)
# plt.axvspan(xmin =plt.xlim()[0], xmax=100000, facecolor="grey", alpha=0.3)
plt.xlabel("Steps",  fontsize=20)
plt.ylabel("Test Reward",  fontsize=20)
plt.savefig(figdir + "patient011_n_TestReward.png", dpi =300)
plt.show()

