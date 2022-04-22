from abc import ABC, abstractmethod
import os
import numpy as np
import torch
import gym
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from collections import deque
from scipy.stats import bernoulli
import json

import os
import yaml
import argparse
from datetime import datetime
from collections import deque
import gym
import random
import torch
import pandas as pd
from PPO import PPO


def exploit(actor, state):
    # Act without randomness.
    state = torch.FloatTensor(
        state[None, ...]).float()  # /255.
    with torch.no_grad():
        action = actor.eval().act(state)
    return action.item()


def EvalActor(path, env, actor, patientNo):
    # para is a string denotes which policy to use
    policy = torch.load(path + patientNo +"_policy.pth")
    actor.load_state_dict(policy)
    actor.eval()

    max_episode_steps = 120
    state_arr = deque()
    action_arr = deque()
    reward_arr = []
    dose_arr = []
    fea, state = env.reset()
    state_arr.append(state)
    episode_steps = 0
    episode_return = 0.0
    done = False
    while (not done) and episode_steps <= max_episode_steps:
        action = actor.act_exploit(fea)
        fea, next_state, reward, done, infos = env.step(action)
        episode_steps += 1
        episode_return += reward
        state = next_state
        action_arr.append(action)
        state_arr.append(state)
        reward_arr.append(reward)
        dose_arr.append(infos["dose"])
    return np.array(action_arr).reshape(-1), np.array(state_arr).reshape(-1,3), np.array(reward_arr).reshape(-1), np.array(dose_arr).reshape(-1,2)


def MakePlot(path, para, env, actor,  plotStyle, patientNo, metasPar):
    actSeq, stateSeq, rwdSeq, doseSeq = EvalActor(path, para, env, actor, patientNo)
    doseCPA, doseLEU = np.array(doseSeq[:,0]).reshape(-1), np.array(doseSeq[:,1]).reshape(-1)
    CPATotalDosage, LEUTotalDosage = doseCPA.sum(), doseLEU.sum()
    CPAHolidayMonth, LEUHolidayMonth = np.where(doseCPA == 0)[0].shape[0], np.where(doseLEU == 0)[0].shape[0]
    CPAFreePercentage, LEUFreePercentage = CPAHolidayMonth/len(doseSeq), LEUHolidayMonth/len(doseSeq)
    # evolution process
    stateDF = pd.DataFrame(stateSeq, columns=['AD', "AI", "PSA"])
    stateDF['DAY'] = np.arange(0, stateSeq.shape[0]) * 28
    plt.style.use(plotStyle)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(stateDF['DAY'], stateDF['PSA'])
    plt.xlabel("Time (Days)", fontsize = 14)
    plt.ylabel("PSA level (ug/ml)", fontsize = 14)

    plt.subplot(1, 2, 2)
    plt.plot(stateDF['DAY'], stateDF['AD'], label="HD")
    plt.plot(stateDF['DAY'], stateDF['AI'], label="HI")
    plt.xlabel("Time (Days)", fontsize = 14)
    plt.ylabel("Cell counts", fontsize = 14)
    plt.legend(loc='upper left',  fontsize=16)
    plt.savefig("../ManualScripts/RLPics/Analysis_patient006/Evolution_" + metasPar +"_"+ patientNo + ".png", dpi=300)
    #plt.savefig("../ManualScripts/RLPics/Evolution_" + patientNo + ".png", dpi=300)
    # plt.show()
    plt.close()
    # dose plot
    COLOR_CPA = "#69b3a2"
    COLOR_LEU = '#FF4500'
    doseDF = pd.DataFrame(doseSeq, columns=['CPA', "LEU"])
    actDF = pd.DataFrame(actSeq)
    actDF["DAY"] = np.arange(0, len(actSeq)) * 28
    doseDF['DAY'] = np.arange(0, len(doseSeq)) * 28
    plt.style.use(plotStyle)
    fig = plt.figure(figsize=(7.5, 5))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(doseDF['DAY'], doseDF['CPA'], color=COLOR_CPA, label='CPA', lw = 1.5)
    ax1.set_xlabel("Days")
    ax1.set_ylabel("CPA (mg/ Day)", fontsize=14)
    # ax1.set_ylim(-5, 205)
    ax1.set_yticks(np.arange(0, 250, 50))
    ax1.tick_params(axis="y", labelcolor=COLOR_CPA)
    ax1.legend(loc=(0.03, 0.9), fontsize=14, facecolor = COLOR_CPA)
    plt.grid(False)
    ax2.plot(doseDF['DAY'], doseDF['LEU'], color=COLOR_LEU, label='LEU', lw = 1.5, ls ='--')
    ax2.set_ylabel("LEU (ml/ Month)",  fontsize=14)
    ax2.tick_params(axis="y", labelcolor=COLOR_LEU)
    ax2.set_yticks(np.arange(0, 22.5, 7.5))
    plt.ylim(-.5, 15)
    ax2.legend(loc=(0.03, 0.8),  fontsize=14, facecolor = COLOR_LEU)
    plt.savefig("../ManualScripts/RLPics/Analysis_patient006/Dosage_" + metasPar +"_"+ patientNo + ".png", dpi=300)
    # plt.savefig("../ManualScripts/RLPics/Dosage_" + patientNo + ".png", dpi=300)
    # plt.show()
    plt.close()
    return [CPATotalDosage, CPAHolidayMonth, CPAFreePercentage, LEUTotalDosage, LEUHolidayMonth, LEUFreePercentage, len(doseSeq)], doseDF, actDF


def run(args):
    patientNo = args.patientNo
    # print(patientNo)
    list_df = args.patients_pars[patientNo]
    A, K, states, pars, best_pars = [np.array(list_df.loc[i, ~np.isnan(list_df.loc[i, :])]) for i in range(5)]
    A = A.reshape(2, 2)
    init_state = states[:3]
    terminate_state = states[3:]

    default_acts = pd.read_csv("../Model_creation/test-sigmoid/model_actions/" + patientNo + "_actions_seqs.csv")
    default_acts = np.array(default_acts)
    weight = np.array([1.46, 0.206, 1.5])
    weight1 = np.array([1.3, 0.7])
    base = 1.15
    m1 = 0.8
    m2 = 12

    patient = (A, K, best_pars, init_state, terminate_state, weight, weight1, base, m1, m2)
    # Create environments.
    env = gym.make(args.env_id, patient = patient).unwrapped
    policypath = args.policyPath
    # policypath =
    actor = CateoricalPolicy(3, 10) #.to("cuda:0")
    AnaDosage, DoseSeq, ActSeq = MakePlot(policypath, args.para, env, actor, args.plotStyle, patientNo, args.metasPar)
    return AnaDosage, DoseSeq, ActSeq


env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'CancerControl-v0' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]
parsdir = "../Model_creation/retrain-sigmoid/model_pars"
parslist = os.listdir(parsdir)
patient_pars = {}
patient_test = []
patient_train = []
# reading the ode parameters and the initial/terminal states
for args in parslist:
    pars_df = pd.read_csv(parsdir + '/' + args)
    patient = args[5:(-4)]
    patient_train.append(patient)
    if patient not in patient_test:
        patient_pars[patient] = pars_df
DoseList = []
paraSetting = ["test"]#['0112', '01512', '0212', '02512', '0312', '03512', '0412', '0512', '05512', '0612', '0712', "07512", "0812", "0912"]
parser = argparse.ArgumentParser()
parser.add_argument('--plotStyle', type=str, default='seaborn')
parser.add_argument('--para', type=str, default='all')
parser.add_argument('--env_id', type=str, default='gym_cancer:CancerControl-v0')
parser.add_argument('--seed', type=int, default= 0) # random.randint(0,100000))
parser.add_argument('--patients_pars', type=dict, default = patient_pars)
parser.add_argument('--patients_train', type=list, default=patient_train)
parser.add_argument('--number', '-n', type=int, help='Patient No., int type, requested', default=6)
parser.add_argument('--patientNo', type=str, default="patient006")
parser.add_argument('--policyPath', type=str, default='./')
parser.add_argument('--metasPar', type = str, default="0412")
args = parser.parse_args()
for i in range(len(paraSetting)):
    paras = paraSetting[i]
    args.metasPar = paras
    args.policyPath = './Patient006Training/' + paras + "/actor/"
    d, _, _ = run(args)
    DoseList.append(d)
    print(args.metasPar)
# DoseDf = pd.DataFrame(np.array(DoseList).reshape(-1, len(paraSetting)),index =paraSetting,  columns=['CPATotalDosage', 'CPAHolidayMonth', 'CPAFreePercentage', 'LEUTotalDosage', 'LEUHolidayMonth', 'LEUFreePercentage'])
# DoseDf.to_csv('./Analysis_patient006_RL_DoseHistory.csv')


# if __name__ == '__main__':
#     parsdir = "../Model_creation/test-sigmoid/model_pars"
#     parslist = os.listdir(parsdir)
#     patient_pars = {}
#     patient_test = []
#     patient_train = []
#     # reading the ode parameters and the initial/terminal states
#     for args in parslist:
#         pars_df = pd.read_csv(parsdir + '/' + args)
#         patient = args[5:(-4)]
#         patient_train.append(patient)
#         if patient not in patient_test:
#             patient_pars[patient] = pars_df
#
#     env_dict = gym.envs.registration.registry.env_specs.copy()
#     for env in env_dict:
#         if 'CancerControl-v0' in env:
#             print("Remove {} from registry".format(env))
#             del gym.envs.registration.registry.env_specs[env]
#
#     DoseList = []
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--plotStyle', type=str, default='seaborn')
#     parser.add_argument('--para', type=str, default='all')
#     parser.add_argument('--env_id', type=str, default='gym_cancer:CancerControl-v0')
#     parser.add_argument('--seed', type=int, default= 0) # random.randint(0,100000))
#     parser.add_argument('--patients_pars', type=dict, default = patient_pars)
#     parser.add_argument('--patients_train', type=list, default=patient_train)
#     parser.add_argument('--number', '-n', type=int, help='Patient No., int type, requested', default=1)
#     parser.add_argument('--patientNo', type=str, default="patient001")
#     parser.add_argument('--policyPath', type=str, default="./Actor/")
#     parser.add_argument('--metasPar', type = str, default=None)
#     args = parser.parse_args()
#     patientKeys = list(patient_pars.keys())
#     NewpatientKeys = []
#     for patientNo in patientKeys:
#         try:
#             args.patientNo = patientNo
#             d1, d2, d3 = run(args)
#             DoseList.append(d1)
#             NewpatientKeys.append(patientNo)
#             d2.to_csv("./PatientDose/" + patientNo + "_doseSeq.csv")
#             d3.to_csv("./PatientAction/" + patientNo + "_actSeq.csv")
#         except (FileNotFoundError, KeyError):
#             # patientKeys.remove(patientNo)
#             print(patientNo)
#     DoseDf = pd.DataFrame(np.array(DoseList).reshape(-1, 7), index = NewpatientKeys,columns=['CPATotalDosage', 'CPAHolidayMonth', 'CPAFreePercentage', 'LEUTotalDosage', 'LEUHolidayMonth', 'LEUFreePercentage', "TotalMonth"])
#     DoseDf.to_csv('./RL_DoseHistory.csv')
#
# #
# #
# #
# #
# #
# #
# #
