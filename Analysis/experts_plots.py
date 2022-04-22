import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.offsetbox as moff
from collections import deque
from scipy.stats import bernoulli
# import xitorch
import random
from Analysis.tMGLV import CancerODEGlv_CPU
from xitorch import integrate
from Analysis.LoadData import LoadData
import seaborn as sns
import scipy.stats as st
from _utils import *
#

handbox = moff.DrawingArea(width = 60, height=21)
extra_path = "../Experts_policy/analysis/prediction/"
if not os.path.exists(extra_path):
    os.mkdir(extra_path)
savepath = "../Experts_figs/analysis/"
parsDir = '../GLV/analysis-dual-sigmoid/model_pars/'
# parsDir = '../GLV/analysis-sigmoid/model_pars/gcp/'
if not os.path.exists(savepath):
    os.mkdir(savepath)
patientlist1 =  [1, 2, 3, 4, 6, 13, 15, 16, 17, 20, 24, 29, 30, 31, 32, 37, 40, 42, 44, 46, 50, 51,
                58, 60, 61, 62, 63, 66, 71, 75, 77, 78, 79, 84, 86, 87, 91, 92, 93, 94, 95, 96, 97,
               100, 102, 104, 105, 106, 108] # kick 56 and 83
patientlist2 = [11, 12, 19, 25, 36, 52, 54, 85, 88, 99, 101]
patientlist = patientlist1 + patientlist2
patientlist.sort()
cell_size = 5.236e-10
mean_v = 5
Mean_psa = 22.1
alldata = LoadData().Double_Drug()
s_endlist = []
s_end_patients = []
metastasis_all_patients = []
max_ai_c_all_patients = []
max_ad_c_all_patients = []
survival_length_all_patients = []
range_c1_all_patients = []
range_c2_all_patients = []
for no in patientlist:
    if len(str(no)) == 1:
        patientNo = "patient00" + str(no)
    elif len(str(no)) == 2:
        patientNo = "patient0" + str(no)
    else:
        patientNo = "patient" + str(no)
    argslist = os.listdir(parsDir+patientNo)
    argslist.sort()
    AD_List = []
    AI_List = []
    PSA_List = []
    X_List = []
    Truncated_flag_list = []
    data = alldata[patientNo]
    if patientNo == "patient002":
        data = data[:84]
    if patientNo == "patient046":
        data[43:46, 1] -= 10
    if patientNo == "patient056":
        data[46, 1] = (data[44, 1] + data[48, 1]) / 2
    if patientNo == "patient086":
        data[1, 1] = (data[1, 1] + data[8, 1]) / 2
    if patientNo == "patient104":
        data = data[:(-3)]
    EXTRA_DOSE = []
    extra_dose_len = 0
    metastasis = []
    max_ai_c = []
    max_ad_c = []
    max_c1 = 0
    min_c1 = 100
    max_c2 = 0
    min_c2 = 100
    survival_length = []
    if no in [11, 12, 19, 25, 36, 54, 85, 88, 99, 101]:
        t = 2
    else:
        t=1
    for patient_args in argslist:
        pars_df = pd.read_csv(parsDir+patientNo +'/' + patient_args)

        #patientNo = "patient011"
        print(patientNo)
        A, K, states, final_pars, best_pars = [torch.from_numpy(np.array(pars_df.loc[i, ~np.isnan(pars_df.loc[i, :])])).float() for
                                               i in range(5)]
        actions_seq = np.array(pd.read_csv("../Experts_policy/" + patientNo + "_actions_seqs.csv"))[:, 0]
        original_steps = actions_seq.shape[0]
        actions_seqs_prediction = deque(actions_seq[np.where(actions_seq != 0)[0]])
        Days = data[:, 6] - data[0, 6]
        PSA = data[:, 1]
        OnOff = data[:, 5]
        index = np.where(np.isnan(PSA))[0]
        PSA = torch.from_numpy(np.delete(PSA, index)).float()
        DAYS = np.delete(Days, index)
        inputs = torch.linspace(start=Days[0], end=Days[-1], steps=int(Days[-1] - Days[0]) + 1, dtype=torch.float)
        cancerode = CancerODEGlv_CPU(patientNo, A=A, K=K, pars=best_pars, type = t)
        Init = states[:3] #torch.tensor([mean_v / Mean_psa * PSA[0] / cell_size, 1e-4 * K[1], PSA[0]],
        #                  dtype=torch.float)  # states[:3]
        OriginalOut = integrate.solve_ivp(cancerode.forward, ts=inputs, y0=Init, params=(), atol=1e-08, rtol=1e-05)
        OriginalOut = OriginalOut.detach().numpy()
        ad = OriginalOut[:, 0]
        ai = OriginalOut[:, 1]
        psa = OriginalOut[:, -1]
        x = inputs.numpy()

        # prospective eval of done
        each_max_ai_c = max(ai/K[1]).item()
        each_max_ad_c = max(ad/K[0]).item()
        original_metastasis_ai_deque = deque(maxlen=121)
        original_metastasis_ad_deque = deque(maxlen=121)
        slicing = np.linspace(start = 0, stop = Days[-1], endpoint = False, num = int(Days[-1] // 28), dtype = int) # DAYS.astype(np.int)
        # adMeasured = ad[slicing]
        # aiMeasured = ai[slicing]

        flagss = None
        Truncated_flag = False
        for ss in slicing:
            adMeasured = ad[ss]
            aiMeasured = ai[ss]
            try:
                metastasis_ad = bernoulli.rvs((adMeasured / K[0])**(2/3), size=1).item() if adMeasured / K[0] > 0.5 else 0
                metastasis_ai = bernoulli.rvs((aiMeasured / K[1])**(2/3), size=1).item() #if aiMeasured / K[1] > 0.5 else 0
            except ValueError:
                print(adMeasured / K[0])
                print(aiMeasured / K[1])
            original_metastasis_ai_deque.append(metastasis_ai)
            original_metastasis_ad_deque.append(metastasis_ad)
            done = bool(adMeasured >= K[0] or aiMeasured >= 0.8 * K[1] or original_steps > 121)
            if done:
                Truncated_flag = True
                Truncated_flag_list.append(Truncated_flag)
                flagss = ss
                print(ss)
                TruncatedAD = ad[:ss]
                TruncatedAD1 = ad[ss:]
                TruncatedAI = ai[:ss]
                TruncatedAI1 = ai[ss:]
                TruncatedPSA = psa[:ss]
                TruncatedPSA1 = psa[ss:]
                TruncatedX = x[:ss]
                TruncatedX1 = x[ss:]
                AD = ad
                AI = ai
                Psa = psa
                X = np.array([ss, x[-1]])
                AD_List.append(AD)
                AI_List.append(AI)
                PSA_List.append(Psa)
                X_List.append(X)
                break
        def Done(x, y, metastasis_ad_deque=original_metastasis_ad_deque, metastasis_ai_deque=original_metastasis_ai_deque,
                 s1=original_steps, s2=0):  # x: ad y: ai
            metastasis_ad = bernoulli.rvs((x / K[0])**(2/3), size=1).item() if x / K[0] > 0.5 else 0
            #metastasis_ai = bernoulli.rvs((y / K[1]), size=1).item() if y / K[1] > 0.5 else 0
            metastasis_ad_deque.append(metastasis_ad)
            #metastasis_ai_deque.append(metastasis_ai)
            mask = bool(
                x >= K[0]
                or y >= 0.8 * K[1]
                or bool(s1 + s2 > 121)
                # or bool(metastasis_ad_deque.count(1) >= 12)
                # or bool(metastasis_ai_deque.count(1) >= 12)
            )
            return mask
        ending = inputs[-1]
        new_inits = torch.from_numpy(np.array([ad[-1], ai[-1], psa[-1]])).float()
        new_ai = ai[-1];
        new_ad = ad[-1];
        new_psa = psa[-1]
        # in the doctor's policy they can only consider the psa level
        # predict
        PredictOut = []
        extraActSeq = []
        new_steps = 0
        if not done: # not Truncated_flag and
            temp = OnOff[::-1]
            __action = temp[0]
            dose_times = 0
            # define the last action is drug administrated or not
            if temp[0]:
                for ii in range(temp.shape[0]):
                    dose_times += 1
                    if temp[ii] == 1 and temp[ii + 1] == 0:
                        break
            max_dosage_times = 8
            while not done:
                if bool(__action) and dose_times < max_dosage_times:
                    action = actions_seqs_prediction.popleft()
                    actions_seqs_prediction.append(action)
                    dose_times += 1
                else:
                    if new_psa > 10 or new_ad/K[0] >= 0.99:
                        action = actions_seqs_prediction.popleft()
                        actions_seqs_prediction.append(action)
                        dose_times += 1
                    else:
                        action = 0
                        dose_times = 0
                __action = action
                extraActSeq.append(action)
                new_steps += 1
                t_stamp = torch.linspace(start=ending, end=ending + 27, steps=28, dtype=torch.float)
                new_out = integrate.solve_ivp(cancerode.forward, ts=t_stamp, y0=new_inits, params=(int(action), ending,), atol=1e-08, rtol=1e-05)
                new_ad = new_out[:, 0].detach().numpy()
                new_ai = new_out[:, 1].detach().numpy()
                new_psa = new_out[:, -1].detach().numpy()
                each_max_ai_c = max(each_max_ai_c, max(new_ai / K[1]).item())
                each_max_ad_c = max(each_max_ad_c, max(new_ad / K[0]).item())
                ending = t_stamp[-1]
                new_psa = new_psa[-1]
                new_ad = new_ad[-1]
                new_ai = new_ai[-1]
                new_inits = torch.from_numpy(np.array([new_ad, new_ai, new_psa])).float()

                done = Done(new_ad, new_ai, s2=new_steps)
                PredictOut.append(new_out.detach().numpy())
            EXTRA_DOSE.append(extraActSeq)
            # extraActDf = pd.DataFrame(extraActSeq)
            # extraActDf.to_csv(extra_path + patient_args[5:-4] + "_extrapolated_doseSeq.csv")
            PredictOut = np.vstack(PredictOut)
            AllOut = np.concatenate((OriginalOut, PredictOut))
            x = np.arange(AllOut.shape[0])
            ad = AllOut[:, 0]
            ai = AllOut[:, 1]
            psa = AllOut[:, 2]
            OriginalX = x[:int(Days[-1])]
            PredictX = x[int(Days[-1]):]
            OriginalAI = ai[:int(Days[-1])]
            PredictAI = ai[int(Days[-1]):]
            OriginalAD = ad[:int(Days[-1])]
            PredictAD = ad[int(Days[-1]):]
            OriginalPSA = psa[:int(Days[-1])]
            PredictPSA = psa[int(Days[-1]):]
            AD = ad
            AI = ai
            Psa = psa
            X = np.array([int(Days[-1]), x[-1]])
            AD_List.append(AD)
            AI_List.append(AI)
            PSA_List.append(Psa)
            X_List.append(X)
            Truncated_flag_list.append(Truncated_flag)
        # Compute the competition intensity range
        phi, gamma = best_pars.detach().numpy()[-4:-2]
        K = K.detach().numpy()
        a = 1 / (1 + np.exp(-gamma * np.array([np.arange(X[-1].item() + 1) / 28 / 12]))).reshape(-1)
        c1 = (AI * 0.5 / K[0] * 4)**phi
        c2 = (AD * a /K[1])**phi
        max_c1 = max(max_c1, max(c1))
        min_c1 = min(min_c1, min(c1))
        max_c2 = max(max_c2, max(c2))
        min_c2 = min(min_c2, min(c2))

        max_ai_c.append(each_max_ai_c)
        max_ad_c.append(each_max_ad_c)
        metastasis.append([original_metastasis_ad_deque.count(1), original_metastasis_ai_deque.count(1)])
        survival_length.append(original_steps + new_steps)
    # OUT = np.concatenate((OUT, new_out.detach().numpy()))
    Flag = sum(Truncated_flag_list)
    if Flag < 5:
        extra_dose_len = np.array([len(i) for i in EXTRA_DOSE])
        ava_extra_dose_len = int(np.mean(extra_dose_len).item()) + 1
        max_extra_dose = EXTRA_DOSE[np.where(extra_dose_len == max(extra_dose_len))[0][0]]
        extraActDf = pd.DataFrame(max_extra_dose[:ava_extra_dose_len])
        extraActDf.to_csv(extra_path + patientNo + "_predicted_doseSeq.csv")
    ad_list = []
    ai_list = []
    psa_list = []
    x_list = []
    X_array = np.stack(X_List)
    add_0_length = max(X_array[:, 1]) - X_array[:, 1]
    for i in range(len(AD_List)):
        AD_List[i] = np.concatenate((AD_List[i], -np.ones(int(add_0_length[i]))))
        AI_List[i] = np.concatenate((AI_List[i], -np.ones(int(add_0_length[i]))))
        PSA_List[i] = np.concatenate((PSA_List[i], -np.ones(int(add_0_length[i]))))
    AD_arr = np.stack(AD_List)
    AI_arr = np.stack(AI_List)
    PSA_arr = np.stack(PSA_List)
    mean_ad = np.array([np.mean(AD_arr[AD_arr[:, i] != -1, i], axis=0) for i in range(AD_arr.shape[1])])
    sd_ad = np.array([st.sem(AD_arr[AD_arr[:, i] != -1, i]) for i in range(AD_arr.shape[1])])
    low_ad_bound, high_ad_bound = st.t.interval(0.95, 10 - 1, loc=mean_ad, scale=sd_ad)

    mean_ai = np.array([np.mean(AI_arr[AI_arr[:, i] != -1, i], axis=0) for i in range(AI_arr.shape[1])])
    sd_ai = np.array([st.sem(AI_arr[AI_arr[:, i] != -1, i]) for i in range(AI_arr.shape[1])])
    low_ai_bound, high_ai_bound = st.t.interval(0.95, 10 - 1, loc=mean_ai, scale=sd_ai)

    mean_psa = np.array([np.mean(PSA_arr[PSA_arr[:, i] != -1, i], axis=0) for i in range(PSA_arr.shape[1])])
    sd_psa = np.array([st.sem(PSA_arr[PSA_arr[:, i] != -1, i]) for i in range(PSA_arr.shape[1])])
    low_psa_bound, high_psa_bound = st.t.interval(0.95, 10 - 1, loc=mean_psa, scale=sd_psa)
    pd.DataFrame(np.stack((mean_ad, mean_ai, mean_psa), axis=1), columns=['ad', 'ai', 'psa']).to_csv("../Experts_states/analysis/"+patientNo+'_experts_states.csv')
    cs = sns.color_palette("Paired")
    # plt.style.use('seaborn')
    plt.style.use(["science", "nature"])
    face_c_pred0 = colorAlpha_to_rgb(cs[0], 0.5)[0]
    face_c_pred1 = colorAlpha_to_rgb(cs[3], 0.2)[0]
    face_c_pred3 = colorAlpha_to_rgb(mplc.to_rgb('grey'), 0.5)[0]
    face_c_pred4 = colorAlpha_to_rgb(mplc.to_rgb('grey'), 0.2)[0]
    face_c_pred5 = colorAlpha_to_rgb(cs[2], 0.5)[0]
    fig = plt.figure(figsize=(5, 4))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.scatter(DAYS, PSA, marker="*", s=20, color=cs[1], zorder = 5)
    # ax1.set_xlim((-66.15, 1389.15))
    if Flag < 5:
        plt.plot(np.arange(int(Days[-1])), mean_psa[:int(Days[-1])], c=cs[1], lw=1.2, zorder=4)

        plt.fill_between(np.arange(int(np.mean(X_array[:, 1] + 1))), low_psa_bound[:int(np.mean(X_array[:, 1] + 1))], high_psa_bound[:int(np.mean(X_array[:, 1] + 1))], color=face_c_pred0, zorder=3)
        predict_x = np.arange(int(Days[-1]), int(np.mean(X_array[:, 1] + 1)))
        plt.plot(predict_x, mean_psa[int(Days[-1]):int(np.mean(X_array[:, 1] + 1))], color=cs[1], lw=1.2, ls='--', zorder=4)

        plt.axvspan(xmin=int(Days[-1]), xmax= np.mean(X_array[:, 1] + 1), ymin=0, facecolor=face_c_pred1, label="PRED")
    else:
        truncated_x = int(np.mean(X_array[Truncated_flag_list, 0]).item())
        s_endlist.append(truncated_x)
        s_end_patients.append(patientNo)
        plt.plot(np.arange(truncated_x), mean_psa[:truncated_x], c=cs[1], lw=1.2, zorder=4)

        plt.fill_between(np.arange(truncated_x), low_psa_bound[:truncated_x], high_psa_bound[:truncated_x], color=face_c_pred0, zorder=3)
        done_x = np.arange(truncated_x, int(Days[-1])+1)
        plt.plot(done_x, mean_psa[truncated_x:int(Days[-1])+1], c='grey', lw=1.2, zorder=4)

        plt.fill_between(done_x, low_psa_bound[truncated_x:int(Days[-1])+1], high_psa_bound[truncated_x:int(Days[-1])+1], color=face_c_pred3, zorder=3)

        plt.axvspan(xmin=truncated_x, xmax=int(Days[-1]), ymin=0, facecolor=face_c_pred4, label="DONE")
    lower_xlim, high_xlim = copy.deepcopy(ax1.get_xlim())
    plt.hlines(y=4, xmin=lower_xlim, xmax=high_xlim, colors=cs[5], lw=1.2, zorder=4, linestyles='--')
    plt.hlines(y=10, xmin=lower_xlim, xmax=high_xlim, colors=cs[5], lw=1.2, zorder=4, linestyles='--')
    # plt.xlabel("Time (Days)", fontsize=22)
    ax1.set_xlim((lower_xlim, high_xlim))
    ax1.set_ylabel("PSA ($\mu$g/L)", fontsize=15)
    ax1.tick_params(labelsize=12)
    ax1.set_xticks(ticks=[], minor=False)
    # ax1.set_xticklabels(labels=[], minor=False)
    # # if max(PSA).item()/10
    # ax1.set_yticks(ticks=[0, 5, 10])
    # ax1.set_yticklabels(labels=[0, 5, 10], minor=False)
    threshold = Line2D([0], [0], color=cs[5],ls='--', lw=1.2)
    a1 = AnyObject()
    a1_c = colorAlpha_to_rgb(cs[3], 0.4)[0] if Flag<5 else colorAlpha_to_rgb(mplc.to_rgb('grey'), 0.4)[0]
    ax1.legend([a1,threshold],['prediction' if Flag<5 else 'done', 'threshold'], loc = 'upper left',handler_map={a1: AnyObjectHandler(color= a1_c, alpha = None,_hatch=None)}, fontsize=12, ncol=2)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(np.arange(int(Days[-1])), mean_ai[:int(Days[-1])], c=cs[3], lw=1.2, label='RST', zorder=4)

    plt.fill_between(np.arange(int(Days[-1])), low_ai_bound[:int(Days[-1])], high_ai_bound[:int(Days[-1])], color=face_c_pred5, zorder=3)
    plt.plot(np.arange(int(Days[-1])), mean_ad[:int(Days[-1])], c=cs[1], lw=1.2, label='RSP', zorder=4)

    plt.fill_between(np.arange(int(Days[-1])), low_ad_bound[:int(Days[-1])], high_ad_bound[:int(Days[-1])], color=face_c_pred0, zorder=3)
    if Flag < 5:
        predict_x = np.arange(int(Days[-1]), int(np.mean(X_array[:, 1] + 1)))
        plt.plot(predict_x, mean_ad[int(Days[-1]): int(np.mean(X_array[:, 1] + 1))], ls='--', c=cs[1], lw=1.2, zorder=4)
        plt.fill_between(predict_x, low_ad_bound[int(Days[-1]): int(np.mean(X_array[:, 1] + 1))], high_ad_bound[int(Days[-1]): int(np.mean(X_array[:, 1] + 1))], color=face_c_pred0, zorder=3)
        plt.plot(predict_x, mean_ai[int(Days[-1]): int(np.mean(X_array[:, 1] + 1))], ls='--', lw=1.2, c=cs[3], zorder=4)
        plt.fill_between(predict_x, low_ai_bound[int(Days[-1]): int(np.mean(X_array[:, 1] + 1))], high_ai_bound[int(Days[-1]): int(np.mean(X_array[:, 1] + 1))], color=face_c_pred5, zorder=3)
        plt.axvspan(xmin=int(Days[-1]), xmax=np.mean(X_array[:, 1] + 1), ymin=0, facecolor=face_c_pred1, label="PRED")
    else:
        truncated_x = int(np.mean(X_array[Truncated_flag_list, 0]).item())
        done_x = np.arange(truncated_x, int(Days[-1]) + 1)
        plt.plot(done_x, mean_ad[truncated_x:int(Days[-1])+1], c='grey', lw=1.2, zorder=4)
        plt.fill_between(done_x, low_ad_bound[truncated_x:int(Days[-1])+1], high_ad_bound[truncated_x:int(Days[-1])+1], color=face_c_pred3, zorder=3)
        plt.plot(done_x, mean_ai[truncated_x:int(Days[-1])+1], c='grey', lw=1.2, zorder=4)
        plt.fill_between(done_x, low_ai_bound[truncated_x:int(Days[-1])+1], high_ai_bound[truncated_x:int(Days[-1])+1], color=face_c_pred3, zorder=3)
        plt.axvspan(xmin=truncated_x, xmax=int(Days[-1]), ymin=0, facecolor=face_c_pred4, label="DONE")
    ax2.set_xlabel("Time (Days)", fontsize=15)
    ax2.set_ylabel("$\#$ of Cell", fontsize=15)
    ax2.tick_params(labelsize=12)
    # ax2.set_yticks(ticks = [0, 0.25 * 10**10, .5 * 10**10])
    # ax2.set_yticklabels(labels=[0, 2.5, 5])
    # plt.text(0.0, 1, '$10^{9} $', fontsize = 8, transform=ax2.transAxes)
    a2 = AnyObject()
    a2_c = colorAlpha_to_rgb(cs[3], 0.4)[0] if Flag<5 else colorAlpha_to_rgb(mplc.to_rgb('grey'), 0.4)[0]
    rss = Line2D([0], [0], color=cs[3], lw=1.2)
    rsp = Line2D([0], [0], color=cs[1], lw=1.2)
    l2 = plt.legend([a2], ['prediction' if Flag < 5 else 'done'], loc=[0.01, 0.75],
                    handler_map={a2: AnyObjectHandler(color=a2_c, alpha=None, _hatch=None)}
                    , fontsize=12)
    l1 = plt.legend([rss, rsp], ['resistance', 'response'], loc=[0.01, 0.45], fontsize=12, ncol=1)

    plt.gca().add_artist(l2)
    plt.tight_layout()
    plt.savefig(savepath + patientNo + "_evolution.eps", dpi=500, bbox_inches='tight')
    plt.show()
    metastasis_all_patients.append(np.mean(np.stack(metastasis), 0))
    max_ai_c_all_patients.append(np.mean(max_ai_c))
    max_ad_c_all_patients.append(np.mean(max_ad_c))
    survival_length_all_patients.append(np.mean(survival_length))
    range_c1_all_patients.append([min_c1.item(), max_c1.item()])
    range_c2_all_patients.append([min_c2.item(), max_c2.item()])

metastasis_all_patients = np.stack(metastasis_all_patients)
max_ai_c_all_patients = np.array(max_ai_c_all_patients).reshape(-1, 1)
max_ad_c_all_patients = np.array(max_ad_c_all_patients).reshape(-1, 1)
survival_length_all_patients = np.array(survival_length_all_patients).reshape(-1,1)
range_c1_all_patients = np.array(range_c1_all_patients).reshape(-1, 2)
range_c2_all_patients = np.array(range_c1_all_patients).reshape(-1, 2)
end_states_all_patients = np.concatenate((metastasis_all_patients, max_ad_c_all_patients, max_ai_c_all_patients,
                                          survival_length_all_patients, range_c1_all_patients, range_c2_all_patients),axis=1)
pd.DataFrame(end_states_all_patients, columns=['m_ad', 'm_ai', 'c_ad', 'c_ai','sl', 'minc1', 'maxc1', 'minc2', 'maxc2'], index=patientlist).to_csv('end_states_all_patients.csv')
pd.DataFrame(s_endlist, index=s_end_patients).to_csv('../Experts_policy/analysis/s_end_list.csv')