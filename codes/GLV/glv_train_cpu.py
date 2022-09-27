from datetime import datetime
import torch
import xitorch
from xitorch import integrate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
from torch import nn, optim
from LoadData import LoadData
from _utils import *
from torch.utils.tensorboard import SummaryWriter
from ode_glv_cpu import *
from collections import deque


def train_glv(args, alldata):
    i = args.number
    fail_deque =  deque(maxlen = 10)
    alpha = 0.25 #Alpha[i] 99 and 25 101  83 are 0.5
    cell_size = 5.236e-10
    if len(str(i)) == 1:
        patientNo = "patient00" + str(i)
    elif len(str(i)) == 2:
        patientNo = "patient0" + str(i)
    else:
        patientNo = "patient" + str(i)
    print(patientNo)
    data = alldata[patientNo]
    if patientNo == "patient002":
        data = data[:84]
    Days = data[:, 6] - data[0, 6]
    OnOff = data[:, 5]
    Cycle = data[:,-3]
    PSA = data[:, 1]
    index = np.where(np.isnan(PSA))[0]
    PSA = torch.from_numpy(np.delete(PSA, index)).float()
    DAYS = np.delete(Days, index)
    OnOff = np.delete(OnOff, index)
    treatInt = [0.]

    for ii in range(1, OnOff.shape[0] - 1):
        if OnOff[ii - 1] == 0 and OnOff[ii] == 1:
            treatInt.append(DAYS[ii])
    treatInt.append(DAYS[-1])
    slicing = np.digitize(treatInt, DAYS, right=True)
    validate_set = []
    validate_days = []
    validate_psa = []
    train_days = []
    train_psa = []
    for kk in np.arange(slicing.shape[0]-1):
        if kk != slicing.shape[0]-2:
            samples = DAYS[(slicing[kk] + 1):(slicing[kk + 1]-1)]
        else:
            samples = DAYS[(slicing[kk] + 1): -1]
        size = int(samples.shape[0] * 0.2)
        loo = np.random.choice(samples, size,  replace=False)
        loo = np.sort(loo)
        if kk != slicing.shape[0] - 2:
            samples1 = DAYS[(slicing[kk]):(slicing[kk + 1])]
            psa_samples1 = PSA[(slicing[kk]):(slicing[kk + 1])]
        else:
            samples1 = DAYS[slicing[kk]:]
            psa_samples1 = PSA[slicing[kk]:]
        if size != 0:
            validate_days.append(samples1[np.isin(samples1, loo)].astype(int))
            validate_psa.append(psa_samples1[np.isin(samples1, loo)].detach().numpy())
            print(loo)
            print(psa_samples1[np.isin(samples1, loo)].detach().numpy())
        train_days.append(samples1[~np.isin(samples1, loo)].astype(int))
        train_psa.append(psa_samples1[~np.isin(samples1, loo)])
    validate_days = np.concatenate(validate_days).reshape(-1)
    print(validate_days.shape[0])
    validate_psa = np.concatenate(validate_psa).reshape(-1)
    print(validate_psa.shape[0])
    mean_v = 5
    mean_psa = 22.1
    K1 = 1.25 * mean_v * (max(PSA)/mean_psa)/cell_size  # 2e+11
    K2 = alpha * K1
    K = torch.tensor([K1, K2]) #1.1 * mean_v * (max(PSA)/mean_psa)/cell_size #
    A = torch.tensor([1., .5, 0.5, 1.], dtype=torch.float).view(2, 2)
    inputs = torch.linspace(start=Days[0], end=Days[-1], steps=int(Days[-1] - Days[0]) + 1, dtype=torch.float)

    # Initialization the pars
    pars = torch.tensor([0.03, 0.03, 1, 1, 1, -0.1, 1]).float().requires_grad_()
    loss = torch.tensor([10000], dtype=torch.float)
    Epoch = 5000
    best_loss = 10000
    best_pars = pars.detach().numpy()
    cancerode = ODEGlv_CPU(data, patientNo, A=A, K=K, pars=pars)
    optimizer = torch.optim.Adam([cancerode.pars], lr = .001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma=0.75)
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=15)

    if not os.path.exists("./analysis-dual-sigmoid"):
        os.mkdir("./analysis-dual-sigmoid")
    if not os.path.exists("./analysis-dual-sigmoid/model_infos/" + patientNo):
        os.makedirs("./analysis-dual-sigmoid/model_infos/" + patientNo)
    if not os.path.exists("./analysis-dual-sigmoid/model_plots/" + patientNo):
        # shutil.rmtree("./retrain-sigmoid/model_plots/" + patientNo)
        os.makedirs("./analysis-dual-sigmoid/model_plots/" + patientNo)
    if not os.path.exists("./analysis-dual-sigmoid/model_pars/" + patientNo):
        os.makedirs("./analysis-dual-sigmoid/model_pars/" + patientNo)
    if not os.path.exists("./analysis-dual-sigmoid/model_validate/" + patientNo):
        os.makedirs("./analysis-dual-sigmoid/model_validate/"+ patientNo)
    t = datetime.now().strftime("%Y%m%d-%H%M")
    summary_dir = "./analysis-dual-sigmoid/model_infos" + '/' + str(patientNo) + '/' + str(t) + "/" + str(args.t)
    writer = SummaryWriter(log_dir=summary_dir)
    for epoch in range(Epoch):
        Init = torch.tensor([0.8 * K1, 1e-5 * K2, PSA[0]], dtype=torch.float)
        _loss = loss.detach().numpy()
        _pars = pars.detach().numpy()
        loss = torch.zeros(1, dtype=torch.float)
        res = Init.detach().numpy().reshape(1, -1)
        optimizer.zero_grad()
        for ii in range(len(treatInt) - 1):
            if ii == len(treatInt) - 2:
                INPUTS = inputs[int(treatInt[ii]):]
            else:
                INPUTS = inputs[int(treatInt[ii]):int(treatInt[ii + 1])]
            ts = INPUTS.requires_grad_(False)
            OUT = integrate.solve_ivp(cancerode.forward, ts=ts, y0=Init, params=(), method="rk45", atol=1e-07, rtol=1e-05)
            res = np.append(res, OUT.detach().numpy(), axis=0)
            d = train_days[ii]
            slicing1 = np.isin(INPUTS.detach().numpy(), d.astype(float))
            EST_PSA = OUT[slicing1, -1]
            # EST_PSA[torch.isnan(EST_PSA)] = 1000
            psa = train_psa[ii]
            if ii == 0:
                psa[0] = Init[-1]
            Init = OUT[-1]
            weights = torch.ones_like(psa)
            low_psa = np.where(psa < 4)[0]
            weights[low_psa] = 2
            weights = weights/sum(weights)

            loss = MSEloss_weight(EST_PSA, psa, weights) #+ loss
            loss.backward(retain_graph=True if ii != len(treatInt) - 2 else False)

        optimizer.step()
        with torch.no_grad():
            pars[:2].clamp_( min = 5e-3, max = 1e-1)
            pars[2:4].clamp_(min=1e-3)
            pars[-1].clamp_(min=1e-3)
            pars[-2].clamp_(max = -1e-8, min= - 1)
        scheduler.step()

        with torch.no_grad():
            res = res[1:]
            ad = res[:, 0]
            ai = res[:, 1]
            p = res[:, 2]
            val_psa = p[validate_days]
            validate_loss = np.mean((val_psa - validate_psa) ** 2)

            writer.add_scalar('Loss', loss.detach().numpy().item(), epoch)
            writer.add_scalar('V-Loss', validate_loss, epoch)
            if epoch % 5 == 0:
                print(pars.detach().numpy())
                if loss.detach().numpy().item() < best_loss:
                    best_loss = loss.detach().numpy().item()
                    best_pars = pars.detach().numpy()

                pred_validate_psa = p[validate_days]
                validate_loss = np.array([sum((pred_validate_psa - validate_psa) ** 2)], dtype=float)
                validate_list = [validate_psa, pred_validate_psa, validate_loss]
                validate_df = pd.DataFrame(validate_list, index=["true", 'predict', 'loss'])
                validate_df.to_csv("./analysis-dual-sigmoid/model_validate/" + patientNo + "/validate_" + str(
                    args.t) + "-" + patientNo + ".csv", index=True)
                print("Validation Loss: {}".format(validate_loss))
                plot_dir = "./analysis-dual-sigmoid"
                plot_evolution(DAYS, PSA,res, patientNo, args.t, plot_dir)

                Init = torch.tensor([mean_v / mean_psa * PSA[0] / cell_size, 1e-5 * K2, PSA[0]], dtype=torch.float)
                init = Init.detach().numpy()
                save_path = "./analysis-dual-sigmoid/model_pars/"+patientNo+"/Args_" + str(args.t) + "-" + patientNo + ".csv"
                cancerode.save_pars(init, best_pars, save_path)
        if epoch > 1500:
            early_stopping(validate_loss, pars)
        if early_stopping.early_stop:
            print("Early stopping")
            break

parser = argparse.ArgumentParser(description='Patient arguments')
parser.add_argument('--number', '-n', default = 1, help='Patient No., int type, requested', type=int)
parser.add_argument('--t',  default=0, type=int)
args = parser.parse_args()

if __name__ == "__main__":
    alldata = LoadData().Double_Drug()
    train_glv(args, alldata)

