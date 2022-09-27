import collections
from datetime import datetime

import numpy as np
from xitorch import integrate
import argparse
import os
from ._utils import *
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from .LoadData import LoadData
from .ode_glv_cpu import *

def mk_dir(patientNo, glv_dir):
    if not os.path.exists(glv_dir):
        os.mkdir(glv_dir)
    if not os.path.exists(glv_dir + "/model_infos/" + patientNo):
        os.makedirs(glv_dir + "/model_infos/" + patientNo)
    if not os.path.exists(glv_dir + "/model_plots/" + patientNo):
        # shutil.rmtree("./retrain-sigmoid/model_plots/" + patientNo)
        os.makedirs(glv_dir + "/model_plots/" + patientNo)
    if not os.path.exists(glv_dir + "/model_pars/" + patientNo):
        os.makedirs(glv_dir + "/model_pars/" + patientNo)
    if not os.path.exists(glv_dir + "/model_validate/" + patientNo):
        os.makedirs(glv_dir + "/model_validate/" + patientNo)
    return None


def glv_train_online(clinical_data, glv_dir, args = None, tuple_args = None,  online_pars=None, real_pars=None, new_data=None, printf=True): # new data is a dict, with CPA/LEU/PSA/Days
    patientNo, pa_t = tuple_args # if tuple_args is not None else args.patientNo, args.pa_t
    alpha = 0.25 # Alpha[i] 99 and 25 101  83 are 0.5
    cell_size = 5.236e-10
    # obtain training data and validate data, we use the first treatment circle as the training data,
    # which can be viewed as a new patient coming. WE first treat him/her as a standard IADT,
    # then use this information to construct the math model. The next circle of the data can be used as
    # validation the predictive power
    OnOff = clinical_data[:, 5]
    for kk,element in enumerate(OnOff):
        if element == 1 and OnOff[kk-1] == 0:
            train_data = clinical_data[:kk+1]
            train_flag = kk
            validate_data = clinical_data[kk+1:min(kk+len(new_data['days']) + 1, clinical_data.shape[0])]  # the future half year prediction
            break
    train_days = train_data[:, 6] - clinical_data[0, 6]
    train_data[:, 6] = train_days
    train_psa = train_data[:, 1]
    index = np.where(np.isnan(train_psa))[0]
    train_days = np.delete(train_days, index)
    train_psa = torch.from_numpy(np.delete(train_psa, index)).float()
    if new_data is not None:
        train_days = np.append(train_days, new_data['days'], axis = 0)
        train_psa = torch.from_numpy(np.append(train_psa, new_data['psa'], axis = 0)).float()
        new_data_length = len(new_data['days'])
        filling = np.ones((new_data_length, 1))
        new_data_array = np.hstack((filling, np.array(new_data['psa']).reshape(-1, 1), np.array(new_data['cpa']).reshape(-1, 1),
                                         np.array(new_data['leu']).reshape(-1, 1), filling, filling, np.array(new_data['days']).reshape(-1, 1)))

        train_data = np.vstack((train_data, new_data_array))
        # if train_data.shape[0] < clinical_data.shape[0]:
        #     validate_data = validate_data[:validate_data.shape[0] - (train_data.shape[0]-train_flag)]

    validate_days = validate_data[:, 6] - clinical_data[0, 6]
    validate_psa = validate_data[:, 1]
    validate_index = ~np.isnan(validate_psa)
    # validate_psa = np.delete(validate_psa, validate_index)
    # validate_days = np.delete(validate_days, index0)
    Days = clinical_data[:, 6] - clinical_data[0, 6]
    Cycle = clinical_data[:,-3]
    PSA = clinical_data[:, 1]
    index = np.where(np.isnan(PSA))[0]
    PSA = torch.from_numpy(np.delete(PSA, index)).float()
    DAYS = np.delete(Days, index)
    OnOff = np.delete(OnOff, index)
    treatInt = [0.]

    for ii in range(1, OnOff.shape[0] - 1):
        if OnOff[ii - 1] == 0 and OnOff[ii] == 1:
            treatInt.append(DAYS[ii])
    treatInt.append(DAYS[-1])
    mean_v = 5
    mean_psa = 22.1
    K1 = 1.25 * mean_v * (max(PSA)/mean_psa)/cell_size  # 2e+11
    K2 = alpha * K1
    K = torch.tensor([K1, K2]) # 1.1 * mean_v * (max(PSA)/mean_psa)/cell_size #
    A = torch.tensor([1., .5, 0.5, 1.], dtype=torch.float).view(2, 2)
    inputs = torch.linspace(start=Days[0], end=Days[-1], steps=int(Days[-1] - Days[0]) + 1, dtype=torch.float)

    # Initialization the pars
    # pars = torch.tensor([0.03, 0.03, 1, 1, 1, -0.1, 1]).float().requires_grad_()
    real_pars = real_pars
    training_pars = torch.from_numpy(online_pars).float().requires_grad_() if online_pars is not None else torch.tensor([0.03, 0.03, 1, 1, 1, -0.1, 1]).float().requires_grad_()
    loss = torch.tensor([10000], dtype=torch.float)
    Epoch = 1000
    best_loss = 10000
    _val_loss = 10000
    online_cancerode = ODEGlv_CPU(train_data, patientNo, A=A, K=K, pars=training_pars)
    validate_cancerode = ODEGlv_CPU(clinical_data, patientNo, A=A, K=K, pars=training_pars)
    optimizer = torch.optim.Adam([online_cancerode.pars], lr = .001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma=0.75)
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=10)
    loss_deduction_rate_deque = collections.deque(maxlen=50)
    val_loss_deduction_rate_deque = collections.deque(maxlen=50)
    t = datetime.now().strftime("%Y%m%d-%H%M")
    summary_dir = glv_dir + "/model_infos" + '/' + str(patientNo) + '/' + str(t) + "/" + str(pa_t)
    writer = SummaryWriter(log_dir=summary_dir)
    print('================ Updating online Environment================')
    for epoch in range(Epoch):
        Init = torch.tensor([0.8 * K1, 1e-5 * K2, PSA[0]], dtype=torch.float)
        _loss = loss.detach().numpy()

        _pars = training_pars.detach().numpy()
        res = Init.detach().numpy().reshape(1, -1)
        optimizer.zero_grad()
        ts = torch.linspace(start=Days[0], end=train_days[-1], steps=int(train_days[-1]) + 1, dtype=torch.float)
        OUT = integrate.solve_ivp(online_cancerode.forward, ts=ts, y0=Init, params=(), method="rk45", atol=1e-07, rtol=1e-05)
        slicing1 = np.isin(ts.detach().numpy(), train_days.astype(float))
        est_psa = OUT[slicing1, -1]
        weights = torch.ones_like(train_psa)
        low_psa = np.where(train_psa < 4)[0]
        weights[low_psa] = 2
        weights = weights / sum(weights)
        loss = MSEloss_weight(est_psa, train_psa, weights)
        loss.backward()
        loss_deduction_rate = (_loss - loss.detach().numpy())/_loss
        loss_deduction_rate_deque.append(loss_deduction_rate)
        _loss = loss.detach().numpy()
        optimizer.step()

        with torch.no_grad():
            training_pars[:2].clamp_(min = 5e-3, max = 1e-1)
            training_pars[2:4].clamp_(min=1e-3)
            training_pars[-1].clamp_(min=1e-3)
            training_pars[-2].clamp_(max = -1e-8, min= - 1)
        scheduler.step()
        res = np.append(res, OUT.detach().numpy(), axis=0)
        if epoch % 2 == 0:
            with torch.no_grad():
                validate_cancerode.pars = training_pars
                # res = res[1:]
                Init = torch.tensor([0.8 * K1, 1e-5 * K2, PSA[0]], dtype=torch.float)
                pred_ts = inputs[0: validate_days[-1].astype(int)+1]
                pred_res = integrate.solve_ivp(validate_cancerode.forward, ts=pred_ts, y0=Init, params=(), method="rk45", atol=1e-07, rtol=1e-05)
                pred_res = pred_res.detach().numpy()
                pred_val_psa = pred_res[validate_days.astype(int), -1]
                validate_loss = np.sum((pred_val_psa[validate_index] - validate_psa[validate_index]) ** 2)
                val_loss_deduction_rate = (_val_loss - validate_loss) / _val_loss
                _val_loss = validate_loss
                val_loss_deduction_rate_deque.append(val_loss_deduction_rate)
                writer.add_scalar('Loss', loss.detach().numpy().item(), epoch)
                writer.add_scalar('V-Loss', validate_loss, epoch)
                if epoch % 5 == 0:
                    if loss.detach().numpy().item() < best_loss:
                        best_loss = loss.detach().numpy().item()
                        best_pars = training_pars.detach().numpy()

                    validate_loss = np.array(np.sum((pred_val_psa[validate_index] - validate_psa[validate_index]) ** 2)).reshape(-1)
                    validate_list = [validate_psa, pred_val_psa, validate_loss]
                    validate_df = pd.DataFrame(validate_list, index=["true", 'predict', 'loss'])
                    validate_df.to_csv(glv_dir + "/model_validate/" + patientNo + "/validate_" + str(
                        pa_t) + "-" + patientNo + ".csv", index=True)
                    if printf:
                        print("Epochs: {}  Training   Loss: {} Validation Loss: {}".format(epoch, round(loss.detach().numpy().item(), 3), round(validate_loss.item(), 3)))
                    plot_evolution(DAYS, PSA, pred_res, patientNo, pa_t, glv_dir)

                    Init = torch.tensor([mean_v / mean_psa * PSA[0] / cell_size, 1e-5 * K2, PSA[0]], dtype=torch.float)
                    init = Init.detach().numpy()
                    ending_states = res[-1]
                    states = np.concatenate((init, ending_states))
                    K_detach = K.detach().numpy()
                    pars_detach = best_pars
                    initial_date = np.array(train_days[-1]).reshape(-1)
                    plist = [initial_date, K_detach, states, pars_detach, best_pars]
                    save_path = glv_dir + "/model_pars/"+patientNo+"/Args_" + str(pa_t) + "-" + patientNo + ".csv"
                    online_cancerode.save_pars(plist, save_path)
        if epoch > 100:
            early_stopping(validate_loss, training_pars)
            if (np.array(loss_deduction_rate_deque) < 0.01).all() and (np.array(val_loss_deduction_rate_deque) < 0.01).all():
                Init = torch.tensor([mean_v / mean_psa * PSA[0] / cell_size, 1e-5 * K2, PSA[0]], dtype=torch.float)
                init = Init.detach().numpy()
                ending_states = res[-1]
                states = np.concatenate((init, ending_states))
                K_detach = K.detach().numpy()
                pars_detach = best_pars
                initial_date = np.array(train_days[-1]).reshape(-1)
                plist = [initial_date, K_detach, states, pars_detach, best_pars]
                save_path = glv_dir + "/model_pars/" + patientNo + "/Args_" + str(pa_t) + "-" + patientNo + ".csv"
                online_cancerode.save_pars(plist, save_path)
                print("Early stopping of GLV online training due to low loss reduction!")
                print('================= Done =================')
                break
        if early_stopping.early_stop:
            Init = torch.tensor([mean_v / mean_psa * PSA[0] / cell_size, 1e-5 * K2, PSA[0]], dtype=torch.float)
            init = Init.detach().numpy()
            ending_states = res[-1]
            states = np.concatenate((init, ending_states))
            K_detach = K.detach().numpy()
            pars_detach = best_pars
            initial_date = np.array(train_days[-1]).reshape(-1)
            plist = [initial_date, K_detach, states, pars_detach, best_pars]
            save_path = glv_dir + "/model_pars/" + patientNo + "/Args_" + str(pa_t) + "-" + patientNo + ".csv"
            online_cancerode.save_pars(plist, save_path)
            print("Early stopping of GLV online training due to validate loss rising!")
            print('================= Done =================')
            break
    if new_data is not None:
        return best_pars


