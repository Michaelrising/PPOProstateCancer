#!/usr/bin/env pyth
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 15:45:32 2021

@author: michael
"""

import torch
import xitorch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
from torch import nn, optim
from .LoadData import LoadData
import shutil

from typing import Callable, Union, Mapping, Any, Sequence, Optional
import xitorch
from xitorch import integrate


###################################################
################### GLV model #####################
###################################################

class CancerODEGlv_CPU(xitorch.EditableModule):
    def __init__(self, patientNo, **params):
        # self.r = params["r"]
        self.A = params["A"]
        self.K = params["K"]
        self.pars = params["pars"]
        self.type = params['type']
        self.data = LoadData()._Patient_data(patientNo)
        if patientNo == "patient002":
            self.data = self.data[:84]
        if patientNo == "patient046":
            self.data[43:46, 1] -= 10
        if patientNo == "patient056":
            self.data[46, 1] = (self.data[44, 1] + self.data[48, 1]) / 2
        if patientNo == "patient086":
            self.data[1, 1] = (self.data[1, 1] + self.data[8, 1]) / 2
        if patientNo == "patient104":
            self.data = self.data[:(-3)]
        # normalization of drug
        self.cpa = torch.tensor([0, 50, 100, 150, 200], dtype=torch.float)
        self.leu = torch.tensor([0, 7.5], dtype=torch.float)
        self._action_set = torch.stack((self.cpa.repeat(2), torch.sort(self.leu.repeat(5))[0]), dim=1)
        self.CPA = torch.from_numpy(self.data[:, 2]).float()
        self.LEU = torch.from_numpy(self.data[:, 3]).float()
        self.__dose = np.array([self.CPA[-2], self.LEU[-2]])
        self.Days = torch.from_numpy(self.data[:, 6] - self.data[0, 6]).float()
        self.OnOff = self.data[:, 5];
        # self.pre_leu()
        self.Response = self.drug_response(
            torch.linspace(start=self.Days[0], end=self.Days[-1], steps=int(self.Days[-1] - self.Days[0]) + 1))
        self.cell_size = 5.236e-10  # 4. / 3. * 3.1415926 * (5e-4cm) ** 3   # cm^3

    def drug_response(self, t):
        drug = torch.zeros((t.shape[0], 2), dtype=torch.float)
        slice0 = torch.bucketize(t, self.Days, right=True) - 1
        slice_75 = torch.where(self.LEU == 7.5)[0]
        dose75_date = self.Days[slice_75]
        slice_225 = torch.where(self.LEU == 22.5)[0]
        dose225_date = self.Days[slice_225]
        slice_30 = torch.where(self.LEU == 30)[0]
        dose30_date = self.Days[slice_30]
        slice_45 = torch.where(self.LEU == 45)[0]
        dose45_date = self.Days[slice_45]

        I0_CPA_dose = self.CPA[slice0]
        I0_LEU_dose = self.LEU[slice0] # LEu dose has 7.5/22.5/30/45, 4 different dosages, for 4/12/16/24 weeks, and we can see that no matter
        dose75 = dose225 = dose30 = dose45 = 1
        _date = -100
        for date in dose75_date.int():
            if abs(date - _date) < 7 or abs(date - _date) == 7:
                dose75 += 1
            else:
                dose75 = 1
            if dose75 == 1:
                temp = torch.zeros(28, dtype = torch.float)  # last 12 weeks
                temp[0:7 * 1] = - 3.75 / 6 * torch.arange(0, 7, 1)
                temp[(7 * 1):(7 * 3)] = (7.5 + 3.75) / (20 - 6) * torch.arange(7, 21, 1) + (
                            7.5 - (7.5 + 3.75) / (20 - 6) * 20)
                temp[(7 * 3):] = 7.5
                I0_LEU_dose[date: (date + 7 * 4)] = temp[0:I0_LEU_dose[date: (date + 7 * 4)].size(0)]
            else:
                I0_LEU_dose[_date: (date + 7 * 4)] = 7.5
            _date = date + 7 * 4
        _date = -100
        for date in dose225_date.int():
            if abs(date - _date) <  7 or abs(date - _date) == 7:
                dose225 += 1
            else:
                dose225 = 1
            if dose225 == 1:
                temp = torch.zeros(7*12, dtype = torch.float) # last 12 weeks
                temp[0:7*1] = - 3.75/6 * torch.arange(0,7,1)
                temp[(7*1):(7*3)] = (7.5 + 3.75)/(20-6) *torch.arange(7, 21, 1) +(7.5 -  (7.5 + 3.75)/(20-6)*20)
                temp[(7*3):] = 7.5
                I0_LEU_dose[date: (date + 7 * 12)] = temp[0:I0_LEU_dose[date: (date + 7 * 12)].size(0)]
            else:
                I0_LEU_dose[_date: (date + 7 * 12)] = 7.5
            _date = date + 7 * 12

        _date = -100
        for date in dose30_date.int():
            if abs(date - _date) <  7 or abs(date - _date) == 7:
                dose30 += 1
            else:
                dose30 = 1
            if dose30 == 1:
                temp = torch.zeros(7*16, dtype = torch.float)  # last 12 weeks
                temp[0:7 * 1] = - 3.75 / 6 * torch.arange(0, 7, 1)
                temp[(7 * 1):(7 * 3)] = (7.5 + 3.75) / (20 - 6) * torch.arange(7, 21, 1) + (
                            7.5 - (7.5 + 3.75) / (20 - 6) * 20)
                temp[(7 * 3):] = 7.5
                I0_LEU_dose[date: (date + 7 * 16)] = temp[0:I0_LEU_dose[date: (date + 7 * 16)].size(0)]
            else:
                I0_LEU_dose[_date: (date + 7 * 16)] = 7.5
            _date = date + 7 * 16
        _date = -100
        for date in dose45_date.int():
            if abs(date - _date) <  7 or abs(date - _date) == 7:
                dose45 += 1
            else:
                dose45 = 1
            if dose45 == 1:
                temp = torch.zeros(7 * 24, dtype = torch.float)  # last 12 weeks
                temp[0:7 * 1] = - 3.75 / 6 * torch.arange(0, 7, 1)
                temp[(7 * 1):(7 * 3)] = (7.5 + 3.75) / (20 - 6) * torch.arange(7, 21, 1) + (
                            7.5 - (7.5 + 3.75) / (20 - 6) * 20)
                temp[(7 * 3):] = 7.5
                I0_LEU_dose[date: (date + 7 * 24)] = temp[0:I0_LEU_dose[date: (date + 7 * 24)].size(0)]
            else:
                I0_LEU_dose[_date: (date + 7 * 24)] = 7.5
            _date = date + 7 * 24

        I0_LEU_dose[torch.cat((torch.where(I0_LEU_dose == 22.5)[0],torch.where(I0_LEU_dose == 30)[0],torch.where(I0_LEU_dose == 45)[0]))] = 7.5
        I0_LEU_dose = I0_LEU_dose / 7.5
        I0_CPA_dose = I0_CPA_dose / 200
        for ii in range(1, I0_LEU_dose.shape[0] - 1):
            if  I0_LEU_dose[ii+1] < 0 and I0_LEU_dose[ii-1] == 1.:
                I0_LEU_dose[ii : (ii + 7 * 3)] = 1.
        drug[:, 0] = I0_CPA_dose
        drug[:, 1] = I0_LEU_dose
        return drug

    def new_dose(self, t, action):
        dose = self._action_set[action]
        slicing = int(t)
        drug = torch.zeros(2)
        if self.__dose[1] != 0:
            if dose[1] != 0:
                drug[1] = 1
        elif self.__dose[1] == 0:
            if dose[1] != 0:
                temp = torch.zeros(7 * 4, dtype=torch.float)  # last 4 weeks
                temp[0:7 * 1] = - 3.75 / 6 * torch.arange(0, 7, 1)
                temp[(7 * 1):(7 * 3)] = (7.5 + 3.75) / (20 - 6) * torch.arange(7, 21, 1) + (
                        7.5 - (7.5 + 3.75) / (20 - 6) * 20)
                temp[(7 * 3):] = 7.5
                drug[1] = temp[slicing]/7.5
        drug[0] = dose[0]/200
        self.__dose = dose
        return drug

    def forward(self, t, y, action = None, ending = None):
        r = self.pars[0:2]
        beta = self.pars[2:4]
        Beta = torch.zeros((2, 2), dtype=torch.float)
        Beta[:, 0] = beta
        phi = self.pars[-3]
        betac = self.pars[-1]
        A = torch.tensor([1., .5, .5, 1.]).view(2, 2)
        A[0,1] = A[1, 0] = self.type/(1 + torch.exp(-self.pars[-2] * torch.tensor([t /28/12])))
        gamma = 0.25  # the half life time for psa is 2.5 days, so each day psa decrease by  25%
        index = int(t) if len(t.shape) == 0 else t.int().cpu().numpy()
        drug = self.Response[index] if action is None else self.new_dose(t - ending, action)
        x = y[0:2]  # cell count
        p = y[-1]  # psa level
        dxdt = torch.multiply(
            r * x, (1 - (x @ A / self.K) ** phi - drug @ Beta))  # -
        dpdt = betac * sum(x) * self.cell_size - gamma * p
        df = torch.zeros(3, dtype=torch.float)
        df[0:2], df[-1] = dxdt, dpdt
        return df

    def getparamnames(self, methodname, prefix=""):
        if methodname == "forward":
            return [prefix + "A",  prefix + "K", prefix + "pars"]  # , prefix+"beta", prefix+"gamma", prefix+"alpha"]
        else:
            raise KeyError()

def MSEloss_weight(inputs, targets, weight):
    return torch.sum(weight * (inputs - targets) ** 2)


def clip_grad(grad, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:

    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grad) == 0:
        return torch.tensor(0.)
    device = grad.device
    total_norm = torch.norm(grad.detach(), norm_type).to(device)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        grad.detach().mul_(clip_coef.to(grad.device))
    return total_norm

# As for the determination of the initial value, we define if psa =10, the cell count for AD is 1