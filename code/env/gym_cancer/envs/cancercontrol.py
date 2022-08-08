#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 14:46:21 2021

@author: michael
"""

# Environment
from abc import ABC
import gym
from gym import spaces
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import bernoulli
from collections import deque
from gym.utils import seeding
import torch
import argparse


class CancerControl(gym.Env, ABC):
    def __init__(self, patient, mode='train', t=0.):
        # patient is a dictionary: the patient specific parameters:
        # A, alpha, K, pars, initial states and the terminal states of original drug administration scheme
        # time step: one day
        self.mode = mode
        self.t = t
        self.K, self.pars, self.init_states, self.weight, \
        self.base, self.m1, self.m2_ad, self.m2_ai, drug_decay, drug_length, self.ad_end_c, self.ai_end_c, self.sl = patient
        # the terminal state of sensitive cancer cell is replaced by the capacity of sensitive cancer cell
        # note that the terminal state of the AI cancer cell is the mainly focus in our problem
        self.ad_end_c = min(self.ad_end_c, 0.98) if self.mode == 'train' else 1
        self.ai_end_c = max(self.ai_end_c, 0.8) if self.mode == 'test' else 0.8
        self.m2_ai = max(self.m2_ai, 6)
        self.m2_ad = max(self.m2_ad, 24)
        self.gamma = 0.99  # RL discount factor
        # observation space is a continuous space
        low = np.array([0., 0., 0., -1, -1, -1, 0], dtype=np.float32)
        high = np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high)
        self.treatOnOff = 1  # default On and changes in every step function
        self.cpa = np.array([0, 50, 100, 150, 200])
        self.leu = np.array([0, 7.5])
        self._action_set = np.stack((np.tile(self.cpa, 2), np.sort(self.leu.repeat(5))), axis=1)
        self.action_space = spaces.Discrete(10)
        self.steps = 0
        self.penalty_index = np.zeros(2, dtype = np.float32)
        self.reward_index = 0
        # the first two denotes the drug dosage, the last two denotes the duration time of each drug's treatment,
        # the longest duration for the first line drug is 300 weeks, 
        # for second line drug, the longest duration is 12 months
        # Note that for LEU, the total dosage should be 7.5*8 ml for one treatment duration
        pp = (self.K, self.pars)
        self.cancerode = CancerODEGlv(*pp)
        self.dose = np.zeros(2)
        self.normalized_coef = np.append(self.K, self.K[0] / (1.1 * 5) * self.cancerode.cell_size * 22.1).reshape(-1)
        # self._dose = np.zeros(2)
        self.dosage_arr = []
        self.leu_on = False
        self._action = None
        self.drug_penalty_decay = drug_decay
        self.drug_penalty_length = drug_length
        self.drug_penalty_index = 0
        self.rest_decay = 0.95
        self.max_episodes_steps = 120
        self.metastasis_ai_deque = deque(maxlen=121)
        self.metastasis_ad_deque = deque(maxlen=121)

    def CancerEvo(self, dose):
        # get the cell counts and the PSA level of each day
        # 28 days for one action
        ###################
        ts = np.linspace(start=self.t, stop=self.t + 28 - 1, num=28, dtype=np.int)
        dose_leu = dose[1]
        temp = np.zeros(ts.shape[0], dtype=np.float)
        if dose_leu != 0:
            if not self.leu_on:
                temp[0:7 * 1] = - 3.75 / 6 * np.linspace(0, 7, 7, endpoint= False)
                temp[(7 * 1):(7 * 3)] = (7.5 + 3.75) / (20 - 6) * np.linspace(7, 21, 14, endpoint= False) + (
                        7.5 - (7.5 + 3.75) / (20 - 6) * 20)
                temp[(7 * 3):] = 7.5
            else:
                temp[:] = 7.5
        else:  # current dosage is 0
            temp[:] = 0

        drug = np.repeat(dose.reshape(1, -1), ts.shape[0], axis=0)
        drug[:, 1] = temp

        self.cancerode.ts = ts
        # normalization the drug concentration
        self.cancerode.drug = drug * np.array([1 / 200, 1 / 7.5])
        y0 = self.states.copy()
        # dose = torch.from_numpy(dose_)
        t_interval = (int(self.t), int(self.t) + 28 - 1)
        out = solve_ivp(self.cancerode.forward, t_span=t_interval, y0=y0, t_eval=ts, method="DOP853") #,  atol=1e-7, rtol=1e-5)
        # out = Solve_ivp.solver(self.cancerode.forward, ts = ts, y0 = y0, params = (), atol=1e-08, rtol = 1e-05)
        dy = self.cancerode.forward(t = int(self.t) + 28 - 1, y = out.y[:,-1].reshape(-1))
        return out.y, dy

    def step(self, action):
        if self.steps == 0:
            self.states = self.init_states.copy()
            self.penalty_index = np.zeros(2, dtype=np.float32)
            self.reward_index = 0
            x0, _ = self.init_states[0:2].copy(), self.init_states[2].copy()
            self.leu_on = False

        # By taking action into next state, and obtain new state and reward
        # the action is the next month's dosage
        # update states
        phi0, _, _ = self._utilize(self.t, action)
        dose_ = self._action_set[action]
        self.dose = dose_
        _dose = self.dosage_arr[-1] if self.steps > 0 else np.zeros(2, dtype=np.float32)
        _dose_leu = _dose[1]
        self.leu_on = bool(_dose_leu)
        # penalty index  for the continuous drug administration, and reward index for no-drug administration
        if (dose_ == 0).all() and (self.penalty_index >= 1).all():
            self.reward_index += 1
            self.penalty_index -= np.ones(2, dtype=np.float32)
        if dose_[0] != 0 and dose_[1] == 0:
            self.penalty_index[0] += 1.
            if self.penalty_index[1] >= 1:
                self.penalty_index[1] -= 1.
            self.reward_index = 0
        if dose_[1] != 0 and dose_[0] == 0:
            self.penalty_index[1] += 1.
            if self.penalty_index[0] >= 1:
                self.penalty_index[0] -= 1
            self.reward_index = 0
        if (dose_ != 0).all():
            self.reward_index = 0
            self.penalty_index += np.ones(2, dtype=np.float32)

        if bool(action):
            self.drug_penalty_index += 1
        else:
            self.drug_penalty_index -= 1 if self.drug_penalty_index > 0 else 0


        evolution, df = self.CancerEvo(dose_)
        self.states = evolution[:, -1].clip(np.array([10, 10, 0])).copy()
        x, psa = self.states[0:2], self.states[-1]
        t_current = self.t + 28
        self.t = t_current
        self._action = action
        phi1, c1, c2 = self._utilize(self.t, action)

        done = bool(
            x[0] >= 0.99 * self.K[0] #self.ad_end_c * self.K[0]
            or x[1] >= self.ai_end_c * self.K[1]
            # or x[1] / x[0] > 20 # from the patients original data
            # or bool(self.LEU_On and dose_[1] != 0)
            # or bool(sum(x) > sum(self.init_states[:2]))
            or self.steps > self.max_episodes_steps
            # or bool(self.metastasis_ad_deque.count(1) > self.m2_ad)
            # or bool(self.metastasis_ai_deque.count(1) > self.m2_ai)
        ) if 1.0 >= x[0]/self.K[0] else True
        if self.mode=='train':
            # metastasis_ai = bernoulli.rvs((x[1] / self.K[1])**(2/3), size=1).item()  # if 1 > x[1]/self.K[1] > self.m1 else 0
            # self.metastasis_ai_deque.append(metastasis_ai)
            metastasis_ad = bernoulli.rvs((x[0] / self.K[0])**(2/3), size=1).item() if 1 > x[0] / self.K[0] > self.m1 else 0
            self.metastasis_ad_deque.append(metastasis_ad)
            done = bool(done or bool(self.metastasis_ad_deque.count(1) > self.m2_ad))
                        #or bool(self.metastasis_ai_deque.count(1) > self.m2_ai))
        # reward
        reward = (121 - self.steps) / 121 * (1 - x[0] / self.init_states[0]) + 1 * c2 if not done else 0
        self.dosage_arr.append(dose_)
        # dosage penalty with effect until the drug are eliminated from body
        dosages = np.array(self.dosage_arr)
        d_decay = np.flipud(self.drug_penalty_decay ** np.arange(min(self.drug_penalty_length, len(self.dosage_arr))))
        # d_decay = d_decay[int(len(self.dosage_arr) - self.drug_penalty_length):len(self.dosage_arr)]
        # c_decay = np.flipud(self.drug_penalty_decay ** np.arange(self.penalty_index[0]))
        # l_decay = np.flipud(self.drug_penalty_decay ** np.arange(self.penalty_index[1]))
        c_dose = (dosages[- self.drug_penalty_length:, 0]*d_decay).sum() / 200
        l_dose = (dosages[- self.drug_penalty_length:, 1]*d_decay).sum() / 7.5
        d = np.array([c_dose, l_dose])
        dosage = np.array([0.6, 0.4]) * np.array(self.dosage_arr).sum(axis=0) / np.array([200, 7.5])
        drug_penalty = 0.6 * sum(self.base**self.drug_penalty_index * d * np.array([.6, .4])) + 0.4 * dosage.sum() # (self.base**self.penalty_index -1) *
        reward += (self.steps + 1) * (1 - x[1]/self.K[1]) - drug_penalty # c2 * self.steps/self.sl + (self.ai_end_c - x[1]/self.K[1])* c2 +  + (1 - x[1]/self.K[1])*(self.steps + 1)
        # # reward *= self.steps/self.max_episodes_steps
        self.steps += 1
        normalized_states = np.log10(self.states + 1) / np.log10(self.normalized_coef)
        normalized_df = np.zeros(3)
        for i, dx in enumerate(df):
            if dx > 1:
                normalized_df[i] = (np.log10(dx) + 1)/np.log10(self.normalized_coef[i])
            elif dx < -1:
                normalized_df[i] = (-np.log10(-dx) - 1) / np.log10(self.normalized_coef[i])
            else:
                normalized_df[i] = dx/np.log10(self.normalized_coef[i])
#        metastasis = np.array([self.metastasis_ad_deque.count(1), self.metastasis_ai_deque.count(1)])/self.m2
        fea = np.concatenate((normalized_states, normalized_df, np.array([self.steps/self.max_episodes_steps]))).reshape(-1)
        return fea, self.states, reward, done, {"evolution": evolution, "dose": dose_}

    def _utilize(self, t, action):
        # potential function is only related to states
        # the potential decreases after using drug but increase when without drug administration
        # what we want to describe in the potential function is:
        # 1) the decrease of the cancer cell is rewarding of the therapy
        # 2) the competition of the cancer cell: stronger, better

        pars = self.pars
        phi = pars[-3]
        # A = self.A.copy()
        A01 = A10 = 1 / (1 + np.exp(-pars[-2] * np.array([t /28/12])))
        A = np.array([1, A01.item(), A10.item(), 1]).reshape(2,2)
        K = self.K
        states = self.states.copy()
        x, psa = states[0:2], states[2]
        # potential for cancer volume of androgen dependent cancer cell
        # normalized_vt = -sum(np.log10(x))/sum(np.log10(K)) * 100
        vt = sum(x)/sum(K)
        # normalized_vai = -np.log10(x[1])/np.log10(K[1]) * 100 if x[1] > 1 else 0
        vai = x[1]/K[1]
        # normalized_c2 = np.log10(x[0])/np.log10(K[0]) * A[0,1] * 100
        c1 = (x[1] * A10 / K[0] * 4)**phi
        c2 = (x[0] * A01 / K[1])**phi # the competition index from AD to AI cells
        c1, c2 = (x @ A / self.K) ** phi
        potential = self.weight @ np.array([-vt, -vai]) # (np.log(vai) + np.log(vad))
        return potential, c1, c2

    def seed(self, seed=None):
        np.random.seed(seed)
        # _, _ = seeding.np_random(seed)
        return

    def reset(self):
        self.steps = 0
        self.states = self.init_states.copy()
        self.cancerode.drug = np.zeros((1,2))
        self.cancerode.ts = np.array([0, 1])
        normalized_states = np.log10(self.states + 1) / np.log10(self.normalized_coef)
        normalized_df = np.zeros(3)
        df = self.cancerode.forward(t = 0, y = self.states.reshape(-1))
        for i, dx in enumerate(df):
            if dx > 1:
                normalized_df[i] = (np.log10(dx) + 1) / np.log10(self.normalized_coef[i])
            elif dx < -1:
                normalized_df[i] = (-np.log10(-dx) - 1) / np.log10(self.normalized_coef[i])
            else:
                normalized_df[i] = dx / np.log10(self.normalized_coef[i])

        # normalized_df = self.cancerode.forward(t = 0, y = self.states.reshape(-1)) # /self.normalized_coef
        # normalized_states = self.states # /self.normalized_coef
        # fea = np.concatenate((normalized_states, normalized_df, np.array([self.steps/self.max_episodes_steps]))).reshape(-1)
        self.t = 0.
        self.dosage_arr = []
        self.penalty_index = np.zeros(2, dtype=np.float32)
        self.reward_index = np.zeros(2, dtype=np.float32)
        self.drug_penalty_index = 0
        self.metastasis_ad_deque.clear()
        self.metastasis_ai_deque.clear()
       #metastasis = np.array([self.metastasis_ad_deque.count(1), self.metastasis_ai_deque.count(1)]) / self.m2
        fea = np.concatenate(
            (normalized_states, normalized_df, np.array([self.steps / self.max_episodes_steps]))).reshape(
            -1)
        self.leu_on = False
        return fea, self.states


class CancerODEGlv:
    def __init__(self, K, pars):
        self.A = np.array([1, 0.5, 0.5, 1]).reshape(2,2)
        self.K = K
        self.pars = pars
        self.ts = np.array([0, 1])
        self.drug = np.zeros((1,2))
        self.cell_size = 5.236e-10  # 4. / 3. * 3.1415926 * (5e-4cm) ** 3   # cm^3

    def forward(self, t, y):
        r = self.pars[0:2]
        beta = self.pars[2:4]
        Beta = np.zeros((2, 2), dtype=np.float32)
        Beta[:, 0] = beta
        phi = self.pars[-3]
        betac = self.pars[-1]
        self.A[0, 1] = self.A[1, 0] = 1 / (1 + np.exp(-self.pars[-2] * np.array([t / 28 / 12])))
        gamma = 0.25

        x = y[0:2] #.clip(10)  # cell count
        p = y[-1] # .clip(0)  # psa level
        index = np.where(np.int(t) == self.ts)[0][0] if (np.int(t) <= self.ts).any() else -1
        drug = self.drug[index].reshape(-1)
        dxdt = np.multiply(r * x, (1 - (x @ self.A / self.K) ** phi - drug @ Beta))
        # dxdt_ = dxdt.copy()
        # dxdt_[x<0] = 0
        # dxdt = dxdt if (x>0).all() else dxdt_
        dpdt = betac * sum(x) * self.cell_size - gamma * p
        # mutation: we assume the mutation rate is 4%, with 5 or more mutations wil become new AI cells
        df = np.append(dxdt, dpdt)
        return df
