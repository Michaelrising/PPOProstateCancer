#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 11:47:27 2021

@author: michael
"""

from gym.envs.registration import register

register(
    id='CancerControl-v0',
    entry_point='gym_cancer.envs:CancerControl',
)