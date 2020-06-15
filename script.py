#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : Train.py.py
# Author: Chongkai LU
# Date  : 3/29/2020

import numpy as np

gt = np.linspace(0, 100, 10000)
pd = np.full(10000, 30)
mae = np.mean(np.abs(pd-gt))