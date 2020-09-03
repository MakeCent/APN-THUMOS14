#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : evaluate.py
# Author: Chongkai LU
# Date  : 6/8/2020
import pandas as pd
import numpy as np
import json
from tools.utils import *

action = 'Shotput'
mode = 'fused'
print('evaluate on {}_{}'.format(action, mode))
np.set_printoptions(suppress=True)
if mode == 'rgb' or mode == 'flow' or mode == 'two-stream':
    prediction_file_location = 'saved/{}_{}_prediction'.format(action, mode)
    with open(prediction_file_location, 'r') as f:
        list_predictions = json.load(f)
    predictions = {k: np.array(v) for k, v in list_predictions.items()}
elif mode == 'fused':
    rgb_prediction_file_location = 'saved/{}_rgb_prediction'.format(action)
    flow_prediction_file_location = 'saved/{}_flow_prediction'.format(action)
    with open(rgb_prediction_file_location, 'r') as f1:
        rgb_list_predictions = json.load(f1)
    with open(flow_prediction_file_location, 'r') as f2:
        flow_list_predictions = json.load(f2)
    predictions = {k: (np.array(rgb_list_predictions[k][:len(v)]) + np.array(v))/2 for k, v in flow_list_predictions.items()}



IoU = 0.5
# min_T, max_T, min_L = 60, 30, 60
down_sampling = 1

ap = {}
det = {}
gt = {}

anndir = "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/test/annotationF"
ac_ta = pd.read_csv(
    "{}/{}_testF.csv".format(anndir, action),
    header=None).values
for min_T in [60, 70, 80]:
    for max_T in [20, 30, 40]:
        for min_L in [30, 60, 80]:
            ap_i, det_i, gt_i = action_ap(predictions, ac_ta, IoU=IoU, min_T=min_T, max_T=max_T, min_L=min_L,
                                          down_sampling=down_sampling, return_detections=True)
            ap["{}-{}-{}".format(min_T, max_T, min_L)] = ap_i
            det["{}-{}-{}".format(min_T, max_T, min_L)] = det_i
            gt["{}-{}-{}".format(min_T, max_T, min_L)] = gt_i

best_parm = max(ap, key=ap.get)
best_ap = ap[best_parm]
print("{}_{}: get best ap {:.2f} under {}".format(action, mode, best_ap, best_parm))

with open('saved/{}_{}_search'.format(action, mode), 'w') as f:
    json.dump(ap, f)