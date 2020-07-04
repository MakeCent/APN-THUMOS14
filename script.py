#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : train.py
# Author: Chongkai LU
# Date  : 3/29/2020
import pandas as pd
from pathlib import Path
import numpy as np
video_names = pd.read_csv("/mnt/louis-consistent/Datasets/THUMOS14/Information/test_videos.txt", header=None).values.squeeze().tolist()
lens = 0
action_idx = {'BaseballPitch': 0, 'BasketballDunk': 1, 'Billiards': 2, 'CleanAndJerk': 3, 'CliffDiving': 4,
              'CricketBowling': 5, 'CricketShot': 6, 'Diving': 7, 'FrisbeeCatch': 8, 'GolfSwing': 9,
              'HammerThrow': 10, 'HighJump': 11, 'JavelinThrow': 12, 'LongJump': 13, 'PoleVault': 14, 'Shotput': 15,
              'SoccerPenalty': 16, 'TennisSwing': 17, 'ThrowDiscus': 18, 'VolleyballSpiking': 19}
for v in video_names:
    gt = {}
    with open("{}_annoationF".format(v), 'a') as f:
        for annfile in sorted(Path("/mnt/louis-consistent/Datasets/THUMOS14/Annotations/test/annotationF").iterdir()):
            action_name = annfile.stem.split("_")[0]
            v_gt = pd.read_csv(str(annfile), header=None)
            v_gt = v_gt.loc[v_gt.iloc[:, 0] == v].iloc[:, 1:].values
            if v_gt.size != 0:
                gt[action_name] = v_gt.tolist()
                lens += len(gt[action_name])
                for g in gt[action_name]:
                    f.write("{},{},{}\n".format(action_idx[action_name], g[0], g[1]))

ll = 0
for k in sorted(Path("/mnt/louis-consistent/Datasets/THUMOS14/Annotations/test/annotationF").iterdir()):
    ll += pd.read_csv(str(k), header=None).shape[0]