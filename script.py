#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : Train.py.py
# Author: Chongkai LU
# Date  : 3/29/2020

# decay = 1e-3 / train_steps
import pandas as pd
from pathlib import Path
annotation_path = "/mnt/louis-consistent/Datasets/THUMOS14/TH14_Temporal_annotations_test/annotationF"
annotation_path = Path(annotation_path)
d = {}
for gtp in annotation_path.iterdir():
    action_name = gtp.stem
    gt = pd.read_csv(gtp, header=None)
    ls = gt.iloc[:, 2]-gt.iloc[:,1]
    al = ls.sum()
    d[action_name]=al
pdd = pd.DataFrame({'action'})