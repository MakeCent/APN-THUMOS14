#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : Train.py.py
# Author: Chongkai LU
# Date  : 3/29/2020
from load_data import *
from utilities import *
import numpy as np
import pandas as pd
from pathlib import Path
from load_data import *
from utilities import *
from action_detection import action_search
action = "GolfSwing"
model_path = "/mnt/louis-consistent/Saved/THUMOS14_output/GolfSwing/Model/30-15.67.h5"
annfile = "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/test/annotationF/{}_testF.csv".format(action)
temporal_annotation = pd.read_csv(annfile, header=None)
video_names = temporal_annotation.iloc[:, 0].unique()
test_dir = "/mnt/louis-consistent/Datasets/THUMOS14/Images/Test"
action_detected = []
ground_truth = []
tps = []
for v in video_names:
    gt = temporal_annotation.loc[temporal_annotation.iloc[:, 0] == v].iloc[:, 1:].values
    ground_truth.append(gt)

    video_path = Path(test_dir, v)
    img_list = find_imgs(video_path)
    ds = build_dataset_from_slices(img_list, batch_size=1, shuffle=False)
    strategy = tf.distribute.MirroredStrategy()
    n_mae = normalize_mae(100)  # make mae loss normalized into range 0 - 100.
    with strategy.scope():
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(loss='mse', metrics=[n_mae])
        prediction = model.predict(ds, verbose=1)

    # %% Detect actions
    ads = action_search(prediction, min_T=75, max_T=20, min_L=35)
    ads = np.array(ads)
    action_detected.append(ads)
    tps.append(calc_truepositive(ads, gt, 0.5))

num_gt = np.vstack(ground_truth).shape[0]
loss = np.hstack(action_detected)[:, 2]
ap = average_precision(np.vstack(tps), num_gt, loss)
