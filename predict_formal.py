#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : train.py
# Author: Chongkai LU
# Date  : 3/29/2020
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from load_data import *
from utils import *
from custom_class import BiasLayer
action = "GolfSwing"
model_path = "/mnt/louis-consistent/Saved/THUMOS14_output/GolfSwing/Model/2020-07-01-01:18:02/50-24.19.h5"
ordinal = True
annfile = "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/test/annotationF/{}_testF.csv".format(action)
temporal_annotation = pd.read_csv(annfile, header=None)
video_names = temporal_annotation.iloc[:, 0].unique()
test_dir = "/mnt/louis-consistent/Datasets/THUMOS14/Images/test"
n_mae = normalize_mae(100)  # make mae loss normalized into range 0 - 100.
# %% Model Predict
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = tf.keras.models.load_model(model_path, compile=False, custom_objects={'BiasLayer': BiasLayer})
    if ordinal:
        model.compile(loss='binary_crossentropy', metrics=[mae_od])
    else:
        model.compile(loss='mse', metrics=[n_mae])

predictions = {}
ground_truth = {}
for v in video_names:
    gt = temporal_annotation.loc[temporal_annotation.iloc[:, 0] == v].iloc[:, 1:].values
    ground_truth[v] = gt

    video_path = Path(test_dir, v)
    img_list = find_imgs(video_path)
    ds = build_dataset_from_slices(img_list, batch_size=1, shuffle=False)

    prediction = model.predict(ds, verbose=1)
    predictions[v] = np.squeeze(prediction)

# %% Detect actions
action_detected = {}
tps = {}
for v, prediction in predictions.items():
    if ordinal:
        prediction = ordinal2completeness(prediction)
    else:
        pass
    ads = action_search(prediction, min_T=80, max_T=25, min_L=80)
    action_detected[v] = ads
    tps[v] = calc_truepositive(ads, ground_truth[v], 0.5)

num_gt = sum([len(gt) for gt in ground_truth.values()])
loss = np.vstack(list(action_detected.values()))[:, 2]
tp_values = np.hstack(list(tps.values()))
ap = average_precision(tp_values, num_gt, loss)

plot_prediction(prediction)
plot_detection(prediction, gt, ads)
