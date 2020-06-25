#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : Train.py.py
# Author: Chongkai LU
# Date  : 6/9/2020
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from load_data import *
from utilities import *
from action_detection import action_search
import matplotlib.pyplot as plt
action = "GolfSwing"
model_path = "/mnt/louis-consistent/Saved/THUMOS14_output/GolfSwing/Model/30-15.67.h5"
video_path = "/mnt/louis-consistent/Datasets/THUMOS14/Images/Test/video_test_0000028"
val_video_path = "/mnt/louis-consistent/Datasets/THUMOS14/Images/Validation/video_validation_0000281"
img_list = find_imgs(video_path)
val_img_list = find_imgs(val_video_path)
untrimmed_video = build_dataset_from_slices(img_list, batch_size=1, shuffle=False)
val_untrimmed_video = build_dataset_from_slices(val_img_list, batch_size=1, shuffle=False)

## Also test on train and validation dataset and trimmed test dataset
root = {'train': "/mnt/louis-consistent/Datasets/THUMOS14/Images/Train",
        'val': "/mnt/louis-consistent/Datasets/THUMOS14/Images/Validation",
        'test': "/mnt/louis-consistent/Datasets/THUMOS14/Images/Test"}
annfile = {
    'train': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/train/annotationF/{}_trainF.csv".format(
        action),
    'val': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/validation/annotationF/{}_valF.csv".format(
        action),
    'test': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/test/annotationF/{}_testF.csv".format(
        action)}

t = pd.read_csv(annfile['test'], header=None)
val_t = pd.read_csv(annfile['val'], header=None)
gt = t.loc[t.iloc[:, 0] == Path(video_path).stem].iloc[:, 1:].values  # temporal annotations of the untrimmed video
val_gt = val_t.loc[val_t.iloc[:, 0] == Path(val_video_path).stem].iloc[:, 1:].values


datalist = {x: read_from_annfile(root[x], annfile[x], (0,100)) for x in ['train', 'val', 'test']}
train_dataset = build_dataset_from_slices(*datalist['train'], batch_size=1, shuffle=False)
val_dataset = build_dataset_from_slices(*datalist['val'], batch_size=1, shuffle=False)
test_dataset = build_dataset_from_slices(*datalist['test'], batch_size=1, shuffle=False)


strategy = tf.distribute.MirroredStrategy()
n_mae = normalize_mae(100)  # make mae loss normalized into range 0 - 100.


class LossCallback(tf.keras.callbacks.Callback):

    def on_test_begin(self, logs=None):
        self.n_mae = []
        self.loss = []

    def on_test_batch_end(self, batch, logs=None):
        self.loss.append(logs['loss'])
        self.n_mae.append(logs['n_mae'])

train_records = LossCallback()
val_records = LossCallback()
test_records = LossCallback()
with strategy.scope():
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(loss='mse', metrics=[n_mae])
    train_prediction = model.predict(train_dataset, verbose=1)
    val_prediction = model.predict(val_dataset, verbose=1)
    test_prediction = model.predict(test_dataset, verbose=1)
    train_evaluation = model.evaluate(train_dataset, verbose=1, callbacks=[train_records])
    val_evaluation = model.evaluate(val_dataset, verbose=1, callbacks=[val_records])
    test_evaluation = model.evaluate(test_dataset, verbose=1, callbacks=[test_records])

    video_prediction = model.predict(untrimmed_video, verbose=1)
    val_video_prediction = model.predict(val_untrimmed_video, verbose=1)

# boxplot
# # get_boxplot(datalist['test'][1], test_records['n_mae'])

# %% Detect actions
ads = action_search(video_prediction, min_T=75, max_T=20, min_L=35)
val_ads = action_search(val_video_prediction, min_T=75, max_T=20, min_L=35)
ads = np.array(ads)
val_ads = np.array(val_ads)

plot_prediction(val_video_prediction)
plot_prediction(video_prediction)
plot_detection(val_video_prediction, val_gt, val_ads)
plot_detection(test_prediction, gt, ads)
