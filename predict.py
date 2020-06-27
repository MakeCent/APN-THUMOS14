#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : train.py.py
# Author: Chongkai LU
# Date  : 6/9/2020
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from load_data import *
from utilities import *
from action_detection import action_search

# %% Test on a Untrimmed video
action = "GolfSwing"
rgb_model_path = "/mnt/louis-consistent/Saved/THUMOS14_output/GolfSwing/Model/30-15.67.h5"
rgb_video_path = "/mnt/louis-consistent/Datasets/THUMOS14/Images/Test/video_test_0000028"
rgb_img_list = find_imgs(rgb_video_path)
rgb_untrimmed_video = build_dataset_from_slices(rgb_img_list, batch_size=1, shuffle=False)

flow_model_path = "/mnt/louis-consistent/Saved/THUMOS14_output/GolfSwing/Model/30-15.67.h5"
flow_video_path = "/mnt/louis-consistent/Datasets/THUMOS14/OpticalFlows/test/video_test_0000028"
flow_img_list = find_imgs(flow_video_path)
flow_untrimmed_video = stack_optical_flow(flow_img_list, batch_size=1, shuffle=False)

# %% Also test on trimmed train, validation, and test dataset
rgb_root = {'train': "/mnt/louis-consistent/Datasets/THUMOS14/Images/train",
            'val': "/mnt/louis-consistent/Datasets/THUMOS14/Images/Validation",
            'test': "/mnt/louis-consistent/Datasets/THUMOS14/Images/Test"}

flow_root = {'train': "/mnt/louis-consistent/Datasets/THUMOS14/Images/train",
             'val': "/mnt/louis-consistent/Datasets/THUMOS14/Images/validation",
             'test': "/mnt/louis-consistent/Datasets/THUMOS14/Images/test"}

annfile = {
    'train': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/train/annotationF/{}_trainF.csv".format(
        action),
    'val': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/validation/annotationF/{}_valF.csv".format(
        action),
    'test': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/test/annotationF/{}_testF.csv".format(
        action)}

t = pd.read_csv(annfile['test'], header=None)
gt = t.loc[t.iloc[:, 0] == Path(rgb_video_path).stem].iloc[:, 1:].values  # temporal annotations of the untrimmed video


rgb_datalist = {x: read_from_annfile(rgb_root[x], annfile[x], (0, 100)) for x in ['train', 'val', 'test']}
rgb_train_dataset = build_dataset_from_slices(*rgb_datalist['train'], batch_size=1, shuffle=False)
rgb_val_dataset = build_dataset_from_slices(*rgb_datalist['val'], batch_size=1, shuffle=False)
rgb_test_dataset = build_dataset_from_slices(*rgb_datalist['test'], batch_size=1, shuffle=False)

flow_datalist = {x: read_from_annfile(flow_root[x], annfile[x], (0, 100), mode='flow') for x in ['train', 'val', 'test']}
flow_train_dataset = stack_optical_flow(*flow_datalist['train'], batch_size=1, shuffle=False)
flow_val_dataset = stack_optical_flow(*flow_datalist['val'], batch_size=1, shuffle=False)
flow_test_dataset = stack_optical_flow(*flow_datalist['test'], batch_size=1, shuffle=False)


strategy = tf.distribute.MirroredStrategy()
n_mae = normalize_mae(100)  # make mae loss normalized into range 0 - 100.

class LossCallback(tf.keras.callbacks.Callback):

    def on_test_begin(self, logs=None):
        self.n_mae = []
        self.loss = []

    def on_test_batch_end(self, batch, logs=None):
        self.loss.append(logs['loss'])
        self.n_mae.append(logs['n_mae'])

rgb_train_records = LossCallback()
rgb_val_records = LossCallback()
rgb_test_records = LossCallback()

flow_train_records = LossCallback()
flow_val_records = LossCallback()
flow_test_records = LossCallback()

with strategy.scope():
    rgb_model = tf.keras.models.load_model(rgb_model_path, compile=False)
    rgb_model.compile(loss='mse', metrics=[n_mae])
    rgb_train_prediction = rgb_model.predict(rgb_train_dataset, verbose=1)
    rgb_val_prediction = rgb_model.predict(rgb_val_dataset, verbose=1)
    rgb_test_prediction = rgb_model.predict(rgb_test_dataset, verbose=1)
    rgb_train_evaluation = rgb_model.evaluate(rgb_train_dataset, verbose=1, callbacks=[rgb_train_records])
    rgb_val_evaluation = rgb_model.evaluate(rgb_val_dataset, verbose=1, callbacks=[rgb_val_records])
    rgb_test_evaluation = rgb_model.evaluate(rgb_test_dataset, verbose=1, callbacks=[rgb_test_records])
    rgb_video_prediction = rgb_model.predict(rgb_untrimmed_video, verbose=1)

    flow_model = tf.keras.models.load_model(flow_model_path, compile=False)
    flow_model.compile(loss='mse', metrics=[n_mae])
    flow_train_prediction = flow_model.predict(flow_train_dataset, verbose=1)
    flow_val_prediction = flow_model.predict(flow_val_dataset, verbose=1)
    flow_test_prediction = flow_model.predict(flow_test_dataset, verbose=1)
    flow_train_evaluation = flow_model.evaluate(flow_train_dataset, verbose=1, callbacks=[flow_train_records])
    flow_val_evaluation = flow_model.evaluate(flow_val_dataset, verbose=1, callbacks=[flow_val_records])
    flow_test_evaluation = flow_model.evaluate(flow_test_dataset, verbose=1, callbacks=[flow_test_records])
    flow_video_prediction = flow_model.predict(flow_untrimmed_video, verbose=1)

# boxplot
# # get_boxplot(datalist['test'][1], test_records['n_mae'])

# %% Detect actions
rgb_ads = action_search(rgb_video_prediction, min_T=75, max_T=20, min_L=35)
rgb_ads = np.array(rgb_ads)
plot_prediction(rgb_video_prediction)
plot_detection(rgb_test_prediction, gt, rgb_ads)

flow_ads = action_search(flow_video_prediction, min_T=75, max_T=20, min_L=35)
flow_ads = np.array(flow_ads)
plot_prediction(flow_video_prediction)
plot_detection(flow_test_prediction, gt, flow_ads)

fused_prediction = (rgb_video_prediction + flow_test_prediction)/2
fused_ads = action_search(fused_prediction, min_T=75, max_T=20, min_L=35)
fused_ads = np.array(flow_ads)
plot_prediction(fused_prediction)
plot_detection(fused_prediction, gt, flow_ads)