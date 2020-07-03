#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : train.py.py
# Author: Chongkai LU
# Date  : 6/9/2020
import numpy as np
from custom_class import *
from pathlib import Path
import tensorflow as tf
from load_data import *
from utils import *

# %% Test on a Untrimmed video
action = "GolfSwing"
y_range = (1, 100)
n_mae = normalize_mae(y_range[1] - y_range[0] + 1)

rgb_model_path = "/mnt/louis-consistent/Saved/THUMOS14_output/GolfSwing/Model/2020-07-01-01:18:02/50-24.19.h5"
# rgb_video_path = "/mnt/louis-consistent/Datasets/THUMOS14/Images/test/video_test_0000028"
# rgb_img_list = find_imgs(rgb_video_path)
# rgb_untrimmed_video = build_dataset_from_slices(rgb_img_list, batch_size=1, shuffle=False)

ordinal = True
flow_model_path = "/mnt/louis-consistent/Saved/THUMOS14_output/GolfSwing/50-22.09.h5"
# flow_video_path = "/mnt/louis-consistent/Datasets/THUMOS14/OpticalFlows/test/video_test_0000028"
# flow_img_list = find_flows(flow_video_path)
# flow_untrimmed_video = stack_optical_flow(flow_img_list, batch_size=1, shuffle=False)

rgb_loss, rgb_metric = 'binary_crossentropy', mae_od
flow_loss, flow_metric = 'binary_crossentropy', mae_od
# %% Also test on trimmed train, validation, and test dataset
rgb_root = {'train': "/mnt/louis-consistent/Datasets/THUMOS14/Images/train",
            'val': "/mnt/louis-consistent/Datasets/THUMOS14/Images/validation",
            'test': "/mnt/louis-consistent/Datasets/THUMOS14/Images/test"}

flow_root = {'train': "/mnt/louis-consistent/Datasets/THUMOS14/OpticalFlows/train",
             'val': "/mnt/louis-consistent/Datasets/THUMOS14/OpticalFlows/validation",
             'test': "/mnt/louis-consistent/Datasets/THUMOS14/OpticalFlows/test"}

annfile = {
    'train': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/train/annotationF/{}_trainF.csv".format(
        action),
    'val': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/validation/annotationF/{}_valF.csv".format(
        action),
    'test': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/test/annotationF/{}_testF.csv".format(
        action)}

# t = pd.read_csv(annfile['test'], header=None)
# video_gt = t.loc[t.iloc[:, 0] == Path(rgb_video_path).stem].iloc[:, 1:].values  # temporal annotations of the untrimmed video

# %% Build datasets
rgb_datalist = {x: read_from_annfile(rgb_root[x], annfile[x], (1, 100), ordinal=ordinal) for x in ['train', 'val', 'test']}
rgb_train_dataset = build_dataset_from_slices(*rgb_datalist['train'], batch_size=1, shuffle=False)
rgb_val_dataset = build_dataset_from_slices(*rgb_datalist['val'], batch_size=1, shuffle=False)
rgb_test_dataset = build_dataset_from_slices(*rgb_datalist['test'], batch_size=1, shuffle=False)

flow_datalist = {x: read_from_annfile(flow_root[x], annfile[x], (1, 100), mode='flow', ordinal=ordinal, stack_length=10) for x in ['train', 'val', 'test']}
flow_train_dataset = build_dataset_from_slices(*flow_datalist['train'], batch_size=1, shuffle=False)
flow_val_dataset = build_dataset_from_slices(*flow_datalist['val'], batch_size=1, shuffle=False)
flow_test_dataset = build_dataset_from_slices(*flow_datalist['test'], batch_size=1, shuffle=False)


strategy = tf.distribute.MirroredStrategy()

# rgb_train_records = LossCallback()
# rgb_val_records = LossCallback()
rgb_test_records = RGBLossCallback()

# flow_train_records = LossCallback()
# flow_val_records = LossCallback()
flow_test_records = FLowLossCallback()

# %% Prediction on trimmed videos and single untrimmed video
with strategy.scope():
    rgb_model = tf.keras.models.load_model(rgb_model_path, compile=False, custom_objects={'BiasLayer': MultiAction_BiasLayer})
    rgb_model = tf.keras.Sequential([rgb_model, tf.keras.layers.Reshape((100,))])
    rgb_model.compile(loss=rgb_loss, metrics=[rgb_metric])
    # rgb_train_prediction = rgb_model.predict(rgb_train_dataset, verbose=1)
    # rgb_val_prediction = rgb_model.predict(rgb_val_dataset, verbose=1)
    # rgb_test_prediction = rgb_model.predict(rgb_test_dataset, verbose=1)
    # rgb_train_evaluation = rgb_model.evaluate(rgb_train_dataset, verbose=1, callbacks=[rgb_train_records])
    # rgb_val_evaluation = rgb_model.evaluate(rgb_val_dataset, verbose=1, callbacks=[rgb_val_records])
    # rgb_test_evaluation = rgb_model.evaluate(rgb_test_dataset, verbose=1, callbacks=[rgb_test_records])
    # rgb_video_prediction = rgb_model.predict(rgb_untrimmed_video, verbose=1)

    flow_model = tf.keras.models.load_model(flow_model_path, compile=False, custom_objects={'BiasLayer': MultiAction_BiasLayer})
    flow_model = tf.keras.Sequential([flow_model, tf.keras.layers.Reshape((100,))])
    flow_model.compile(loss=flow_loss, metrics=[flow_metric])
    # flow_train_prediction = flow_model.predict(flow_train_dataset, verbose=1)
    # flow_val_prediction = flow_model.predict(flow_val_dataset, verbose=1)
    # flow_test_prediction = flow_model.predict(flow_test_dataset, verbose=1)
    # flow_train_evaluation = flow_model.evaluate(flow_train_dataset, verbose=1, callbacks=[flow_train_records])
    # flow_val_evaluation = flow_model.evaluate(flow_val_dataset, verbose=1, callbacks=[flow_val_records])
    # flow_test_evaluation = flow_model.evaluate(flow_test_dataset, verbose=1, callbacks=[flow_test_records])
    # flow_video_prediction = flow_model.predict(flow_untrimmed_video, verbose=1)

# boxplot
# # get_boxplot(datalist['test'][1], test_records['n_mae'])

# %% Predict on untrimmed videos
import pandas as pd
temporal_annotation = pd.read_csv(annfile['test'], header=None)
video_names = temporal_annotation.iloc[:, 0].unique()
rgb_untrimmed_predictions = {}
flow_untrimmed_predictions = {}
fused_untrimmed_predictions = {}
ground_truth = {}
for v in video_names:
    gt = temporal_annotation.loc[temporal_annotation.iloc[:, 0] == v].iloc[:, 1:].values
    ground_truth[v] = gt

    rgb_video_path = Path(rgb_root['test'], v)
    img_list = find_imgs(rgb_video_path)
    flow_video_path = Path(flow_root['test'], v)
    flow_list = find_flows(flow_video_path)

    rgb_ds = build_dataset_from_slices(img_list, batch_size=1, shuffle=False)
    rgb_untrimmed_prediction = rgb_model.predict(rgb_ds, verbose=1)
    rgb_untrimmed_predictions[v] = np.squeeze(rgb_untrimmed_prediction)

    flow_ds = build_dataset_from_slices(flow_list, batch_size=1, shuffle=False)
    flow_untrimmed_prediction = flow_model.predict(flow_ds, verbose=1)
    flow_untrimmed_predictions[v] = np.squeeze(flow_untrimmed_prediction)

    num_predict = flow_untrimmed_prediction.shape[0]
    fused_untrimmed_predictions[v] = (flow_untrimmed_prediction + rgb_untrimmed_prediction[:num_predict]) / 2

# %% Detect actions
import numpy as np
num_gt = sum([len(gt) for gt in ground_truth.values()])

iou = 0.5

rgb_action_detected = {}
rgb_tps = {}
for v, prediction in rgb_untrimmed_predictions.items():
    if ordinal:
        prediction = ordinal2completeness(prediction)
    ads = action_search(prediction, min_T=50, max_T=30, min_L=40)
    rgb_action_detected[v] = ads
    rgb_tps[v] = calc_truepositive(ads, ground_truth[v], iou)

rgb_loss = np.vstack(list(rgb_action_detected.values()))[:, 2]
rgb_tp_values = np.hstack(list(rgb_tps.values()))
rgb_ap = average_precision(rgb_tp_values, num_gt, rgb_loss)
plot_prediction(prediction)
plot_detection(prediction, gt, ads)

flow_action_detected = {}
flow_tps = {}
for v, prediction in flow_untrimmed_predictions.items():
    if ordinal:
        prediction = ordinal2completeness(prediction)
    ads = action_search(prediction, min_T=80, max_T=30, min_L=60)
    flow_action_detected[v] = ads
    flow_tps[v] = calc_truepositive(ads, ground_truth[v], iou)

flow_loss = np.vstack(list(flow_action_detected.values()))[:, 2]
flow_tp_values = np.hstack(list(flow_tps.values()))
flow_ap = average_precision(flow_tp_values, num_gt, flow_loss)
plot_prediction(prediction)
plot_detection(prediction, gt, ads)
# Fused by take mean of prediction rgb and flow
fused_action_detected = {}
fused_tps = {}
for v, prediction in fused_untrimmed_predictions.items():
    if ordinal:
        prediction = ordinal2completeness(prediction)
    ads = action_search(prediction, min_T=60, max_T=30, min_L=40)
    fused_action_detected[v] = ads
    fused_tps[v] = calc_truepositive(ads, ground_truth[v], iou)

fused_loss = np.vstack(list(fused_action_detected.values()))[:, 2]
fused_tp_values = np.hstack(list(fused_tps.values()))
fused_ap = average_precision(fused_tp_values, num_gt, fused_loss)

# plot_prediction(prediction)
# plot_detection(prediction, gt, ads)
# Fused by deleted detections not intersect with each other
fused2_action_detected = {}
fused2_tps = {}
for v, rgb_ad in rgb_action_detected.items():
    op_ad = flow_action_detected[v]
    iou_matrix = matrix_iou(rgb_ad[:, :2], op_ad[:, :2])
    ads = []
    while iou_matrix.max() > 0.5:
        max_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        ads.append(rgb_ad[max_idx[0]] if rgb_ad[max_idx[0]][2] < op_ad[max_idx[1]][2] else op_ad[max_idx[1]])
        iou_matrix[max_idx[0], :],  iou_matrix[:, max_idx[1]] = 0, 0
    ads = np.vstack(ads)
    fused2_action_detected[v] = ads
    fused2_tps[v] = calc_truepositive(ads, ground_truth[v], iou)

fused2_loss = np.vstack(list(fused2_action_detected.values()))[:, 2]
fused2_tp_values = np.hstack(list(fused2_tps.values()))
fused2_ap = average_precision(fused2_tp_values, num_gt, fused2_loss)
# %% Single untrimmed video detection
# rgb_ads = action_search(rgb_video_prediction, min_T=50, max_T=30, min_L=35)
# rgb_ads = np.array(rgb_ads)
# plot_prediction(rgb_video_prediction)
# plot_detection(rgb_test_prediction, gt, rgb_ads)
#
# flow_ads = action_search(flow_video_prediction, min_T=50, max_T=30, min_L=35)
# flow_ads = np.array(flow_ads)
# plot_prediction(flow_video_prediction)
# plot_detection(flow_test_prediction, gt, flow_ads)
#
# fused_prediction = (rgb_video_prediction + flow_test_prediction)/2
# fused_ads = action_search(fused_prediction, min_T=50, max_T=30, min_L=35)
# fused_ads = np.array(flow_ads)
# plot_prediction(fused_prediction)
# plot_detection(fused_prediction, gt, flow_ads)