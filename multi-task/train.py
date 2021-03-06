#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : train.py
# Author: Chongkai LU
# Date  : 3/29/2020

from load_data import *
from utils import *
from custom_class import MultiAction_BiasLayer
from pathlib import Path
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.layers import Dense, Dropout, Activation, Reshape
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import wandb
from wandb.keras import WandbCallback
import datetime
import tensorflow as tf
import socket
agent = socket.gethostname()
AUTOTUNE = tf.data.experimental.AUTOTUNE
fix_bug()
now = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
# %% wandb Initialization
# Configurations. If you don't use wandb, just manually set these values.
default_config = dict(
    y_s=1,
    y_e=100,
    learning_rate=0.0001,
    batch_size=32,
    epochs=30,
    agent=agent
)
ordinal = True
mode = 'rgb'
stack_length = 1
weighted = False

cross_pre = True if mode == 'flow' else False
partialBN = True if mode == 'flow' else False
# Just for wandb
tags = ['all', mode]
if ordinal:
    tags.append("od")
if weighted:
    tags.append("weighted")
if stack_length > 1:
    tags.append("stack{}".format(stack_length))
if cross_pre:
    tags.append("cross-pre")
if partialBN:
    tags.append("partialBN")
wandb.init(config=default_config, name=now, tags=tags, notes='baseline, all, od, rgb1')
config = wandb.config
wandbcb = WandbCallback(monitor='val_n_mae', save_model=False)

y_range = (config.y_s, config.y_e)
y_nums = y_range[1] - y_range[0] + 1
learning_rate = config.learning_rate
batch_size = config.batch_size
epochs = config.epochs
action_num = 20

# %% Parameters, Configuration, and Initialization
model_name = now
if mode == 'rgb':
    root = {'train': "/mnt/louis-consistent/Datasets/THUMOS14/Images/train",
            'val': "/mnt/louis-consistent/Datasets/THUMOS14/Images/validation",
            'test': "/mnt/louis-consistent/Datasets/THUMOS14/Images/test"}
else:
    root = {'train': "/mnt/louis-consistent/Datasets/THUMOS14/OpticalFlows/train",
            'val': "/mnt/louis-consistent/Datasets/THUMOS14/OpticalFlows/validation",
            'test': "/mnt/louis-consistent/Datasets/THUMOS14/OpticalFlows/test"}
anndir = {
    'train': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/train/annotationF",
    'val': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/validation/annotationF",
    'test': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/test/annotationF"}

output_path = '/mnt/louis-consistent/Saved/THUMOS14_output/Multi-task'  # Directory to save model and history
history_path = Path(output_path, 'History', model_name)
models_path = Path(output_path, 'Model', model_name)
history_path.mkdir(parents=True, exist_ok=True)
models_path.mkdir(parents=True, exist_ok=True)

# %% Build dataset


# def augment_func(x, y=None, w=None):
#     import tensorflow as tf
#     x = tf.image.random_flip_left_right(x)
#     return x, y, w


datalist = {x: read_from_anndir(root[x], anndir[x], mode=mode, y_range=y_range, ordinal=ordinal, stack_length=stack_length) for x in ['train', 'val', 'test']}
test_dataset = build_dataset_from_slices(*datalist['test'], batch_size=batch_size, shuffle=False)
train_val_datalist = (datalist['train'][0]+datalist['val'][0], datalist['train'][1]+datalist['val'][1])
train_val_dataset = build_dataset_from_slices(*train_val_datalist, batch_size=batch_size)
# %% Build and compile model
lr_sche = LearningRateScheduler(lr_schedule)
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    inputs = tf.keras.Input(shape=(224, 224, 3))
    backbone = ResNet101(weights='imagenet', input_shape=(224, 224, 3), pooling='avg', include_top=False)
    x = backbone(inputs)
    x = Dense(1024, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.5)(x)
    x = Dense(action_num, kernel_initializer='he_uniform', use_bias=False)(x)
    x = MultiAction_BiasLayer(y_nums)(x)
    output = Activation('sigmoid')(x)
    model = Model(inputs, output)
    model_checkpoint = ModelCheckpoint(str(models_path.joinpath('{epoch:02d}-{val_multi_od_metric:.2f}.h5')), period=5)
    lr_sche = LearningRateScheduler(lr_schedule)
    # %% Fine tune
    backbone.trainable = True
    model.compile(loss=multi_binarycrossentropy, optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=[multi_od_metric])
    ftune_his = model.fit(train_val_dataset, validation_data=test_dataset, epochs=epochs,
                          callbacks=[model_checkpoint, wandbcb, lr_sche], verbose=1)

# %% Save history to csv and images
history = ftune_his.history
save_history(history_path, history)
plot_history(history_path, history)

# %% Prediction on Untrimmed Videos
import pandas as pd
import numpy as np

video_names = pd.read_csv("/mnt/louis-consistent/Datasets/THUMOS14/Information/test_videos.txt", header=None).values.squeeze().tolist()
untrimmed_predictions = {}
ground_truth = {}
for v in video_names:
    gt = pd.read_csv(
        "/mnt/louis-consistent/Datasets/THUMOS14/Information/video_wise_annotationF/test/{}_annotationF".format(v),
        header=None).values
    ground_truth[v] = gt

    video_path = Path(root['test'], v)
    img_list = find_imgs(video_path)

    ds = build_dataset_from_slices(img_list, batch_size=1, shuffle=False)
    untrimmed_prediction = model.predict(ds, verbose=1)
    untrimmed_predictions[v] = np.squeeze(untrimmed_prediction)

# %% Detect action and calculate mAP
action_idx = {'BaseballPitch': 0, 'BasketballDunk': 1, 'Billiards': 2, 'CleanAndJerk': 3, 'CliffDiving': 4,
              'CricketBowling': 5, 'CricketShot': 6, 'Diving': 7, 'FrisbeeCatch': 8, 'GolfSwing': 9,
              'HammerThrow': 10, 'HighJump': 11, 'JavelinThrow': 12, 'LongJump': 13, 'PoleVault': 14, 'Shotput': 15,
              'SoccerPenalty': 16, 'TennisSwing': 17, 'ThrowDiscus': 18, 'VolleyballSpiking': 19}

iou = 0.5
all_detected_action = {}
KILL = True
for v, p in untrimmed_predictions.items():
    if ordinal:
        p = ordinal2completeness(p)
    v_ads = []
    for ac_name, ac_idx in action_idx.items():
        ac_prediction = p[:, ac_idx]
        ac_ads = action_search(ac_prediction, min_T=50, max_T=30, min_L=40)
        ac_ads = np.c_[ac_ads, np.ones(ac_ads.shape[0]) * ac_idx]
        v_ads.append(ac_ads)
    v_ads = np.vstack(v_ads)
    if KILL:
        iou_matrix = matrix_iou(v_ads[:, :2], v_ads[:, :2])
        while iou_matrix.max() > 0:
            max_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            iou_matrix = np.delete(iou_matrix, max_idx[0], axis=0) if v_ads[max_idx[0]][2] > v_ads[max_idx[1]][
                2] else np.delete(iou_matrix, max_idx[1], axis=1)
            v_ads = np.delete(v_ads, max_idx[0], axis=0) if v_ads[max_idx[0]][2] > v_ads[max_idx[1]][2] else np.delete(
                v_ads, max_idx[1], axis=0)
        all_detected_action[v] = v_ads
    else:
        pass
    all_detected_action[v] = v_ads

all_detection = np.array(list(all_detected_action.values()))

all_tps = {}
ap = {}
for ac_gt in Path(anndir['test']).iterdir():
    ac_name = ac_gt.stem.split('_')[0]
    ac_idx = action_idx[ac_name]
    ac_ta = pd.read_csv(str(ac_gt), header=None).values
    ac_num_gt = ac_ta.shape[0]
    ac_v = ac_ta[:, 0].squeeze().tolist()
    ac_tps = {}
    ac_detected = {}
    for v in ac_v:
        ac_v_gt = ac_ta[ac_ta[:, 0] == v][:, 1:3]
        v_detected = all_detected_action[v]
        ac_v_detected = v_detected[v_detected[:, 3] == ac_idx]
        ac_v_tps = calc_truepositive(ac_v_detected, ac_v_gt, iou)
        ac_tps[v] = ac_v_tps
    ac_loss = np.array(list(ac_detected.values()))[:, 2]
    ac_tp_values = np.array(list(ac_tps.values())).flatten()
    ac_ap = average_precision(ac_tp_values, ac_num_gt, ac_loss)
    ap[ac_name] = ac_ap


