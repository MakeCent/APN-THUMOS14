#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : i3d.py
# Author: Chongkai LU
# Date  : 24/7/2020
import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import json
import socket
import wandb
from pathlib import Path
from tools.Flated_Inception import Inception_Inflated3d
from tensorflow.keras.layers import Dense, Dropout, Activation, Reshape
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from wandb.keras import WandbCallback
from tools.load_data import *
from tools.utils import *
from tools.custom_class import BiasLayer
agent = socket.gethostname()
AUTOTUNE = tf.data.experimental.AUTOTUNE
fix_bug()
now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# %% wandb Initialization
# Configurations. If you don't use wandb, just manually set these values.
default_config = dict(
    y_s=1,
    y_e=100,
    learning_rate=0.0001,
    batch_size=32,
    epochs=50,
    agent=agent,
    action='BaseballPitch',
    mode='rgb'
)
ordinal = True
mode = default_config['mode']
stack_length = 10
weighted = False
pretrain = True
notes = 'i3d_{}_{}'.format(default_config['action'], mode)

# Just for wandb
tags = [default_config['action'], mode, 'i3d']
if ordinal:
    tags.append("od")
if weighted:
    tags.append("weighted")
if stack_length > 1:
    tags.append("stack{}".format(stack_length))
wandb.init(config=default_config, name=now, tags=tags, notes=notes)
config = wandb.config
wandbcb = WandbCallback(monitor='val_mae_od', save_model=False)

y_range = (config.y_s, config.y_e)
y_nums = y_range[1] - y_range[0] + 1
learning_rate = config.learning_rate
batch_size = config.batch_size
epochs = config.epochs
action_num = 1
action = config.action

# %% Parameters, Configuration, and Initialization
model_name = now
if mode == 'rgb':
    root = {'train': "/mnt/louis-consistent/Datasets/THUMOS14/Images/train",
            'val': "/mnt/louis-consistent/Datasets/THUMOS14/Images/validation",
            'test': "/mnt/louis-consistent/Datasets/THUMOS14/Images/test"}
elif mode == 'flow':
    root = {'train': "/mnt/louis-consistent/Datasets/THUMOS14/OpticalFlows/train",
            'val': "/mnt/louis-consistent/Datasets/THUMOS14/OpticalFlows/validation",
            'test': "/mnt/louis-consistent/Datasets/THUMOS14/OpticalFlows/test"}
elif mode == 'w_flow':
    root = {'train': "/mnt/louis-consistent/Datasets/THUMOS14/Warped_OpticalFlows/train",
            'val': "/mnt/louis-consistent/Datasets/THUMOS14/Warped_OpticalFlows/validation",
            'test': "/mnt/louis-consistent/Datasets/THUMOS14/Warped_OpticalFlows/test"}
annfile = {
    'train': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/train/annotationF/{}_trainF.csv".format(
        action),
    'val': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/validation/annotationF/{}_valF.csv".format(
        action),
    'test': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/test/annotationF/{}_testF.csv".format(
        action)}

output_path = '/mnt/louis-consistent/Saved/THUMOS14_output/{}'.format(action)  # Directory to save model and history
history_path = Path(output_path, 'History', model_name)
models_path = Path(output_path, 'Model', model_name)
history_path.mkdir(parents=True, exist_ok=True)
models_path.mkdir(parents=True, exist_ok=True)

# %% Build dataset
parse_function = thumo14_parse_builder(i3d=True, mode=mode)
datalist = {x: read_from_annfile(root[x], annfile[x], mode=mode, y_range=y_range, ordinal=ordinal, stack_length=stack_length) for x in ['train', 'val', 'test']}
train_val_datalist = [a+b for a, b in zip(datalist['train'], datalist['val'])]
train_dataset = build_dataset_from_slices(*train_val_datalist, batch_size=batch_size, parse_func=parse_function)
test_dataset = build_dataset_from_slices(*datalist['test'], batch_size=batch_size, shuffle=False, parse_func=parse_function)
del train_val_datalist, datalist
model_checkpoint = ModelCheckpoint(str(models_path.joinpath('{epoch:02d}-{val_mae_od:.2f}.h5')), period=1)
with tf.distribute.MirroredStrategy().scope():
    backbone = Inception_Inflated3d(
        include_top=False,
        weights='{}_imagenet_and_kinetics'.format(mode if mode == 'rgb' else 'flow') if pretrain is not None else None,
        input_shape=(stack_length, 224, 224, 3 if mode == 'rgb' else 2))
    x = tf.keras.layers.Reshape((1024,))(backbone.output)
    x = Dense(2048, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.5)(x)
    x = Dense(action_num, kernel_initializer='he_uniform', use_bias=False)(x)
    x = BiasLayer(y_nums)(x)
    output = Activation('sigmoid')(x)
    model = Model(backbone.input, output)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=[mae_od])

ftune_his = model.fit(train_dataset, validation_data=test_dataset, callbacks=[model_checkpoint, wandbcb], epochs=epochs, verbose=1)

# %% Save history to csv and images
history = ftune_his.history
save_history(history_path, history)
plot_history(history_path, history)
tf.keras.backend.clear_session()
# %% Prediction
model = tf.keras.models.load_model(str(sorted(models_path.iterdir())[-1]), compile=False, custom_objects={'BiasLayer': BiasLayer})
model.compile(loss='binary_crossentropy', metrics=[mae_od])
temporal_annotation = pd.read_csv(annfile['test'], header=None)
video_names = temporal_annotation.iloc[:, 0].unique()
untrimmed_predictions = {}
ground_truth = {}
for v in video_names:
    gt = temporal_annotation.loc[temporal_annotation.iloc[:, 0] == v].iloc[:, 1:].values
    ground_truth[v] = gt

    video_path = Path(root['test'], v)
    if mode == 'rgb':
        data_list = find_imgs(video_path, stack_length=stack_length)
    else:
        data_list = find_flows(video_path, stack_length=stack_length)

    ds = build_dataset_from_slices(data_list, batch_size=1, shuffle=False, parse_func=parse_function)
    untrimmed_prediction = model.predict(ds, verbose=1)
    if ordinal == True:
        untrimmed_predictions[v] = ordinal2completeness(np.squeeze(untrimmed_prediction))
    else:
        untrimmed_predictions[v] = np.squeeze(untrimmed_prediction)

with open('saved/{}_{}_prediction'.format(action, mode), 'w') as f:
    list_pre = {str(k): v.tolist() for k, v in untrimmed_predictions.items()}
    json.dump(list_pre, f)

# %% Action search
np.set_printoptions(suppress=True)

IoU = 0.5
# min_T, max_T, min_L = 60, 30, 60
down_sampling = 1

ap = {}
det = {}
gt = {}

ac_ta = pd.read_csv(
    "{}/{}_testF.csv".format("/mnt/louis-consistent/Datasets/THUMOS14/Annotations/test/annotationF", action),
    header=None).values
for min_T in [60, 70, 80]:
    for max_T in [20, 30, 40]:
        for min_L in [30, 60, 80]:
            ap_i, det_i, gt_i = action_ap(untrimmed_predictions, ac_ta, IoU=IoU, min_T=min_T, max_T=max_T, min_L=min_L,
                                             down_sampling=down_sampling, return_detections=True)
            ap["{}-{}-{}".format(min_T, max_T, min_L)] = ap_i
            det["{}-{}-{}".format(min_T, max_T, min_L)] = det_i
            gt["{}-{}-{}".format(min_T, max_T, min_L)] = gt_i

best_parm = max(ap, key=ap.get)
best_ap = ap[best_parm]
print("{}_{}: get best ap {:.2f} under {}".format(action, mode, best_ap, best_parm))
wandb.run.summary["average_precision"] = best_ap

with open('saved/{}_{}_search'.format(action, mode), 'w') as f:
    json.dump(ap, f)
