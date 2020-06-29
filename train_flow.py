#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : train.py
# Author: Chongkai LU
# Date  : 3/29/2020

from load_data import *
from utils import *
from pathlib import Path
import numpy as np
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.layers import Dense, Dropout
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
default_config = dict(
    loss='mse',
    y_s=1,
    y_e=100,
    learning_rate=0.0001,
    batch_size=32,
    epochs=50,
    action="GolfSwing",
    agent=agent
)
wandb.init(config=default_config, name=now, notes='train with optical_flow (10), mse loss')
config = wandb.config
wandbcb = WandbCallback(monitor='val_n_mae', save_model=False)

loss = config.loss
y_range = (config.y_s, config.y_e)
y_nums = y_range[1] - y_range[0] + 1
learning_rate = config.learning_rate
batch_size = config.batch_size
epochs = config.epochs
action = config.action

# %% Parameters, Configuration, and Initialization
model_name = now
root = {'train': "/mnt/louis-consistent/Datasets/THUMOS14/OpticalFlows/train",
        'val': "/mnt/louis-consistent/Datasets/THUMOS14/OpticalFlows/validation",
        'test': "/mnt/louis-consistent/Datasets/THUMOS14/OpticalFlows/test"}
annfile = {
    'train': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/train/annotationF/{}_trainF.csv".format(
        action),
    'val': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/validation/annotationF/{}_valF.csv".format(
        action),
    'test': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/test/annotationF/{}_testF.csv".format(
        action)}
output_path = '/mnt/louis-consistent/Saved/THUMOS14_output'  # Directory to save model and history
history_path = Path(output_path, action, 'History', model_name)
models_path = Path(output_path, action, 'Model', model_name)
history_path.mkdir(parents=True, exist_ok=True)
models_path.mkdir(parents=True, exist_ok=True)

# %% Build dataset

datalist = {x: read_from_annfile(root[x], annfile[x], y_range, mode='flow') for x in ['train', 'val', 'test']}
test_dataset = stack_optical_flow(*datalist['test'], batch_size=batch_size, shuffle=False)
train_val_datalist = (datalist['train'][0]+datalist['val'][0], datalist['train'][1]+datalist['val'][1])
train_val_dataset = stack_optical_flow(*train_val_datalist, batch_size=batch_size)

# %% Build and compile model
n_mae = normalize_mae(y_range[1] - y_range[0])  # make mae loss normalized into range 0 - 100.
strategy = tf.distribute.MirroredStrategy()
pretrained = ResNet101(weights='imagenet', input_shape=(224, 224, 3), pooling='avg', include_top=False)
weights = pretrained.layers[2].get_weights()[0]
biases = pretrained.layers[2].get_weights()[1]
extended_kernels = np.repeat(weights.mean(axis=2)[:, :, np.newaxis], 20, axis=2)
with strategy.scope():
    inputs = tf.keras.Input(shape=(224, 224, 20))
    backbone = ResNet101(weights=None, input_shape=(224, 224, 20), pooling='avg', include_top=False)
    backbone.layers[2].set_weights([extended_kernels, biases])
    for layer in pretrained.layers:
        if layer.name != 'conv1_conv' and layer.get_weights() != []:
            backbone.get_layer(layer.name).set_weights(layer.get_weights())
    x = backbone(inputs)
    x = Dense(64, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, kernel_initializer='he_uniform')(x)
    model = Model(inputs, output)
    model_checkpoint = ModelCheckpoint(str(models_path.joinpath('{epoch:02d}-{val_n_mae:.2f}.h5')), period=5)
    lr_sche = LearningRateScheduler(lr_schedule)
    # %% Fine tune whole model
    backbone.trainable = True
    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=[n_mae])
    ftune_his = model.fit(train_val_dataset, validation_data=test_dataset, epochs=epochs,
                          callbacks=[model_checkpoint, wandbcb, lr_sche], verbose=1)

# %% Save history to csv and images
history = ftune_his.history
save_history(history_path, history)
plot_history(history_path, history)

# %% Prediction on untrimmed videos
import pandas as pd
import numpy as np
temporal_annotation = pd.read_csv(annfile['test'], header=None)
video_names = temporal_annotation.iloc[:, 0].unique()
predictions = {}
ground_truth = {}
for v in video_names:
    gt = temporal_annotation.loc[temporal_annotation.iloc[:, 0] == v].iloc[:, 1:].values
    ground_truth[v] = gt

    video_path = Path(root['test'], v)
    flow_list = find_flows(video_path)
    ds = stack_optical_flow(flow_list, batch_size=1, shuffle=False)
    prediction = model.predict(ds, verbose=1)
    predictions[v] = np.array([ordinal2completeness(p) for p in prediction])

# %% Detect actions
import numpy as np
action_detected = {}
tps = {}
for v, prediction in predictions.items():
    ads = action_search(prediction, min_T=50, max_T=30, min_L=40)
    action_detected[v] = ads
    tps[v] = calc_truepositive(ads, ground_truth[v], 0.5)

num_gt = sum([len(gt) for gt in ground_truth.values()])
loss = np.vstack(list(action_detected.values()))[:, 2]
tp_values = np.hstack(list(tps.values()))
ap = average_precision(tp_values, num_gt, loss)
