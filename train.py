#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : train.py
# Author: Chongkai LU
# Date  : 3/29/2020

from load_data import *
from utilities import *
from pathlib import Path
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
wandb.init(config=default_config, name=now, notes='baseline:no ordinal no flow, train and val for train, no pretrain on imagenet.')
config = wandb.config
wandbcb = WandbCallback(monitor='val_n_mae', save_model=False)

loss = config.loss
y_range = (config.y_s, config.y_e)
y_nums = y_range[1] - y_range[0]
learning_rate = config.learning_rate
batch_size = config.batch_size
epochs = config.epochs
action = config.action

# %% Parameters, Configuration, and Initialization
model_name = now
root = {'train': "/mnt/louis-consistent/Datasets/THUMOS14/Images/train",
        'val': "/mnt/louis-consistent/Datasets/THUMOS14/Images/validation",
        'test': "/mnt/louis-consistent/Datasets/THUMOS14/Images/test"}
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

# def augment_func(x, y):
#     import tensorflow as tf
#     x = tf.image.random_flip_left_right(x)
#     return x, y

datalist = {x: read_from_annfile(root[x], annfile[x], y_range) for x in ['train', 'val', 'test']}
test_dataset = build_dataset_from_slices(*datalist['test'], batch_size=batch_size, shuffle=False)
train_val_datalist = (datalist['train'][0]+datalist['val'][0], datalist['train'][1]+datalist['val'][1])
train_val_dataset = build_dataset_from_slices(*train_val_datalist, batch_size=batch_size, augment=None)

# %% Build and compile model
n_mae = normalize_mae(y_nums)  # make mae loss normalized into range 0 - 100.
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    inputs = tf.keras.Input(shape=(224, 224, 3))
    backbone = ResNet101(weights=None, input_shape=(224, 224, 3), pooling='avg', include_top=False)
    x = backbone(inputs)
    x = Dense(64, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, kernel_initializer='he_uniform')(x)
    model = Model(inputs, output)
    model_checkpoint = ModelCheckpoint(str(models_path.joinpath('{epoch:02d}-{val_n_mae:.2f}.h5')), period=5)
    lr_sche = LearningRateScheduler(lr_schedule)
    # %% Fine tune
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
temporal_annotation = pd.read_csv(annfile['test'], header=None)
video_names = temporal_annotation.iloc[:, 0].unique()
predictions = {}
ground_truth = []
for v in video_names:
    gt = temporal_annotation.loc[temporal_annotation.iloc[:, 0] == v].iloc[:, 1:].values
    ground_truth.append(gt)

    video_path = Path(root['test'], v)
    img_list = find_imgs(video_path)
    ds = build_dataset_from_slices(img_list, batch_size=1, shuffle=False)
    prediction = model.predict(ds, verbose=1)
    predictions[v] = prediction

# %% Detect actions
import numpy as np
from action_detection import action_search
action_detected = []
tps = []
for k, prediction in predictions.items():
    ads = action_search(prediction, min_T=65, max_T=30, min_L=35)
    ads = np.array(ads)
    action_detected.append(ads)
    tps.append(calc_truepositive(ads, ground_truth[k], 0.5))

num_gt = np.vstack(ground_truth).shape[0]
loss = np.vstack(action_detected)[:, 2]
ap = average_precision(np.hstack(tps), num_gt, loss)
