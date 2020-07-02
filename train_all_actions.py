#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : train.py
# Author: Chongkai LU
# Date  : 3/29/2020

from load_data import *
from utils import *
from custom_class import BiasLayer
from pathlib import Path
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.layers import Dense, Dropout, Activation
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
    y_s=1,
    y_e=100,
    learning_rate=0.0001,
    batch_size=64,
    epochs=100,
    agent=agent
)
wandb.init(config=default_config, name=now, notes='one model for all action, multi-task train')
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
root = {'train': "/mnt/louis-consistent/Datasets/THUMOS14/Images/train",
        'val': "/mnt/louis-consistent/Datasets/THUMOS14/Images/validation",
        'test': "/mnt/louis-consistent/Datasets/THUMOS14/Images/test"}
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
def augment_func(x, y):
    import tensorflow as tf
    x = tf.image.random_flip_left_right(x)
    return x, y
datalist = {x: read_from_anndir(root[x], anndir[x], y_range=y_range, ordinal=True, mode='rgb', stack_length=10) for x in ['train', 'val', 'test']}
test_dataset = build_dataset_from_slices(*datalist['test'], batch_size=batch_size, shuffle=False)
train_val_datalist = [a + b for a, b in zip(datalist['train'], datalist['val'])]
train_val_dataset = build_dataset_from_slices(*train_val_datalist, batch_size=batch_size, augment=augment_func)
# %% Build and compile model
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
    x = BiasLayer(y_nums)(x)
    output = Activation('sigmoid')(x)
    model = Model(inputs, output)
    model_checkpoint = ModelCheckpoint(str(models_path.joinpath('{epoch:02d}-{val_mae_od:.2f}.h5')), period=5)
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

# # %% Prediction on untrimmed videos
# import pandas as pd
# temporal_annotation = pd.read_csv(annfile['test'], header=None)
# video_names = temporal_annotation.iloc[:, 0].unique()
# predictions = {}
# ground_truth = {}
# for v in video_names:
#     gt = temporal_annotation.loc[temporal_annotation.iloc[:, 0] == v].iloc[:, 1:].values
#     ground_truth[v] = gt
#
#     video_path = Path(root['test'], v)
#     img_list = find_imgs(video_path)
#     ds = build_dataset_from_slices(img_list, batch_size=1, shuffle=False)
#     prediction = model.predict(ds, verbose=1)
#     predictions[v] = prediction
#
# # %% Detect actions
# import numpy as np
# from action_detection import action_search
# action_detected = {}
# tps = {}
# for v, prediction in predictions.items():
#     ads = action_search(prediction, min_T=65, max_T=30, min_L=35)
#     action_detected[v] = ads
#     tps[v] = calc_truepositive(ads, ground_truth[v], 0.5)
#
# num_gt = sum([len(gt) for gt in ground_truth.values()])
# loss = np.vstack(list(action_detected.values()))[:, 2]
# tp_values = np.hstack(list(tps.values()))
# ap = average_precision(tp_values, num_gt, loss)
