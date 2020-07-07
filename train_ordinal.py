#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : train.py
# Author: Chongkai LU
# Date  : 3/29/2020

from load_data import *
from utils import *
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
from custom_class import BiasLayer
import numpy as np

agent = socket.gethostname()
AUTOTUNE = tf.data.experimental.AUTOTUNE
fix_bug()
now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# %% wandb Initialization
default_config = dict(
    loss='binary_crossentropy',
    y_s=1,
    y_e=100,
    learning_rate=0.0001,
    batch_size=32,
    epochs=50,
    action="GolfSwing",
    agent=agent
)
ordinal = True
mode = 'rgb'
stack_length = 10
weighted = False

cross_pre = True #if mode == 'flow' else False
partialBN = True #if mode == 'flow' else False

notes = "rgb10, cross-pre and partialBN, 'bad'"
# Just for wandb
tags = [mode]
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

wandb.init(config=default_config, tags=tags, name=now, notes=notes)
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
if mode == 'rgb':
    root = {'train': "/mnt/louis-consistent/Datasets/THUMOS14/Images/train",
            'val': "/mnt/louis-consistent/Datasets/THUMOS14/Images/validation",
            'test': "/mnt/louis-consistent/Datasets/THUMOS14/Images/test"}
elif mode == 'flow':
    root = {'train': "/mnt/louis-consistent/Datasets/THUMOS14/OpticalFlows/train",
            'val': "/mnt/louis-consistent/Datasets/THUMOS14/OpticalFlows/validation",
            'test': "/mnt/louis-consistent/Datasets/THUMOS14/OpticalFlows/test"}
else:
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

datalist = {x: read_from_annfile(root[x], annfile[x], y_range, mode=mode, ordinal=True, stack_length=stack_length) for x in
            ['train', 'val', 'test']}
train_val_datalist = [a + b for a, b in zip(datalist['train'], datalist['val'])]
test_dataset = build_dataset_from_slices(*datalist['test'], batch_size=batch_size, shuffle=False)
train_val_dataset = build_dataset_from_slices(*train_val_datalist, batch_size=batch_size)
# %% Build and compile model

if cross_pre:
    pretrained = ResNet101(weights='imagenet', input_shape=(224, 224, 3), pooling='avg', include_top=False)
    weights = pretrained.layers[2].get_weights()[0]
    biases = pretrained.layers[2].get_weights()[1]
    extended_kernels = np.repeat(weights.mean(axis=2)[:, :, np.newaxis], stack_length*3 if mode == 'rgb' else stack_length*2, axis=2)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    backbone = ResNet101(weights='imagenet' if mode == 'rgb' and stack_length == '1' else None,
                         input_shape=(224, 224, stack_length * 3 if mode == 'rgb' else stack_length * 2), pooling='avg',
                         include_top=False)
    if partialBN:
        backbone = insert_layer_nonseq(backbone, 'conv1_bn', bn_factory, position='replace', new_training=True, other_training=False)
    if cross_pre:
        backbone.layers[2].set_weights([extended_kernels, biases])
        for layer in pretrained.layers:
            if layer.name != 'conv1_conv' and layer.get_weights() != []:
                backbone.get_layer(layer.name).set_weights(layer.get_weights())
    x = Dense(2048, activation='relu', kernel_initializer='he_uniform')(backbone.output)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, kernel_initializer='he_uniform', use_bias=False)(x)
    x = BiasLayer(y_nums)(x)
    output = Activation('sigmoid')(x)
    model = Model(backbone.input, output)
    model_checkpoint = ModelCheckpoint(str(models_path.joinpath('{epoch:02d}-{val_mae_od:.2f}.h5')), period=5)
    # %% Fine tune
    backbone.trainable = True
    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=[mae_od])

ftune_his = model.fit(train_val_dataset, validation_data=test_dataset, epochs=epochs,
                      callbacks=[model_checkpoint, wandbcb], verbose=1)

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
    img_list = find_imgs(video_path, stack_length=10)
    ds = build_dataset_from_slices(img_list, batch_size=1, shuffle=False)
    prediction = model.predict(ds, verbose=1)
    predictions[v] = np.squeeze(prediction)

# %% Detect actions
action_detected = {}
tps = {}
for v, prediction in predictions.items():
    prediction = ordinal2completeness(prediction)
    ads = action_search(prediction, min_T=50, max_T=30, min_L=40)
    action_detected[v] = ads
    tps[v] = calc_truepositive(ads, ground_truth[v], 0.5)

num_gt = sum([len(gt) for gt in ground_truth.values()])
loss = np.vstack(list(action_detected.values()))[:, 2]
tp_values = np.hstack(list(tps.values()))
ap = average_precision(tp_values, num_gt, loss)
wandb.log({'ap': ap})
