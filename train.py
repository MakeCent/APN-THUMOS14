#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : Train.py.py
# Author: Chongkai LU
# Date  : 3/29/2020

import numpy as np
from load_data import *
from utilities import *
from pathlib import Path
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import wandb
from wandb.keras import WandbCallback
import datetime
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE
fix_bug()
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# %% wandb Initialization
hyperparameter_defaults = dict(
    loss='mae',
    y_s=0,
    y_e=100,
    learning_rate=0.0001,
    learning_rate2=0.00001,
    batch_size=32,
    epochs=30,
    epochs2=10,
    action="BaseballPitch"
)
wandb.init(config=hyperparameter_defaults, name=now)
config = wandb.config
wandbcb = WandbCallback(monitor='val_n_mae', save_model=False)

loss = config.loss
y_range = (config.y_s, config.y_e)
learning_rate = config.learning_rate
learning_rate2 = config.learning_rate2
batch_size = config.batch_size
epochs = config.epochs
epochs2 = config.epochs2
action = config.action

# %% Parameters, Configuration, and Initialization
model_name = "resnet50_imagenet_{}_{}_{}".format(config.loss, *y_range)
root = {'train': "/mnt/louis-consistent/Datasets/THUMOS14/Validation",
        'val': "/mnt/louis-consistent/Datasets/THUMOS14/Test"}
annfile = {
    'train': "/mnt/louis-consistent/Datasets/THUMOS14/Temporal_annotations_validation/annotationF/{}_valF.csv".format(
        action),
    'val': "/mnt/louis-consistent/Datasets/THUMOS14/Temporal_annotations_test/annotationF/{}_testF.csv".format(
        action)}
output_path = '/mnt/louis-consistent/Saved/THUMOS14_output'  # Directory to save model and history
history_path = Path(output_path, config.action, 'History', model_name)
models_path = Path(output_path, config.action, 'Model', model_name)
history_path.mkdir(parents=True, exist_ok=True)
models_path.mkdir(parents=True, exist_ok=True)

# %% Build dataset
datasets = {x: build_dataset(root[x], annfile[x], y_range, target_size=(224, 224)) for x in ['train', 'val']}
ds_size = {x: tf.data.experimental.cardinality(datasets[x]).numpy() for x in ['train', 'val']}
train_dataset = datasets['train'].cache().shuffle(ds_size['train']).repeat().batch(batch_size).prefetch(
    buffer_size=AUTOTUNE)
val_dataset = datasets['val'].cache().repeat().batch(batch_size).prefetch(buffer_size=AUTOTUNE)
train_steps = ds_size['train'] // batch_size
val_steps = ds_size['val'] // batch_size

# %% Build and compile model
n_mae = normalize_mae(y_range[1] - y_range[0])  # make mae loss normalized into range 0 - 100.
strategy = tf.distribute.MirroredStrategy()
# Make sure all model construction and compilation is in the scope()
with strategy.scope():
    inputs = tf.keras.Input(shape=(224, 224, 3))
    norm_layer = tf.keras.layers.experimental.preprocessing.Normalization()
    mean = np.array([127.5] * 3)
    var = mean ** 2
    # Scale inputs to [-1, +1]
    x = norm_layer(inputs)
    norm_layer.set_weights([mean, var])
    res_net = ResNet50(weights='imagenet', input_shape=(224, 224, 3), pooling='avg', include_top=False)
    # Freeze all BatchNorm layers. Attribute "training" does NOT represent the state of train phase, it will only
    # affects batchnorm layers to keep them in inference mode.
    x = res_net(x, training=False)
    x = Dense(64, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, kernel_initializer='he_uniform')(x)
    model = Model(inputs, output)
    model_checkpoint = ModelCheckpoint(str(models_path.joinpath('{epoch:02d}-{val_n_mae:.2f}.h5')), period=5)
    # %% Transfer Learning
    res_net.trainable = False
    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=[n_mae])
    transf_his = model.fit(train_dataset, steps_per_epoch=train_steps, validation_data=val_dataset,
                           validation_steps=val_steps, epochs=epochs, callbacks=[model_checkpoint, wandbcb], verbose=1)

    # %% Fine tune
    res_net.trainable = True
    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate2), metrics=[n_mae])
    ftune_his = model.fit(train_dataset, steps_per_epoch=train_steps, validation_data=val_dataset,
                          validation_steps=val_steps, epochs=epochs, callbacks=[model_checkpoint, wandbcb], verbose=1)


# %% Save history to csv and images
history = combine_his(transf_his.history, ftune_his.history)
save_history(history_path, history)
plot_history(history_path, history)

