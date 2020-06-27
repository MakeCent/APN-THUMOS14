#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : Train.py.py
# Author: Chongkai LU
# Date  : 3/29/2020

from load_data import *
from utilities import *
from pathlib import Path
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import wandb
from wandb.keras import WandbCallback
import datetime
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE
fix_bug()
now = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
# %% wandb Initialization
default_config = dict(
    loss='binary_crossentropy',
    y_s=1,
    y_e=100,
    learning_rate=0.0001,
    learning_rate2=0.0001,
    batch_size=32,
    epochs=0,
    epochs2=50,
    action="GolfSwing"
)
wandb.init(config=default_config, name=now, notes='change y_s to 1, add a fc2 layer, extend epoch to 50')
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
model_name = now
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
output_path = '/mnt/louis-consistent/Saved/THUMOS14_output'  # Directory to save model and history
history_path = Path(output_path, action, 'History', model_name)
models_path = Path(output_path, action, 'Model', model_name)
history_path.mkdir(parents=True, exist_ok=True)
models_path.mkdir(parents=True, exist_ok=True)

# %% Build dataset

# def augment_func(x):
#     import tensorflow as tf
#     x = tf.image.random_flip_left_right(x)
#     return x

datalist = {x: read_from_annfile(root[x], annfile[x], y_range) for x in ['train', 'val', 'test']}
test_dataset = build_dataset_from_slices(*datalist['test'], batch_size=batch_size, shuffle=False)
train_val_datalist = (datalist['train'][0]+datalist['val'][0], datalist['train'][1]+datalist['val'][1])
train_val_dataset = build_dataset_from_slices(*train_val_datalist, batch_size=batch_size)
# %% Build and compile model
# n_mae = normalize_mae(y_range[1] - y_range[0])  # make mae loss normalized into range 0 - 100.
strategy = tf.distribute.MirroredStrategy()
# Make sure all model construction and compilation is in the scope()
with strategy.scope():
    inputs = tf.keras.Input(shape=(224, 224, 3))
    backbone = ResNet101(weights='imagenet', input_shape=(224, 224, 3), pooling='avg', include_top=False)
    x = backbone(inputs)
    x = Dense(1024, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, kernel_initializer='he_uniform', use_bias=False)(x)
    x = BiasLayer(100)(x)
    output = Activation('sigmoid')(x)
    model = Model(inputs, output)
    model_checkpoint = ModelCheckpoint(str(models_path.joinpath('{epoch:02d}-{val_mae_od:.2f}.h5')), period=5)
    lr_sche = LearningRateScheduler(lr_schedule)
    if epochs > 0:
        # # %% Transfer Learning
        backbone.trainable = False
        model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=[mae_od])
        transf_his = model.fit(train_val_dataset, validation_data=test_dataset, epochs=epochs,
                               callbacks=[model_checkpoint, wandbcb, lr_sche], verbose=1)
    if epochs2 > 0:
        # %% Fine tune
        backbone.trainable = True
        model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate2), metrics=[mae_od])
        ftune_his = model.fit(train_val_dataset, validation_data=test_dataset, epochs=epochs2,
                              callbacks=[model_checkpoint, wandbcb, lr_sche], verbose=1)

# %% Save history to csv and images
history = ftune_his.history
save_history(history_path, history)
plot_history(history_path, history)
