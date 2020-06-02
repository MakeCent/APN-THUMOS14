#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : Train.py.py
# Author: Chongkai LU
# Date  : 3/29/2020

import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
from load_data import load_trimmed_images
from utilities import *
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
import wandb
from wandb.keras import WandbCallback
import datetime

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# if not formal_training then just a memo experiments will be conducted
formal_training = True

hyperparameter_defaults = dict(
    loss='mae',
    y_s=0,
    y_e=100,
    learning_rate=0.0001,
    batch_size=32,
    epochs=30 if formal_training is True else 5,
    action="BaseballPitch"
)
wandb.init(config=hyperparameter_defaults, entity="makecent", project="thumos14", name=now)
config = wandb.config
wandbcb = WandbCallback(monitor='val_n_mae', save_model=False)

y_range = (config.y_s, config.y_e)  # range of progression-label
train_directory = "/mnt/louis-consistent/Datasets/THUMOS14/Validation"
test_directory = "/mnt/louis-consistent/Datasets/THUMOS14/Test"
model_name = "resnet50_imagenet_{}_{}_{}".format(config.loss, *y_range)
output_path = '/mnt/louis-consistent/Saved/THUMOS14_output'  # Directory to save model and fit history

# Define default files path and create folders for storing outputs
train_ground_truth_path = "/mnt/louis-consistent/Datasets/THUMOS14/TH14_Temporal_annotations_validation/annotationF/{}_valF.csv".format(
    config.action)
test_ground_truth_path = "/mnt/louis-consistent/Datasets/THUMOS14/TH14_Temporal_annotations_test/annotationF/{}_testF.csv".format(
    config.action)
history_path = Path(output_path, config.action, 'History', model_name)
models_path = Path(output_path, config.action, 'Model', model_name)
history_path.mkdir(parents=True, exist_ok=True)
models_path.mkdir(parents=True, exist_ok=True)

# Load data
train_X, train_y, test_X, test_y = load_trimmed_images(train_ground_truth_path, test_ground_truth_path, train_directory,
                                                       test_directory, *y_range)
# Formal training or tiny training that just for testing if code works
if not formal_training:
    train_X, train_y = np.random.choice(train_X, size=500), np.random.choice(train_y, size=500)
    test_X, test_y = np.random.choice(test_X, size=50), np.random.choice(test_y, size=50)
else:
    pass

# # Build datagenerators
train_generator = build_datagenerators(train_X, train_y, preprocess_input, target_size=(224, 224),
                                       batch_size=config.batch_size, class_mode='raw')
test_generator = build_datagenerators(test_X, test_y, preprocess_input, target_size=(224, 224),
                                      batch_size=config.batch_size, class_mode='raw')
train_steps = train_generator.n // train_generator.batch_size
val_steps = test_generator.n // test_generator.batch_size

# Build and compile model
n_mae = normalize_mae(y_range[1] - y_range[0])  # make mae loss normalized into range 0 - 100.
strategy = tf.distribute.MirroredStrategy()
# Make sure all model construction and compilation is in the scope()
with strategy.scope():
    res_net = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')
    parallel_model = build_model(res_net, dense_layer=(64, 32), out_activation=None)
    optimizer = tf.keras.optimizers.Adam(config.learning_rate)
    parallel_model.compile(loss=config.loss, optimizer=optimizer, metrics=[n_mae])

# Start to train
model_checkpoint = ModelCheckpoint(str(models_path.joinpath('{epoch:02d}-{val_n_mae:.2f}.h5')), verbose=1)
history = parallel_model.fit(x=train_generator, steps_per_epoch=train_steps, validation_data=test_generator,
                             validation_steps=val_steps, epochs=config.epochs, verbose=1,
                             callbacks=[model_checkpoint, wandbcb])

# Save history to csv and images
history_pd = pd.DataFrame(data=history.history)
history_pd.to_csv(history_path.joinpath('history.csv'))
plot_history(history, ['loss'], figname=history_path.joinpath('loss.png'))
plot_history(history, ['n_mae'], figname=history_path.joinpath('n_mae.png'))
