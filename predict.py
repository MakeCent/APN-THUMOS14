#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : Train.py.py
# Author: Chongkai LU
# Date  : 6/9/2020

import tensorflow as tf
from load_data import *
from utilities import *
import matplotlib.pyplot as plt
action = "BaseballPitch"
model_path = "/mnt/louis-consistent/Saved/THUMOS14_output/BaseballPitch/Model/2020-06-11 17:14:09/30-26.10.h5"
video_path = "/mnt/louis-consistent/Datasets/THUMOS14/Test/video_test_0000324"
dataset = dataset_single_video(video_path)
train = dataset_trimmed("/mnt/louis-consistent/Datasets/THUMOS14/Validation", "/mnt/louis-consistent/Datasets/THUMOS14/Temporal_annotations_validation/annotationF/{}_valF.csv".format(action), (0, 100), target_size=(224, 224))
train = train.batch(32)
tf.data.experimental.cardinality(dataset).numpy()
strategy = tf.distribute.MirroredStrategy()
n_mae = normalize_mae(100)  # make mae loss normalized into range 0 - 100.
with strategy.scope():
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(loss='mse', metrics=[n_mae])
    train_prediction = model.predict(train, verbose=1)
    test_prediction = model.predict(dataset, verbose=1)
    train_evaluation = model.evaluate(train, verbose=1)

import numpy as np
plt.figure(figsize=(15, 5))
plt.plot(prediction[:10], '-')
plt.yticks(np.arange(0, 100, 20.0))
plt.xlabel('Frame Index')
plt.ylabel('Completeness')
plt.grid()
plt.show()
