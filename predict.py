#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : Train.py.py
# Author: Chongkai LU
# Date  : 6/9/2020
import numpy as np
import tensorflow as tf
from load_data import *
from utilities import *
import matplotlib.pyplot as plt
action = "GolfSwing"
model_path = "/mnt/louis-consistent/Saved/THUMOS14_output/GolfSwing/Model/2020-06-12 20:34:46/30-21.75.h5"
video_path = "/mnt/louis-consistent/Datasets/THUMOS14/Test/video_test_0000028"
test = single_test_video(video_path, batch_size=128)
train = dataset_trimmed("/mnt/louis-consistent/Datasets/THUMOS14/Validation", "/mnt/louis-consistent/Datasets/THUMOS14/Temporal_annotations_validation/annotationF/{}_valF.csv".format(action), (0, 100))
train = train.batch(32)
tf.data.experimental.cardinality(test).numpy()
strategy = tf.distribute.MirroredStrategy()
n_mae = normalize_mae(100)  # make mae loss normalized into range 0 - 100.


class LossCallback(tf.keras.callbacks.Callback):
    def on_test_begin(self, logs=None):
        self.losses = []

    def on_test_batch_end(self, batch, logs=None):
        self.losses.append(logs['loss'])


with strategy.scope():
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(loss='mse', metrics=[n_mae])
    train_prediction = model.predict(train, verbose=1)
    test_prediction = model.predict(test, verbose=1)
    train_evaluation = model.evaluate(train, verbose=1, callbacks=[LossCallback()])


np.array(train_evaluation.losses)


import numpy as np
plt.figure(figsize=(15, 5))
plt.plot(test_prediction[:10], '-')
plt.yticks(np.arange(0, 100, 20.0))
plt.xlabel('Frame Index')
plt.ylabel('Completeness')
plt.grid()
plt.show()
