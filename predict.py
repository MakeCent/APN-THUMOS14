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
model_path = "/mnt/louis-consistent/Saved/THUMOS14_output/GolfSwing/Model/2020-06-23 22:41:56/30-15.67.h5"
video_path = "/mnt/louis-consistent/Datasets/THUMOS14/Images/Test/video_test_0000028"
img_list = find_imgs(video_path)
test_dataset = build_dataset_from_slices(img_list, batch_size=128, shuffle=False)
train_root = "/mnt/louis-consistent/Datasets/THUMOS14/Images/Train"
train_annfile = "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/train/annotationF/{}_trainF.csv".format(action)
train_list = read_from_annfile(train_root, train_annfile, (0, 100))
train_dataset = build_dataset_from_slices(*train_list, shuffle=False)
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
    train_prediction = model.predict(train_dataset, verbose=1)
    test_prediction = model.predict(test_dataset, verbose=1)
    train_evaluation = model.evaluate(train_dataset, verbose=1, callbacks=[LossCallback()])


np.array(train_evaluation.losses)


import numpy as np
plt.figure(figsize=(15, 5))
plt.plot(test_prediction, '-')
plt.yticks(np.arange(0, 100, 20.0))
plt.xlabel('Frame Index')
plt.ylabel('Completeness')
plt.grid()
plt.show()
