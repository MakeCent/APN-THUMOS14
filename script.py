#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : Train.py.py
# Author: Chongkai LU
# Date  : 3/29/2020
from load_data import *
from utilities import *
import tensorflow as tf
import datetime
action = "GolfSwing"
y_range = (0, 100)
batch_size = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
model_name = now
root = {'train': "/mnt/louis-consistent/Datasets/THUMOS14/Images/Train",
        'val': "/mnt/louis-consistent/Datasets/THUMOS14/Images/Validation"}
annfile = {
    'train': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/train/annotationF/{}_trainF.csv".format(
        action),
    'val': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/validation/annotationF/{}_valF.csv".format(
        action)}
datasets = {x: dataset_trimmed(root[x], annfile[x], y_range) for x in ['train', 'val']}
ds_size = {x: tf.data.experimental.cardinality(datasets[x]).numpy() for x in ['train', 'val']}
train_dataset = datasets['train'].cache().batch(batch_size).prefetch(buffer_size=AUTOTUNE)
val_dataset = datasets['val'].cache().batch(batch_size).prefetch(buffer_size=AUTOTUNE)

img = train_dataset.take(1).as_numpy_iterator().next()[0][0]