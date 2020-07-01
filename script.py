#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : train.py
# Author: Chongkai LU
# Date  : 3/29/2020

from load_data import *
from utils import *
from pathlib import Path
import numpy as np
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
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

pretrained = ResNet101(weights='imagenet', input_shape=(224, 224, 3), pooling='avg', include_top=False)
weights = pretrained.layers[2].get_weights()[0]
biases = pretrained.layers[2].get_weights()[1]
extended_kernels = np.repeat(weights.mean(axis=2)[:, :, np.newaxis], 20, axis=2)


def bn_factory():
    return BatchNormalization(name='conv1_bn')

inputs = tf.keras.Input(shape=(224, 224, 20))
backbone = ResNet101(weights=None, input_shape=(224, 224, 20), pooling='avg', include_top=False)
backbone = insert_layer_nonseq(backbone, 'conv1_bn', bn_factory, position='replace', new_training=True, other_training=False)
start = []
end = []
start = []
end = []
for v, p in predictions.items():
    p = ordinal2completeness(p)
    end.extend(p[ground_truth[v][:,1]])
    start.extend(p[ground_truth[v][:,0]])
from matplotlib import pyplot as plt
plt.boxplot([start, end])
plt.title("unweighted, 50-21.79")
plt.show()