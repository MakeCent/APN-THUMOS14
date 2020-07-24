#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : two-stream_fine-tune.py
# Author: Chongkai LU
# Date  : 19/7/2020
import tensorflow as tf
from tools.custom_class import BiasLayer, BiasLayer
from tools.load_data import *
from tools.utils import *


rgb_model_path = "/mnt/louis-consistent/Saved/THUMOS14_output/Multi-task/Model/2020-07-07-21-08-04/13-19.10.h5"
flow_model_path = ""
y_nums = 100
y_range = (1, 100)
batch_size = 32

rgb_root = {'train': "/mnt/louis-consistent/Datasets/THUMOS14/Images/train",
            'val': "/mnt/louis-consistent/Datasets/THUMOS14/Images/validation",
            'test': "/mnt/louis-consistent/Datasets/THUMOS14/Images/test"}
flow_root = {'train': "/mnt/louis-consistent/Datasets/THUMOS14/OpticalFlows/train",
             'val': "/mnt/louis-consistent/Datasets/THUMOS14/OpticalFlows/validation",
             'test': "/mnt/louis-consistent/Datasets/THUMOS14/OpticalFlows/test"}
anndir = {
    'train': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/train/annotationF",
    'val': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/validation/annotationF",
    'test': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/test/annotationF"}

rgb_datalist = {x: read_from_anndir(rgb_root[x], anndir[x], mode='rgb', y_range=y_range, ordinal=True, stack_length=10) for x in ['train', 'val', 'test']}
rgb_train_datalist = [a+b for a, b in zip(rgb_datalist['train'], rgb_datalist['val'])]

flow_datalist = {x: read_from_anndir(flow_root[x], anndir[x], mode='flow', y_range=y_range, ordinal=True, stack_length=10) for x in ['train', 'val', 'test']}
flow_train_datalist = [a+b for a, b in zip(flow_datalist['train'], flow_datalist['val'])]

two_stream_train_datalist = [list(i) for i in zip(rgb_train_datalist[0], flow_train_datalist[0])]
two_stream_test_datalist = [list(i) for i in zip(rgb_datalist['test'][0], flow_datalist['test'][0])]

train_dataset = build_dataset_from_slices(two_stream_train_datalist, flow_train_datalist[1], shuffle=True, i3d=True, mode='two_stream')
test_dataset = build_dataset_from_slices(two_stream_test_datalist, flow_datalist['test'][1], shuffle=False, i3d=True, mode='two_stream')

with tf.distribute.MirroredStrategy():
    rgb_model = tf.keras.models.load_model(rgb_model_path)
    flow_model = tf.keras.models.load_model(flow_model_path)
    x = tf.keras.layers.concatenate([rgb_model.get_layer('backbone').output, flow_model.get_layer('backbone').output])
    x = tf.keras.layers.Dense(2048, activation='relu', kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(2048, activation='relu', kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(20, kernel_initializer='he_uniform', use_bias=False)(x)
    x = BiasLayer(y_nums)(x)
    output = tf.keras.layers.Activation('sigmoid')(x)
    model = tf.keras.Model(inputs=[rgb_model.input, flow_model.input], outputs=output, optimizer=tf.keras.optimizers.Adam(0.0001))
    model.compile(loss=multi_binarycrossentropy, metrics=[multi_od_metric])

his = model.fit(train_dataset, validation_data=test_dataset, verbose=1)
