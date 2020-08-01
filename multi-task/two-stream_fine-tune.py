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

root = {'train': "/mnt/louis-consistent/Datasets/THUMOS14/Images/train",
            'val': "/mnt/louis-consistent/Datasets/THUMOS14/Images/validation",
            'test': "/mnt/louis-consistent/Datasets/THUMOS14/Images/test"}
anndir = {
    'train': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/train/annotationF",
    'val': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/validation/annotationF",
    'test': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/test/annotationF"}

parse = thumo14_parse_builder(mode='two_stream', i3d=True)

rgb_datalist = {x: read_from_anndir(anndir[x], root=root[x], mode='rgb', y_range=y_range, ordinal=True, stack_length=10) for x in ['train', 'val', 'test']}
rgb_train_datalist = [a+b for a, b in zip(rgb_datalist['train'], rgb_datalist['val'])]
rgb_test_datalist = rgb_datalist['test']

train_dataset = build_dataset_from_slices(*rgb_train_datalist, shuffle=True, parse_func=parse)
test_dataset = build_dataset_from_slices(*rgb_test_datalist, shuffle=False, parse_func=parse)

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
