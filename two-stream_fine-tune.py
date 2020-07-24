#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : two-stream_fine-tune.py
# Author: Chongkai LU
# Date  : 7/2/2020
import tensorflow as tf
from custom_class import MultiAction_BiasLayer, BiasLayer
from load_data import *
from utils import *
y_nums = 100
y_range = (1, 100)
batch_size = 32
action = 'GolfSwing'
rgb_root = {'train': "/mnt/louis-consistent/Datasets/THUMOS14/Images/train",
        'val': "/mnt/louis-consistent/Datasets/THUMOS14/Images/validation",
        'test': "/mnt/louis-consistent/Datasets/THUMOS14/Images/test"}
flow_root = {'train': "/mnt/louis-consistent/Datasets/THUMOS14/OpticalFlows/train",
        'val': "/mnt/louis-consistent/Datasets/THUMOS14/OpticalFlows/validation",
        'test': "/mnt/louis-consistent/Datasets/THUMOS14/OpticalFlows/test"}
annfile = {
    'train': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/train/annotationF/{}_trainF.csv".format(
        action),
    'val': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/validation/annotationF/{}_valF.csv".format(
        action),
    'test': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/test/annotationF/{}_testF.csv".format(
        action)}

rgb_datalist = {x: read_from_annfile(rgb_root[x], annfile[x], mode='rgb', y_range=y_range, ordinal=True) for x in ['train', 'val', 'test']}
rgb_train_val_datalist = (a+b for a, b in zip(rgb_datalist['train'], rgb_datalist['val']))
rgb_train_val_dataset = build_dataset_from_slices(*rgb_train_val_datalist, batch_size=-1, shuffle=False, prefetch=False)
rgb_test_dataset = build_dataset_from_slices(*rgb_datalist['test'], batch_size=-1, shuffle=False, prefetch=False)

flow_datalist = {x: read_from_annfile(flow_root[x], annfile[x], mode='flow', y_range=y_range, ordinal=True, stack_length=10) for x in ['train', 'val', 'test']}
flow_train_val_datalist = (a+b for a, b in zip(*flow_datalist['train'], flow_datalist['val']))
flow_train_val_dataset = build_dataset_from_slices(*flow_train_val_datalist, batch_size=-1, shuffle=False, prefetch=False)
flow_test_dataset = build_dataset_from_slices(*flow_datalist['test'], batch_size=-1, shuffle=False, prefetch=False)

train_dataset = tf.data.Dataset.zip((rgb_train_val_dataset, flow_train_val_dataset))
test_dataset = tf.data.Dataset.zip((rgb_test_dataset, flow_test_dataset))
rgb_model_path = ""
flow_model_path = ""

with tf.distribute.MirroredStrategy():
    rgb_model = tf.keras.models.load_model(rgb_model_path)
    flow_model = tf.keras.models.load_model(flow_model_path)
    x = tf.keras.layers.concatenate([rgb_model.get_layer('backbone').output, flow_model.get_layer('backbone').output])
    x = tf.keras.layers.Dense(2048)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(2048)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = BiasLayer(y_nums)(x)
    output = tf.keras.layers.Activation('sigmoid')(x)
    model = tf.keras.Model(inputs=[rgb_model.input, flow_model.input], outputs=output, optimizer=tf.keras.optimizers.Adam(0.0001))
    model.compile(loss='binary_crossentropy', metrics=[mae_od])

his = model.fit(train_dataset, validation_data=test_dataset, verbose=1)
