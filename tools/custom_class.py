#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : custom_class.py
# Author: Chongkai LU
# Date  : 29/6/2020

import tensorflow as tf


class LossCallback(tf.keras.callbacks.Callback):

    def on_test_begin(self, logs=None):
        self.n_mae = []
        self.loss = []

    def on_test_batch_end(self, batch, logs=None):
        self.loss.append(logs['loss'])
        self.n_mae.append(logs['n_mae'])


class BiasLayer(tf.keras.layers.Layer):
    def __init__(self, units, *args, **kwargs):
        super(BiasLayer, self).__init__(*args, **kwargs)
        self.units = units

    def build(self, input_shape):
        self.bias = self.add_weight('bias',
                                    shape=(input_shape[-1], self.units),
                                    initializer='zeros',
                                    trainable=True)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'units': self.units})
        return config

    def call(self, inputs):
        return tf.repeat(tf.expand_dims(inputs, -1), self.units, axis=-1) + self.bias


class RGBLossCallback(tf.keras.callbacks.Callback):

    def on_test_begin(self, logs=None):
        self.n_mae = []
        self.loss = []

    def on_test_batch_end(self, batch, logs=None):
        self.loss.append(logs['loss'])
        self.n_mae.append(logs['n_mae'])


class FLowLossCallback(tf.keras.callbacks.Callback):

    def on_test_begin(self, logs=None):
        self.mae_od = []
        self.loss = []

    def on_test_batch_end(self, batch, logs=None):
        self.loss.append(logs['loss'])
        self.mae_od.append(logs['mae_od'])