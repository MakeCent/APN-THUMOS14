#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : two-stream_fine-tune.py
# Author: Chongkai LU
# Date  : 19/7/2020
import tensorflow as tf
from tools.custom_class import BiasLayer
from tools.load_data import *
from tools.utils import *
from pathlib import Path
from wandb.keras import WandbCallback
import socket
import datetime
import wandb
now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
model_name =now
agent = socket.gethostname()
# %% wandb Initialization
# Configurations. If you don't use wandb, just manually set these values.
default_config = dict(
    learning_rate=0.0001,
    batch_size=32,
    epochs=20,
    agent=agent
)

stack_length = 10
notes = ''

# Just for wandb
tags = ['all', 'two_stream', 'i3d']
tags.append("od")
tags.append("stack10")
wandb.init(config=default_config, name=now, tags=tags, notes=notes)
config = wandb.config
wandbcb = WandbCallback(monitor='val_multi_od_metric', save_model=False)

y_range = (1, 100)
y_nums = 100
learning_rate = config.learning_rate
batch_size = config.batch_size
epochs = config.epochs
action_num = 20

# %% Basic information
rgb_model_path = "/mnt/louis-consistent/Saved/THUMOS14_output/Multi-task/Model/2020-07-07-21-08-04/13-19.10.h5"
flow_model_path = "/mnt/louis-consistent/Saved/THUMOS14_output/Multi-task/Model/2020-07-19-22-18-54/20-17.84.h5"

root = {'train': "/mnt/louis-consistent/Datasets/THUMOS14/Images/train",
            'val': "/mnt/louis-consistent/Datasets/THUMOS14/Images/validation",
            'test': "/mnt/louis-consistent/Datasets/THUMOS14/Images/test"}
anndir = {
    'train': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/train/annotationF",
    'val': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/validation/annotationF",
    'test': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/test/annotationF"}

output_path = '/mnt/louis-consistent/Saved/THUMOS14_output/Multi-task'  # Directory to save model and history
history_path = Path(output_path, 'History', model_name)
models_path = Path(output_path, 'Model', model_name)
history_path.mkdir(parents=True, exist_ok=True)
models_path.mkdir(parents=True, exist_ok=True)
# %% Build dataset
parse = parse_builder(mode='two_stream', i3d=True)

rgb_datalist = {x: read_from_anndir(anndir[x], root=root[x], mode='rgb', y_range=y_range, ordinal=True, stack_length=10) for x in ['train', 'val', 'test']}
rgb_train_datalist = [a+b for a, b in zip(rgb_datalist['train'], rgb_datalist['val'])]
rgb_test_datalist = rgb_datalist['test']

train_dataset = build_dataset_from_slices(*rgb_train_datalist, shuffle=True, parse_func=parse)
test_dataset = build_dataset_from_slices(*rgb_test_datalist, shuffle=False, parse_func=parse)
del rgb_datalist, rgb_test_datalist, rgb_test_datalist

# %% Start training
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(str(models_path.joinpath('{epoch:02d}-{val_multi_od_metric:.2f}.h5')), period=1)
with tf.distribute.MirroredStrategy():
    rgb_model = tf.keras.models.load_model(rgb_model_path)
    flow_model = tf.keras.models.load_model(flow_model_path)
    for layer in flow_model.layers:
        layer._name = 'flow_' + layer._name
    x = tf.keras.layers.concatenate([rgb_model.get_layer('reshape').output, flow_model.get_layer('flow_reshape').output])
    x = tf.keras.layers.Dense(2048, activation='relu', kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(2048, activation='relu', kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(action_num, kernel_initializer='he_uniform', use_bias=False)(x)
    x = BiasLayer(y_nums)(x)
    output = tf.keras.layers.Activation('sigmoid')(x)
    model = tf.keras.Model(inputs={"rgb_input": rgb_model.input, "flow_input": flow_model.input}, outputs=output)
    model.compile(loss=multi_binarycrossentropy, metrics=[multi_od_metric], optimizer=tf.keras.optimizers.Adam(learning_rate))

his = model.fit(train_dataset, validation_data=test_dataset, verbose=1, callback=[WandbCallback, model_checkpoint])
