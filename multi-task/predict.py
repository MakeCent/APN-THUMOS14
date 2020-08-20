#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : train.py.py
# Author: Chongkai LU
# Date  : 6/9/2020
from tools.custom_class import *
from tools.load_data import *
from tools.utils import *
from pathlib import Path
import numpy as np
import pandas as pd
import json

# %% Test on a Untrimmed video
mode = 'two_stream'
y_range = (1, 100)
n_mae = normalize_mae(y_range[1] - y_range[0] + 1)

model_path = "/mnt/louis-consistent/Saved/THUMOS14_output/Multi-task/Model/Two-Stream_Fine-Tune/05-17.30.h5"
loss, metric = 'binary_crossentropy', mae_od
# %% Also test on trimmed train, validation, and test dataset
if mode == 'rgb' or mode == 'two_stream':
    root = {'train': "/mnt/louis-consistent/Datasets/THUMOS14/Images/train",
            'val': "/mnt/louis-consistent/Datasets/THUMOS14/Images/validation",
            'test': "/mnt/louis-consistent/Datasets/THUMOS14/Images/test"}
else:
    root = {'train': "/mnt/louis-consistent/Datasets/THUMOS14/OpticalFlows/train",
            'val': "/mnt/louis-consistent/Datasets/THUMOS14/OpticalFlows/validation",
            'test': "/mnt/louis-consistent/Datasets/THUMOS14/OpticalFlows/test"}
anndir = {
    'train': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/train/annotationF",
    'val': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/validation/annotationF",
    'test': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/test/annotationF"}

# %% Load model

model = tf.keras.models.load_model(model_path, compile=False, custom_objects={'BiasLayer': BiasLayer})
model = tf.keras.Sequential([model, tf.keras.layers.Reshape((100,))])
model.compile(loss=loss, metrics=[metric])

# %% Predict on untrimmed videos
parse = thumo14_parse_builder(mode=mode, i3d=True)
temporal_annotation = pd.read_csv("/mnt/louis-consistent/Datasets/THUMOS14/Information/test_videos.txt", header=None)
video_names = temporal_annotation.iloc[:, 0].unique()
predictions = {}
for v in video_names:
    video_path = Path(root['test'], v)
    if mode == 'rgb' or mode == 'two_stream':
        data_list = find_imgs(video_path)
    else:
        data_list = find_flows(video_path)

    ds = build_dataset_from_slices(data_list, batch_size=1, shuffle=False, parse_func=parse)
    prediction = model.predict(ds, verbose=1)
    predictions[v] = ordinal2completeness(np.squeeze(prediction))

with open('saved/Multi-task_{}_prediction'.format(mode), 'w') as f:
    list_pre = {str(k): v.tolist() for k, v in predictions.items()}
    json.dump(list_pre, f)
