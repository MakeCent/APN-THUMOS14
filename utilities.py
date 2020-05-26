#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : utilities.py
# Author: Chongkai LU
# Date  : 26/5/2020


def annotation_time2frame(mp4path, annotation_path):
    """
    Transfer timestamp-style temporal annotation file to frame-style
    :param mp4path:
        String. Directory where mp4 files locate.
    :param annotation_path:
        String. Directory where temporal annotation files lcoate.
    :return:
        None. create .csv files in same directory with annotation_path.
    Examples:
    mp4path = "/mnt/louis-consistent/Datasets/THUMOS14/TH14_test_set_mp4"
    annotation_path = "/mnt/louis-consistent/Datasets/THUMOS14/TH14_Temporal_annotations_test/annotation"
    """
    import pandas as pd
    import cv2 as cv
    import csv
    from pathlib import Path

    annotation_path= Path(annotation_path)
    for gtp in annotation_path.iterdir():
        train_ground_truth = pd.read_csv(gtp, sep='\s+', header=None)
        with open(annotation_path.joinpath(gtp.stem+'F'+'.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for _, row in train_ground_truth.iterrows():
                vn, start, end = row.values
                vc = cv.VideoCapture(mp4path + '/' + vn + '.mp4')
                vc.set(cv.CAP_PROP_POS_MSEC, start*1000)
                startf = int(vc.get(cv.CAP_PROP_POS_FRAMES))
                vc.set(cv.CAP_PROP_POS_MSEC, end*1000)
                endf = int(vc.get(cv.CAP_PROP_POS_FRAMES))
                writer.writerow([vn, startf, endf])


def normalize_mae(y_range):
    """
    Calculate MAE loss and normalize it to range 0 to 100.
    :param y_range:
        Float. The original MAE range length.
    :return:
        Function. Used as a loss function for keras. While it returns normalized mae loss.
    """
    from tensorflow.keras.losses import mean_absolute_error

    def n_mae(*args, **kwargs):
        mae = mean_absolute_error(*args, **kwargs)
        return mae / y_range * 100.0

    return n_mae


def build_model(backbone, dense_layer=(64, 32), out_activation=None):
    """
    Construct Sequential model. Add dense layers on the top of backbone network.
    :param backbone:
        Model. A tensorflow model.
    :param dense_layer:
        Tuple. Include integers that will be the number of nodes in each dense layers
    :param out_activation:
        String. To indicate the activation function in the output laydf. It must can be recognized by tensorflow.
    :return:
        Model. A tensorflow keras Sequential model.
    """
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Dropout
    model = tf.keras.Sequential()
    model.add(backbone)
    for index, value in enumerate(dense_layer):
        model.add(Dense(value, activation='relu', name='fc'+str(index+2), kernel_initializer='he_uniform'))
        model.add(Dropout(0.5))
    model.add(Dense(1, activation=out_activation, name='output_layer', kernel_initializer='he_uniform'))
    return model


def plot_history(history, keys, figname):
    from matplotlib import pyplot as plt
    import numpy as np
    n = len(keys)
    plt.figure(figsize=(15, 5))
    for i, key in enumerate(keys):
        plt.subplot(1, n, i + 1)
        train = plt.plot(history.epoch, history.history[key], label='Train ' + key.title())
        plt.plot(history.epoch, history.history['val_' + key], '--', color=train[0].get_color(), label='Test ' + key.title())
        plt.legend()
        plt.xticks(np.arange(min(history.epoch), max(history.epoch) + 1, 5.0))
        plt.xlim([0, max(history.epoch)])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid()
    plt.savefig(figname)


