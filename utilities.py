#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : utilities.py
# Author: Chongkai LU
# Date  : 26/5/2020


def fix_bug():
    import socket
    import tensorflow as tf
    if socket.gethostname() == "louis-2":
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        for physical_device in physical_devices:
            tf.config.experimental.set_memory_growth(physical_device, True)


def video2images(video_path, save_path, suffix='mp4'):
    """
    Convert mp4 videos to images and store them in folders.
    :param video_path:
        String. Directory that contain video files.
    :param save_path:
        String. Direcotory that will be used to store images. Sub-folders will be made here.
    :param suffix:
        String. Video suffix. e.g. 'mp4'
    :return:
        Nothing.
    Example:
    video_path = "/mnt/louis-consistent/Datasets/THUMOS14/TH14_test_set_mp4"
    save_path = "/mnt/louis-consistent/Datasets/THUMOS14/Test"2
    suffix = "mp4"
    """
    from pathlib import Path
    import cv2
    videos = Path(video_path).glob('*.'+suffix)
    for video in videos:
        image_path = Path(save_path, video.stem)
        image_path.mkdir()
        vidcap = cv2.VideoCapture(video.__str__())
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite("%s/%s.jpg" % (image_path, str(count).zfill(5)), image)     # save frame as JPEG file
            success, image = vidcap.read()
            print('Read a new frame: ', success)
            count += 1


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


def build_datagenerators(data_paths, labels, preprocess_input, **kwargs):
    import pandas as pd
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    df = pd.DataFrame({'paths': data_paths, 'labels': labels})
    preprocessed = ImageDataGenerator(preprocessing_function=preprocess_input)

    data_generator = preprocessed.flow_from_dataframe(df, x_col='paths', y_col='labels', **kwargs)
    return data_generator


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


def combine_his(*historys):
    history = {}
    for key in historys[0]:
        history[key] = [item for his in historys for item in his[key]]
    return history


def plot_history(path, history, keys=None):
    from matplotlib import pyplot as plt
    import numpy as np
    from pathlib import Path

    if keys is None:
        keys = list(history.keys())
    if isinstance(path, str):
        path = Path(path)

    for i, key in enumerate(keys):
        plt.figure(figsize=(15, 5))
        if 'val' in key:
            continue
        train = plt.plot(history[key], label='Train ' + key.title())
        if 'val_{}'.format(key) in history:
            plt.plot(history['val_{}'.format(key)], '--', color=train[0].get_color(), label='Val ' + key.title())
        plt.legend()
        plt.xticks(np.arange(0, len(history[key]) + 1, 5.0))
        # plt.xlim([0, max(history.epoch)])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid()
        plt.savefig(path.joinpath('{}.png'.format(key)))


def save_history(path, history):
    import pandas as pd
    history_pd = pd.DataFrame(data=history)
    history_pd.to_csv(path.joinpath('history.csv'), index=False)


def lr_schedule(epoch, lr):
    # if epoch == 20:
    #     return lr * 0.2
    # if epoch == 40:
    #     return lr * 0.2
    # else:
        return lr



