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


def normalize_mae(y_nums):
    """
    Calculate MAE loss and normalize it to range 0 to 100.
    :param y_nums:
        Float. The original MAE length.
    :return:
        Function. Used as a loss function for keras. While it returns normalized mae loss.
    """
    from tensorflow.keras.losses import mean_absolute_error

    def n_mae(*args, **kwargs):
        mae = mean_absolute_error(*args, **kwargs)
        return mae / (y_nums - 1) * 100.0

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
        train = plt.plot(history[key], label='train ' + key.title())
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


def boxplot_split(split_as, split_on):
    boxed = []
    cuts = [(n, n+1) for n in range(100)]
    for c in cuts:
        boxed.append(list(split_on[(split_as > c[0]) * (split_as <= c[1])].squeeze()))
    return boxed


def get_boxplot(labels, n_maes):
    import numpy as np
    from matplotlib import pyplot as plt
    split_for_box = boxplot_split(np.arrat(labels), np.array(n_maes))
    plt.figure()
    plt.boxplot(split_for_box)
    plt.show()


def plot_detection(video_prediction, gt, ads):
    from matplotlib import pyplot as plt
    import numpy as np
    plt.figure(figsize=(15, 5))
    plt.plot(video_prediction, '-')
    plt.vlines(gt[:, 0], 0, 100, colors='r', linestyles='solid', label='ground truth')
    plt.vlines(gt[:, 1], 0, 100, colors='r', linestyles='solid', label='ground truth')
    plt.vlines(ads[:, 0], 0, 100, colors='k', linestyles='dashed', label='ground truth')
    plt.vlines(ads[:, 1], 0, 100, colors='k', linestyles='dashed', label='ground truth')
    plt.yticks(np.arange(0, 100, 20.0))
    plt.xlabel('Frame Index')
    plt.ylabel('Completeness')
    plt.grid()
    plt.show()


def plot_prediction(video_prediction):
    from matplotlib import pyplot as plt
    import numpy as np
    plt.figure(figsize=(15, 5))
    plt.plot(video_prediction, '-')
    plt.yticks(np.arange(0, 100, 20.0))
    plt.xlabel('Frame Index')
    plt.ylabel('Completeness')
    plt.grid()
    plt.show()

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
                                    shape=self.units,
                                    initializer='zeros',
                                    trainable=True)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'units': self.units,})
        return config

    def call(self, input):
        return input + self.bias


def matrix_iou(gt, ads):
    import numpy as np

    def iou(a, b):
        ov = 0
        union = max(a[1], b[1]) - min(a[0], b[0])
        intersection = min(a[1], b[1]) - max(a[0], a[0])
        if intersection > 0:
            ov = intersection/union
        return ov
    ov_m = np.zeros([gt.shape[0], ads.shape[0]])
    for i in range(gt.shape[0]):
        for j in range(ads.shape[0]):
            ov_m[i, j] = iou(gt[i, :], ads[j, :])
    return ov_m


def calc_truepositive(action_detected, temporal_annotations, iou_T):
    """
    Give the predicted action intervals and ground truth intervals, using IoU threshold to get true positive proposals.
    :param action_detected: Array. Shape (N, 3). Float. 1st and 2st columns contain start and ending frame indexes of detected actions.
            3st column for confidence/loss. Here is for loss, which means tp will be sort with ascend order.
    :param temporal_annotations: Array. Shape (M, 2). Float. 1st and 2st columns contain start and ending frame indexes of ground truthes.
    :param iou_T: Float. IoU thredhold. A detected action can be true positive only if it has IoU larger than threhold with a ground truth.
    :return: Array. Shape (N,). Composing 0 and 1 corresponding to each detected action, 1 means true positive.
    """
    import numpy as np
    num_detection = action_detected.shape[0]
    iou_matrix = matrix_iou(temporal_annotations, action_detected[:, :2])
    tp = np.zeros(num_detection, dtype=np.int)
    for r in range(iou_matrix.shape[0]):
        max_iou = iou_matrix[r, :].max()
        max_idx = iou_matrix[r, :].argmax()
        if max_iou > iou_T:
            iou_matrix[:, max_idx] = 0
            tp[max_idx] = 1
    return tp


def average_precision(tp, num_gt, loss):
    """
    Compute average precision with given true positive indicator and number of ground truth.
    :param tp: Array. Shape (N,). Comprising 0 and 1. Represents the T or F of proposed predictions.
    :param num_gt: Int. Number of ground truth samples.
    :param loss: Array. Shape (N,). loss of each predictions used for sort the tp. For using confidence, code need to be revised.
    :return: Array. Shape (1,). Average Precision.
    """
    import numpy as np
    tp = tp[loss.argsort()]
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(1-tp)
    precisions = cum_tp / (cum_tp + cum_fp)
    AP = np.sum(precisions*tp)/num_gt
    return AP


def mae_od(y_true, y_pred):
    import tensorflow as tf
    predict_completeness = tf.math.count_nonzero(y_pred > 0.5, axis=-1)
    true_completeness = tf.math.count_nonzero(y_true > 0.5, axis=-1)
    mean_absolute_error = tf.math.reduce_mean(tf.math.abs(predict_completeness - true_completeness), axis=-1)
    return mean_absolute_error


def ordinal2completeness(array):
    import numpy as np
    completeness = np.count_nonzero(array > 0.5)
    return completeness
