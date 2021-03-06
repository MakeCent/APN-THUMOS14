#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : utils.py
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
    video_path = "/mnt/louis-consistent/Datasets/THUMOS14/Videos/validation_mp4"
    save_path = "/mnt/louis-consistent/Datasets/THUMOS14/Images/validation"
    suffix = "mp4"
    """
    from pathlib import Path
    import cv2
    videos = Path(video_path).glob('*.' + suffix)
    Path(save_path).mkdir(parents=True, exist_ok=True)
    for video in videos:
        image_path = Path(save_path, video.stem)
        image_path.mkdir()
        vidcap = cv2.VideoCapture(video.__str__())
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite("%s/%s.jpg" % (image_path, str(count).zfill(5)), image)  # save frame as JPEG file
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
    mp4path = "/mnt/louis-consistent/Datasets/THUMOS14/Videos/validation_mp4"
    annotation_path = "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/validation/annotation"
    """
    import pandas as pd
    import cv2 as cv
    import csv
    from pathlib import Path

    annotation_path = Path(annotation_path)
    annotationF_path = annotation_path.parent.joinpath('annotationF')
    annotationF_path.mkdir(parents=True, exist_ok=True)
    for gtp in annotation_path.glob('[!A]*.txt'):
        train_ground_truth = pd.read_csv(gtp, sep='\s+', header=None)
        with open(annotationF_path.joinpath(gtp.stem + 'F.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for _, row in train_ground_truth.iterrows():
                vn, start, end = row.values
                vc = cv.VideoCapture(mp4path + '/' + vn + '.mp4')
                vc.set(cv.CAP_PROP_POS_MSEC, start * 1000)
                startf = int(vc.get(cv.CAP_PROP_POS_FRAMES))
                vc.set(cv.CAP_PROP_POS_MSEC, end * 1000)
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
        model.add(Dense(value, activation='relu', name='fc' + str(index + 2), kernel_initializer='he_uniform'))
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
    cuts = [(n, n + 1) for n in range(100)]
    for c in cuts:
        boxed.append(list(split_on[(split_as > c[0]) * (split_as <= c[1])].squeeze()))
    return boxed


def get_boxplot(labels, n_maes):
    import numpy as np
    from matplotlib import pyplot as plt
    split_for_box = boxplot_split(np.array(labels), np.array(n_maes))
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


def compute_iou(a, b):
    ov = 0
    union = max(a[1], b[1]) - min(a[0], b[0])
    intersection = min(a[1], b[1]) - max(a[0], b[0])
    if intersection > 0:
        ov = intersection / union
    return ov


def matrix_iou(gt, ads):
    import numpy as np
    ov_m = np.zeros([gt.shape[0], ads.shape[0]])
    for i in range(gt.shape[0]):
        for j in range(ads.shape[0]):
            ov_m[i, j] = compute_iou(gt[i, :], ads[j, :])
    return ov_m


def calc_truepositive(action_detected, temporal_annotations, iou_T):
    """
    Give the predicted action intervals and ground truth intervals, using IoU threshold to get true positive proposals.
    :param action_detected: Array. Shape (N, 4). Float. 1st and 2st columns contain start and ending frame indexes of detected actions.
            3st column for confidence/loss. Here is for loss, which means tp will be sort with ascend order. 4st for action index.
    :param temporal_annotations: Array. Shape (M, 2). Float. 1st and 2st columns contain start and ending frame indexes of ground truthes.
    :param iou_T: Float. IoU thredhold. A detected action can be true positive only if it has IoU larger than threhold with a ground truth.
    :return: Array. Shape (N,). Composing 0 and 1 corresponding to each detected action, 1 means true positive.
    """
    import numpy as np
    num_detection = action_detected.shape[0]
    if num_detection == 0:
        return np.array([], dtype=np.int)
    iou_matrix = matrix_iou(temporal_annotations, action_detected[:, :2])
    tp = np.zeros(num_detection, dtype=np.int)
    while True:
        max_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        if iou_matrix[max_idx] > iou_T:
            iou_matrix[:, max_idx[1]] = 0
            iou_matrix[max_idx[0], :] = 0
            tp[max_idx[1]] = 1
        else:
            break
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
    cum_fp = np.cumsum(1 - tp)
    precisions = cum_tp / (cum_tp + cum_fp)
    AP = np.sum(precisions * tp) / num_gt
    return AP


def mae_od(y_true, y_pred):
    import tensorflow as tf
    predict_completeness = tf.math.count_nonzero(y_pred > 0.5, axis=-1)
    true_completeness = tf.math.count_nonzero(y_true > 0.5, axis=-1)
    mean_absolute_error = tf.math.abs(predict_completeness - true_completeness)
    return mean_absolute_error


def ordinal2completeness(array):
    import numpy as np
    completeness = np.count_nonzero(array > 0.5, axis=-1)
    return completeness


def multi_binarycrossentropy(y_true, y_pred):
    import tensorflow as tf
    # May need manually set for simplicity, otherwise you may need to warp this function with a new function.
    y_nums = tf.constant(100, dtype=tf.int64)
    # change y_true to int used for indexing
    y_true = tf.cast(y_true, tf.int64)
    # indexing row of y_pred by action index stored in y_true. [batch_size, action_num, y_nums] --> [batch_size, y_nums]
    y_pred = tf.gather_nd(y_pred, y_true[:, 0, :], batch_dims=1)
    # convert int completeness to ordinal vector. [batch_size, 2, 1] --> [batch_size, y_nums]
    y_true = tf.map_fn(
        lambda x: tf.concat([tf.repeat(tf.constant(1, dtype=tf.int64), x), tf.repeat(tf.constant(0, dtype=tf.int64), y_nums - x)], axis=0),
        y_true[:, 1, :])
    multi_ordinal_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return multi_ordinal_loss


def multi_od_metric(y_true, y_pred):
    import tensorflow as tf
    # change y_true to int used for indexing
    y_true = tf.cast(y_true, tf.int64)
    # ordinal to int. [batch_size, action_num, y_nums] --> [batch_size, action_num]
    y_pred = tf.math.count_nonzero(y_pred > 0.5, axis=-1, dtype=tf.dtypes.float64)
    # indexing row of y_pred by action index.  [batch_size, action_num] --> [batch_size]
    y_pred = tf.gather_nd(y_pred, y_true[:, 0, :], batch_dims=1)
    # only keep completeness values of y_true. [batch_size, 2, 1] --> [batch_size]
    y_true = tf.squeeze(y_true[:, 1, :])
    # # change back y_true to float used for compute loss
    y_true = tf.cast(y_true, tf.float64)
    multi_mae_od = tf.math.abs(y_true - y_pred)
    return multi_mae_od


def action_search(completeness_array, min_T, max_T, min_L):
    import numpy as np
    """
    Detect (temporal localization) complete action on completeness list.
    :param completeness_array: Numpy Array. List of float numbers, completeness of frames
    :param min_T: Int. Minimum completeness value threshold used to find end frame candidates.
    :param max_T: Int. Maximum completeness value threshold used to find start frame candidates.
    :param min_L: Int. Minimum complete action length used
    :return: List. List of list. each list represent a detected action illustrated as [start_inx(int) end_inx(int) loss(float)]
    Examples:
    min_T, max_T, min_L = 75, 20, 35
    """

    P = completeness_array.squeeze()
    C_startframe = np.where(P < max_T)[0]  # "C_" represent variable for candidates.
    C_endframe = np.where(P > min_T)[0]
    action_detected = []
    for s_i in C_startframe:
        for e_i in C_endframe:
            C_action_length = e_i - s_i + 1
            if C_action_length > min_L:
                if not action_detected:
                    iou_vector = np.array([0])
                else:
                    iou_vector = matrix_iou(np.array([[s_i, e_i]]), np.array(action_detected)).squeeze(axis=0)
                if iou_vector.max() < 0.95:
                    action_template = np.linspace(0, 100, C_action_length)
                    predicted_sequence = P[s_i:e_i + 1]
                    mse = ((action_template - predicted_sequence) ** 2).mean()
                    action_candidate = [s_i, e_i, mse]
                    if mse < 833:
                        if iou_vector.max() == 0:
                            action_detected.append(action_candidate)
                            continue
                        else:
                            intersection_actions_idx = np.argwhere(iou_vector > 0).squeeze(axis=-1)
                            beat_any_one = False
                            for i in intersection_actions_idx:
                                if action_candidate[2] < action_detected[i][2]:
                                    beat_any_one = True
                                    action_detected.pop(i)
                            if beat_any_one:
                                action_detected.append(action_candidate)
    action_detected.sort(key=lambda x: x[2])
    return np.array(action_detected).reshape(-1, 3)


def action_ap(predictions, temporal_annotations, IoU, min_T=60, max_T=30, min_L=60, down_sampling=1, action_idx=None, return_detections=True):
    """
    Calculate average precision for one action class.
    :param predictions: Dict {Video name(Str): Array}. Predicted action progression sequences of test videos
    :param temporal_annotations: Array. Shape [N, 3]. First column for video names, 2nd and 3nd for start and end frames indexs.
    :param IoU: Float. IoU thresholds.
    :param min_T: Int or Float. Arguments for function "action search". Minimum action progression threshold for ending frames.
    :param max_T: Int or Float. Arguments for function "action search". Maximum action progression threshold for starting frames.
    :param min_L: Int or Float. Arguments for function "action search". Minimum action length threshold.
    :param down_sampling: Int. If larger than 1, then down-sampling the predictions sequence.
    :param action_idx: Int or None. If not None, which means the prediction arrays are for multi-tasks, then
    a slicing operation  will be conducted on the prediction array to get prediction of specific action by action_idx.
    :param return_detections: Boolean. If True, then detections and ground truth will also be returned.
    :return: Float or Tuple. If return_detections is True then return (ap(Float), detections(Dict), ground_truths(Dict))
    else return ap(Float).
    """
    import numpy as np
    video_list = np.unique(temporal_annotations[:, 0])
    detections = {}
    grount_truths = {}
    true_positives = {}

    for v in video_list:
        v_temporal_annotations = temporal_annotations[temporal_annotations[:, 0] == v][:, 1:]
        v_predictions = predictions[v]

        if action_idx is not None:
            v_predictions = v_predictions[:, action_idx]
        if down_sampling > 1:
            v_predictions = v_predictions[::down_sampling]
            v_temporal_annotations = v_temporal_annotations // down_sampling

        v_detections = action_search(v_predictions, min_T=min_T, max_T=max_T, min_L=min_L)
        v_true_positives = calc_truepositive(v_detections, v_temporal_annotations, IoU)
        detections[v] = v_detections
        true_positives[v] = v_true_positives
        grount_truths[v] = v_temporal_annotations

    num_of_ground_truth = temporal_annotations.shape[0]
    losses = np.vstack(list(detections.values()))[:, 2]
    true_positives_array = np.hstack(list(true_positives.values()))
    ap = average_precision(true_positives_array, num_of_ground_truth, losses)

    if return_detections:
        return ap, detections, grount_truths
    else:
        return ap


def imshow(array, **kwargs):
    from matplotlib import pyplot as plt
    img = (array + 1) / 2
    plt.figure()
    plt.imshow(img, **kwargs)
    plt.show()
