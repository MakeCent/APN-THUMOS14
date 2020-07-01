#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : load_data.py
# Author: LU Chongkai
# Date  : 23/5/2019

# %% Special Function: Can only be used in this program. Most of them aim to make main file more concise.


def format_img(image, label=None, weight=None):
    import tensorflow as tf
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (224, 224))
    if label is None:
        return image
    elif weight is None:
        return image, label
    else:
        return image, label, weight


def generate_labels(length, y_range, ordinal=False, multi_action=False, action_index=0):
    """
    Given number of frames of a trimmed action video. Return a label array monotonically increasing in y_range.
    Mainly based on numpy.linspace function
    :param length:
        Int. Number of completeness labels need to be generate.
    :param y_range:
        Tuple. Range of the completeness label values vary in.
    :param ordinal:
        Boolean. If True, then each completeness label will be convert to a ordinal vector. e.g. 3 -> [1,1,1,0,0,...]
    :param multi_action:
        Boolean. Since in multi_task case ordinal label vector will be too large (20 X 100), hence take too much space
        to store. Therefore in this case each label will be set as single value even ordinal is True. The vector
        transfer will be conducted on loss and metric function instead. To identify the action class, each label will be
        along with a action index.
    :param action_index:
        Int. Identify the action class of correspond label
    :return:
        Array.
    """
    import numpy as np
    y_nums = y_range[1] - y_range[0] + 1
    completeness = np.linspace(*y_range, num=length, dtype=np.float32)
    if ordinal:
        rounded_completeness = np.round(completeness).astype(np.int)
        ordinal_completeness = np.array([[1] * int(c) + [0] * int(y_nums - c) for c in rounded_completeness],
                                        dtype=np.float32)
        if multi_action:
            return np.expand_dims(np.insert(rounded_completeness[..., np.newaxis], 0, action_index, axis=1), axis=-1)
        else:
            return ordinal_completeness
    else:
        if multi_action:
            return np.insert(completeness[..., np.newaxis], 1, action_index, axis=1)
        else:
            return completeness


def read_from_annfile(root, annfile, y_range, mode='rgb', ordinal=False,
                      stack_length=10, multi_action=False, action_index=0):
    """
    According to the temporal annotation file. Create list of image paths and list of labels.
    :param root: String. Directory where all images are stored.
    :param annfile: String. Path where temporal annotation locates.
    :param y_range: Tuple. Label range.
    :return: List, List. List of image paths and list of labels.
    """
    import pandas as pd
    import numpy as np
    temporal_annotations = pd.read_csv(annfile, header=None)

    if mode == 'rgb':
        img_paths, labels, weights = [], [], []
        for i_r, row in temporal_annotations.iterrows():
            action_length = row.values[2] + 1 - row.values[1]
            img_paths.extend(["{}/{}/{}.jpg".format(root, row.values[0], str(num).zfill(5)) for num in
                              np.arange(row.values[1], row.values[2] + 1)])
            labels.extend(generate_labels(action_length, y_range=y_range, ordinal=ordinal, multi_action=multi_action))
        return img_paths, labels
            labels.extend(label_func(action_length, y_range=y_range, ordinal=ordinal))
            if weighted:
                w_10 = [3, 2, 1, 1, 1, 1, 1, 1, 2, 3]
                weights.extend(np.hstack([w*p for w, p in zip(w_10, np.array_split(np.ones(action_length), 10))]))
        if weighted:
            return img_paths, labels, weights
        else:
            return img_paths, labels

    elif mode == 'flow':
        flow_paths, labels = [], []
        for i_r, row in temporal_annotations.iterrows():
            flow_paths.append(["{}/{}/{}/{}_{}.jpg".format(root, row.values[0], d, d, str(num + 1).zfill(5)) for num in
                               np.arange(row.values[1], row.values[2]) for d in ['flow_x', 'flow_y']])
        stacked_flow_list = []
        for v_fl in flow_paths:
            v_stacked_flow = [v_fl[2 * i:2 * i + stack_length * 2] for i in range(0, len(v_fl) // 2 - stack_length + 1)]
            labels.extend(generate_labels(len(v_stacked_flow), y_range, ordinal, multi_action, action_index))
            stacked_flow_list.extend(v_stacked_flow)
        return stacked_flow_list, labels


def read_from_anndir(root, anndir, y_range, mode='rgb', orinal=False, stack_length=10, multi_action=True):
    """
    According to the temporal annotation file. Create list of image paths and list of labels.
    :param root: String. Directory where all images are stored.
    :param anndir: String. Path where temporal annotation locates.
    :param y_range: Tuple. Label range.
    :return: List, List. List of image paths and list of labels.
    """
    from pathlib import Path

    action_idx = {'BaseballPitch': 0, 'BasketballDunk': 1, 'Billiards': 2, 'CleanAndJerk': 3, 'CliffDiving': 4,
                  'CricketBowling': 5, 'CricketShot': 6, 'Diving': 7, 'FrisbeeCatch': 8, 'GolfSwing': 9,
                  'HammerThrow': 10, 'HighJump': 11, 'JavelinThrow': 12, 'LongJump': 13, 'PoleVault': 14, 'Shotput': 15,
                  'SoccerPenalty': 16, 'TennisSwing': 17, 'ThrowDiscus': 18, 'VolleyballSpiking': 19}

    datalist, ylist = [], []
    for annfile in sorted(Path(anndir).iterdir()):
        action_name = str(annfile.stem).split('_')[0]
        action_list, label_list = read_from_annfile(root, str(annfile), y_range, mode, orinal, stack_length,
                                                    multi_action, action_idx[action_name])
        datalist.extend(action_list)
        ylist.extend(label_list)
    return datalist, ylist


# %% Basic function: Can be easily used in other programs.


def decode_img(file_path, label=None, weight=None):
    """
    Read image from path
    :param label: Unknown.
    :param file_path: String.
    :return: Image Tensor.
    """
    import tensorflow as tf
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img)
    img = tf.cast(img, tf.float32)
    if label is None:
        return img
    elif weight is None:
        return img, label
    else:
        return img, label, weight


def build_dataset_from_slices(data_list, labels_list=None, weighs=None, batch_size=32, augment=None, shuffle=True, prefetch=True):
    """
    Given image paths and labels, create tf.data.Dataset instance.
    :param data_list: List. Consists of strings. Each string is a path of one image.
    :param labels_list: List. Consists of labels. None means for only prediction
    :param transform: List. transform functions applied on images.
    :return: tf.data.Dataset. if labels_list is provided, will produce a labeled dataset. Images dataset otherwise.
    """
    import tensorflow as tf
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    if labels_list is None:
        dataset = tf.data.Dataset.from_tensor_slices(data_list)
    elif weighs is None:
        dataset = tf.data.Dataset.from_tensor_slices((data_list, labels_list))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((data_list, labels_list, weighs))
    if shuffle:
        dataset = dataset.shuffle(len(data_list))
    dataset = dataset.map(decode_img, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(format_img, num_parallel_calls=AUTOTUNE)
    if augment:
        dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    if prefetch:
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


def stack_optical_flow(flow_list, labels_list=None, batch_size=32, augment=None, shuffle=True, prefetch=True):
    import tensorflow as tf
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    if labels_list is None:
        dataset = tf.data.Dataset.from_tensor_slices(flow_list)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((flow_list, labels_list))
    if shuffle:
        dataset = dataset.shuffle(len(flow_list))

    def stack_decode_format(filepath_list, labels=None):
        filepath_list = tf.unstack(filepath_list, axis=-1)
        flow_snip = []
        for flow_path in filepath_list:
            decoded = decode_img(flow_path)
            flow_snip.append(format_img(decoded))
        parsed = tf.concat(flow_snip, axis=-1)
        if labels is None:
            return parsed
        else:
            return parsed, labels

    dataset = dataset.map(stack_decode_format, num_parallel_calls=AUTOTUNE)
    if augment:
        dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    if prefetch:
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


def find_imgs(video_path, suffix='jpg'):
    """
    Find all images in a given path.
    :param video_path: String. Target video folder
    :return: List. Consists of strings. Each string is a image path; Sorted.
    """
    from pathlib import Path
    if isinstance(video_path, str):
        video_path = Path(video_path)
    imgs_list = [str(jp) for jp in sorted(video_path.glob('*.{}'.format(suffix)))]
    return imgs_list


def find_flows(video_path, suffix='jpg', stack_length=10):
    """
    Find all images in a given path.
    :param video_path: String. Target video folder
    :return: List. Consists of strings. Each string is a image path; Sorted.
    """
    from pathlib import Path
    if isinstance(video_path, str):
        video_path = Path(video_path)
    flow_x = sorted(video_path.glob('flow_x/*.{}'.format(suffix)))
    flow_y = sorted(video_path.glob('flow_y/*.{}'.format(suffix)))
    v_fl = [str(e) for xy in zip(flow_x, flow_y) for e in xy]
    v_stacked_flow = [v_fl[2 * i:2 * i + stack_length * 2] for i in range(0, len(v_fl) // 2 - stack_length + 1)]
    return v_stacked_flow


def prepare_for_training(ds, batch_size, cache=True, shuffle_buffer_size=1000):
    """
    Given a tf.data.Dataset that contain all data but has not been prepared for training. This do preparation for it.
    :param ds: tf.data.Dataset. Contain all data and labels.
    :param batch_size: Int.
    :param cache: Boolean.
    :param shuffle_buffer_size: Int.
    :return: tf.data.Dataset. A dataset prepared for training. batched, shuffled, prefetch and cached(may)
    """
    import tensorflow as tf
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds
