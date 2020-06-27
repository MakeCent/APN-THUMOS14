#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : load_data.py
# Author: LU Chongkai
# Date  : 23/5/2019

# %% Special Function: Can only be used in this program. Most of them aim to make main file more concise.


def format_img(image, label=None):
    import tensorflow as tf
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (224, 224))
    if label is None:
        return image
    else:
        return image, label


def read_from_annfile(root, annfile, y_range, mode='rgb', orinal=False, stack_length=10):
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
    y_nums = y_range[1] - y_range[0] + 1

    def generate_labels(length):
        completeness = np.linspace(*y_range, num=length, dtype=np.float32)
        if orinal:
            rounded_completeness = np.round(completeness)
            ordinal_completeness = np.array([[1]*int(c) + [0]*int(y_nums-c) for c in rounded_completeness])
            return ordinal_completeness
        else:
            return completeness

    if mode == 'rgb':
        img_paths, labels = [], []
        for i_r, row in temporal_annotations.iterrows():
            action_length = row.values[2] + 1 - row.values[1]
            img_paths.extend(["{}/{}/{}.jpg".format(root, row.values[0], str(num).zfill(5)) for num in
                              np.arange(row.values[1], row.values[2] + 1)])
            labels.extend(generate_labels(action_length))
        return img_paths, labels

    elif mode == 'flow':
        flow_paths, labels = [], []
        for i_r, row in temporal_annotations.iterrows():
            flow_paths.append(["{}/{}/{}/{}_{}.jpg".format(root, row.values[0], d, d, str(num+1).zfill(5)) for num in
                               np.arange(row.values[1], row.values[2]) for d in ['flow_x', 'flow_y']])
        stacked_flow_list = []
        for v_fl in flow_paths:
            v_stacked_flow = [v_fl[2*i:2*i+stack_length*2] for i in range(0, len(v_fl)//2 - stack_length + 1)]
            labels.extend(generate_labels(len(v_stacked_flow)))
            stacked_flow_list.extend(v_stacked_flow)
        return stacked_flow_list, labels


# %% Basic function: Can be easily used in other programs.


def decode_img(file_path, label=None):
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
    else:
        return img, label


def build_dataset_from_slices(data_list, labels_list=None, batch_size=32, augment=None, shuffle=True, prefetch=True):
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
    else:
        dataset = tf.data.Dataset.from_tensor_slices((data_list, labels_list))
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
