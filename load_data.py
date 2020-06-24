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


def read_from_annfile(root, annfile, y_range):
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
    img_paths, labels = [], []
    for i_r, row in temporal_annotations.iterrows():
        action_length = row.values[2] + 1 - row.values[1]
        img_paths.extend(["{}/{}/{}.jpg".format(root, row.values[0], str(num).zfill(5)) for num in
                          np.arange(row.values[1], row.values[2] + 1)])
        labels.extend(np.linspace(*y_range, num=action_length, dtype=np.float32))
    return img_paths, labels


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
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    if label is None:
        return img
    else:
        return img, label


def build_dataset_from_slices(imgs_list, labels_list=None, batch_size=32, augment=None, shuffle=True, prefetch=True):
    """
    Given image paths and labels, create tf.data.Dataset instance.
    :param imgs_list: List. Consists of strings. Each string is a path of one image.
    :param labels_list: List. Consists of labels. None means for only prediction
    :param transform: List. transform functions applied on images.
    :return: tf.data.Dataset. if labels_list is provided, will produce a labeled dataset. Images dataset otherwise.
    """
    import tensorflow as tf
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    if labels_list is None:
        dataset = tf.data.Dataset.from_tensor_slices(imgs_list)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((imgs_list, labels_list))
    if shuffle:
        dataset = dataset.shuffle(len(imgs_list))
    dataset = dataset.map(decode_img, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(format_img, num_parallel_calls=AUTOTUNE)
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
    vp = Path(video_path)
    imgs_list = [str(jp) for jp in sorted(vp.glob('*.{}'.format(suffix)))]
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

