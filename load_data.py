#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : load_data.py
# Author: LU Chongkai
# Date  : 23/5/2019

# %% Special Function: Can only be used in this program. Most of them aim to make main file more concise.


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


def dataset_trimmed(root, annfile, y_range, target_size=(224, 224)):
    """
    Create tf.data.Dataset according to image paths and labels paths. This is for training
    :param root: String. Directory where all images are stored.
    :param annfile: annfile: String. Path where temporal annotation locates.
    :param y_range: Tuple. Label range.
    :param target_size: Tuple. image size to be resized.
    :return: tf.data.Dataset. Contain all trimmed images and labels. But not prepared for training yet, need batch,
    shuffle, cache, prefetch ...
    """
    import tensorflow as tf
    imgs_list, labels_list = read_from_annfile(root, annfile, y_range)

    def transfroms_func(x):
        x = tf.image.resize(x, target_size)
        x = tf.image.random_flip_left_right(x)
        return x

    ds = build_dataset(imgs_list, labels_list, transform=transfroms_func)
    return ds


def dataset_single_video(video_path, target_size=(224, 224)):
    """
    Used for prediction. Create tf.data.Dataset contains all images in a video path. Hence this is untrimmed.
    :param video_path: String. Path where all video images locate.
    :param target_size: Tuple. image size to be resized.
    :return: tf.data.Dataset. Contain all images in that given video path
    """
    import tensorflow as tf
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    imgs_list = find_imgs(video_path)
    ds = build_dataset(imgs_list, transform=format_data)
    ds = ds.batch(128)
    # ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


# %% Basic function: Can be easily used in other programs.


def decode_img(file_path):
    import tensorflow as tf
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def build_dataset(imgs_list, labels_list=None, transform=None):
    """
    Given image paths and labels, create tf.data.Dataset instance.
    :param imgs_list: List. Consists of strings. Each string is a path of one image.
    :param labels_list: List. Consists of labels.
    :param transform: Function. transform function applied on images.
    :return: tf.data.Dataset. if labels_list is provided, will produce a labeled dataset. Images dataset otherwise.
    """
    import tensorflow as tf
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    img_ds = tf.data.Dataset.from_tensor_slices(imgs_list)
    img_ds = img_ds.map(decode_img, num_parallel_calls=AUTOTUNE)
    ds = img_ds.map(transform, num_parallel_calls=AUTOTUNE)
    if labels_list:
        labels_ds = tf.data.Dataset.from_tensor_slices(labels_list)
        ds = tf.data.Dataset.zip((ds, labels_ds))
    return ds


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


def format_data(image, label):
    import tensorflow as tf
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (224, 224))
    return image, label
