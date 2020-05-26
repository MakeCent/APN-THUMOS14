#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : Train.py.py
# Author: Chongkai LU
# Date  : 3/29/2020

import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
# from create_model import create_model
from tensorflow.keras.applications import ResNet50
from load_data import load_trimmed_images
from utilities import *
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras_preprocessing.image import ImageDataGenerator

# Hyper-parameters that mush be check before training
action = "BaseballPitch"  # action names
loss = 'mae'
y_range = (0, 100)  # range of progression-label
model_name = "resnet50_imagenet_{}_{}_{}".format(loss, *y_range)
formal_training = False     # if false then just a memo experiments will be conducted.
BATCH_SIZE = 32
EPOCHS = 30

# Define default files path
train_directory = "/mnt/louis-consistent/Datasets/THUMOS14/Validation"
test_directory = "/mnt/louis-consistent/Datasets/THUMOS14/Test"
train_ground_truth_path = "/mnt/louis-consistent/Datasets/THUMOS14/TH14_Temporal_annotations_validation/annotationF/{}_valF.csv".format(action)
test_ground_truth_path = "/mnt/louis-consistent/Datasets/THUMOS14/TH14_Temporal_annotations_test/annotationF/{}_testF.csv".format(action)
output_path = '/mnt/louis-consistent/Saved/THUMOS14_output'  # Directory to save model and fit history
Path(output_path, action, 'History', model_name).mkdir(parents=True, exist_ok=True)  # Create folders
Path(output_path, action, 'Model', model_name).mkdir(parents=True, exist_ok=True)

# Load train data
train_X, train_y, test_X, test_y = load_trimmed_images(train_ground_truth_path, test_ground_truth_path, train_directory,
                                                       test_directory, *y_range)

# Formal training or tiny training that just for testing if code works
if not formal_training:
    train_X, train_y = np.random.choice(train_X, size=500), np.random.choice(train_y, size=500)
    test_X, test_y = np.random.choice(test_X, size=50), np.random.choice(test_y, size=50)
    EPOCHS = 5
else:
    pass

# Build datagenerators
train_df = pd.DataFrame({'paths': train_X, 'labels': train_y})
test_df = pd.DataFrame({'paths': test_X, 'labels': test_y})
train_datagenerator = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagenerator = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagenerator.flow_from_dataframe(train_df, x_col='paths', y_col='labels',
                                                          target_size=(224, 224), class_mode='other',
                                                          batch_size=BATCH_SIZE)
test_generator = test_datagenerator.flow_from_dataframe(test_df, x_col='paths', y_col='labels', target_size=(224, 224),
                                                        class_mode='other', batch_size=BATCH_SIZE)
STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VAL = test_generator.n // test_generator.batch_size

# Build model
n_mae = normalize_mae(y_range[1] - y_range[0])  # custom metric for mae, make mae loss normalized into range 0 - 100.
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    res_net = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')
    parallel_model = build_model(res_net, dense_layer=(64, 32), out_activation=None)
optimizer = tf.keras.optimizers.Adam(0.0001, decay=1e-3 / STEP_SIZE_TRAIN)
parallel_model.compile(loss=loss, optimizer=optimizer, metrics=[n_mae])
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    Path(output_path, action, 'Model', model_name, model_name + '-{epoch:02d}-{val_n_mae:.2f}.h5').__str__(),
    monitor='val_n_mae', verbose=1)

# Start to train
history = parallel_model.fit(x=train_generator, steps_per_epoch=STEP_SIZE_TRAIN, validation_data=test_generator,
                             validation_steps=STEP_SIZE_VAL, epochs=EPOCHS, verbose=1,
                             callbacks=[model_checkpoint])

# Save scores and history image
scores = min(history.history['val_n_mae'])
history_pd = pd.DataFrame(data=history.history)
history_pd.to_csv(Path(output_path, action, 'History', model_name, 'History_{}.csv'.format(model_name)))
print("Validation Loss: {:.2f}".format(scores))
plot_history(history, ['loss'],
             figname=Path(output_path, action, 'History', model_name, 'History_{}_loss.png'.format(model_name)))
plot_history(history, ['n_mae'],
             figname=Path(output_path, action, 'History', model_name, 'History_{}_nmae.png'.format(model_name)))
